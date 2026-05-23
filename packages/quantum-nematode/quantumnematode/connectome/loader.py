"""Connectome loader — read vendored *Nature* SI XLSX files into ``Connectome``.

Two public loaders:

- ``load_cook_2019_hermaphrodite()`` — reads Cook et al. 2019 SI 5
  (``data/connectome/cook_2019_si5_connectome_adjacency.xlsx``) and returns
  the full 302-neuron hermaphrodite connectome.
- ``load_witvliet_2021_adult()`` — reads Witvliet et al. 2021 dataset 8
  (``data/connectome/witvliet_2020_dataset8_adult.xlsx``) and returns the
  adult worm's nerve-ring subset (~150-200 neurons).

The Cook 2019 SI 5 sheet layout (per the file's TITLE AND LEGEND):

- One sheet per (sex x connection-type):
  ``hermaphrodite chemical``, ``herm gap jn symmetric``,
  ``herm gap jn asymmetric``, plus the male equivalents.
- Each sheet's adjacency matrix has:
    * Row 0: organ-zone groupings ("PHARYNX" etc.; spans multiple columns)
    * Row 1: blank
    * Row 2: postsynaptic neuron names (columns 3 onward)
    * Col 0: organ-zone groupings
    * Col 1: blank
    * Col 2: presynaptic neuron names (rows 3 onward)
    * Data block: rows 3+, cols 3+
- Cell values are *EM serial-section counts* (a measure of total
  connectivity that takes into account both synapse number AND synapse
  size, per Cook 2019's own documentation).

Cook 2019 uses zero-padded ventral-cord motor-neuron names
(``VC01``-``VC06``, ``VD07``-``VD13``, ``AS01``-``AS11``) but the project's
canonical naming (matching cect / WormAtlas) uses unpadded forms
(``VC1``-``VC6``, ``VD7``-``VD13``, ``AS1``-``AS11``). The loader
normalises padded names via the ``_unpad_neuron_name()`` helper.

The Cook 2019 sheet also contains non-neuron cells in its column headers:
muscles (``dBWML*``, ``vm*``), glia (``CEPshDL``, ``GLR*``), etc. These
are filtered out by intersecting with ``NEURON_CLASSIFICATION``; only the
302 hermaphrodite neurons make it into the returned ``Connectome``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from quantumnematode.connectome.model import (
    ChemicalSynapse,
    Connectome,
    GapJunction,
    Neuron,
)
from quantumnematode.connectome.neurons import (
    CANONICAL_NAME_ALIASES,
    NEURON_CLASSIFICATION,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Vendored data paths (resolved relative to repo root)
# ---------------------------------------------------------------------------

# This file lives at:
#   <repo>/packages/quantum-nematode/quantumnematode/connectome/loader.py
# Walk 4 parents up to reach <repo>:
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "data" / "connectome"

COOK_2019_HERMAPHRODITE_PATH = _DATA_DIR / "cook_2019_si5_connectome_adjacency.xlsx"
WITVLIET_2021_ADULT_PATH = _DATA_DIR / "witvliet_2020_dataset8_adult.xlsx"


# ---------------------------------------------------------------------------
# Name-normalisation
# ---------------------------------------------------------------------------

_ZERO_PAD_RE = re.compile(r"(\D+)0+(\d+)")


def _unpad_neuron_name(name: str) -> str:
    """Strip zero-padding from numerical suffixes (``VC01`` -> ``VC1``).

    Cook 2019 zero-pads ventral-cord motor-neuron suffixes; the project's
    canonical naming uses unpadded forms (matching cect + WormAtlas).
    """
    return _ZERO_PAD_RE.sub(r"\1\2", name)


def _canonicalise(name: str) -> str:
    """Apply zero-pad stripping and any explicit aliases."""
    unpadded = _unpad_neuron_name(name)
    return CANONICAL_NAME_ALIASES.get(unpadded, unpadded)


# ---------------------------------------------------------------------------
# Cook 2019 loader
# ---------------------------------------------------------------------------


def _parse_cook_2019_adjacency_sheet(
    df: pd.DataFrame,
    *,
    valid_neurons: set[str],
) -> dict[tuple[str, str], int]:
    """Walk a Cook-2019-shaped sheet and emit ``{(pre, post): weight}``.

    Drops non-neuron cells (muscles, glia, etc.) and zero-weight cells.
    """
    raw_post = df.iloc[2, 3:].tolist()
    raw_pre = df.iloc[3:, 2].tolist()

    post_neurons: list[str | None] = [
        _canonicalise(str(n)) if isinstance(n, str) else None for n in raw_post
    ]
    pre_neurons: list[str | None] = [
        _canonicalise(str(n)) if isinstance(n, str) else None for n in raw_pre
    ]

    edges: dict[tuple[str, str], int] = {}
    data = df.iloc[3:, 3:].to_numpy()

    for i, pre in enumerate(pre_neurons):
        if pre is None or pre not in valid_neurons:
            continue
        for j, post in enumerate(post_neurons):
            if post is None or post not in valid_neurons:
                continue
            val = data[i, j]
            try:
                weight = int(val)
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            edges[(pre, post)] = weight

    return edges


def _to_gap_junctions(
    edges: Iterable[tuple[tuple[str, str], int]],
) -> list[GapJunction]:
    """Convert directed-pair weights into canonical undirected gap junctions.

    Gap junctions are undirected; Cook 2019 reports them symmetrically (both
    ``A -> B`` and ``B -> A`` cells carry the same weight in the "symmetric"
    sheet; the "asymmetric" sheet captures any asymmetric junctions). We
    fold both directions into a single canonical
    ``(neuron_a, neuron_b)`` entry with ``neuron_a < neuron_b``.
    """
    canonical: dict[tuple[str, str], int] = {}
    for (a, b), weight in edges:
        if a == b:
            # Self-loop gap junction: vanishingly rare; skip.
            continue
        key = (a, b) if a < b else (b, a)
        canonical[key] = max(canonical.get(key, 0), weight)
    return [GapJunction(neuron_a=a, neuron_b=b, weight=w) for (a, b), w in canonical.items()]


def load_cook_2019_hermaphrodite() -> Connectome:
    """Load the Cook 2019 hermaphrodite connectome from vendored SI 5.

    Returns a ``Connectome`` with 302 neurons, all hermaphrodite chemical
    synapses (directed, weighted), and all gap junctions (undirected, in
    canonical form with ``neuron_a < neuron_b``). Both symmetric and
    asymmetric gap-junction sheets are merged.

    Output is deterministic: chemical synapses sorted by ``(pre, post)``,
    gap junctions sorted by ``(neuron_a, neuron_b)``, neurons keyed by
    canonical name.

    Raises
    ------
    FileNotFoundError
        If the vendored XLSX file isn't present (e.g. LFS-pull hasn't run).
    ValueError
        If the loaded data fails Pydantic validation (e.g. references a
        neuron name not in ``NEURON_CLASSIFICATION``).
    """
    if not COOK_2019_HERMAPHRODITE_PATH.is_file():
        msg = (
            f"Cook 2019 SI 5 not found at {COOK_2019_HERMAPHRODITE_PATH}. "
            "Run `git lfs pull` to fetch the vendored data."
        )
        raise FileNotFoundError(msg)

    valid = set(NEURON_CLASSIFICATION)

    def _read_sheet(name: str) -> pd.DataFrame:
        df = pd.read_excel(
            COOK_2019_HERMAPHRODITE_PATH,
            engine="openpyxl",
            sheet_name=name,
            header=None,
        )
        # pd.read_excel can return a dict when sheet_name is a list; with a
        # single string it always returns a single DataFrame, but pyright
        # can't narrow that, so assert it here.
        if not isinstance(df, pd.DataFrame):
            msg = f"Expected DataFrame from sheet {name!r}, got {type(df).__name__}"
            raise TypeError(msg)
        return df

    chem_edges = _parse_cook_2019_adjacency_sheet(
        _read_sheet("hermaphrodite chemical"),
        valid_neurons=valid,
    )
    gj_sym_edges = _parse_cook_2019_adjacency_sheet(
        _read_sheet("herm gap jn symmetric"),
        valid_neurons=valid,
    )
    gj_asym_edges = _parse_cook_2019_adjacency_sheet(
        _read_sheet("herm gap jn asymmetric"),
        valid_neurons=valid,
    )

    # Build deterministically-sorted edge lists
    chemical_synapses = [
        ChemicalSynapse(pre=pre, post=post, weight=w)
        for (pre, post), w in sorted(chem_edges.items())
    ]

    combined_gj = list(gj_sym_edges.items()) + list(gj_asym_edges.items())
    gap_junctions = sorted(
        _to_gap_junctions(combined_gj),
        key=lambda gj: (gj.neuron_a, gj.neuron_b),
    )

    # Build the neuron dict — keyed by canonical name, classification from
    # the static table
    neurons = {
        name: Neuron(name=name, cell_class=cls, neurotransmitter=nt)
        for name, (cls, nt) in sorted(NEURON_CLASSIFICATION.items())
    }

    return Connectome(
        neurons=neurons,
        chemical_synapses=chemical_synapses,
        gap_junctions=gap_junctions,
        source="cook_2019_hermaphrodite",
        version="Cook et al. 2019 Nature, SI 5 (vendored snapshot)",
    )


# ---------------------------------------------------------------------------
# Witvliet 2021 loader
# ---------------------------------------------------------------------------


def _find_witvliet_column(
    cols_lower: dict[str, object],
    candidates: tuple[str, ...],
) -> object:
    """Look up a column by lower-cased name from a candidate list."""
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    msg = (
        f"Witvliet 2021 dataset: expected one of {candidates!r} in columns, "
        f"got {list(cols_lower.values())!r}."
    )
    raise ValueError(msg)


def _coerce_weight(value: object) -> int | None:
    """Coerce a Witvliet weight cell to ``int`` or ``None`` if not coercible."""
    if isinstance(value, bool):
        # Reject bool early — int(True) == 1 would otherwise silently survive.
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_witvliet_edge_list(
    df: pd.DataFrame,
    *,
    valid_neurons: set[str],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Parse a Witvliet long-format sheet into (chemical_edges, gj_edges).

    Witvliet 2021 publishes connectomes as long-format edge lists rather
    than adjacency matrices. Columns are (pre, post, type, weight) or
    similar — this sniffs the header row to find the relevant columns.
    """
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    col_pre = _find_witvliet_column(
        cols_lower,
        ("pre", "presynaptic", "from", "source", "neuron1"),
    )
    col_post = _find_witvliet_column(
        cols_lower,
        ("post", "postsynaptic", "to", "target", "neuron2"),
    )
    col_type = _find_witvliet_column(
        cols_lower,
        ("type", "synapse_type", "connection_type", "syntype"),
    )
    col_weight = _find_witvliet_column(
        cols_lower,
        ("weight", "synapses", "count", "n_synapses"),
    )

    chem_edges: dict[tuple[str, str], int] = {}
    gj_edges: dict[tuple[str, str], int] = {}

    for _, row in df.iterrows():
        raw_pre = row[col_pre]
        raw_post = row[col_post]
        if not isinstance(raw_pre, str) or not isinstance(raw_post, str):
            continue
        pre = _canonicalise(raw_pre.strip())
        post = _canonicalise(raw_post.strip())
        if pre not in valid_neurons or post not in valid_neurons:
            continue

        weight = _coerce_weight(row[col_weight])
        if weight is None or weight <= 0:
            continue

        type_raw = row[col_type]
        if not isinstance(type_raw, str):
            continue
        type_lc = type_raw.strip().lower()

        if "chem" in type_lc:
            chem_edges[(pre, post)] = chem_edges.get((pre, post), 0) + weight
        elif "elec" in type_lc or "gap" in type_lc:
            gj_edges[(pre, post)] = gj_edges.get((pre, post), 0) + weight

    return chem_edges, gj_edges


def load_witvliet_2021_adult() -> Connectome:
    """Load the Witvliet 2021 adult connectome (dataset 8).

    Returns a ``Connectome`` covering the adult worm's nerve ring
    (~150-200 of the 302 whole-animal neurons). Used for cross-validation
    against the Cook 2019 hermaphrodite connectome.

    Raises
    ------
    FileNotFoundError
        If the vendored XLSX file isn't present.
    ValueError
        If the sheet structure doesn't match the expected long-format
        edge-list layout, or if the loaded data fails Pydantic validation.
    """
    if not WITVLIET_2021_ADULT_PATH.is_file():
        msg = (
            f"Witvliet 2021 dataset 8 not found at {WITVLIET_2021_ADULT_PATH}. "
            "Run `git lfs pull` to fetch the vendored data."
        )
        raise FileNotFoundError(msg)

    valid = set(NEURON_CLASSIFICATION)

    df = pd.read_excel(WITVLIET_2021_ADULT_PATH, engine="openpyxl")
    chem_edges, gj_edges = _parse_witvliet_edge_list(df, valid_neurons=valid)

    chemical_synapses = [
        ChemicalSynapse(pre=pre, post=post, weight=w)
        for (pre, post), w in sorted(chem_edges.items())
    ]
    gap_junctions = sorted(
        _to_gap_junctions(gj_edges.items()),
        key=lambda gj: (gj.neuron_a, gj.neuron_b),
    )

    # Only include neurons that participate in this dataset
    used_neurons: set[str] = set()
    for syn in chemical_synapses:
        used_neurons.add(syn.pre)
        used_neurons.add(syn.post)
    for gj in gap_junctions:
        used_neurons.add(gj.neuron_a)
        used_neurons.add(gj.neuron_b)

    neurons = {
        name: Neuron(
            name=name,
            cell_class=NEURON_CLASSIFICATION[name][0],
            neurotransmitter=NEURON_CLASSIFICATION[name][1],
        )
        for name in sorted(used_neurons)
    }

    return Connectome(
        neurons=neurons,
        chemical_synapses=chemical_synapses,
        gap_junctions=gap_junctions,
        source="witvliet_2021_adult_dataset8",
        version="Witvliet et al. 2021 Nature, dataset 8 (vendored snapshot)",
    )
