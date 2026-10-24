"""A neuromodulator layer as data: ligands, their producers, and the couples that carry them.

A layer has one pool per ligand. A ligand's concentration moves toward the mean positive settled
rate of its producer cells, with its own timescale. Every cell carrying one of its receptors reads
that pool. The layer's current on the network is a sum of rows over neurons, each row scaled by its
input and a strength:

- **couples:** a ligand acting through a receptor gene on the cells that express it, with a fixed
  sign (a softplus strength starting at 1) or a fitted one (a free signed strength starting at 0);
- **links:** a ligand acting on named cells with a fixed sign and no receptor gene, each tagged to
  the receptor gene whose loss removes it;
- **drives:** sensory inputs onto named cells, outside the ligands.

``load_data_layer`` reads the vendored layer placed from Bentley et al. 2016 and Ripoll-Sánchez et
al. 2023. ``hand_placed_spec`` writes the four-pathway serotonin and PDF circuit in the same form.
``receptor_placement_null`` moves each receptor gene's cells, class by class, to classes of the
same size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from quantumnematode.connectome.neuron_classes import class_members, neuron_classes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantumnematode.connectome.neuromodulators import ModulatorCircuit

LAYER_PATH = Path(__file__).resolve().parents[4] / "data" / "neuromodulators" / "layer.json"
DEFAULT_TIMESCALE_S = 60.0
LayerSensitivity = Literal["none", "ec50_500nM"]
LayerSource = Literal["bentley2016", "ripollsanchez2023"]
# The synthesis gene whose loss removes each monoamine; a peptide is removed by its own gene.
MONOAMINE_GENES = {
    "tph-1": "serotonin",
    "cat-2": "dopamine",
    "tbh-1": "octopamine",
    "tdc-1": "tyramine",
}


@dataclass(frozen=True)
class Row:
    """One row of the layer's current: an input acting with a strength on named cells.

    ``kind`` is ``couple``, ``link`` or ``drive``. A couple or link reads ligand ``ligand``'s pool;
    a drive reads sensory input ``drive``. ``sign`` is +1 or -1 for a fixed row and ``None`` for a
    fitted one. ``receptor`` is the gene whose knockout removes the row (a couple's own receptor, or
    the gene a link is tagged to).
    """

    kind: Literal["couple", "link", "drive"]
    cells: tuple[str, ...]
    sign: int | None
    ligand: int = -1
    drive: int = -1
    receptor: str | None = None
    name: str = ""


@dataclass(frozen=True)
class LayerSpec:
    """A neuromodulator layer on one connectome's neurons."""

    ligands: tuple[str, ...]
    producers: tuple[tuple[str, ...], ...]
    rows: tuple[Row, ...]
    drives: tuple[str, ...]
    skipped: tuple[str, ...] = ()

    @property
    def n_inputs(self) -> int:
        """Return the width of the layer's input: each ligand's concentration, then each drive."""
        return len(self.ligands) + len(self.drives)

    def input_index(self, row: Row) -> int:
        """Return a row's position in the input vector ``[concentrations..., drives...]``."""
        return row.ligand if row.kind != "drive" else len(self.ligands) + row.drive


def load_data_layer(  # noqa: C901 - one filter per option
    names: Sequence[str],
    path: Path = LAYER_PATH,
    *,
    sensitivity: LayerSensitivity = "none",
    sources: Sequence[LayerSource] | None = None,
) -> LayerSpec:
    """Read the vendored layer onto the neurons in ``names``.

    ``sensitivity="ec50_500nM"`` keeps only the couples within Ripoll-Sánchez's potency rule.
    ``sources`` keeps only the cells those sources place; a couple left with no cells, or a ligand
    with no producers, is dropped.
    """
    data = json.loads(path.read_text())
    present = set(names)
    keep = set(sources) if sources is not None else None
    skipped: set[str] = set()

    def cells_of(tagged: dict[str, list[str]]) -> tuple[str, ...]:
        out = []
        for cell, tags in sorted(tagged.items()):
            if keep is not None and not keep.intersection(tags):
                continue
            if cell in present:
                out.append(cell)
            else:
                skipped.add(cell)
        return tuple(out)

    producers = {name: cells_of(entry["producers"]) for name, entry in data["ligands"].items()}
    ligands = tuple(name for name in sorted(producers) if producers[name])
    index = {name: i for i, name in enumerate(ligands)}
    rows = []
    for couple in data["couples"]:
        if sensitivity == "ec50_500nM" and not couple["within_500nM"]:
            continue
        if couple["ligand"] not in index:
            continue
        cells = cells_of(couple["cells"])
        if not cells:
            continue
        rows.append(
            Row(
                kind="couple",
                cells=cells,
                sign=couple["sign"] if couple["sign_class"] == "fixed" else None,
                ligand=index[couple["ligand"]],
                receptor=couple["receptor"],
                name=couple["id"],
            ),
        )
    drives = tuple(d["name"] for d in data["drives"])
    for i, drive in enumerate(data["drives"]):
        cells = tuple(c for c in drive["cells"] if c in present)
        if cells:
            rows.append(Row(kind="drive", cells=cells, sign=1, drive=i, name=drive["name"]))
    return LayerSpec(
        ligands=ligands,
        producers=tuple(producers[name] for name in ligands),
        rows=tuple(rows),
        drives=drives,
        skipped=tuple(sorted(skipped)),
    )


def hand_placed_spec(circuit: ModulatorCircuit) -> LayerSpec:
    """Write the hand-placed serotonin and PDF circuit as a layer, row for row.

    The rows follow the hand-placed field's order: serotonin through MOD-1, PDF through PDFR-1,
    PDF's inhibition of NSM (a link tagged to pdfr-1), and the ingestion drive onto the serotonin
    cells.
    """
    return LayerSpec(
        ligands=("serotonin", "pdf"),
        producers=(circuit.serotonin_release, circuit.pdf_release),
        rows=(
            Row(
                "couple",
                circuit.mod1,
                circuit.mod1_sign,
                ligand=0,
                receptor="mod-1",
                name="serotonin:mod-1",
            ),
            Row(
                "couple",
                circuit.pdfr1,
                circuit.pdfr1_sign,
                ligand=1,
                receptor="pdfr-1",
                name="pdf:pdfr-1",
            ),
            Row("link", circuit.pdf_inhibits, -1, ligand=1, receptor="pdfr-1", name="pdf:NSM"),
            Row("drive", circuit.serotonin_release, 1, drive=0, name="ingestion"),
        ),
        drives=("ingestion",),
        skipped=circuit.skipped,
    )


def knocked_out(spec: LayerSpec, genes: Sequence[str]) -> tuple[frozenset[int], frozenset[int]]:
    """Return the ligand indices and row indices that ``genes``' knockouts remove.

    A producer gene holds its ligand at zero: a monoamine by its synthesis gene, a peptide by its
    own gene. A receptor gene removes every couple through it and every link tagged to it. A gene
    that is neither raises.
    """
    ligands: set[int] = set()
    rows: set[int] = set()
    for gene in genes:
        ligand = MONOAMINE_GENES.get(gene, gene)
        found = False
        if ligand in spec.ligands:
            ligands.add(spec.ligands.index(ligand))
            found = True
        hit = {i for i, row in enumerate(spec.rows) if row.receptor == gene}
        if hit:
            rows |= hit
            found = True
        if not found:
            msg = f"{gene!r} is neither a producer nor a receptor gene in this layer"
            raise ValueError(msg)
    return frozenset(ligands), frozenset(rows)


def receptor_placement_null(spec: LayerSpec, seed: int, names: Sequence[str]) -> LayerSpec:
    """Return the layer with each receptor gene's cells moved, class by class, to same-size classes.

    For each receptor gene, its cells are grouped by anatomical class: k cells of a class of size n.
    Each group moves to a class of size n drawn without replacement, landing on the member positions
    its source cells held (by suffix or digit, in member order). So every receptor gene keeps its
    class count and cell count, left/right pairs move together, and a gene keeps one placement
    across all its couples. Producers, links, drives and signs are held. A class whose size no other
    class shares maps to itself.
    """
    classes = neuron_classes(names)
    members = class_members(classes)
    by_size: dict[int, list[str]] = {}
    for neuron_class, cells in sorted(members.items()):
        by_size.setdefault(len(cells), []).append(neuron_class)
    rng = np.random.default_rng(seed)
    moved: dict[str, dict[str, str]] = {}
    for gene in sorted(
        {row.receptor for row in spec.rows if row.kind == "couple" and row.receptor},
    ):
        cells = sorted(
            {
                c
                for row in spec.rows
                if row.kind == "couple" and row.receptor == gene
                for c in row.cells
            },
        )
        source_classes = sorted({classes[c] for c in cells})
        targets: dict[str, str] = {}
        used: dict[int, set[str]] = {}
        for neuron_class in source_classes:
            size = len(members[neuron_class])
            pool = [c for c in by_size[size] if c not in used.setdefault(size, set())]
            target = pool[int(rng.integers(len(pool)))]
            used[size].add(target)
            targets[neuron_class] = target
        mapping = {}
        for cell in cells:
            source = classes[cell]
            position = members[source].index(cell)
            mapping[cell] = members[targets[source]][position]
        moved[gene] = mapping
    rows = tuple(
        replace(row, cells=tuple(sorted(moved[row.receptor][c] for c in row.cells)))
        if row.kind == "couple" and row.receptor in moved
        else row
        for row in spec.rows
    )
    return replace(spec, rows=rows)
