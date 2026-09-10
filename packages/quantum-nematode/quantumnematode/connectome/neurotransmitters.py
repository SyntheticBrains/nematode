"""Neurotransmitter identities and the synapse signs derived from them.

The 302-entry classification table in :mod:`quantumnematode.connectome.neurons` carries a
release identity per neuron, taken from the CRISPR knock-in neurotransmitter atlas vendored
under ``data/connectome/``. Those values are **committed literals**: this module's reader is
what generated them and what a test re-derives them with, so nothing at run time opens the
spreadsheet.

Sign, and its limits
--------------------
A synapse's sign is set by the **post-synaptic receptor**, not by the transmitter the
pre-synaptic cell releases. Glutamate is excitatory through non-NMDA receptors and inhibitory
through glutamate-gated chloride channels; GABA is inhibitory at most sites and excitatory at
some neuromuscular junctions. :data:`TRANSMITTER_SIGN` is therefore a **per-neuron
approximation** — the sign a synapse most often has given what its source releases — and not a
claim about any individual synapse. Receptor-class expression refines it to a per-synapse
model; until then a neuron whose transmitter does not imply a sign leaves its synapses on the
sign they were initialised with.

Uptake is not release: an atlas entry qualified as uptake records that the cell takes a
transmitter up, which gives no release identity and therefore no sign.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from collections.abc import Iterator

ATLAS_FILENAME = "elife-95402-supp2-v1.xlsx"
ATLAS_PATH = Path(__file__).resolve().parents[4] / "data" / "connectome" / ATLAS_FILENAME
ATLAS_SHA256 = "0013e4b5f366b82a6b0ec0d682c3bace4027545c957823841293de93feafc0e2"
ATLAS_SHEET = "Supp File 2"
_HEADER_ROW = 3  # zero-based; row 4 in the sheet
_NEURON_COL = 2
_TRANSMITTER_COL = 20
# The sheet's "Neurotransmitter(s)" heading spans three columns: the primary identity and up to
# two co-transmitters. Reading only the first is why ADF and HSN look purely cholinergic and RIM
# purely glutamatergic -- their serotonergic and tyraminergic release identities sit here.
_IDENTITY_COLS = (20, 21, 22)

# Aminergic release identities. These carry no fast synaptic sign (they are absent from
# TRANSMITTER_SIGN) and are the sources of the modulatory pathways the wiring routes.
AMINERGIC = frozenset({"DA", "5-HT", "octopamine", "tyramine"})

# Markers the atlas itself writes on an identity that is something other than plain release.
# Order matters and each is deliberate:
#
# "alternative synthesis"  the atlas's own hedge -- I5, VC4, VC5 carry "5-HT (alternative
#                          synthesis/uptake mechanism)". Checked BEFORE the uptake rule, whose
#                          synthesis clause would otherwise admit them.
# "male"                   PVW's serotonergic note is sex-specific and questioned in the sheet.
_HEDGED_QUALIFIERS = ("alternative synthesis", "male")
# A precursor is not the transmitter: MI carries "5-HTP", the molecule serotonin is made from.
_PRECURSOR_LABELS = frozenset({"5-HTP"})

SynapseSign = Literal[1, -1]

# The atlas writes two ventral-cord motor neurons under joint labels; the project's canonical
# names are the unjoined ones. Mapped explicitly so a future atlas revision cannot drop them
# silently — the loader raises if either disappears.
ATLAS_NAME_MAP: dict[str, str] = {"DB1/3": "DB1", "DB3/1": "DB3"}

# Release identity -> the sign its synapses are given. Absent means "no sign": the monoamines,
# which modulate rather than transmit fast, and every orphan or uptake-only identity.
TRANSMITTER_SIGN: dict[str, SynapseSign] = {
    "ACh": 1,
    "Glu": 1,
    "GABA": -1,
}

_ANNOTATION = re.compile(r"\s*-\s*NEW$", flags=re.IGNORECASE)


def normalise_identity(raw: str) -> tuple[str, str | None]:
    """Split an atlas transmitter cell into ``(base label, qualifier)``.

    The atlas annotates its identities: a leading ``*`` marks a footnote, a trailing ``- NEW``
    marks an identity this study added, case varies on the orphan labels, and a parenthetical
    carries qualifiers such as ``(uptake)`` or ``(orphan, unc-47 expression)``.
    """
    text = _ANNOTATION.sub("", str(raw).strip().lstrip("*").strip()).strip()
    qualifier: str | None = None
    if "(" in text:
        base, _, rest = text.partition("(")
        qualifier = rest.rstrip(")").strip() or None
        text = base.strip()
    label = text.strip()
    if "unknown" in label.lower():
        # The sheet writes the hedge in more than one position: "unknown (orphan)" and
        # URB's "bas-1-depen unknown monoamine?" are both identities the atlas declines to name.
        label = "unknown"
    return label, qualifier


def release_identity(raw: str) -> str | None:
    """Return the transmitter a cell releases, or ``None`` when the atlas records none.

    Orphan labels give none. Uptake-only entries give none, but "synthesis + uptake" does: a
    neuron that makes its transmitter and also recovers it is releasing it.
    """
    label, qualifier = normalise_identity(raw)
    if not label or label == "unknown" or label in _PRECURSOR_LABELS:
        return None
    marked = f"{label} {qualifier or ''}".lower()
    if any(mark in marked for mark in _HEDGED_QUALIFIERS):
        # The hedge may sit inside the parenthesis ("alternative synthesis/uptake mechanism")
        # or outside it ("male - 5-HT (...)"), so both halves are searched.
        return None
    if qualifier and "uptake" in qualifier.lower() and "synthesis" not in qualifier.lower():
        # Uptake ALONE is not release: AIM and RIH carry "5-HT (uptake)" and take the amine up
        # without making it. "synthesis + uptake" -- ADF's serotonin, RIM's tyramine -- is a
        # neuron that both makes and recovers its transmitter, which is release.
        return None
    return label


def sign_for(transmitter: str | None) -> SynapseSign | None:
    """Return the sign a release identity implies, or ``None`` when it implies none."""
    return TRANSMITTER_SIGN.get(transmitter) if transmitter else None


def _rows(path: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    import openpyxl

    if not path.is_file():
        msg = f"Neurotransmitter atlas not found at {path}. Run `git lfs pull` to fetch it."
        raise FileNotFoundError(msg)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[ATLAS_SHEET]
        for index, row in enumerate(worksheet.iter_rows(values_only=True)):
            if index <= _HEADER_ROW:
                continue
            name = row[_NEURON_COL]
            if not name:
                continue
            cells = tuple(
                "" if len(row) <= col or row[col] is None else str(row[col])
                for col in _IDENTITY_COLS
            )
            yield str(name).strip(), cells
    finally:
        # A read-only workbook holds the file open; close it on normal completion and on an
        # early generator exit alike.
        workbook.close()


def read_atlas_identities(path: Path = ATLAS_PATH) -> dict[str, tuple[str, ...]]:
    """Read the atlas into ``{canonical neuron name: every release identity, primary first}``.

    A neuron may release more than one transmitter: ADF and HSN release serotonin beside
    acetylcholine, RIM tyramine beside glutamate. The sheet writes the primary identity in one
    column and the co-transmitters in the next two, so all three are read. Uptake-only
    annotations, the atlas's own "alternative synthesis" hedges, its one precursor entry and its
    male-only note yield no release identity -- see ``_NON_RELEASE_QUALIFIERS``.
    """
    out: dict[str, tuple[str, ...]] = {}
    seen_mapped: set[str] = set()
    for raw_name, cells in _rows(path):
        name = ATLAS_NAME_MAP.get(raw_name, raw_name)
        if raw_name in ATLAS_NAME_MAP:
            seen_mapped.add(raw_name)
        identities = tuple(
            identity for identity in (release_identity(cell) for cell in cells if cell) if identity
        )
        out[name] = identities
    missing = set(ATLAS_NAME_MAP) - seen_mapped
    if missing:
        msg = (
            f"Atlas no longer carries the joint labels {sorted(missing)}; the name map in "
            "ATLAS_NAME_MAP needs revisiting against the new revision."
        )
        raise ValueError(msg)
    return out


def read_atlas_transmitters(path: Path = ATLAS_PATH) -> dict[str, str | None]:
    """Read the atlas into ``{canonical neuron name: primary release identity or None}``.

    The primary identity is the one synapse signs are derived from; it is the first column's,
    unchanged by the co-transmitter columns this module also reads. Used to generate the
    committed table and, in tests, to prove the committed values still match the file.
    """
    # Derived from the FIRST identity column itself, not from the compacted list: if that
    # column's entry is one the exclusion rule drops, the neuron has no primary identity, and
    # taking `identities[0]` would silently promote a co-transmitter into the sign table.
    out: dict[str, str | None] = {}
    for raw_name, cells in _rows(path):
        name = ATLAS_NAME_MAP.get(raw_name, raw_name)
        out[name] = release_identity(cells[0]) if cells and cells[0] else None
    return out


def aminergic_neurons(identities: dict[str, tuple[str, ...]]) -> set[str]:
    """Return the neurons releasing an amine in any of their identities.

    These are the sources of the modulatory pathways: dopamine, serotonin, octopamine and
    tyramine. A neuron qualifies on any of its identities, so the cholinergic-and-serotonergic
    ADF and the glutamatergic-and-tyraminergic RIM are both in.
    """
    return {
        name
        for name, carried in identities.items()
        if any(identity in AMINERGIC for identity in carried)
    }


def instructed_neurons(connectome: object) -> set[str]:
    """Neurons receiving a chemical synapse from an aminergic neuron.

    This is a model of aminergic reach **by synaptic connectivity**, and it is a lower bound:
    dopamine, serotonin, octopamine and tyramine are released by volume onto receptors expressed
    by cells that need not be synaptic partners of the releasing neuron -- the "wireless"
    monoamine layer of the multilayer connectome (Bentley et al., PLoS Comput Biol, 2016). It is
    therefore a falsifiable proxy for which cells the amines instruct, not the set of them.

    Derived from the connectome passed in, so a rewired substrate yields its own pathway rather
    than inheriting the wild type's.
    """
    from quantumnematode.connectome.neurons import (
        NEURON_CLASSIFICATION,
        NEURON_CO_TRANSMITTERS,
    )

    sources = {
        name
        for name, (_cls, primary) in NEURON_CLASSIFICATION.items()
        if (primary in AMINERGIC)
        or any(identity in AMINERGIC for identity in NEURON_CO_TRANSMITTERS.get(name, ()))
    }
    synapses = connectome.chemical_synapses  # type: ignore[attr-defined]
    return {synapse.post for synapse in synapses if synapse.pre in sources}
