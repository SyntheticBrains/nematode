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
    if label.lower().startswith("unknown"):
        label = "unknown"
    return label, qualifier


def release_identity(raw: str) -> str | None:
    """Return the transmitter a cell releases, or ``None`` when the atlas records none.

    Uptake-qualified entries and the orphan labels give no release identity.
    """
    label, qualifier = normalise_identity(raw)
    if not label or label == "unknown":
        return None
    if qualifier and "uptake" in qualifier.lower():
        return None
    return label


def sign_for(transmitter: str | None) -> SynapseSign | None:
    """Return the sign a release identity implies, or ``None`` when it implies none."""
    return TRANSMITTER_SIGN.get(transmitter) if transmitter else None


def _rows(path: Path) -> Iterator[tuple[str, str]]:
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
            raw = row[_TRANSMITTER_COL] if len(row) > _TRANSMITTER_COL else None
            yield str(name).strip(), ("" if raw is None else str(raw))
    finally:
        # A read-only workbook holds the file open; close it on normal completion and on an
        # early generator exit alike.
        workbook.close()


def read_atlas_transmitters(path: Path = ATLAS_PATH) -> dict[str, str | None]:
    """Read the vendored atlas into ``{canonical neuron name: release identity or None}``.

    Used to generate the committed table and, in tests, to prove the committed values still
    match the file. Nothing on a run-time path calls it.
    """
    out: dict[str, str | None] = {}
    seen_mapped: set[str] = set()
    for raw_name, raw_identity in _rows(path):
        name = ATLAS_NAME_MAP.get(raw_name, raw_name)
        if raw_name in ATLAS_NAME_MAP:
            seen_mapped.add(raw_name)
        out[name] = release_identity(raw_identity)
    missing = set(ATLAS_NAME_MAP) - seen_mapped
    if missing:
        msg = (
            f"Atlas no longer carries the joint labels {sorted(missing)}; the name map in "
            "ATLAS_NAME_MAP needs revisiting against the new revision."
        )
        raise ValueError(msg)
    return out
