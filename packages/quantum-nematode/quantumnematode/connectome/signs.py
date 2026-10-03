"""Per-connection signs for the chemical synapses.

:data:`~quantumnematode.connectome.neurotransmitters.TRANSMITTER_SIGN` gives every synapse of a
neuron the same sign, read from what the neuron releases. A synapse's sign is set by the
post-synaptic receptor, so that rule is wrong wherever one transmitter acts through receptors of
both kinds: glutamate excites through AMPA-type receptors and inhibits through glutamate-gated
chloride channels, so AWC's glutamatergic synapses onto AIY are inhibitory although the rule signs
them positive.

This module signs each chemical connection separately, taking the first of four steps that gives a
sign:

1. **Physiology** — a cited measurement of that connection's effect, from the vendored table of
   overrides.
2. **Expression** — the polarity Fenyves et al. 2020 predict from the presynaptic transmitters and
   the postsynaptic receptor genes, used only where every transmitter their prediction rests on is
   one of the presynaptic cell's release identities in the atlas the rest of the package reads. They
   name a primary and sometimes a secondary transmitter per cell; a prediction is set aside if the
   primary is not a release identity, or if the secondary is not and the primary alone would not
   give the same polarity. Their tables come in two files, built on two reconstructions; where both
   predict a sign they must agree.
3. **Rule** — the per-neuron transmitter rule.
4. **None** — no fast sign: the presynaptic cell releases nothing the rule signs.

Nothing in the package reads these signs yet.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION, NEURON_CO_TRANSMITTERS
from quantumnematode.connectome.neurotransmitters import sign_for

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from collections.abc import Iterator

    from quantumnematode.connectome.model import Connectome

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "connectome"

FENYVES_S1_PATH = DATA_DIR / "fenyves_2020_s1_data.xlsx"
FENYVES_S1_SHA256 = "85959066fd7cbdbc2024d0ebb323b71c4365f4083bc85e555ee973f470697c47"
FENYVES_S1_SHEET = "5. Sign prediction"
FENYVES_S5_PATH = DATA_DIR / "fenyves_2020_s5_data.xlsx"
FENYVES_S5_SHA256 = "35902e0f43842ed25a65b4cc1c95028bcd73db121c6ad4d831ddc5ce5612ec81"
FENYVES_S5_SHEET = "5. Sign prediction (Cook)"
PHYSIOLOGY_OVERRIDES_PATH = DATA_DIR / "sign_overrides_physiology.csv"

# The sheets' layout: two header rows, then one row per connection with the source neuron in column
# A, its primary transmitter in B, the target in D, the edge type in F and the polarity in Q.
_FIRST_ROW = 2
_PRE_COL, _TRANSMITTER_COL, _SECONDARY_COL, _POST_COL, _TYPE_COL, _POLARITY_COL = 0, 1, 2, 3, 5, 16
# Columns G-L: whether the target expresses an excitatory (+) or inhibitory (-) receptor for each
# transmitter the source releases, primary or secondary.
_RECEPTOR_COLS: dict[str, tuple[int, int]] = {"Glu": (6, 7), "ACh": (8, 9), "GABA": (10, 11)}
_POLARITIES = frozenset({"+", "-", "complex", "no pred"})
_POLARITY_SIGN: dict[str, Literal[1, -1]] = {"+": 1, "-": -1}

# The sheets write ventral-cord motor neurons zero-padded (VB01); the package writes VB1.
_PADDED = re.compile(r"^(AS|DA|DB|DD|VA|VB|VC|VD)0*(\d+)$")

_OVERRIDE_HEADER = ("pre", "post", "sign", "citation", "evidence")

# Every source the physiology table cites, by the key its rows use.
PHYSIOLOGY_CITATIONS: dict[str, str] = {
    "chalasani2007": "Chalasani et al. 2007, Nature 450:63, doi:10.1038/nature06292",
    "huo2024": "Huo et al. 2024, PNAS 121:e2410789121, doi:10.1073/pnas.2410789121",
    "li2014": "Li et al. 2014, Cell 159:751, doi:10.1016/j.cell.2014.09.056",
    "lin2024": "Lin et al. 2024, Nature Communications 15:297, doi:10.1038/s41467-023-44638-5",
    "piggott2011": "Piggott et al. 2011, Cell 147:922, doi:10.1016/j.cell.2011.08.053",
    "roberts2016": "Roberts et al. 2016, eLife 5:e12572, doi:10.7554/eLife.12572",
    "wang2020": "Wang et al. 2020, eLife 9:e56942, doi:10.7554/eLife.56942",
    "zhang2025": "Zhang et al. 2025, Nature Communications 16:4405, doi:10.1038/s41467-025-59668-4",
}

SignSource = Literal["physiology", "expression", "rule", "none"]


class ConnectionSign(BaseModel):
    """The sign of one chemical connection and the step that gave it."""

    model_config = ConfigDict(frozen=True)

    pre: str = Field(..., min_length=1)
    post: str = Field(..., min_length=1)
    sign: Literal[1, 0, -1]
    source: SignSource
    citation: str | None = Field(
        default=None,
        description="The citation key, for a physiology sign; otherwise None.",
    )


class PhysiologyOverride(BaseModel):
    """One cited measurement of a chemical connection's sign."""

    model_config = ConfigDict(frozen=True)

    pre: str = Field(..., min_length=1)
    post: str = Field(..., min_length=1)
    sign: Literal[1, -1]
    citation: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)


class FenyvesSheet(BaseModel):
    """The predictions one Fenyves et al. 2020 sheet makes for the connectome's chemical edges."""

    model_config = ConfigDict(frozen=True)

    predictions: dict[tuple[str, str], str]
    transmitters: dict[str, str | None]
    ignored_rows: int = Field(..., ge=0, description="Rows naming an edge the connectome lacks.")
    secondaries: dict[str, str | None] = Field(
        default_factory=dict,
        description="Each source neuron's secondary transmitter, column C, or None.",
    )
    primary_only: dict[tuple[str, str], str] = Field(
        default_factory=dict,
        description="The polarity the receptor columns give for the primary transmitter alone.",
    )


def _unpad(name: object) -> str:
    text = str(name).strip()
    match = _PADDED.match(text)
    return f"{match.group(1)}{int(match.group(2))}" if match else text


def _check_digest(path: Path, expected: str) -> None:
    """Refuse a missing file, or one whose digest is not the vendored file's."""
    if not path.is_file():
        msg = f"{path.name} not found at {path}. Run `git lfs pull` to fetch the vendored data."
        raise FileNotFoundError(msg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        msg = (
            f"{path.name} has SHA256 {digest}, not the recorded {expected}; it is not the vendored "
            "file. If it is a Git LFS pointer, run `git lfs pull`."
        )
        raise ValueError(msg)


def _transmitter(raw: object) -> str | None:
    return None if raw in (0, None, "") else str(raw)


def _primary_only_polarity(row: tuple[object, ...], primary: str | None) -> str:
    """Return the polarity the sheet's formula gives when only the primary transmitter counts."""
    if primary not in _RECEPTOR_COLS:
        return "no pred"
    plus_col, minus_col = _RECEPTOR_COLS[primary]
    plus, minus = bool(row[plus_col]), bool(row[minus_col])
    if plus and minus:
        return "complex"
    return "+" if plus else "-" if minus else "no pred"


def _sheet_row(row: tuple[object, ...], where: str) -> tuple[str, str, str | None, str] | None:
    """Validate one sheet row, returning ``(pre, post, transmitter, polarity)`` or None if blank."""
    if row[_PRE_COL] in (None, ""):
        if any(cell not in (None, "") for cell in row):
            msg = f"{where}: cells without a source neuron"
            raise ValueError(msg)
        return None
    pre, post = _unpad(row[_PRE_COL]), _unpad(row[_POST_COL])
    for name in (pre, post):
        if name not in NEURON_CLASSIFICATION:
            msg = f"{where}: unknown neuron {name}"
            raise ValueError(msg)
    if row[_TYPE_COL] != "chemical":
        msg = f"{where}: edge type {row[_TYPE_COL]!r}, expected 'chemical'"
        raise ValueError(msg)
    polarity = row[_POLARITY_COL]
    if polarity not in _POLARITIES:
        msg = (
            f"{where}: polarity {polarity!r} is not one the sheet computes; a copy saved without "
            "its cached formula values reads as blank"
        )
        raise ValueError(msg)
    return pre, post, _transmitter(row[_TRANSMITTER_COL]), str(polarity)


def read_fenyves_sheet(
    path: Path,
    sheet: str,
    expected_sha256: str,
    edges: set[tuple[str, str]],
) -> FenyvesSheet:
    """Read one sign-prediction sheet, keeping the rows that name one of ``edges``.

    The sheet's cells are spreadsheet formulas. The values read are the ones the file was saved
    with, so a copy re-saved by a program that does not compute them is refused rather than read as
    blank.
    """
    import openpyxl

    _check_digest(path, expected_sha256)
    # A read-only workbook holds its file open until closed, so it is closed however parsing ends.
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return _parse_sheet(workbook[sheet].iter_rows(values_only=True), path, sheet, edges)
    finally:
        workbook.close()


def _parse_sheet(
    rows: Iterator[tuple[object, ...]],
    path: Path,
    sheet: str,
    edges: set[tuple[str, str]],
) -> FenyvesSheet:
    """Parse a sign-prediction sheet's rows, keeping the ones that name one of ``edges``."""
    predictions: dict[tuple[str, str], str] = {}
    transmitters: dict[str, str | None] = {}
    secondaries: dict[str, str | None] = {}
    primary_only: dict[tuple[str, str], str] = {}
    ignored = 0
    for index, row in enumerate(rows):
        if index < _FIRST_ROW:
            continue
        where = f"{path.name} {sheet!r} row {index + 1}"
        parsed = _sheet_row(row, where)
        if parsed is None:
            continue
        pre, post, transmitter, polarity = parsed
        if pre in transmitters and transmitters[pre] != transmitter:
            msg = f"{where}: {pre}'s transmitter changes from {transmitters[pre]} to {transmitter}"
            raise ValueError(msg)
        transmitters[pre] = transmitter
        secondary = _transmitter(row[_SECONDARY_COL])
        if pre in secondaries and secondaries[pre] != secondary:
            msg = (
                f"{where}: {pre}'s secondary transmitter changes from {secondaries[pre]} "
                f"to {secondary}"
            )
            raise ValueError(msg)
        secondaries[pre] = secondary
        if (pre, post) not in edges:
            ignored += 1
            continue
        if (pre, post) in predictions:
            msg = f"{where}: {pre}>{post} listed twice"
            raise ValueError(msg)
        predictions[(pre, post)] = polarity
        primary_only[(pre, post)] = _primary_only_polarity(row, transmitter)
    if not predictions:
        msg = f"{path.name} {sheet!r}: no predictions read"
        raise ValueError(msg)
    return FenyvesSheet(
        predictions=predictions,
        transmitters=transmitters,
        ignored_rows=ignored,
        secondaries=secondaries,
        primary_only=primary_only,
    )


def read_physiology_overrides(
    path: Path = PHYSIOLOGY_OVERRIDES_PATH,
) -> list[PhysiologyOverride]:
    """Read the table of cited per-connection signs, refusing a malformed or uncited row."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != _OVERRIDE_HEADER:
            msg = f"{path.name}: the header must be {','.join(_OVERRIDE_HEADER)}, not {header}"
            raise ValueError(msg)
        overrides: list[PhysiologyOverride] = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(_OVERRIDE_HEADER):
                msg = f"{path.name} row {line}: {len(row)} fields, expected {len(_OVERRIDE_HEADER)}"
                raise ValueError(msg)
            pre, post, sign, citation, evidence = row
            if sign not in ("1", "-1"):
                msg = f"{path.name} row {line}: sign {sign!r}, expected 1 or -1"
                raise ValueError(msg)
            if citation not in PHYSIOLOGY_CITATIONS:
                msg = f"{path.name} row {line}: unknown citation {citation!r}"
                raise ValueError(msg)
            overrides.append(
                PhysiologyOverride(
                    pre=pre,
                    post=post,
                    sign=1 if sign == "1" else -1,
                    citation=citation,
                    evidence=evidence,
                ),
            )
    return overrides


def _release_identities(neuron: str) -> tuple[str, ...]:
    primary = NEURON_CLASSIFICATION[neuron][1]
    return ((primary,) if primary else ()) + NEURON_CO_TRANSMITTERS.get(neuron, ())


def _expression_sign(
    edge: tuple[str, str],
    sheets: list[FenyvesSheet],
) -> Literal[1, -1] | None:
    signs: set[Literal[1, -1]] = {
        _POLARITY_SIGN[sheet.predictions[edge]]
        for sheet in sheets
        if sheet.predictions.get(edge) in _POLARITY_SIGN
    }
    if len(signs) > 1:
        msg = f"the Fenyves files disagree on {edge[0]}>{edge[1]}"
        raise ValueError(msg)
    return next(iter(signs)) if signs else None


def _fenyves_transmitter(pre: str, sheets: list[FenyvesSheet]) -> str | None:
    named = {sheet.transmitters[pre] for sheet in sheets if pre in sheet.transmitters}
    if len(named) > 1:
        msg = f"the Fenyves files disagree on {pre}'s transmitter: {sorted(map(str, named))}"
        raise ValueError(msg)
    return next(iter(named)) if named else None


def _fenyves_secondary(pre: str, sheets: list[FenyvesSheet]) -> str | None:
    named = {sheet.secondaries[pre] for sheet in sheets if pre in sheet.secondaries}
    if len(named) > 1:
        msg = (
            f"the Fenyves files disagree on {pre}'s secondary transmitter: "
            f"{sorted(map(str, named))}"
        )
        raise ValueError(msg)
    return next(iter(named)) if named else None


def _rests_on_release_identities(edge: tuple[str, str], sheets: list[FenyvesSheet]) -> bool:
    """Whether every transmitter the edge's prediction rests on is one the cell releases.

    The primary is checked by the caller. A secondary the atlas does not give the cell is harmless
    only if the primary alone gives the same polarity in every sheet that predicts one.
    """
    identities = _release_identities(edge[0])
    secondary = _fenyves_secondary(edge[0], sheets)
    if secondary is None or secondary in identities:
        return True
    return all(
        sheet.primary_only[edge] == sheet.predictions[edge]
        for sheet in sheets
        if sheet.predictions.get(edge) in _POLARITY_SIGN
    )


def per_connection_signs(
    connectome: Connectome | None = None,
    *,
    overrides_path: Path = PHYSIOLOGY_OVERRIDES_PATH,
) -> dict[tuple[str, str], ConnectionSign]:
    """Sign every chemical connection of ``connectome``, by the first step that gives a sign.

    Parameters
    ----------
    connectome
        The wiring whose chemical edges are signed; the Cook 2019 hermaphrodite by default. The
        Emmons 2024 release has the same chemical edges, so it gives the same table.
    overrides_path
        The table of cited per-connection signs.

    Returns
    -------
    dict
        ``(pre, post)`` to its :class:`ConnectionSign`, one entry per chemical edge.

    Raises
    ------
    ValueError
        If a vendored file's digest is not the recorded one, the two Fenyves files disagree on a
        connection's sign or a cell's transmitters, or an override names a connection the wiring
        does not have or names one twice.
    """
    if connectome is None:
        from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite

        connectome = load_cook_2019_hermaphrodite()
    edges = [(synapse.pre, synapse.post) for synapse in connectome.chemical_synapses]
    edge_set = set(edges)

    overrides: dict[tuple[str, str], PhysiologyOverride] = {}
    for override in read_physiology_overrides(overrides_path):
        key = (override.pre, override.post)
        if key in overrides:
            msg = f"the physiology table lists {key[0]}>{key[1]} twice"
            raise ValueError(msg)
        if key not in edge_set:
            msg = f"the physiology table signs {key[0]}>{key[1]}, which the wiring does not have"
            raise ValueError(msg)
        overrides[key] = override

    sheets = [
        read_fenyves_sheet(FENYVES_S1_PATH, FENYVES_S1_SHEET, FENYVES_S1_SHA256, edge_set),
        read_fenyves_sheet(FENYVES_S5_PATH, FENYVES_S5_SHEET, FENYVES_S5_SHA256, edge_set),
    ]

    signed: dict[tuple[str, str], ConnectionSign] = {}
    for pre, post in edges:
        # Checked on every edge, overridden or not, so the two files' agreement holds everywhere.
        expression = _expression_sign((pre, post), sheets)
        if override := overrides.get((pre, post)):
            signed[(pre, post)] = ConnectionSign(
                pre=pre,
                post=post,
                sign=override.sign,
                source="physiology",
                citation=override.citation,
            )
            continue
        if expression is not None:
            transmitter = _fenyves_transmitter(pre, sheets)
            # A prediction resting on a transmitter the atlas does not give the cell is set aside:
            # it predicts the receptor response to a release the cell does not make.
            if (
                transmitter is not None
                and transmitter in _release_identities(pre)
                and _rests_on_release_identities((pre, post), sheets)
            ):
                signed[(pre, post)] = ConnectionSign(
                    pre=pre,
                    post=post,
                    sign=expression,
                    source="expression",
                )
                continue
        rule = sign_for(NEURON_CLASSIFICATION[pre][1])
        signed[(pre, post)] = ConnectionSign(
            pre=pre,
            post=post,
            sign=rule if rule is not None else 0,
            source="rule" if rule is not None else "none",
        )
    return signed
