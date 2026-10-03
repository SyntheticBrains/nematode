"""Per-connection chemical signs from vendored sources.

Covers the connectome-substrate requirement "Per-connection chemical signs from vendored sources":
the vendored files are the ones recorded, the Cook 2019 table has the recorded composition, the
table matches Wormlight's, the sources disagree or point at nothing, and nothing an experiment
reads changes.

The counts are pinned rather than recomputed from the same code, because they are what the
derivation was checked against, edge by edge, before the files were vendored.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from quantumnematode.connectome import (
    load_cook_2019_hermaphrodite,
    load_emmons_2024_hermaphrodite,
)
from quantumnematode.connectome import signs as sg

if TYPE_CHECKING:
    from quantumnematode.connectome import Connectome

PROVENANCE = sg.DATA_DIR / "PROVENANCE.md"
BRAIN_DIR = Path(sg.__file__).resolve().parents[1] / "brain"

# Wormlight's chemical sign table at commit 1190b3e, without the edges below: every
# "pre>post:sign:source" line, sorted and joined by newlines, hashed. Taken from its
# public/data/wormlight.v1.json.
WORMLIGHT_1190B3E_DIGEST = "797f7b41836bde20af53d17042d0291858bbeadaf98fd23a18582dd43cb5984e"

# Where this table deliberately differs from Wormlight's: Wormlight signs these by Fenyves et al.'s
# prediction, but each prediction rests on a secondary transmitter the atlas does not give the cell,
# and the primary alone predicts nothing, so here they fall to the per-neuron rule.
SECONDARY_SET_ASIDE = {
    ("AIML", "RID"), ("AIML", "SMBVL"), ("AIML", "URXL"), ("AIMR", "ALA"), ("AIMR", "URXR"),
    ("AVAL", "AVDL"), ("AVAL", "AVDR"), ("AVAL", "LUAL"), ("AVAL", "LUAR"), ("AVAL", "SABVL"),
    ("AVAR", "AVDL"), ("AVAR", "AVDR"), ("AVAR", "LUAL"), ("AVAR", "LUAR"), ("AVAR", "SABVR"),
    ("AVBL", "AVDR"), ("AVBR", "AVDL"), ("RIBL", "DVC"), ("RIBL", "RIAL"), ("RIBR", "RIAR"),
    ("RIBR", "RIH"), ("RIBR", "RMDVR"), ("RIBR", "RMED"),
}  # fmt: skip


@pytest.fixture(scope="module")
def table() -> dict[tuple[str, str], sg.ConnectionSign]:
    """Build the Cook 2019 table once for the module; reading both sheets takes seconds."""
    return sg.per_connection_signs()


def _write_overrides(path: Path, rows: list[str]) -> Path:
    path.write_text("pre,post,sign,citation,evidence\n" + "".join(f"{r}\n" for r in rows))
    return path


class TestVendoredFiles:
    """Scenario: The vendored files are the ones recorded."""

    @pytest.mark.parametrize(
        ("path", "digest"),
        [
            (sg.FENYVES_S1_PATH, sg.FENYVES_S1_SHA256),
            (sg.FENYVES_S5_PATH, sg.FENYVES_S5_SHA256),
        ],
    )
    def test_digest_matches_and_is_recorded(self, path: Path, digest: str) -> None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert digest in PROVENANCE.read_text(encoding="utf-8")

    def test_a_changed_file_is_refused(self, tmp_path: Path) -> None:
        copy = tmp_path / sg.FENYVES_S5_PATH.name
        shutil.copyfile(sg.FENYVES_S5_PATH, copy)
        with copy.open("ab") as handle:
            handle.write(b"\0")
        with pytest.raises(ValueError, match="not the recorded"):
            sg.read_fenyves_sheet(copy, sg.FENYVES_S5_SHEET, sg.FENYVES_S5_SHA256, set())

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="git lfs pull"):
            sg.read_fenyves_sheet(
                tmp_path / "absent.xlsx",
                sg.FENYVES_S5_SHEET,
                sg.FENYVES_S5_SHA256,
                set(),
            )

    def test_the_physiology_table_is_recorded(self) -> None:
        assert sg.PHYSIOLOGY_OVERRIDES_PATH.name in PROVENANCE.read_text(encoding="utf-8")
        overrides = sg.read_physiology_overrides()
        assert len(overrides) == 51
        assert {o.citation for o in overrides} == set(sg.PHYSIOLOGY_CITATIONS)


class TestComposition:
    """Scenario: The Cook 2019 table has the recorded composition."""

    def test_every_edge_is_signed_once(self, table) -> None:
        edges = [(s.pre, s.post) for s in load_cook_2019_hermaphrodite().chemical_synapses]
        assert len(table) == len(edges) == 3709
        assert set(table) == set(edges)

    def test_sources_and_signs(self, table) -> None:
        assert Counter(entry.source for entry in table.values()) == {
            "physiology": 51,
            "expression": 1676,
            "rule": 1449,
            "none": 533,
        }
        assert Counter((entry.source, entry.sign) for entry in table.values()) == {
            ("physiology", 1): 19,
            ("physiology", -1): 32,
            ("expression", 1): 1291,
            ("expression", -1): 385,
            ("rule", 1): 1324,
            ("rule", -1): 125,
            ("none", 0): 533,
        }

    def test_sections_by_source(self, table) -> None:
        sections: Counter[str] = Counter()
        for synapse in load_cook_2019_hermaphrodite().chemical_synapses:
            sections[table[(synapse.pre, synapse.post)].source] += synapse.weight
        assert sections == {"physiology": 850, "expression": 10962, "rule": 7098, "none": 2055}

    def test_awc_to_aiy_is_inhibitory_where_the_rule_says_excitatory(self, table) -> None:
        entry = table[("AWCL", "AIYL")]
        assert (entry.sign, entry.source, entry.citation) == (-1, "physiology", "chalasani2007")
        assert sg.sign_for(sg.NEURON_CLASSIFICATION["AWCL"][1]) == 1

    def test_signs_opposite_to_the_rule(self, table) -> None:
        opposite = Counter(
            entry.source
            for (pre, _post), entry in table.items()
            if sg.sign_for(sg.NEURON_CLASSIFICATION[pre][1]) not in (None, entry.sign)
        )
        assert opposite == {"physiology": 32, "expression": 310}

    def test_sheet_coverage(self, table) -> None:
        edges = set(table)
        s1 = sg.read_fenyves_sheet(
            sg.FENYVES_S1_PATH,
            sg.FENYVES_S1_SHEET,
            sg.FENYVES_S1_SHA256,
            edges,
        )
        s5 = sg.read_fenyves_sheet(
            sg.FENYVES_S5_PATH,
            sg.FENYVES_S5_SHEET,
            sg.FENYVES_S5_SHA256,
            edges,
        )
        assert (len(s1.predictions), s1.ignored_rows) == (3516, 122)
        assert (len(s5.predictions), s5.ignored_rows) == (3237, 5)
        assert len(set(s1.predictions) & set(s5.predictions)) == 3117

    def test_the_emmons_2024_release_gives_the_same_table(self, table) -> None:
        assert sg.per_connection_signs(load_emmons_2024_hermaphrodite()) == table


class TestWormlightAgreement:
    """Scenario: The table matches Wormlight's."""

    def test_every_other_edge_matches_its_sign_and_source(self, table) -> None:
        rows = sorted(
            f"{pre}>{post}:{e.sign}:{e.source}"
            for (pre, post), e in table.items()
            if (pre, post) not in SECONDARY_SET_ASIDE
        )
        digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        assert digest == WORMLIGHT_1190B3E_DIGEST

    def test_the_differing_edges_fall_to_the_rule(self, table) -> None:
        assert {table[edge].source for edge in SECONDARY_SET_ASIDE} == {"rule"}
        edges = set(table)
        sheets = [
            sg.read_fenyves_sheet(
                sg.FENYVES_S1_PATH,
                sg.FENYVES_S1_SHEET,
                sg.FENYVES_S1_SHA256,
                edges,
            ),
            sg.read_fenyves_sheet(
                sg.FENYVES_S5_PATH,
                sg.FENYVES_S5_SHEET,
                sg.FENYVES_S5_SHA256,
                edges,
            ),
        ]
        # Wormlight's sign for each is the sheets' prediction; 11 of the 23 change sign here.
        flipped = [
            e for e in SECONDARY_SET_ASIDE if sg._expression_sign(e, sheets) != table[e].sign
        ]
        assert len(flipped) == 11


class TestRefusals:
    """Scenario: The sources disagree or point at nothing."""

    def test_an_override_on_an_absent_edge(self, tmp_path: Path) -> None:
        path = _write_overrides(tmp_path / "o.csv", ["AWCL,VB1,-1,chalasani2007,x"])
        with pytest.raises(ValueError, match="AWCL>VB1, which the wiring does not have"):
            sg.per_connection_signs(overrides_path=path)

    def test_an_override_listed_twice(self, tmp_path: Path) -> None:
        row = "AWCL,AIYL,-1,chalasani2007,x"
        path = _write_overrides(tmp_path / "o.csv", [row, row])
        with pytest.raises(ValueError, match="twice"):
            sg.per_connection_signs(overrides_path=path)

    def test_an_unknown_citation(self, tmp_path: Path) -> None:
        path = _write_overrides(tmp_path / "o.csv", ["AWCL,AIYL,-1,nobody2099,x"])
        with pytest.raises(ValueError, match="unknown citation"):
            sg.read_physiology_overrides(path)

    @pytest.mark.parametrize(
        ("rows", "message"),
        [
            (["AWCL,AIYL,0,chalasani2007,x"], "expected 1 or -1"),
            (["AWCL,AIYL,-1,chalasani2007"], "fields"),
        ],
    )
    def test_a_malformed_row(self, tmp_path: Path, rows: list[str], message: str) -> None:
        path = _write_overrides(tmp_path / "o.csv", rows)
        with pytest.raises(ValueError, match=message):
            sg.read_physiology_overrides(path)

    def test_a_wrong_header(self, tmp_path: Path) -> None:
        path = tmp_path / "o.csv"
        path.write_text("pre,post,sign\n")
        with pytest.raises(ValueError, match="header"):
            sg.read_physiology_overrides(path)

    def test_the_two_files_disagreeing(self) -> None:
        edge = ("AWCL", "AIYL")
        plus = sg.FenyvesSheet(predictions={edge: "+"}, transmitters={}, ignored_rows=0)
        minus = sg.FenyvesSheet(predictions={edge: "-"}, transmitters={}, ignored_rows=0)
        with pytest.raises(ValueError, match="disagree on AWCL>AIYL"):
            sg._expression_sign(edge, [plus, minus])


class TestPrecedence:
    """The first step that gives a sign wins, on hand-built inputs."""

    @staticmethod
    def _wiring(*edges: tuple[str, str]) -> Connectome:
        # Only the chemical edges are read, so a stand-in carrying them is enough.
        wiring = SimpleNamespace(
            chemical_synapses=[SimpleNamespace(pre=pre, post=post) for pre, post in edges],
        )
        return cast("Connectome", wiring)

    def _run(self, monkeypatch, tmp_path, sheet: sg.FenyvesSheet, overrides: list[str], *edges):
        monkeypatch.setattr(sg, "read_fenyves_sheet", lambda *args: sheet)
        path = _write_overrides(tmp_path / "o.csv", overrides)
        return sg.per_connection_signs(self._wiring(*edges), overrides_path=path)

    def test_physiology_beats_expression(self, monkeypatch, tmp_path) -> None:
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "+"},
            transmitters={"AWCL": "Glu"},
            ignored_rows=0,
        )
        got = self._run(monkeypatch, tmp_path, sheet, ["AWCL,AIYL,-1,chalasani2007,x"], edge)
        assert (got[edge].sign, got[edge].source) == (-1, "physiology")

    def test_expression_beats_the_rule(self, monkeypatch, tmp_path) -> None:
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "-"},
            transmitters={"AWCL": "Glu"},
            ignored_rows=0,
        )
        got = self._run(monkeypatch, tmp_path, sheet, [], edge)
        assert (got[edge].sign, got[edge].source) == (-1, "expression")

    def test_a_prediction_on_a_transmitter_the_cell_lacks_is_set_aside(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        # AWC releases glutamate; a prediction resting on GABA predicts a release AWC does not make.
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "-"},
            transmitters={"AWCL": "GABA"},
            ignored_rows=0,
        )
        got = self._run(monkeypatch, tmp_path, sheet, [], edge)
        assert (got[edge].sign, got[edge].source) == (1, "rule")

    def test_a_prediction_resting_on_an_unreleased_secondary_is_set_aside(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        # AWC releases glutamate only. The sheet's minus rests on a secondary GABA, and glutamate
        # alone predicts nothing, so the prediction describes a release AWC does not make.
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "-"},
            transmitters={"AWCL": "Glu"},
            ignored_rows=0,
            secondaries={"AWCL": "GABA"},
            primary_only={edge: "no pred"},
        )
        got = self._run(monkeypatch, tmp_path, sheet, [], edge)
        assert (got[edge].sign, got[edge].source) == (1, "rule")

    def test_an_unreleased_secondary_that_changes_nothing_is_harmless(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "-"},
            transmitters={"AWCL": "Glu"},
            ignored_rows=0,
            secondaries={"AWCL": "GABA"},
            primary_only={edge: "-"},
        )
        got = self._run(monkeypatch, tmp_path, sheet, [], edge)
        assert (got[edge].sign, got[edge].source) == (-1, "expression")

    def test_complex_and_no_prediction_fall_through(self, monkeypatch, tmp_path) -> None:
        edge = ("AWCL", "AIYL")
        sheet = sg.FenyvesSheet(
            predictions={edge: "complex"},
            transmitters={"AWCL": "Glu"},
            ignored_rows=0,
        )
        got = self._run(monkeypatch, tmp_path, sheet, [], edge)
        assert got[edge].source == "rule"


class TestNothingReadsIt:
    """Scenario: Nothing an experiment reads changes."""

    def test_no_brain_module_reads_the_table(self) -> None:
        readers = [
            path.name
            for path in BRAIN_DIR.rglob("*.py")
            if "connectome.signs" in path.read_text(encoding="utf-8")
            or "per_connection_signs" in path.read_text(encoding="utf-8")
        ]
        assert readers == []
