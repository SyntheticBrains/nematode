"""The horizon campaign's reading, and what it refuses to score.

Two failures these pin, both of which would look exactly like a clean result. A campaign missing
runs would shrink a horizon's pairing silently and still assign `does_not_transfer`, which is a
verdict about evidence that was never collected. And a record asserting why a contrast is
unavailable, rather than deriving it, can contradict the numbers printed beside it.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_horizon_multistep as hm  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]

_COMPETENT = ms.COMPETENT_THRESHOLD + 5.0
_DEAD = 1.0


def _scanned(*, drop: tuple[str, str, int] | None = None) -> dict:
    """Build a complete scan, optionally missing one cell."""
    out: dict = {h: {"learning": {}, "frozen": {}, "logs": {}} for h in hm.HORIZONS}
    for horizon in hm.HORIZONS:
        for arm in ("learning", "frozen"):
            for seed in hm.SEEDS:
                if drop == (horizon, arm, seed):
                    continue
                out[horizon][arm][seed] = object()
    return out


class TestAnIncompleteCampaignIsNotScored:
    def test_a_complete_campaign_passes(self) -> None:
        hm.require_complete(_scanned())

    def test_one_missing_run_refuses_a_verdict(self) -> None:
        # Without this the horizon would simply pair fewer seeds and still return a verdict.
        with pytest.raises(ValueError, match="incomplete"):
            hm.require_complete(_scanned(drop=("td099", "frozen", 5)))

    def test_the_refusal_names_the_missing_cell(self) -> None:
        with pytest.raises(ValueError, match=r"td0999/learning seeds \[3\]"):
            hm.require_complete(_scanned(drop=("td0999", "learning", 3)))

    def test_every_registered_cell_is_required(self) -> None:
        # Three horizons, two arms, eight seeds: the campaign as registered.
        assert len(hm.HORIZONS) * 2 * len(hm.SEEDS) == 48


class TestScanKeepsTheCellsApart:
    """Label parsing and the duplicate guard, which decide what each cell contains."""

    def _dir(self, tmp_path: Path, names: list[str]) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir()
        for name in names:
            (logs / name).write_text("")
        return tmp_path

    def _name(self, horizon: str, seed: int, *, frozen: bool = False) -> str:
        return f"{hm._STEM}_{horizon}{'_frozen' if frozen else ''}-seed{seed}.log"

    def _stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hm, "read_log", lambda *_a, **_k: object())

    def test_learning_and_frozen_are_separated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub(monkeypatch)
        root = self._dir(
            tmp_path,
            [self._name("td09", 1), self._name("td09", 1, frozen=True)],
        )
        out = hm.scan(root)
        assert list(out["td09"]["learning"]) == [1]
        assert list(out["td09"]["frozen"]) == [1]

    def test_a_duplicate_cell_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Two logs claiming one cell: the second would otherwise replace the first and the
        # campaign would score as if one run had happened. `seed1` and `seed01` are distinct
        # filenames that parse to the same seed, which is how this arises in practice.
        self._stub(monkeypatch)
        root = self._dir(
            tmp_path,
            [self._name("td09", 1), f"{hm._STEM}_td09-seed01.log"],
        )
        with pytest.raises(ValueError, match="duplicate run"):
            hm.scan(root)

    def test_an_unrecognised_label_is_dropped_not_raised(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Malformed logs are still skipped with a warning; only duplicates are fatal.
        self._stub(monkeypatch)
        root = self._dir(tmp_path, ["something_else-seed1.log", self._name("td099", 4)])
        out = hm.scan(root)
        assert "unrecognised label" in capsys.readouterr().out
        assert list(out["td099"]["learning"]) == [4]

    def test_an_unregistered_horizon_is_dropped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub(monkeypatch)
        root = self._dir(tmp_path, [self._name("td5", 1)])
        out = hm.scan(root)
        assert all(not out[h]["learning"] for h in hm.HORIZONS)


class TestTheCompetenceNoteIsDerived:
    def test_it_names_the_arm_that_lacks_a_competent_seed(self) -> None:
        # The case this campaign hit: the frozen arm has one, the learning arm none.
        out = hm._full_clear(
            dict.fromkeys(hm.SEEDS, _DEAD),
            {1: _COMPETENT, **dict.fromkeys(hm.SEEDS[1:], _DEAD)},
        )
        assert out["level_contrast_available"] is False
        assert out["competent_seeds"] == {"learning": [], "frozen": [1]}
        assert "the learning arm has no seed" in out["note"]
        assert "[1]" in out["note"]

    def test_it_does_not_claim_no_competent_seed_when_there_is_one(self) -> None:
        # The defect: a hardcoded note asserting none while reporting one beside it.
        out = hm._full_clear(
            dict.fromkeys(hm.SEEDS, _DEAD),
            {1: _COMPETENT, **dict.fromkeys(hm.SEEDS[1:], _DEAD)},
        )
        assert "neither arm" not in out["note"]

    def test_neither_arm_competent_says_so(self) -> None:
        out = hm._full_clear(dict.fromkeys(hm.SEEDS, _DEAD), dict.fromkeys(hm.SEEDS, _DEAD))
        assert out["level_contrast_available"] is False
        assert "neither arm" in out["note"]

    def test_both_arms_competent_makes_the_contrast_available(self) -> None:
        competent = dict.fromkeys(hm.SEEDS, _COMPETENT)
        out = hm._full_clear(competent, dict(competent))
        assert out["level_contrast_available"] is True
        assert "available" in out["note"]

    def test_the_threshold_is_the_committed_one(self) -> None:
        out = hm._full_clear(dict.fromkeys(hm.SEEDS, _DEAD), dict.fromkeys(hm.SEEDS, _DEAD))
        assert out["threshold"] == ms.COMPETENT_THRESHOLD


class _Record:
    """Minimal stand-in for a run's parsed record."""

    def __init__(self, success: float, foods: float) -> None:
        self.success = success
        self.foods = foods


def _campaign(learning: float, frozen: float, *, competent: bool = False) -> dict:
    """Build a complete scanned campaign where every learning arm scores `learning`."""
    out: dict = {}
    for horizon in hm.HORIZONS:
        out[horizon] = {"learning": {}, "frozen": {}, "logs": {}}
        for seed in hm.SEEDS:
            clear = _COMPETENT if (competent and seed == 1) else _DEAD
            out[horizon]["learning"][seed] = _Record(_DEAD, learning)
            out[horizon]["frozen"][seed] = _Record(clear, frozen)
    return out


class TestTheReadingEndToEnd:
    def _analyse(self, monkeypatch: pytest.MonkeyPatch, scanned: dict) -> dict:
        # Drift needs exports on disk; it is reported, not scored, so it is stubbed here.
        monkeypatch.setattr(
            hm,
            "drift",
            lambda *_a, **_k: {"per_seed": {}, "mean_relative": float("nan"), "n_read": 0},
        )
        return hm.analyse(scanned)

    def test_a_worse_learning_arm_does_not_transfer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The campaign's own shape: the learning arm below its control everywhere.
        out = self._analyse(monkeypatch, _campaign(learning=0.2, frozen=2.2))
        assert out["verdict"]["verdict"] == "does_not_transfer"
        assert all(not c["beats_control"] for c in out["horizons"].values())

    def test_a_large_consistent_gain_at_every_horizon_is_void(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Including the pinned one, which contradicts the pilot: something else moved.
        out = self._analyse(monkeypatch, _campaign(learning=4.0, frozen=1.0))
        assert out["verdict"]["verdict"] == "void"

    def test_the_record_carries_the_registered_minimum_and_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = self._analyse(monkeypatch, _campaign(learning=0.2, frozen=2.2))
        assert out["min_effect_foods"] == hm.MIN_EFFECT_FOODS
        assert out["committed_yardstick_foods"] == hm.COMMITTED_YARDSTICK_FOODS
        assert "confound" in out["committed_is_reference_only"]

    def test_the_competence_note_reaches_the_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = self._analyse(monkeypatch, _campaign(learning=0.2, frozen=2.2, competent=True))
        clear = next(iter(out["horizons"].values()))["full_clear"]
        assert clear["competent_seeds"] == {"learning": [], "frozen": [1]}
        assert "the learning arm has no seed" in clear["note"]

    def test_the_record_is_strict_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        out = self._analyse(monkeypatch, _campaign(learning=0.2, frozen=2.2))
        text = json.dumps(hm._jsonable(out), allow_nan=False)
        json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the record"))

    def test_it_prints_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hm._print(self._analyse(monkeypatch, _campaign(learning=0.2, frozen=2.2)))
        printed = capsys.readouterr().out
        assert "does_not_transfer" in printed
        assert "trace_decay" in printed


class TestMainWritesTheRecords:
    def test_it_scores_a_complete_campaign_and_writes_both_records(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import csv
        import json

        logs = tmp_path / "logs"
        logs.mkdir()
        for horizon in hm.HORIZONS:
            for frozen in (False, True):
                for seed in hm.SEEDS:
                    suffix = "_frozen" if frozen else ""
                    (logs / f"{hm._STEM}_{horizon}{suffix}-seed{seed}.log").write_text("")
        monkeypatch.setattr(
            hm,
            "read_log",
            lambda log, *_a, **_k: _Record(_DEAD, 2.2 if "_frozen-" in log.name else 0.2),
        )
        monkeypatch.setattr(
            hm,
            "drift",
            lambda *_a, **_k: {"per_seed": {}, "mean_relative": float("nan"), "n_read": 0},
        )
        out_json, out_csv = tmp_path / "h.json", tmp_path / "h.csv"
        assert (
            hm.main(
                [
                    "--campaign-dir",
                    str(tmp_path),
                    "--out",
                    str(out_json),
                    "--csv",
                    str(out_csv),
                ],
            )
            == 0
        )
        record = json.loads(out_json.read_text())
        assert record["verdict"]["verdict"] == "does_not_transfer"
        rows = list(csv.DictReader(out_csv.open(newline="")))
        assert len(rows) == len(hm.HORIZONS) * len(hm.SEEDS)
        assert float(rows[0]["delta"]) < 0

    def test_it_refuses_an_incomplete_campaign(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / f"{hm._STEM}_td09-seed1.log").write_text("")
        monkeypatch.setattr(hm, "read_log", lambda *_a, **_k: _Record(_DEAD, 0.2))
        with pytest.raises(ValueError, match="incomplete"):
            hm.main(["--campaign-dir", str(tmp_path)])


class TestTheRegisteredRule:
    def _cell(self, *, effect: float, q: float) -> dict:
        return {"graded": {"effect": effect, "p_improve": 0.0, "q_improve": q, "defined": True}}

    def test_a_significant_shift_below_the_minimum_does_not_count(self) -> None:
        # The reason the minimum exists: the rank test fires on sign consistency, so a tiny
        # shift can be significant on a platform whose arms sit near zero.
        assert hm.MIN_EFFECT_FOODS == 0.5
        cells = {h: self._cell(effect=0.2, q=0.01) for h in hm.HORIZONS}
        for cell in cells.values():
            cell["beats_control"] = cell["graded"]["q_improve"] <= ms.SIG_Q and (
                cell["graded"]["effect"] >= hm.MIN_EFFECT_FOODS
            )
        assert not any(c["beats_control"] for c in cells.values())

    def test_the_pinned_horizon_beating_its_control_is_void_not_a_result(self) -> None:
        # It contradicts the pilot that motivated the change, so something else moved.
        cells = {h: {"beats_control": h == "td09"} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "void"

    def test_no_horizon_beating_its_control_does_not_transfer(self) -> None:
        cells = {h: {"beats_control": False} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "does_not_transfer"

    def test_a_raised_horizon_alone_transfers(self) -> None:
        cells = {h: {"beats_control": h == "td099"} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "transfers"
