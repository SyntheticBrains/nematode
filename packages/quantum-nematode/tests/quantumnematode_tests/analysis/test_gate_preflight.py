"""The gate preflight: a panel's registered gates evaluated on existing runs before launch.

It must find a panel's runs by stem, refuse to pass a level with no evidence, and report a
saturated, floor-failing or near-bar level as blocking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "campaigns").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]

STEMS = {
    "full": {"wt_learn": "wl", "wt_frozen": "wf", "rn_learn": "rl", "rn_frozen": "rf"},
    "narrow": {"wt_learn": "wl", "wt_frozen": "wf", "rn_learn": "nl", "rn_frozen": "nf"},
}


def _logs(tmp_path: Path, stems: list[str], seeds: tuple[int, ...]) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for seed in seeds:
            (logs / f"{stem}-seed{seed}.log").write_text("x")
    return logs


def _gates(wt: float, rn: float, *, passes: bool = True, n: int = 1) -> dict[str, Any]:
    return {
        "gate_passes": passes,
        "saturated": wt >= 90.0 and rn >= 90.0,
        "wt": {"plateau_success": wt, "floor_success": 0.5, "n_seeds": n},
        "rn": {"plateau_success": rn, "floor_success": 0.5, "n_seeds": n},
    }


class TestEvidence:
    def test_a_shared_wild_type_run_serves_every_level(self, tmp_path: Path) -> None:
        logs = _logs(tmp_path, ["wl", "wf", "rl", "rf"], (1, 2))
        rows = gp.evidence(STEMS, [logs])
        assert sum(1 for arm, level, _s, _l in rows if arm == "wt_learn") == 4
        assert {level for _a, level, _s, _l in rows} == {"full", "narrow"}

    def test_foreign_logs_are_ignored(self, tmp_path: Path) -> None:
        logs = _logs(tmp_path, ["other"], (1,))
        assert gp.evidence(STEMS, [logs]) == []


class TestPreflight:
    def test_a_level_without_evidence_blocks_the_launch(self, tmp_path, monkeypatch) -> None:
        logs = _logs(tmp_path, ["wl", "wf", "rl", "rf"], (1, 2))
        monkeypatch.setattr(gp.ops, "learning_gates", lambda *a, **k: _gates(70.0, 72.0, n=2))
        got = gp.preflight(STEMS, [logs])
        assert got["levels"]["full"]["status"] == "readable"
        assert got["levels"]["narrow"]["status"] == "no_evidence"
        assert got["launch"] is False

    @pytest.mark.parametrize(
        "case",
        [
            (91.4, 92.3, True, "saturated"),
            (86.0, 80.0, True, "near_bar"),
            (70.0, 72.0, False, "fails_floor"),
            (76.4, 72.2, True, "readable"),
        ],
    )
    def test_each_status(self, tmp_path, monkeypatch, case) -> None:
        wt, rn, passes, status = case
        logs = _logs(tmp_path, ["wl", "wf", "rl", "rf", "nl", "nf"], (1,))
        monkeypatch.setattr(
            gp.ops,
            "learning_gates",
            lambda *a, **k: _gates(wt, rn, passes=passes),
        )
        got = gp.preflight(STEMS, [logs])
        assert {lv["status"] for lv in got["levels"].values()} == {status}
        assert got["launch"] is (status == "readable")

    def test_a_seed_the_gate_could_not_score_blocks_the_launch(self, tmp_path, monkeypatch) -> None:
        logs = _logs(tmp_path, ["wl", "wf", "rl", "rf", "nl", "nf"], (1, 2))
        # Two seeds selected, but one log yields no plateau, so the gate scores only one.
        monkeypatch.setattr(gp.ops, "learning_gates", lambda *a, **k: _gates(70.0, 72.0, n=1))
        got = gp.preflight(STEMS, [logs])
        assert {lv["status"] for lv in got["levels"].values()} == {"incomplete_evidence"}
        assert got["levels"]["full"]["n_scored"] == 1
        assert got["launch"] is False

    def test_the_same_run_in_two_directories_is_refused(self, tmp_path) -> None:
        first = _logs(tmp_path / "a", ["wl", "wf", "rl", "rf"], (1,))
        second = _logs(tmp_path / "b", ["wl"], (1,))
        with pytest.raises(ValueError, match="has two runs"):
            gp.preflight(STEMS, [first, second])

    def test_a_panel_module_with_the_wrong_arms_is_refused(self, monkeypatch) -> None:
        bad = SimpleNamespace(STEMS={"full": {"wt_learn": "a"}})
        monkeypatch.setattr(gp.importlib, "import_module", lambda name: bad)
        with pytest.raises(ValueError, match="expected"):
            gp.panel_stems("bad")

    def test_the_thermal_panel_resolves(self) -> None:
        stems = gp.panel_stems("thermal_null_strength")
        assert set(stems) == {"full", "chemical", "gap_held"}
