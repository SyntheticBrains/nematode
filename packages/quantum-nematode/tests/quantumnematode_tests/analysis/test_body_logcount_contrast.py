"""The body contrast on log counts: its window, every verdict-map row, the gates and the refusals.

Covers `add-body-logcount-contrast` (design Decisions B, D and F) and the
`architecture-comparison-protocol` scenarios "The late window disagrees with the verdict" and "The
gates are read on another window": the verdict and the gates read episodes 1-3,000 only.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import body_logcount_contrast as bc  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = (4001, 4002, 4003)


def _series(early: float, late: float, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [rng.random(bc.WINDOW) < early, rng.random(bc.HORIZON - bc.WINDOW) < late],
    ).astype(float)


def _write(log: Path, series: np.ndarray, config: str | None = None, code: str = "0") -> None:
    stem = log.stem.rpartition("-seed")[0]
    lines = [
        f"Run: {i + 1}   Status: {'SUCCESS' if v else 'FAILED'}  Reason: x"
        for i, v in enumerate(series)
    ]
    lines.append(f"  Config: /session/{config or stem + '.yml'}")
    log.write_text("\n".join(lines) + "\n")
    log.with_suffix(".exit").write_text(code + "\n")


def _panel(
    where: Path,
    *,
    wt: tuple[float, float] = (0.5, 0.5),
    rn: tuple[float, float] = (0.5, 0.5),
    floor: float = 0.0,
) -> None:
    rng = np.random.default_rng(1)
    rates = {
        "wt_learn": wt,
        "rn_learn": rn,
        "wt_frozen": (floor, floor),
        "rn_frozen": (floor, floor),
    }
    for role, stem in bc.STEMS[bc.LEVEL].items():
        for seed in _SEEDS:
            _write(where / f"{stem}-seed{seed}.log", _series(*rates[role], rng))


class TestArms:
    def test_288_runs_on_new_bands(self) -> None:
        plan = bc.plan()
        assert plan["runs"] == 288
        assert plan["overlapping_bands"] == []
        assert tuple(range(4001, 4017)) == bc.FLOOR_SEEDS
        assert not set(bc.SEEDS) & (set(bc.PILOT_SEEDS) | set(bc.SMOKE_SEEDS))
        for config in plan["configs"]:
            assert (_root / config).is_file()

    def test_an_overlapping_band_is_named(self) -> None:
        assert bc.overlapping_bands((2330, 4001)) == [(2301, 2332)]


class TestWindow:
    def test_episodes_after_the_window_cannot_move_the_reading(self, tmp_path: Path) -> None:
        """Scenario "The late window disagrees with the verdict"."""
        _panel(tmp_path, wt=(0.5, 0.9), rn=(0.5, 0.1))
        runs = bc.load_runs([tmp_path], _SEEDS, _SEEDS)
        window = bc.gap(runs, _SEEDS)
        late = bc.gap(runs, _SEEDS, bc.WINDOW, bc.HORIZON)
        assert abs(window["mean"]) < 0.05
        assert late["mean"] > 0.7

    def test_the_plateau_is_the_window_s_final_quarter(self) -> None:
        series = np.zeros(bc.HORIZON)
        series[2250:3000] = 1.0
        assert bc.plateau(series) == 100.0
        assert bc.plateau(series, bc.HORIZON) == 0.0

    def test_the_gates_read_the_window(self, tmp_path: Path) -> None:
        """Scenario "The gates are read on another window": saturation only after 3,000 is not."""
        _panel(tmp_path, wt=(0.4, 0.99), rn=(0.4, 0.99))
        runs = bc.load_runs([tmp_path], _SEEDS, _SEEDS)
        assert not bc.gates(runs, _SEEDS, _SEEDS)["saturated"]
        assert bc.gates(runs, _SEEDS, _SEEDS, bc.HORIZON)["saturated"]

    def test_a_floor_as_high_as_the_learner_fails(self, tmp_path: Path) -> None:
        _panel(tmp_path, wt=(0.3, 0.3), rn=(0.3, 0.3), floor=0.3)
        runs = bc.load_runs([tmp_path], _SEEDS, _SEEDS)
        assert not bc.gates(runs, _SEEDS, _SEEDS)["beats_floor"]

    def test_episodes_to_30pct_is_positive_when_the_wild_type_is_sooner(
        self,
        tmp_path: Path,
    ) -> None:
        _panel(tmp_path, wt=(0.6, 0.6), rn=(0.1, 0.6))
        runs = bc.load_runs([tmp_path], _SEEDS, _SEEDS)
        assert bc.episodes_gap(runs, _SEEDS)["mean"] > 0


def _primary(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "q": q}


_OPEN: dict[str, Any] = {"readable": True}


class TestVerdictMap:
    def test_gates_failing_read_unreadable(self) -> None:
        out = bc.read({"readable": False}, _primary(0.1, 0.08, 0.12, 0.001))
        assert out["verdict"] == "unreadable"

    def test_placement(self) -> None:
        out = bc.read(_OPEN, _primary(0.08, 0.06, 0.10, 0.001))
        assert (out["verdict"], out["with_the_point_worm"]) == ("placement", "a dated decision")

    def test_below_minimum_either_way(self) -> None:
        out = bc.read(_OPEN, _primary(0.03, 0.015, 0.045, 0.01))
        assert (out["verdict"], out["direction"]) == ("below_minimum", "wild_type")
        out = bc.read(_OPEN, _primary(-0.03, -0.045, -0.015, 0.01))
        assert (out["verdict"], out["direction"]) == ("below_minimum", "null")

    def test_no_wiring_effect_takes_the_stop_branch(self) -> None:
        out = bc.read(_OPEN, _primary(0.004, -0.02, 0.025, 0.6))
        assert out["verdict"] == "no_wiring_effect"
        assert out["with_the_point_worm"] == "the STOP branch is taken"

    def test_reversed(self) -> None:
        assert bc.read(_OPEN, _primary(-0.08, -0.10, -0.06, 0.001))["verdict"] == "reversed"

    def test_unresolved(self) -> None:
        assert bc.read(_OPEN, _primary(0.02, 0.004, 0.036, 0.09))["verdict"] == "unresolved"


class TestRefusals:
    def test_a_complete_panel_loads(self, tmp_path: Path) -> None:
        _panel(tmp_path)
        runs = bc.load_runs([tmp_path], _SEEDS, _SEEDS)
        assert all(len(runs[r]) == len(_SEEDS) for r in bc.STEMS[bc.LEVEL])

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        _panel(tmp_path)
        stem = bc.STEMS[bc.LEVEL]["rn_frozen"]
        (tmp_path / f"{stem}-seed4003.log").unlink()
        with pytest.raises(bc.ContrastError, match=r"rn_frozen: \[4003\]"):
            bc.load_runs([tmp_path], _SEEDS, _SEEDS)

    def test_a_failed_run_is_refused(self, tmp_path: Path) -> None:
        _panel(tmp_path)
        stem = bc.STEMS[bc.LEVEL]["wt_learn"]
        (tmp_path / f"{stem}-seed4001.exit").write_text("1\n")
        with pytest.raises(bc.ContrastError, match=r"wt_learn: \[4001\]"):
            bc.load_runs([tmp_path], _SEEDS, _SEEDS)

    def test_a_log_recording_another_config_is_refused(self, tmp_path: Path) -> None:
        _panel(tmp_path)
        stem = bc.STEMS[bc.LEVEL]["wt_learn"]
        _write(tmp_path / f"{stem}-seed4001.log", np.zeros(bc.HORIZON), config="other.yml")
        with pytest.raises(bc.ContrastError, match=r"other\.yml"):
            bc.load_runs([tmp_path], _SEEDS, _SEEDS)

    def test_a_short_run_is_refused(self, tmp_path: Path) -> None:
        _panel(tmp_path)
        stem = bc.STEMS[bc.LEVEL]["wt_frozen"]
        _write(tmp_path / f"{stem}-seed4002.log", np.zeros(bc.HORIZON - 1))
        with pytest.raises(bc.ContrastError, match="5999 of 6000"):
            bc.load_runs([tmp_path], _SEEDS, _SEEDS)


class TestPreflight:
    def test_a_learning_pilot_reads_readable(self, tmp_path: Path) -> None:
        _panel(tmp_path, wt=(0.5, 0.8), rn=(0.5, 0.8))
        out = bc.preflight([tmp_path], _SEEDS)
        assert (out["status"], out["launch"]) == ("readable", True)

    def test_a_pilot_near_the_bar_blocks_the_launch(self, tmp_path: Path) -> None:
        _panel(tmp_path, wt=(0.87, 0.9), rn=(0.6, 0.6))
        assert bc.preflight([tmp_path], _SEEDS)["status"] == "near_bar"

    def test_a_missing_pilot_has_no_evidence(self, tmp_path: Path) -> None:
        out = bc.preflight([tmp_path], _SEEDS)
        assert (out["status"], out["launch"]) == ("no_evidence", False)
