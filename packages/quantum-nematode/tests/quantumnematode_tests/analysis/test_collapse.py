"""The collapse statistic on synthetic success series.

Covers the architecture-comparison-protocol requirement "Collapse incidence is reported beside every
long learning curve": the registered definition (onset, censoring at the horizon, recovery), a
never-learning run, Wilson intervals, Kaplan-Meier time to collapse, the sensitivity grid and
Logbook 092's block rule beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import collapse as cl  # noqa: E402  # pyright: ignore[reportMissingImports]


def _series(*segments: tuple[int, float]) -> np.ndarray:
    """Concatenate deterministic segments of ``(length, success rate)``, evenly spaced successes."""
    parts = []
    for length, rate in segments:
        idx = np.arange(length)
        parts.append((np.floor((idx + 1) * rate) > np.floor(idx * rate)).astype(float))
    return np.concatenate(parts)


class TestDefinition:
    def test_a_learner_that_holds_its_peak_does_not_collapse(self) -> None:
        reading = cl.detect(_series((1000, 0.1), (5000, 0.8)))
        assert not reading.collapsed
        assert reading.peak == pytest.approx(0.8, abs=0.01)
        assert reading.final == pytest.approx(0.8, abs=0.01)

    def test_a_fall_to_zero_has_its_onset_where_the_level_crosses_half(self) -> None:
        reading = cl.detect(_series((3000, 0.8), (3000, 0.0)))
        assert reading.collapsed
        # The 100-episode mean falls below 0.4 once more than half its window is past the fall.
        assert reading.onset == pytest.approx(3051, abs=2)
        assert reading.peak_at_onset == pytest.approx(0.8, abs=0.01)
        assert reading.recovered is False

    def test_a_dip_shorter_than_the_persistence_is_not_a_collapse(self) -> None:
        reading = cl.detect(_series((3000, 0.8), (250, 0.0), (2750, 0.8)))
        assert not reading.collapsed

    def test_a_late_fall_is_censored_at_the_horizon(self) -> None:
        reading = cl.detect(_series((5800, 0.8), (200, 0.0)))
        assert not reading.collapsed
        assert reading.horizon == 6000

    def test_a_recovery_within_1000_episodes_is_recorded(self) -> None:
        reading = cl.detect(_series((3000, 0.8), (500, 0.0), (2500, 0.8)))
        assert reading.collapsed
        assert reading.recovered is True

    def test_a_never_learning_run_does_not_collapse(self) -> None:
        reading = cl.detect(np.zeros(6000))
        assert not reading.collapsed
        assert reading.peak == 0.0

    def test_an_onset_needs_a_full_window(self) -> None:
        assert not cl.detect(_series((150, 1.0), (150, 0.0))).collapsed


class TestPanel:
    def test_wilson(self) -> None:
        lo, hi = cl.wilson(0, 10)
        assert lo == 0.0
        assert hi == pytest.approx(0.2775, abs=1e-3)
        lo, hi = cl.wilson(5, 10)
        assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))

    def test_kaplan_meier_censors_the_uncollapsed_at_their_horizon(self) -> None:
        readings = [
            cl.Collapse(6000, 0.8, 0.0, 3000, 0.8, recovered=False),
            cl.Collapse(6000, 0.8, 0.0, 4000, 0.8, recovered=False),
            cl.Collapse(6000, 0.8, 0.8, None, None, None),
            cl.Collapse(6000, 0.8, 0.8, None, None, None),
        ]
        km = cl.kaplan_meier(readings)
        assert km["steps"] == [[3000, 0.75], [4000, 0.5]]
        assert km["median"] == 4000

    def test_the_grid_has_27_cells_and_one_registered(self) -> None:
        grid = cl.sensitivity({"a": _series((3000, 0.8), (3000, 0.0))})
        assert len(grid) == 27
        assert sum(cell["registered"] for cell in grid) == 1

    def test_the_092_rule(self) -> None:
        assert cl.rule_092(_series((3000, 0.8), (3000, 0.1)))["collapsed"] is True
        assert cl.rule_092(_series((6000, 0.2)))["never_learned"] is True
        assert cl.rule_092(_series((6000, 0.8)))["collapsed"] is False

    def test_a_panel_keeps_every_run_and_reads_both_horizons(self) -> None:
        series = {
            "a-seed1": _series((3000, 0.8), (3000, 0.0)),
            "a-seed2": _series((6000, 0.8)),
            "a-seed3": np.zeros(6000),
        }
        result = cl.panel(series)
        assert set(result["runs"]) == set(series)
        assert result["incidence"]["collapsed"] == 1
        assert result["incidence"]["n"] == 3
        assert result["runs"]["a-seed1"]["auc_to_3000"] == pytest.approx(0.8, abs=0.01)
        assert result["runs"]["a-seed1"]["auc_to_horizon"] == pytest.approx(0.4, abs=0.01)
        assert result["rule_092_incidence"] == 1


def test_the_series_is_read_off_the_run_lines(tmp_path: Path) -> None:
    """Only ``Run:`` lines count, in order."""
    log = tmp_path / "x-seed1.log"
    log.write_text(
        "noise\nRun: 1 Status: SUCCESS Eaten: 20/20\nRun: 2 Status: FAILED Eaten: 3/20\n"
        "Run: 3 Status: SUCCESS Eaten: 20/20\n",
    )
    assert list(cl.success_series(log)) == [1.0, 0.0, 1.0]
