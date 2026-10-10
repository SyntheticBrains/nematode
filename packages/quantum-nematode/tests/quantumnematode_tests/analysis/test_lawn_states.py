"""D.1's analysis: intake plateaus and the learning gate, windows on lawns, bouts and depletion.

Covers the patchy-lawns change's readings: a learning arm passes when its intake plateau beats its
floor's, paired by seed; a window is on a lawn only when all of its positions are; bout durations
count complete bouts within on-lawn runs; the cell's density is recovered from intake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import lawn_states as ls  # noqa: E402  # pyright: ignore[reportMissingImports]


def _log(path: Path, intakes: list[float]) -> Path:
    lines = [
        f"Run: {i + 1:<3} Status: SUCCESS Reason: max_steps            Steps: 720    "
        f"Reward:   10.00  Satiety: 50.0   Intake: {v:<8.3f} "
        for i, v in enumerate(intakes)
    ]
    path.write_text("\n".join(["noise", *lines, "tail"]) + "\n")
    return path


def test_the_plateau_is_the_final_quarter_of_intake(tmp_path: Path) -> None:
    """Eight episodes' intakes: the plateau is the mean of the last two."""
    log = _log(tmp_path / "run.log", [1, 1, 1, 1, 1, 1, 4, 6])
    assert ls.intake_plateau(log) == pytest.approx(5.0)
    assert ls.intake_plateau(tmp_path / "missing.log") is None


def test_the_learning_gate_pairs_by_seed() -> None:
    """Above the floor on every seed passes; level with it does not."""
    floor = {s: 3.0 + 0.1 * s for s in range(8)}
    assert ls.learning_gate({s: v + 5.0 for s, v in floor.items()}, floor)["passes"]
    assert not ls.learning_gate(dict(floor), floor)["passes"]


def _steps(xs: list[float], on: list[bool], intake: list[float]) -> list[BehaviourStep]:
    return [
        BehaviourStep(
            step=i,
            x=x,
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=o,
            intake=b,
            satiety=1.0,
        )
        for i, (x, o, b) in enumerate(zip(xs, on, intake, strict=True))
    ]


def test_a_window_is_on_a_lawn_only_when_all_its_positions_are() -> None:
    """Leaving the lawn at the last sample takes that window off the lawn."""
    xs = [0.0, 0.5, 1.0, 1.5, 2.0]
    steps = _steps(xs, [True, True, True, True, False], [0.0, 0.1, 0.1, 0.05, 0.0])
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    episode = ls.episode_windows(steps, classifier, intake_fraction=0.1, quality=1.0)
    assert episode.states[0] != rd.OFF_FOOD
    assert episode.states[1] == rd.OFF_FOOD
    assert episode.density[0] == pytest.approx(1.0)
    assert episode.density[1] == pytest.approx(0.5)


def test_bouts_count_only_complete_bouts_within_lawn_runs() -> None:
    """Edge bouts are open and left out; a bout ending in an off-lawn window is left out too."""
    d, r, off = rd.DWELLING, rd.ROAMING, rd.OFF_FOOD
    states = np.array([d, d, r, r, r, d, d, off, r, r])
    assert ls.bout_durations(states, rd.ROAMING) == [30.0]
    assert ls.bout_durations(states, rd.DWELLING) == []


def test_run_readings_pool_the_episodes() -> None:
    """Roaming share on lawns, and roaming where grazed against where fresh."""
    d, r, off = rd.DWELLING, rd.ROAMING, rd.OFF_FOOD
    episode = ls.Episode(
        states=np.array([d, r, r, off]),
        density=np.array([0.9, 0.2, 0.3, np.nan]),
    )
    out = ls.run_readings([episode])
    assert out["on_lawn_windows"] == 3
    assert out["roaming_fraction"] == pytest.approx(2 / 3)
    assert out["roaming_where_grazed"] == 1.0
    assert out["roaming_where_fresh"] == 0.0


def _run(arm: str, seed: int, roaming: float) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": seed,
        "roaming_fraction": roaming,
        "n_roaming_bouts": 1,
        "n_dwelling_bouts": 1,
        "roaming_where_grazed": 0.5,
        "roaming_where_fresh": 0.4,
    }


class TestPanel:
    def test_a_learner_that_dwells_well_above_its_floor_dwells(self) -> None:
        """Dwelling share 0.6 above a floor of none, on every seed, reads ``dwells``."""
        runs = [_run("internal", s, 0.4 - 0.01 * (s % 3)) for s in range(16)]
        runs += [_run("internal_frozen", s, 1.0) for s in range(16)]
        out = ls.read_panel({"passes": True}, runs)
        assert out["state"] == "move_wt"
        assert out["verdict"] == "dwells"

    def test_a_learner_level_with_its_floor_does_not(self) -> None:
        """No difference on any seed reads ``no_dwelling_at_minimum``."""
        runs = [_run("internal", s, 0.9 + 0.001 * (s % 2)) for s in range(16)]
        runs += [_run("internal_frozen", s, 0.9 + 0.001 * ((s + 1) % 2)) for s in range(16)]
        assert ls.read_panel({"passes": True}, runs)["verdict"] == "no_dwelling_at_minimum"

    def test_a_failed_learning_gate_is_unreadable(self) -> None:
        """Without the gate, nothing is classified."""
        assert ls.read_panel({"passes": False}, [])["verdict"] == "unreadable"

    def test_the_readings_beside(self) -> None:
        """Bouts of both states are counted per arm; depletion is grazed minus fresh."""
        runs = [
            _run(arm, s, 0.5) for arm in ("internal", "blind", "internal_frozen") for s in (1, 2)
        ]
        out = ls.beside(runs)
        assert out["seeds_with_both_bouts"] == {"internal": 2, "blind": 2}
        assert out["roaming_grazed_minus_fresh"]["internal"][1] == pytest.approx(0.1)
