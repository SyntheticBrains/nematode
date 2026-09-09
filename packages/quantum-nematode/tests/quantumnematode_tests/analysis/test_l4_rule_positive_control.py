"""The positive control's harness: the arms, the pass rule, and what makes a result void."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_rule_positive_control as pc  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.plasticity.positive_control import (  # noqa: E402
    ContextualAssociation,
)

_TASK = ContextualAssociation.default()
_FLOOR = _TASK.cue_blind_floor(pc.NOISE)
_OPTIMUM = _TASK.optimum(pc.NOISE)


def _runs(**scores: float) -> list[dict]:
    """Synthetic runs: one score per arm, repeated across every seed and rate."""
    out: list[dict] = []
    for seed in pc.SEEDS:
        for arm in ("analytic", "hebbian"):
            out.append(_run(arm, seed, scores.get(arm, _FLOOR - 0.1)))
        for rate in pc.RATE_GRID:
            out.append(_run("three_factor", seed, scores.get("three_factor", _FLOOR - 0.1), rate))
    return out


def _run(arm: str, seed: int, score: float, rate: float | None = None) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "rate": rate,
        "score": score,
        "modulator": 0.1,
        "mean_abs_delta": 1e-4,
        "alignment": 0.05,
    }


class TestThePinnedProtocol:
    def test_the_instrument_is_the_panels_recipe(self) -> None:
        assert pc.PLASTICITY_RATE == 1e-3
        assert pc.WEIGHT_BOUND == 3.0
        assert pc.TRACE_DECAY == 0.9
        assert pytest.approx(np.exp(-1.0)) == pc.NOISE  # the arms' frozen initial_log_std

    def test_the_pass_rule_is_the_registered_one(self) -> None:
        assert tuple(range(1, 9)) == pc.SEEDS
        assert pc.TRIALS == 20_000
        assert pc.PASS_SEEDS == 7
        assert pc.PASS_FRACTION == 0.5
        assert pc.RATE_GRID == (1e-4, 1e-3, 1e-2)


class TestThePassRule:
    def test_an_arm_at_the_optimum_passes(self) -> None:
        result = pc.assess([_OPTIMUM] * 8, _FLOOR, _OPTIMUM)
        assert result["passes"]

    def test_an_arm_at_the_floor_does_not(self) -> None:
        assert not pc.assess([_FLOOR] * 8, _FLOOR, _OPTIMUM)["passes"]

    def test_beating_the_floor_is_not_enough_without_the_halfway_mark(self) -> None:
        # Just above the floor on every seed, but nowhere near halfway.
        result = pc.assess([_FLOOR + 0.01] * 8, _FLOOR, _OPTIMUM)
        assert result["seeds_above_floor"] == 8
        assert not result["passes"]

    def test_halfway_is_not_enough_without_the_seeds(self) -> None:
        halfway = _FLOOR + 0.5 * (_OPTIMUM - _FLOOR)
        scores = [halfway + 1.0] * 4 + [_FLOOR - 1.0] * 4  # mean clears, seeds do not
        result = pc.assess(scores, _FLOOR, _OPTIMUM)
        assert result["seeds_above_floor"] == 4
        assert not result["passes"]

    def test_an_incomplete_arm_cannot_pass(self) -> None:
        assert not pc.assess([_OPTIMUM] * 7, _FLOOR, _OPTIMUM)["passes"]


class TestTheOutcome:
    def test_a_learning_rule_passes(self) -> None:
        out = pc.analyse(_runs(analytic=_OPTIMUM, three_factor=_OPTIMUM), _TASK)
        assert out["outcome"] == "pass"
        assert out["void_reason"] is None

    def test_a_rule_that_does_not_learn_fails(self) -> None:
        out = pc.analyse(_runs(analytic=_OPTIMUM), _TASK)
        assert out["outcome"] == "fail"

    def test_a_failed_reference_voids_the_control(self) -> None:
        # The reference is the only thing between a broken control and a false conclusion.
        out = pc.analyse(_runs(three_factor=_OPTIMUM), _TASK)
        assert out["outcome"] == "void"
        assert "carries no information" in out["void_reason"]

    def test_a_floor_arm_that_solves_it_voids_the_control(self) -> None:
        out = pc.analyse(_runs(analytic=_OPTIMUM, hebbian=_OPTIMUM, three_factor=_OPTIMUM), _TASK)
        assert out["outcome"] == "void"
        assert "leaks its answer" in out["void_reason"]

    def test_any_rate_passing_counts_as_a_pass(self) -> None:
        runs = _runs(analytic=_OPTIMUM)
        for run in runs:
            if run["arm"] == "three_factor" and run["rate"] == 1e-2:
                run["score"] = _OPTIMUM
        out = pc.analyse(runs, _TASK)
        assert out["outcome"] == "pass"
        assert out["arms"]["three_factor"]["by_rate"]["0.01"]["passes"]
        assert not out["arms"]["three_factor"]["by_rate"]["0.001"]["passes"]

    def test_the_diagnosis_is_kept_whatever_the_outcome(self) -> None:
        out = pc.analyse(_runs(analytic=_OPTIMUM), _TASK)
        assert out["outcome"] == "fail"
        for key in ("modulator", "mean_abs_delta", "alignment"):
            assert not np.isnan(out["diagnosis"]["three_factor"][key])

    def test_the_record_reports_the_budget_actually_run(self) -> None:
        assert pc.analyse(_runs(), _TASK, trials=250)["protocol"]["trials"] == 250


class TestTheArmsThemselves:
    def test_the_reference_arm_learns_the_task(self) -> None:
        # The control's validity check, run for real at a short budget.
        run = pc.run_arm("analytic", seed=1, task=_TASK, trials=1500)
        assert run["score"] > _FLOOR + 0.5 * (_OPTIMUM - _FLOOR)

    def test_the_unmodulated_arm_does_not(self) -> None:
        run = pc.run_arm("hebbian", seed=1, task=_TASK, trials=1500)
        assert run["score"] < _FLOOR + 0.5 * (_OPTIMUM - _FLOOR)

    def test_every_trial_starts_from_a_cleared_trace(self) -> None:
        # Without this the eligibility gating a trial's reward carries the previous cue.
        from quantumnematode.brain.arch._mlp_topology import MLPTopology

        seen: list[float] = []
        original = MLPTopology.reset_traces

        def spy(self: MLPTopology) -> None:
            seen.append(float(sum(float(t.abs().sum()) for t in self.eligibility_traces)))
            original(self)

        MLPTopology.reset_traces = spy  # type: ignore[method-assign]
        try:
            pc.run_arm("three_factor", seed=1, task=_TASK, trials=5)
        finally:
            MLPTopology.reset_traces = original  # type: ignore[method-assign]
        assert len(seen) == 5
