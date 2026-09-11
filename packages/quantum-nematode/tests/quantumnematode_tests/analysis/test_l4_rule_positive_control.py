"""The positive control's harness: the arms, the pass rule, and what makes a result void."""

from __future__ import annotations

import sys
import warnings
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
        out.extend(
            _run(arm, seed, scores.get(arm, _FLOOR - 0.1)) for arm in ("analytic", "hebbian")
        )
        out.extend(
            _run("three_factor", seed, scores.get("three_factor", _FLOOR - 0.1), rate)
            for rate in pc.RATE_GRID
        )
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
        "alignment_decay": 0.05,
        "alignment_floor": 0.05,
        "node_noise": None,
        "schedule": None,
        "normalise_trace": True,
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

    def test_the_annealed_schedule_is_the_registered_one(self) -> None:
        # Registered before the run: 0.2 is the scale that passed this control, 0.02 sits an
        # order of magnitude below it, and the decay spans the first half of the budget so the
        # score window lies entirely at the floor.
        assert pc.ANNEAL_INITIAL == 0.2
        assert pc.ANNEAL_FINAL == 0.02
        assert pc.ANNEAL_FRACTION == 0.5
        schedule = pc.annealed_schedule(pc.TRIALS)
        assert schedule.episodes == 10_000
        assert schedule.scale_at(0) == pytest.approx(0.2)
        assert schedule.scale_at(pc.TRIALS) == pytest.approx(0.02)

    def test_the_score_window_lies_entirely_at_the_floor(self) -> None:
        # The score is the mean over the last BLOCK * 10 trials. If the decay reached into
        # that window, a pass would be scored partly on exploration the arm will not do when
        # it runs.
        schedule = pc.annealed_schedule(pc.TRIALS)
        window_start = pc.TRIALS - pc.BLOCK * 10
        assert window_start > schedule.episodes


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
        # Four scores high enough that the MEAN clears the halfway mark, four below the floor:
        # the seed-count clause alone must reject it. `assess` permits above-optimum scores,
        # which is what makes the mean clause reachable while half the seeds fail.
        scores = [2.0] * 4 + [_FLOOR - 1.0] * 4
        result = pc.assess(scores, _FLOOR, _OPTIMUM)
        assert result["mean"] > result["halfway_threshold"]  # the mean clause passes
        assert result["seeds_above_floor"] == 4  # the seed clause does not
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

    def test_an_arm_with_no_finite_alignment_reports_nan_without_warning(self) -> None:
        # A run can end with no readable alignment (every block degenerate); the diagnosis must
        # carry NaN for it rather than raising or warning on an empty slice.
        runs = _runs(analytic=_OPTIMUM)
        for run in runs:
            if run["arm"] == "three_factor":
                run["alignment"] = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = pc.analyse(runs, _TASK)
        assert np.isnan(out["diagnosis"]["three_factor"]["alignment"])
        assert np.isnan(out["diagnosis"]["three_factor"]["alignment_median"])
        assert not np.isnan(out["diagnosis"]["three_factor"]["modulator"])

    def test_the_alignment_carries_a_median_beside_the_mean(self) -> None:
        # The mean of a long-tailed per-run statistic overstates the typical run, so the record
        # carries both and the prose can quote the same number the harness computed.
        runs = _runs(analytic=_OPTIMUM)
        for index, run in enumerate(r for r in runs if r["arm"] == "three_factor"):
            run["alignment"] = 1.0 if index == 0 else 0.0
        diagnosis = pc.analyse(runs, _TASK)["diagnosis"]["three_factor"]
        assert diagnosis["alignment"] > diagnosis["alignment_median"]
        assert diagnosis["alignment_median"] == pytest.approx(0.0)

    def test_the_diagnosis_is_broken_out_per_rate(self) -> None:
        out = pc.analyse(_runs(analytic=_OPTIMUM), _TASK)
        by_rate = out["diagnosis"]["three_factor"]["by_rate"]
        assert set(by_rate) == {str(rate) for rate in pc.RATE_GRID}
        assert not np.isnan(by_rate[str(pc.PLASTICITY_RATE)]["alignment"])

    def test_the_record_reports_the_budget_actually_run(self) -> None:
        assert pc.analyse(_runs(), _TASK, trials=250)["protocol"]["trials"] == 250


class TestTheAlignmentSign:
    def test_descending_the_loss_aligns_positively(self) -> None:
        import torch

        update = [torch.tensor([1.0, 0.0])]
        descent = [torch.tensor([2.0, 0.0])]  # already the -gradient direction
        assert pc._block_alignment(update, descent) == pytest.approx(1.0)

    def test_ascending_the_loss_aligns_negatively(self) -> None:
        import torch

        assert pc._block_alignment(
            [torch.tensor([-1.0, 0.0])],
            [torch.tensor([2.0, 0.0])],
        ) == pytest.approx(-1.0)

    def test_the_rule_arm_produces_a_readable_alignment(self) -> None:
        # A real run of the arm under test must yield an alignment at all; what its value says
        # about the rule is the control's finding, not this test's business.
        run = pc.run_arm("three_factor", seed=1, task=_TASK, trials=300)
        assert not np.isnan(run["alignment"])

    def test_the_reference_arm_aligns_with_its_own_descent_direction(self) -> None:
        # End to end on the same accumulation path the rule uses: the analytic arm IS gradient
        # descent, so the sign convention must put it at +1. This is what makes the rule arm's
        # near-zero alignment a finding rather than a possible sign error in the measurement.
        run = pc.run_arm("analytic", seed=1, task=_TASK, trials=300)
        assert run["alignment"] == pytest.approx(1.0, abs=1e-6)

    def test_a_degenerate_block_has_no_alignment(self) -> None:
        import torch

        assert pc._block_alignment([torch.zeros(2)], [torch.ones(2)]) is None


class TestTheAnnealedArm:
    def test_the_harness_advances_the_schedule_itself(self) -> None:
        # This control never builds a brain: it drives the topology directly. If the counter
        # only advanced from a brain, an annealed arm would run at its initial scale for every
        # trial and pass a gate the schedule was never tested by. Recording the scale the
        # topology actually perturbs at, rather than the metadata the row reports, is what
        # makes this test able to fail in that case.
        from quantumnematode.brain.arch._mlp_topology import MLPTopology

        seen: list[float] = []
        original = MLPTopology.forward

        def spy(self: MLPTopology, features: object) -> object:
            seen.append(self.current_node_noise)
            return original(self, features)  # type: ignore[arg-type]

        MLPTopology.forward = spy  # type: ignore[method-assign, assignment]
        try:
            run = pc.run_arm(
                "node_perturbation_annealed",
                seed=1,
                task=_TASK,
                trials=200,
                node_noise=pc.ANNEAL_INITIAL,
                schedule=pc.annealed_schedule(200),
            )
        finally:
            MLPTopology.forward = original  # type: ignore[method-assign]

        assert len(seen) == 200
        # The first trial runs at the initial scale, the scale falls, and by the end of the
        # decay it is at the floor and stays there.
        assert seen[0] == pytest.approx(pc.ANNEAL_INITIAL)
        assert seen[-1] == pytest.approx(pc.ANNEAL_FINAL)
        assert seen[100] == pytest.approx(pc.ANNEAL_FINAL)
        assert seen[50] < seen[0]
        assert run["schedule"] == {"initial": 0.2, "final": 0.02, "decay_trials": 100}

    def test_a_non_aligned_budget_files_blocks_by_phase(self) -> None:
        # 2500 is a permitted budget whose decay ends at 1250, mid-block. Without a flush at
        # that point the block spanning it would be filed whole by whichever phase its last
        # trial fell in, mixing decay trials into the floor average.
        schedule = pc.annealed_schedule(2500)
        assert schedule.episodes % pc.BLOCK != 0, "this budget must straddle a block boundary"
        run = pc.run_arm(
            "node_perturbation_annealed",
            seed=1,
            task=_TASK,
            trials=2500,
            node_noise=pc.ANNEAL_INITIAL,
            schedule=schedule,
        )
        assert not np.isnan(run["alignment_decay"])
        assert not np.isnan(run["alignment_floor"])

    def test_the_registered_budget_needs_no_extra_flush(self) -> None:
        # The decay length is a multiple of BLOCK there, so the boundary condition never fires
        # on its own and the blocks are exactly as they were before it was added.
        assert pc.annealed_schedule(pc.TRIALS).episodes % pc.BLOCK == 0

    def test_it_records_the_pinned_rate(self) -> None:
        # It runs at the pinned rate like every other learning arm; a null there would make
        # the row and its CSV line under-describe the run.
        run = pc.run_arm(
            "node_perturbation_annealed",
            seed=1,
            task=_TASK,
            trials=200,
            node_noise=pc.ANNEAL_INITIAL,
            schedule=pc.annealed_schedule(200),
        )
        assert run["rate"] == pc.PLASTICITY_RATE

    def test_it_perturbs_and_records_its_regime(self) -> None:
        run = pc.run_arm(
            "node_perturbation_annealed",
            seed=1,
            task=_TASK,
            trials=200,
            node_noise=pc.ANNEAL_INITIAL,
            schedule=pc.annealed_schedule(200),
        )
        assert run["node_noise"] == pc.ANNEAL_INITIAL
        # The rate regime travels with the number: the trace carries the perturbation, so
        # without trace normalisation the decay would cut the effective rate as well.
        assert run["normalise_trace"] is True

    def test_the_two_phases_are_reported_apart(self) -> None:
        # A low floor-phase alignment is what a good schedule looks like, so averaging the
        # phases would hide the failure signature rather than expose it.
        run = pc.run_arm(
            "node_perturbation_annealed",
            seed=1,
            task=_TASK,
            trials=400,
            node_noise=pc.ANNEAL_INITIAL,
            schedule=pc.annealed_schedule(400),
        )
        assert not np.isnan(run["alignment_decay"])
        assert not np.isnan(run["alignment_floor"])
        assert run["alignment_decay"] != run["alignment_floor"]

    def test_an_unscheduled_arm_reports_no_decay_phase(self) -> None:
        run = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=200,
            node_noise=0.2,
        )
        assert run["schedule"] is None
        # No schedule means no decay phase to separate: every block is at the fixed scale.
        assert np.isnan(run["alignment_decay"])
        assert not np.isnan(run["alignment_floor"])


class TestTheDelay:
    """A delay makes the eligibility horizon measurable; without one it is not."""

    def test_zero_delay_is_the_committed_control(self) -> None:
        # The anchor. An extension that moves the numbers at zero delay has replaced the
        # instrument rather than extended it.
        plain = pc.run_arm("node_perturbation", seed=1, task=_TASK, trials=600, node_noise=0.2)
        delayed = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=600,
            node_noise=0.2,
            delay=0,
        )
        assert delayed["score"] == plain["score"]
        assert delayed["credited_share"] == 1.0

    def test_the_horizon_is_unmeasurable_without_a_delay(self) -> None:
        # Why the delay exists: the undelayed control resets the trace every trial, so the decay
        # has nothing to act across and every setting gives the same answer.
        scores = {
            decay: pc.run_arm(
                "node_perturbation",
                seed=1,
                task=_TASK,
                trials=600,
                node_noise=0.2,
                trace_decay=decay,
            )["score"]
            for decay in (0.0, 0.9, 0.99)
        }
        assert len(set(scores.values())) == 1

    def test_a_delay_makes_it_measurable(self) -> None:
        scores = {
            decay: pc.run_arm(
                "node_perturbation",
                seed=1,
                task=_TASK,
                trials=600,
                node_noise=0.2,
                delay=5,
                trace_decay=decay,
            )["score"]
            for decay in (0.0, 0.9, 0.99)
        }
        assert len(set(scores.values())) > 1

    def test_the_run_records_the_knobs_it_used(self) -> None:
        run = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=200,
            node_noise=0.2,
            delay=5,
            trace_decay=0.99,
            homeostasis=False,
        )
        assert run["delay"] == 5
        assert run["trace_decay"] == 0.99
        assert run["homeostasis"] is False
        assert run["credited_share"] == pytest.approx(_TASK.credited_share(0.99, 5))


class TestTheKnobsReachTheRule:
    def test_homeostasis_changes_the_outcome(self) -> None:
        on = pc.run_arm("node_perturbation", seed=1, task=_TASK, trials=600, node_noise=0.2)
        off = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=600,
            node_noise=0.2,
            homeostasis=False,
        )
        assert on["score"] != off["score"]

    def test_the_exploration_noise_changes_the_outcome(self) -> None:
        quiet = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=600,
            node_noise=0.2,
            noise=0.22,
        )
        loud = pc.run_arm(
            "node_perturbation",
            seed=1,
            task=_TASK,
            trials=600,
            node_noise=0.2,
            noise=1.0,
        )
        assert quiet["score"] != loud["score"]
        assert quiet["action_noise"] == 0.22


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
        # The magnitudes are what matter, not the call count: the first trial starts from a
        # cleared trace, and every later one arrives carrying the previous trial's eligibility,
        # which is exactly what the reset exists to discard.
        assert seen[0] == pytest.approx(0.0)
        assert all(value > 0.0 for value in seen[1:])
