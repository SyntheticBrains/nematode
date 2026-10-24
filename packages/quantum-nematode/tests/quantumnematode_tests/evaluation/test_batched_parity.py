"""The batched evaluator against the per-step runner: state, trajectory and batch-size parity.

Covers the batched-evaluation spec's scenarios "One call runs many animals", "Batch size does not
change an animal", "A deliberately broken evaluator fails", "Random lawn placement is refused" and
"The classifier reads a batched animal". The statistic-level layer is in
``test_batched_statistics.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.evaluation.batched import BatchedEvaluator, UnsupportedConfigError
from quantumnematode.validation import roaming_dwelling as rd

from . import _parity as p

_STEPS = 300
_STATE_STEPS = 50


@pytest.mark.parametrize("protocol", p.PROTOCOLS)
@pytest.mark.parametrize("controller", p.CONTROLLERS)
def test_state_and_trajectory_parity(controller: str, protocol: str) -> None:
    """Features, means and derivatives within 1e-5 over 50 steps; trajectories over 300."""
    config, single, batched = p.build(controller, protocol)
    refs = p.references(config, single, _STEPS)
    trajectories, log = p.run_batched(config, batched, list(p.PARITY_SEEDS), _STEPS)
    assert p.state_mismatches(refs, log, _STATE_STEPS) == []
    assert p.trajectory_mismatches(refs, trajectories, log) == []


def test_batch_size_does_not_change_an_animal() -> None:
    """A seed alone and inside a batch of 64 makes the same bouts, in the same places."""
    config, _, batched = p.build("mlp", "uniform_lawn_flavell")
    alone, _ = p.run_batched(config, batched, [5001], _STEPS)
    seeds = list(range(5001, 5033)) * 2
    crowd, _ = p.run_batched(config, batched, seeds, _STEPS)
    assert np.array_equal(alone.primitive[0], crowd.primitive[0])
    horizon = p.POSITION_HORIZON
    gap = np.hypot(
        alone.x[0, :horizon] - crowd.x[0, :horizon],
        alone.y[0, :horizon] - crowd.y[0, :horizon],
    )
    assert np.max(gap) <= p.POSITION_TOLERANCE
    assert np.array_equal(crowd.primitive[0], crowd.primitive[32])


def test_one_call_runs_many_animals() -> None:
    """256 seeds of an assay come back as 256 records the length of the assay."""
    config, _, batched = p.build("constant", "scheer_lawn")
    seeds = [5001 + (i % 32) for i in range(256)]
    trajectories = BatchedEvaluator(config, batched, seeds).run()
    assert trajectories.x.shape == (256, config.max_steps)
    assert trajectories.assay is not None


def _derivative_scale_written(evaluator: BatchedEvaluator) -> None:
    evaluator.derivative_scale = 50.0


def _sides_swapped(evaluator: BatchedEvaluator) -> None:
    evaluator.sweep = -evaluator.sweep


def test_a_wrong_derivative_scale_fails_state_parity() -> None:
    """Leaving the derivative scale at its written value is caught at the state level."""
    config, single, batched = p.build("mlp", "food_removal_calhoun")
    refs = p.references(config, single, _STATE_STEPS)
    _, log = p.run_batched(
        config,
        batched,
        list(p.PARITY_SEEDS),
        _STATE_STEPS,
        tamper=_derivative_scale_written,
    )
    assert p.state_mismatches(refs, log, _STATE_STEPS)


def test_swapped_head_sweep_sides_fail_both_layers() -> None:
    """Swapping the head-sweep sides is caught at the state and the trajectory level."""
    config, single, batched = p.build("mlp", "food_removal_calhoun")
    refs = p.references(config, single, _STEPS)
    trajectories, log = p.run_batched(
        config,
        batched,
        list(p.PARITY_SEEDS),
        _STEPS,
        tamper=_sides_swapped,
    )
    assert p.state_mismatches(refs, log, _STATE_STEPS)
    assert p.trajectory_mismatches(refs, trajectories, log)


def test_random_lawn_placement_is_refused() -> None:
    """Every animal must share one lawn's geometry."""
    config, _, batched = p.build("constant", "food_removal_calhoun")
    environment = config.environment
    lawns = environment.foraging.lawns.model_copy(update={"placement": "random"})
    foraging = environment.foraging.model_copy(update={"lawns": lawns})
    changed = config.model_copy(
        update={"environment": environment.model_copy(update={"foraging": foraging})},
    )
    with pytest.raises(UnsupportedConfigError, match="placement"):
        BatchedEvaluator(changed, batched, [5001])


def test_the_classifier_reads_a_batched_animal() -> None:
    """A batched Flavell animal's record classifies through the lawn-state windows unchanged."""
    import sys

    sys.path.insert(0, str(p.REPO / "scripts" / "analysis"))
    import lawn_states as ls  # pyright: ignore[reportMissingImports]

    config, _, batched = p.build("pirouette", "uniform_lawn_flavell")
    trajectories = BatchedEvaluator(
        config.model_copy(update={"max_steps": 600}),
        batched,
        [5001],
    ).run()
    steps = trajectories.to_behaviour_steps(0)
    lawns = config.environment.foraging.lawns
    episode = ls.episode_windows(
        steps,
        rd.GaussianClassifier(hmm=rd.load_calibrated_hmm()),
        lawns.intake_fraction,
        1.0,
        step_seconds=1.0,
    )
    assert len(episode.states) == (len(steps) - 1) // 10
    assert steps[0].phase == "assay"


def test_recorded_gradients_match_the_per_step_capture() -> None:
    """With the gradient recorded, a batched capture's gradient fields are the per-step runner's."""
    config, single, batched = p.build("pirouette", "food_removal_calhoun")
    refs = p.references(config, single, 50)
    evaluator = BatchedEvaluator(
        config.model_copy(update={"max_steps": 50}),
        batched,
        list(p.PARITY_SEEDS),
        record_gradient=True,
    )
    trajectories = evaluator.run()
    for i, ref in enumerate(refs):
        steps = trajectories.to_behaviour_steps(i)
        for t, step in enumerate(ref.behaviour):
            assert steps[t].grad_strength == pytest.approx(step.grad_strength, abs=1e-6)
            # A pull of rounding size, as at the lawn's exact centre, has no meaningful direction.
            if step.grad_strength > 1e-9:
                assert steps[t].grad_dir == pytest.approx(step.grad_dir, abs=1e-6)
    assert max(abs(s.grad_strength) for s in refs[0].behaviour) > 0
