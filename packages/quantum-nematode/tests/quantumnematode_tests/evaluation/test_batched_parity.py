"""The batched evaluator against the per-step runner: state, trajectory and batch-size parity.

Covers the batched-evaluation spec's scenarios "One call runs many animals", "Batch size does not
change an animal", "A deliberately broken evaluator fails", "Random lawn placement is refused" and
"The classifier reads a batched animal", and the locomotor-primitives scenario "Both runners draw
the same", with a lawn-blind evaluator failing. The statistic-level layer is in
``test_batched_statistics.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from quantumnematode.evaluation.batched import BatchedEvaluator, UnsupportedConfigError
from quantumnematode.validation import roaming_dwelling as rd

from . import _parity as p

if TYPE_CHECKING:
    from pathlib import Path

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


def _lawn_ignored(evaluator: BatchedEvaluator) -> None:
    evaluator.food_dependent_reversals = False


def test_ignoring_the_lawn_fails_trajectory_parity() -> None:
    """An evaluator that gives on-lawn reversals the off-food kinematics drifts off the runner."""
    config, single, batched = p.build("constant", "uniform_lawn_flavell")
    refs = p.references(config, single, _STEPS)
    trajectories, log = p.run_batched(
        config,
        batched,
        list(p.PARITY_SEEDS),
        _STEPS,
        tamper=_lawn_ignored,
    )
    assert p.trajectory_mismatches(refs, trajectories, log)


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


def test_the_data_layer_matches_the_per_step_brain() -> None:
    """Covers the neuromodulator-layer scenario "Parity with fitted strengths set".

    A data-layer connectome with nonzero fitted strengths: features (which carry every ligand's
    concentration), action means and derivatives match the per-step brain within 1e-5 over 50 steps.
    """
    import torch
    from quantumnematode.evaluation.controllers import BatchedConnectome

    path = p.config_path("connectome", "food_removal_calhoun")
    brain, config, _ = p.harness.build_brain(path, p.PARITY_SEEDS[0], {"modulator_layer": "data"})
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        weights = brain.topology.modulator_layer_w
        weights.copy_(0.5 * torch.randn(weights.shape, generator=generator))
    refs = p.references(config, brain, _STATE_STEPS)
    _, log = p.run_batched(config, BatchedConnectome(brain), list(p.PARITY_SEEDS), _STATE_STEPS)
    assert p.state_mismatches(refs, log, _STATE_STEPS) == []
    assert np.max(np.abs(np.asarray(refs[0].modulators))) > 0


_ENCODED_SEEDS = list(range(5001, 5033))


@pytest.mark.slow
@pytest.mark.parametrize("protocol", ["scheer_lawn", "uniform_lawn_scheer"])
@pytest.mark.parametrize("wiring", ["wild_type", "rewired_degree_preserving"])
def test_the_encoded_brain_matches_the_per_step_brain(wiring: str, protocol: str) -> None:
    """Covers the batched-evaluation scenario "The encoded brain matches the per-step runner".

    An encoded connectome with a vector away from the box's centre, on the wild type and on the
    degree-preserving null, on seeds 5001-5032: features (which carry food contact and every
    ligand's concentration), action means and derivatives within 1e-5 over 50 steps, and
    trajectories over 300.
    """
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain, config = p.encoded_brain(protocol, wiring)
    refs = p.references(config, brain, _STEPS, _ENCODED_SEEDS)
    trajectories, log = p.run_batched(
        config,
        BatchedConnectome(brain),
        _ENCODED_SEEDS,
        _STEPS,
    )
    assert p.state_mismatches(refs, log, _STATE_STEPS) == []
    assert p.trajectory_mismatches(refs, trajectories, log) == []
    assert np.max(np.abs(np.asarray(refs[0].modulators))) > 0
    if protocol == "uniform_lawn_scheer":
        assert min(float(np.asarray(f)[3]) for f in refs[0].features) > 0


def test_an_encoded_animal_is_the_same_alone_and_in_a_batch() -> None:
    """The encoded brain on seeds 5001-5032: a seed alone and in the batch draws the same bouts."""
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain, config = p.encoded_brain("uniform_lawn_scheer")
    batched = BatchedConnectome(brain)
    seeds = list(range(5001, 5033))
    crowd, _ = p.run_batched(config, batched, seeds, _STEPS)
    for i in (0, 17, 31):
        alone, _ = p.run_batched(config, batched, [seeds[i]], _STEPS)
        assert np.array_equal(alone.primitive[0], crowd.primitive[i])
        assert np.allclose(alone.x[0], crowd.x[i], atol=1e-9)


@pytest.mark.slow
def test_the_slow_noise_brain_matches_the_per_step_brain() -> None:
    """Covers the batched-evaluation scenario "The two runners agree under slow noise".

    The encoded wild type with slow speed noise on Scheer's small lawn, seeds 5001-5032: features,
    action means and derivatives within 1e-5 over 50 steps, the sampled actions themselves within
    1e-5, and trajectories over 300. The speed's draws are correlated from second to second.
    """
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain, config = p.encoded_brain("scheer_lawn", encoder_speed_noise="ou")
    assert brain._speed_noise_rho is not None
    assert brain._speed_noise_rho > 0.5
    refs = p.references(config, brain, _STEPS, _ENCODED_SEEDS)
    batched = BatchedConnectome(brain)
    actions = p.logging_actions(batched)
    trajectories, log = p.run_batched(config, batched, _ENCODED_SEEDS, _STEPS)
    assert p.state_mismatches(refs, log, _STATE_STEPS) == []
    assert p.action_mismatches(refs, actions, _STATE_STEPS) == []
    assert p.trajectory_mismatches(refs, trajectories, log) == []
    speed = np.array([np.asarray(a)[0] for a in refs[0].actions])
    assert np.corrcoef(speed[:-1], speed[1:])[0, 1] > 0.5


@pytest.mark.slow
def test_the_procedure_brain_matches_the_per_step_brain() -> None:
    """Covers the batched-evaluation scenario "The two runners agree" with the procedure's options.

    The encoded wild type under per-ligand tying with the soft-capped readout, pool centres, the
    graded box, the inhibition floor, PDFR-1's and SER-5's cited signs and slow speed noise, on
    Scheer's small lawn, seeds 5001-5032: features, action means and derivatives within 1e-5 over
    50 steps, the sampled actions within 1e-5, and trajectories over 300.
    """
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain, config = p.encoded_brain(
        "scheer_lawn",
        encoder_modulator_tying="per_ligand",
        encoder_readout="soft_cap",
        encoder_pool_centres=True,
        encoder_box="graded",
        encoder_inhibition_floor=0.09,
        encoder_speed_noise="ou",
        modulator_receptor_signs=["pdfr-1", "ser-5"],
    )
    assert brain.topology.encoder.pool_centres is not None
    refs = p.references(config, brain, _STEPS, _ENCODED_SEEDS)
    batched = BatchedConnectome(brain)
    actions = p.logging_actions(batched)
    trajectories, log = p.run_batched(config, batched, _ENCODED_SEEDS, _STEPS)
    assert p.state_mismatches(refs, log, _STATE_STEPS) == []
    assert p.action_mismatches(refs, actions, _STATE_STEPS) == []
    assert p.trajectory_mismatches(refs, trajectories, log) == []


def test_a_slow_noise_animal_is_the_same_alone_and_in_a_batch() -> None:
    """Covers the batched-evaluation scenario "An animal alone matches the animal in a batch".

    The encoded brain with slow speed noise: each of three seeds alone and inside a batch of 64
    makes the same bouts over 300 steps, so each animal's noise state is its own.
    """
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain, config = p.encoded_brain("uniform_lawn_scheer", encoder_speed_noise="ou")
    batched = BatchedConnectome(brain)
    seeds = list(range(5001, 5065))
    crowd, _ = p.run_batched(config, batched, seeds, _STEPS)
    for i in (0, 33, 63):
        alone, _ = p.run_batched(config, batched, [seeds[i]], _STEPS)
        assert np.array_equal(alone.primitive[0], crowd.primitive[i])
        assert np.allclose(alone.x[0], crowd.x[i], atol=1e-9)


def _short_food_removal(tmp_path: Path) -> Path:
    """Write López-Cruz's food removal with a 100-s fed phase and a 200-s assay phase."""
    source = p.config_path("encoded", "food_removal_lopezcruz")
    text = source.read_text().replace("max_steps: 1080", "max_steps: 60", 1)
    text = text.replace("duration_s: 2700.0", "duration_s: 100.0", 1)
    text = text.replace("duration_s: 2700.0", "duration_s: 200.0", 1)
    path = tmp_path / source.name
    path.write_text(text)
    return path


_PHASE_START = 100


@pytest.mark.slow
def test_a_phase_start_keeps_the_slow_noise_state(tmp_path: Path) -> None:
    """Covers the batched-evaluation scenario "A phase start keeps the noise state".

    Food removal at 100 s, seeds 5001-5004: both runners carry the speed noise across the phase
    start and sample the same actions throughout. A batched controller that restarts the noise at
    the phase, or draws it independently, fails the same check.
    """
    from quantumnematode.evaluation.controllers import BatchedConnectome

    path = _short_food_removal(tmp_path)
    brain, config = p.encoded_brain(
        "food_removal_lopezcruz",
        path=path,
        encoder_speed_noise="ou",
    )
    steps = 2 * _PHASE_START
    refs = p.references(config, brain, steps)
    batched = BatchedConnectome(brain)
    actions = p.logging_actions(batched)
    p.run_batched(config, batched, list(p.PARITY_SEEDS), steps)
    assert p.action_mismatches(refs, actions, steps) == []

    class RestartsAtThePhase(BatchedConnectome):
        calls = 0

        def act(self, senses, noise):  # type: ignore[no-untyped-def]
            if self.calls == _PHASE_START:
                self.speed_noise = None
            self.calls += 1
            return super().act(senses, noise)

    class Independent(BatchedConnectome):
        def _sample_slow_speed(self, mean, log_std, noise, rho):  # type: ignore[no-untyped-def]
            return super()._sample_slow_speed(mean, log_std, noise, 0.0)

    for tampered in (RestartsAtThePhase(brain), Independent(brain)):
        logged = p.logging_actions(tampered)
        p.run_batched(config, tampered, list(p.PARITY_SEEDS), steps)
        assert p.action_mismatches(refs, logged, steps) != []
