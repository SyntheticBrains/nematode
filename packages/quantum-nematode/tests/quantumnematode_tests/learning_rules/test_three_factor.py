"""Tests for the reward-modulated three-factor plasticity rule.

The rule's claims are cheap to state and easy to violate silently, so each
one is pinned: the update is the three-factor product, the modulator is a
prediction error rather than a reward, no gradient machinery is engaged,
only chemical synapses change, and plasticity stays bounded.
"""

from __future__ import annotations

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules.three_factor import (
    BASELINE_KEY,
    MEAN_ABS_DELTA_KEY,
    PREDICTION_ERROR_KEY,
    SATURATED_FRACTION_KEY,
    ConnectomeThreeFactorRule,
    ThreeFactorBatch,
)

_SEED = 4242


def _brain(**overrides: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        action_mode="continuous",
        learning_rule="three_factor",
        enable_activity_traces=True,
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _rule(topology: ConnectomeTopology, **overrides: object) -> ConnectomeThreeFactorRule:
    kwargs: dict[str, object] = {
        "plasticity_rate": 0.1,
        "weight_decay": 0.0,
        "weight_bound": 10.0,
        "baseline_rate": 0.5,
        "freeze_updates": False,
        "modulated": True,
    }
    kwargs.update(overrides)
    return ConnectomeThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


def _params(strength: float = 0.45, angle: float = 0.15) -> BrainParams:
    return BrainParams(food_gradient_strength=strength, food_gradient_direction=angle)


def _seed_trace(topology: ConnectomeTopology, value: float = 0.5) -> torch.Tensor:
    """Put a known, masked eligibility pattern on the topology."""
    with torch.no_grad():
        topology.activity_traces.copy_(
            topology.apply_weight_mask(torch.full_like(topology.activity_traces, value)),
        )
    return topology.activity_traces.detach().clone()


class TestThreeFactorProduct:
    """The update is the product of rate, prediction error, and trace."""

    def test_update_equals_the_product(self) -> None:
        brain = _brain()
        topology = brain.topology
        trace = _seed_trace(topology)
        rule = _rule(topology, plasticity_rate=0.1, baseline_rate=0.0)
        before = topology.w_chem.detach().clone()

        rule.step(topology, ThreeFactorBatch(reward=2.0))

        # baseline_rate 0 keeps the baseline at its 0.0 start, so delta == reward.
        expected = before + 0.1 * 2.0 * trace
        assert torch.allclose(topology.w_chem, expected, rtol=0.0, atol=1e-7)

    def test_zero_prediction_error_leaves_weights_alone(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology, weight_decay=0.0)
        rule.baseline = 1.0
        before = topology.w_chem.detach().clone()

        rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert torch.equal(topology.w_chem, before)

    def test_zero_trace_leaves_weights_alone(self) -> None:
        brain = _brain()
        topology = brain.topology
        topology.activity_traces.zero_()
        rule = _rule(topology, weight_decay=0.0)
        before = topology.w_chem.detach().clone()

        rule.step(topology, ThreeFactorBatch(reward=5.0))

        assert torch.equal(topology.w_chem, before)

    def test_updates_stay_on_the_edge_set(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology)

        rule.step(topology, ThreeFactorBatch(reward=3.0))

        off_edges = topology.m_chem == 0
        assert torch.all(topology.w_chem[off_edges] == 0)


class TestModulatorIsPredictionError:
    """The third factor measures surprise, not reward magnitude."""

    def test_constant_reward_stops_driving_weights(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology, weight_decay=0.0, baseline_rate=0.5)

        magnitudes = []
        for _ in range(12):
            report = rule.step(topology, ThreeFactorBatch(reward=1.0))
            magnitudes.append(abs(report.extra[PREDICTION_ERROR_KEY]))

        assert magnitudes[0] > magnitudes[-1]
        assert magnitudes[-1] == pytest.approx(0.0, abs=1e-3)

    def test_surprise_moves_weights_more_than_expectation(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology, weight_decay=0.0, baseline_rate=0.5)

        for _ in range(15):
            rule.step(topology, ThreeFactorBatch(reward=1.0))
        expected_delta = rule.step(topology, ThreeFactorBatch(reward=1.0)).extra[MEAN_ABS_DELTA_KEY]
        surprising_delta = rule.step(topology, ThreeFactorBatch(reward=5.0)).extra[
            MEAN_ABS_DELTA_KEY
        ]

        assert surprising_delta > expected_delta

    def test_baseline_survives_episode_reset(self) -> None:
        """It estimates the task's reward level, not one episode's."""
        brain = _brain()
        rule = _rule(brain.topology)
        _seed_trace(brain.topology)
        for _ in range(5):
            rule.step(brain.topology, ThreeFactorBatch(reward=2.0))
        baseline = rule.baseline
        assert baseline != 0.0

        rule.reset_episode()

        assert rule.baseline == baseline


class TestNoGradientMachinery:
    """The rule is not a gradient method wearing a hat."""

    def test_no_gradients_are_produced(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology)

        rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert topology.w_chem.grad is None

    def test_update_works_with_autograd_disabled(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology)
        before = topology.w_chem.detach().clone()

        with torch.no_grad():
            rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert not torch.equal(topology.w_chem, before)


class TestOnlyChemicalSynapsesChange:
    """Everything outside the trace's support is left at initialisation."""

    def test_gains_and_readout_are_untouched(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _rule(topology)
        gains = topology.food_gains.detach().clone()
        readout = topology.readout.detach().clone()
        log_std = topology.log_std.detach().clone()

        rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert torch.equal(topology.food_gains, gains)
        assert torch.equal(topology.readout, readout)
        assert torch.equal(topology.log_std, log_std)


class TestBoundedPlasticity:
    """Hebbian rules diverge; this one is bounded and says so."""

    def test_weights_stay_within_the_bound(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=1.0)
        rule = _rule(topology, plasticity_rate=0.5, weight_bound=0.25, weight_decay=0.0)

        for _ in range(40):
            rule.step(topology, ThreeFactorBatch(reward=10.0))
            rule.baseline = 0.0  # hold the drive at full strength

        assert topology.w_chem.abs().max().item() <= 0.25 + 1e-6

    def test_decay_is_non_increasing_without_reward_surprise(self) -> None:
        brain = _brain()
        topology = brain.topology
        topology.activity_traces.zero_()
        rule = _rule(topology, weight_decay=0.5)
        rule.baseline = 0.0
        before = topology.w_chem.abs().sum().item()

        for _ in range(5):
            rule.step(topology, ThreeFactorBatch(reward=0.0))

        assert topology.w_chem.abs().sum().item() <= before

    def test_a_synapse_may_cross_zero(self) -> None:
        """No Dale's-law clamp: signs here are arbitrary draws, not biology."""
        brain = _brain()
        topology = brain.topology
        with torch.no_grad():
            topology.w_chem.copy_(topology.apply_weight_mask(torch.ones_like(topology.w_chem)))
        _seed_trace(topology, value=1.0)
        rule = _rule(topology, plasticity_rate=1.0, weight_decay=0.0, baseline_rate=0.0)

        rule.step(topology, ThreeFactorBatch(reward=-3.0))

        edges = topology.m_chem
        assert torch.any(topology.w_chem[edges] < 0)

    def test_saturation_is_reported(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=1.0)
        rule = _rule(topology, plasticity_rate=0.5, weight_bound=0.1, weight_decay=0.0)

        report = rule.step(topology, ThreeFactorBatch(reward=10.0))
        for _ in range(30):
            rule.baseline = 0.0
            report = rule.step(topology, ThreeFactorBatch(reward=10.0))

        assert report.extra[SATURATED_FRACTION_KEY] > 0.0


class TestTelemetry:
    """A saturating or inert rule must be visible while it runs."""

    def test_report_carries_every_health_signal(self) -> None:
        brain = _brain()
        _seed_trace(brain.topology)
        rule = _rule(brain.topology)

        report = rule.step(brain.topology, ThreeFactorBatch(reward=1.0))

        for key in (
            PREDICTION_ERROR_KEY,
            BASELINE_KEY,
            MEAN_ABS_DELTA_KEY,
            SATURATED_FRACTION_KEY,
        ):
            assert key in report.extra


class TestGuards:
    """Misuse fails loudly rather than training nothing."""

    def test_foreign_topology_rejected(self) -> None:
        brain = _brain()
        other = _brain()
        rule = _rule(brain.topology)

        with pytest.raises(ValueError, match="topology other than"):
            rule.step(other.topology, ThreeFactorBatch(reward=1.0))

    def test_topology_without_traces_rejected(self) -> None:
        """The update would be identically zero, so refuse rather than idle."""
        traceless = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(seed=_SEED, action_mode="continuous"),
            device=DeviceType.CPU,
        ).topology
        rule = _rule(traceless)

        with pytest.raises(ValueError, match="requires activity traces"):
            rule.step(traceless, ThreeFactorBatch(reward=1.0))


class TestReportedChangeIsTheEffectiveChange:
    """Telemetry must describe what the weights did, not what was proposed."""

    def test_saturated_weights_report_zero_change(self) -> None:
        """A rule pinned at its clamp has become a constant function.

        Reporting the proposed update instead would show a healthy learning
        signal for a rule doing nothing at all — defeating the single metric
        meant to detect that.
        """
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=1.0)
        rule = _rule(
            topology,
            plasticity_rate=0.5,
            weight_bound=0.05,
            weight_decay=0.0,
            baseline_rate=0.0,
        )

        # Drive hard enough that every synapse reaches the bound.
        for _ in range(10):
            rule.step(topology, ThreeFactorBatch(reward=10.0))

        before = topology.w_chem.detach().clone()
        report = rule.step(topology, ThreeFactorBatch(reward=10.0))

        assert torch.equal(before, topology.w_chem), "expected a fully saturated topology"
        assert report.extra[MEAN_ABS_DELTA_KEY] == 0.0
        assert report.extra[SATURATED_FRACTION_KEY] > 0.0

    def test_reported_change_matches_measured_change(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=0.4)
        rule = _rule(topology, plasticity_rate=0.2, weight_bound=10.0)

        before = topology.w_chem.detach().clone()
        report = rule.step(topology, ThreeFactorBatch(reward=2.0))
        measured = (topology.w_chem - before).abs().mean().item()

        assert report.extra[MEAN_ABS_DELTA_KEY] == pytest.approx(measured, abs=1e-9)

    def test_partial_clamping_is_reported_as_partial(self) -> None:
        """Between "free" and "pinned" the reported change must track reality."""
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=1.0)
        # A bound just above the initial spread: some synapses clamp, most do not.
        rule = _rule(topology, plasticity_rate=0.05, weight_bound=2.0, weight_decay=0.0)

        before = topology.w_chem.detach().clone()
        report = rule.step(topology, ThreeFactorBatch(reward=5.0))
        measured = (topology.w_chem - before).abs().mean().item()

        assert measured > 0.0
        assert report.extra[MEAN_ABS_DELTA_KEY] == pytest.approx(measured, abs=1e-9)


class TestPreviousStateSurvivesStateDictRoundTrip:
    """Restoring topology state must restore whether that state is usable."""

    def test_validity_flag_round_trips(self) -> None:
        """Otherwise a restored previous step is silently treated as absent.

        ``prev_activity`` is a buffer and would restore; a plain attribute
        beside it would not, leaving the pair inconsistent and dropping one
        step of eligibility after every restore.
        """
        source = _brain()
        with torch.no_grad():
            source.topology.prev_activity.fill_(0.3)
            source.topology.prev_activity_valid.fill_(True)  # noqa: FBT003 — buffer write

        target = _brain()
        target.topology.load_state_dict(source.topology.state_dict())

        assert bool(target.topology.prev_activity_valid)
        assert torch.equal(target.topology.prev_activity, source.topology.prev_activity)

    def test_restored_state_accrues_eligibility_immediately(self) -> None:
        """The step after a restore credits the restored previous activity."""
        source = _brain()
        with torch.no_grad():
            source.topology.prev_activity.fill_(0.3)
            source.topology.prev_activity_valid.fill_(True)  # noqa: FBT003 — buffer write

        target = _brain()
        target.topology.load_state_dict(source.topology.state_dict())
        assert torch.all(target.topology.activity_traces == 0)

        torch.manual_seed(_SEED)
        target.run_brain(
            _params(),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )

        # Without the restored flag this first step would accrue nothing.
        assert target.topology.activity_traces.abs().sum() > 0

    def test_reset_clears_the_flag(self) -> None:
        brain = _brain()
        with torch.no_grad():
            brain.topology.prev_activity_valid.fill_(True)  # noqa: FBT003 — buffer write

        brain.topology.reset_traces()

        assert not bool(brain.topology.prev_activity_valid)


def _hebbian(topology: ConnectomeTopology, **overrides: object) -> ConnectomeThreeFactorRule:
    kwargs: dict[str, object] = {"modulated": False}
    kwargs.update(overrides)
    return _rule(topology, **kwargs)


class TestUnmodulatedMode:
    """The ablation floor: co-activity learning with reward observed, not used."""

    def test_update_omits_the_modulator(self) -> None:
        brain = _brain()
        topology = brain.topology
        trace = _seed_trace(topology)
        rule = _hebbian(topology, plasticity_rate=0.1, weight_decay=0.0)
        before = topology.w_chem.detach().clone()

        rule.step(topology, ThreeFactorBatch(reward=7.0))

        assert torch.allclose(topology.w_chem, before + 0.1 * trace, rtol=0.0, atol=1e-7)

    def test_reward_changes_nothing(self) -> None:
        """The defining property of the floor."""
        results = []
        for reward in (-9.0, 0.0, 12.0):
            brain = _brain()
            _seed_trace(brain.topology)
            rule = _hebbian(brain.topology, weight_decay=0.0)
            rule.step(brain.topology, ThreeFactorBatch(reward=reward))
            results.append(brain.topology.w_chem.detach().clone())

        for other in results[1:]:
            assert torch.equal(results[0], other)

    def test_reward_stream_is_still_observed(self) -> None:
        """A mislabelled or non-applied arm must be visible from telemetry."""
        modulated_brain, unmodulated_brain = _brain(), _brain()
        _seed_trace(modulated_brain.topology)
        _seed_trace(unmodulated_brain.topology)
        modulated = _rule(modulated_brain.topology)
        unmodulated = _hebbian(unmodulated_brain.topology)

        rewards = [1.0, 3.0, -2.0]
        modulated_reports = [
            modulated.step(modulated_brain.topology, ThreeFactorBatch(reward=r)) for r in rewards
        ]
        unmodulated_reports = [
            unmodulated.step(unmodulated_brain.topology, ThreeFactorBatch(reward=r))
            for r in rewards
        ]

        for expected, actual in zip(modulated_reports, unmodulated_reports, strict=True):
            assert actual.extra[PREDICTION_ERROR_KEY] == pytest.approx(
                expected.extra[PREDICTION_ERROR_KEY],
            )
            assert actual.extra[BASELINE_KEY] == pytest.approx(expected.extra[BASELINE_KEY])

    def test_modes_agree_when_the_prediction_error_is_one(self) -> None:
        """Only the modulator differs; at a modulator of 1 they must coincide."""
        modulated_brain, unmodulated_brain = _brain(), _brain()
        _seed_trace(modulated_brain.topology)
        _seed_trace(unmodulated_brain.topology)
        # baseline_rate 0 pins the baseline at 0, so a reward of 1 gives delta 1.
        modulated = _rule(modulated_brain.topology, baseline_rate=1e-12, weight_decay=0.0)
        unmodulated = _hebbian(unmodulated_brain.topology, baseline_rate=1e-12, weight_decay=0.0)

        modulated.step(modulated_brain.topology, ThreeFactorBatch(reward=1.0))
        unmodulated.step(unmodulated_brain.topology, ThreeFactorBatch(reward=1.0))

        assert torch.allclose(
            modulated_brain.topology.w_chem,
            unmodulated_brain.topology.w_chem,
            rtol=0.0,
            atol=1e-7,
        )

    def test_stabilisation_still_applies(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology, value=1.0)
        rule = _hebbian(topology, plasticity_rate=0.5, weight_bound=0.1, weight_decay=0.0)

        report = rule.step(topology, ThreeFactorBatch(reward=1.0))
        for _ in range(40):
            report = rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert topology.w_chem.abs().max().item() <= 0.1 + 1e-6
        assert report.extra[SATURATED_FRACTION_KEY] > 0.0
        # Effective-change telemetry: saturated means no movement, and says so.
        assert report.extra[MEAN_ABS_DELTA_KEY] == 0.0

    def test_masking_is_shared_with_the_modulated_path(self) -> None:
        brain = _brain()
        topology = brain.topology
        _seed_trace(topology)
        rule = _hebbian(topology)

        rule.step(topology, ThreeFactorBatch(reward=1.0))

        assert torch.all(topology.w_chem[topology.m_chem == 0] == 0)
