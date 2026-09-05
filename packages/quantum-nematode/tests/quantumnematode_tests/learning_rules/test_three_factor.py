"""Tests for the reward-modulated three-factor plasticity rule.

The rule's claims are cheap to state and easy to violate silently, so each
one is pinned: the update is the three-factor product, the modulator is a
prediction error rather than a reward, no gradient machinery is engaged,
only chemical synapses change, and plasticity stays bounded.
"""

from __future__ import annotations

import pytest
import torch
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


def _rule(topology: ConnectomeTopology, **overrides: float) -> ConnectomeThreeFactorRule:
    kwargs: dict[str, float] = {
        "plasticity_rate": 0.1,
        "weight_decay": 0.0,
        "weight_bound": 10.0,
        "baseline_rate": 0.5,
    }
    kwargs.update(overrides)
    return ConnectomeThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


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
