"""A perturbation scale that decays over episodes rather than staying fixed.

The scale has two incompatible jobs. Large enough to learn with, it takes a competent policy
apart before the rule acts: a frozen control at 0.2 fell from 38.7 to 8.9 with no weight ever
written. Small enough to run a policy under, it produces an eligibility indistinguishable from
noise. These pin the schedule that separates them, and the two places it must not be advanced:
resetting the traces, which a policy load also does, and a load itself.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch._node_noise_schedule import NodeNoiseSchedule
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from torch import nn

_SEED = 4471
_INITIAL = 0.2
_FINAL = 0.02
_EPISODES = 100


def _schedule() -> NodeNoiseSchedule:
    return NodeNoiseSchedule(initial=_INITIAL, final=_FINAL, episodes=_EPISODES)


def _actor() -> nn.Sequential:
    torch.manual_seed(_SEED)
    return nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 1))


def _mlp(*, scheduled: bool, noise: float = _INITIAL) -> MLPTopology:
    return MLPTopology(
        _actor(),
        enable_activity_traces=True,
        trace_decay=0.0,
        node_noise=noise,
        node_noise_schedule=_schedule() if scheduled else None,
        perturbation_seed=_SEED,
    )


def _brain(**over: object) -> ConnectomePPOBrain:
    config: dict[str, object] = {
        "seed": _SEED,
        "action_mode": "continuous",
        "learning_rule": "three_factor",
        "enable_activity_traces": True,
        "plasticity_eligibility": "node_perturbation",
        "plasticity_node_noise": _INITIAL,
    }
    config.update(over)
    return ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(**config),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )


class TestTheShape:
    def test_it_starts_at_the_initial_scale(self) -> None:
        assert _schedule().scale_at(0) == pytest.approx(_INITIAL)

    def test_it_reaches_the_floor_exactly_at_the_anneal_length(self) -> None:
        assert _schedule().scale_at(_EPISODES) == pytest.approx(_FINAL)

    def test_it_stays_at_the_floor_after(self) -> None:
        schedule = _schedule()
        assert schedule.scale_at(_EPISODES * 5) == pytest.approx(_FINAL)

    def test_it_falls_monotonically(self) -> None:
        schedule = _schedule()
        scales = [schedule.scale_at(step) for step in range(_EPISODES + 1)]
        assert all(later < earlier for earlier, later in pairwise(scales))

    def test_it_is_geometric_not_linear(self) -> None:
        # Halfway through the decay a geometric path sits at the geometric mean of the
        # bounds, which is well below the arithmetic midpoint a linear one would reach.
        halfway = _schedule().scale_at(_EPISODES // 2)
        assert halfway == pytest.approx((_INITIAL * _FINAL) ** 0.5, rel=1e-6)
        assert halfway < (_INITIAL + _FINAL) / 2


class TestTheHelperRefusesWhatIsNotADecayToAPositiveFloor:
    def test_a_zero_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="final scale must be positive"):
            NodeNoiseSchedule(initial=_INITIAL, final=0.0, episodes=_EPISODES)

    def test_a_floor_at_or_above_the_initial_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be below initial scale"):
            NodeNoiseSchedule(initial=_FINAL, final=_INITIAL, episodes=_EPISODES)

    def test_a_non_positive_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="anneal length must be positive"):
            NodeNoiseSchedule(initial=_INITIAL, final=_FINAL, episodes=0)


class TestTheTopologyAdvancesOnlyWhereAStepBegins:
    def test_the_scale_follows_the_schedule(self) -> None:
        topo = _mlp(scheduled=True)
        assert topo.current_node_noise == pytest.approx(_INITIAL)
        for _ in range(_EPISODES):
            topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_FINAL)

    def test_resetting_the_traces_does_not_advance_it(self) -> None:
        # The rule's load-time reset also clears the traces. If the schedule rode on that,
        # a load would step it forward instead of restarting it.
        topo = _mlp(scheduled=True)
        topo.advance_schedule()
        before = topo.current_node_noise
        for _ in range(10):
            topo.reset_traces()
        assert topo.current_node_noise == pytest.approx(before)

    def test_a_reset_returns_it_to_the_initial_scale(self) -> None:
        topo = _mlp(scheduled=True)
        for _ in range(_EPISODES):
            topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_FINAL)
        topo.reset_schedule()
        assert topo.current_node_noise == pytest.approx(_INITIAL)

    def test_without_a_schedule_the_scale_never_moves(self) -> None:
        topo = _mlp(scheduled=False)
        for _ in range(_EPISODES * 2):
            topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_INITIAL)


class TestAHarnessWithoutABrainCanDriveIt:
    def test_the_seam_alone_reaches_the_floor(self) -> None:
        # The positive control builds the topology directly and never constructs a brain, so
        # a counter that only advanced from a brain would leave the control at its initial
        # scale throughout — passing a gate the schedule was never tested by.
        topo = _mlp(scheduled=True)
        for _ in range(_EPISODES):
            topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_FINAL)


class TestTheDrawUsesTheScheduledScale:
    def _perturbation_norm(self, topo: MLPTopology) -> float:
        topo.reset_traces()
        topo.forward(torch.zeros(4))
        return float(
            sum(float(p.abs().sum()) for p in topo.plastic_perturbations),
        )

    def test_a_later_step_perturbs_less(self) -> None:
        early = self._perturbation_norm(_mlp(scheduled=True))
        late_topo = _mlp(scheduled=True)
        for _ in range(_EPISODES):
            late_topo.advance_schedule()
        late = self._perturbation_norm(late_topo)
        # Same generator seed and same draw order, so the ratio is the scale ratio.
        assert late == pytest.approx(early * (_FINAL / _INITIAL), rel=1e-5)

    def test_the_perturbation_the_trace_carries_is_the_one_the_unit_acted_on(self) -> None:
        topo = _mlp(scheduled=True)
        topo.advance_schedule()
        topo.reset_traces()
        features = torch.ones(4)
        baseline = MLPTopology(
            _actor(),
            enable_activity_traces=True,
            trace_decay=0.0,
            node_noise=0.0,
            perturbation_seed=_SEED,
        )
        perturbed = topo.forward(features)
        unperturbed = baseline.forward(features)
        # The output moved, so the unit acted on the perturbation the trace now carries.
        assert not torch.allclose(perturbed, unperturbed)
        assert any(float(p.abs().sum()) > 0.0 for p in topo.plastic_perturbations)


class TestTheBrainAdvancesItPerEpisode:
    def _drive(self, brain: ConnectomePPOBrain) -> None:
        brain.prepare_episode()
        brain.run_brain(
            BrainParams(food_gradient_strength=0.3, food_gradient_direction=0.5),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )

    def test_each_episode_lowers_the_scale(self) -> None:
        brain = _brain(
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        )
        scales = []
        for _ in range(4):
            self._drive(brain)
            scales.append(brain.topology.current_node_noise)
        assert all(later < earlier for earlier, later in pairwise(scales))

    def test_without_the_fields_the_scale_is_constant(self) -> None:
        brain = _brain()
        for _ in range(4):
            self._drive(brain)
            assert brain.topology.current_node_noise == pytest.approx(_INITIAL)


class TestRefusals:
    def test_a_half_specified_schedule_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be set together"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_eligibility="node_perturbation",
                plasticity_node_noise=_INITIAL,
                plasticity_node_noise_final=_FINAL,
            )

    def test_a_length_without_a_floor_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be set together"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_eligibility="node_perturbation",
                plasticity_node_noise=_INITIAL,
                plasticity_node_noise_anneal_episodes=_EPISODES,
            )

    def test_a_zero_floor_is_refused(self) -> None:
        # Not a pedantic bound: the substrates decide at forward time whether they are
        # perturbing by testing the scale against zero, and where they are not the trace
        # carries post-synaptic activity. A schedule reaching zero would silently restore
        # the Hebbian eligibility rather than silencing the arm.
        with pytest.raises(ValidationError, match="must be positive"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_eligibility="node_perturbation",
                plasticity_node_noise=_INITIAL,
                plasticity_node_noise_final=0.0,
                plasticity_node_noise_anneal_episodes=_EPISODES,
            )

    def test_an_inverted_schedule_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="the schedule is a decay"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_eligibility="node_perturbation",
                plasticity_node_noise=_FINAL,
                plasticity_node_noise_final=_INITIAL,
                plasticity_node_noise_anneal_episodes=_EPISODES,
            )

    def test_the_construction_guard_catches_a_copied_config(self) -> None:
        config = ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        ).model_copy(update={"plasticity_node_noise_final": 0.0})
        with pytest.raises(ValueError, match="must be positive"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)

    def test_the_construction_guard_catches_a_copied_half_schedule(self) -> None:
        config = ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        ).model_copy(update={"plasticity_node_noise_anneal_episodes": None})
        with pytest.raises(ValueError, match="must be set together"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)
