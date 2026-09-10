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
from quantumnematode.brain.modules import ModuleName
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

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_bound_is_refused(self, bad: float) -> None:
        # Every ordering check below is a comparison, and every comparison against NaN is
        # False, so a NaN bound would slip past them and make every scale NaN -- an arm that
        # perturbs nothing while reporting that it does.
        with pytest.raises(ValueError, match="must be finite"):
            NodeNoiseSchedule(initial=bad, final=_FINAL, episodes=_EPISODES)
        with pytest.raises(ValueError, match="must be finite"):
            NodeNoiseSchedule(initial=_INITIAL, final=bad, episodes=_EPISODES)


class TestTheTopologyAdvancesOnlyWhereAStepBegins:
    def test_the_first_step_runs_at_the_initial_scale(self) -> None:
        # The counter records steps BEGUN, so the running one is indexed one lower. Advancing
        # at the start of a step and reading the count directly would put the very first step
        # one notch into the decay and never run the arm at the scale it registered.
        topo = _mlp(scheduled=True)
        topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_INITIAL)

    def test_the_scale_follows_the_schedule(self) -> None:
        topo = _mlp(scheduled=True)
        assert topo.current_node_noise == pytest.approx(_INITIAL)
        # One advance per step begun: after E + 1 the E-th step is running, at the floor.
        for _ in range(_EPISODES + 1):
            topo.advance_schedule()
        assert topo.current_node_noise == pytest.approx(_FINAL)

    def test_each_step_reads_one_index_lower_than_the_count(self) -> None:
        topo = _mlp(scheduled=True)
        schedule = _schedule()
        for step in range(5):
            topo.advance_schedule()
            assert topo.current_node_noise == pytest.approx(schedule.scale_at(step))

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
        for _ in range(_EPISODES + 1):
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
        for _ in range(_EPISODES + 1):
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
        for _ in range(_EPISODES + 1):
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

    def test_the_first_episode_runs_at_the_initial_scale(self) -> None:
        brain = _brain(
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        )
        self._drive(brain)
        assert brain.topology.current_node_noise == pytest.approx(_INITIAL)

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


class TestTheYardstickBrainPerturbsToo:
    """The MLP brain is the matched-rule yardstick, so it must perturb as the connectome does."""

    def _mlp_brain(self, **over: object):
        from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig

        config: dict[str, object] = {
            "seed": _SEED,
            "action_mode": "continuous",
            "learning_rule": "three_factor",
            "enable_activity_traces": True,
            "plastic_layers": "hidden",
            "sensory_modules": [ModuleName.FOOD_CHEMOTAXIS],
        }
        config.update(over)
        return MLPPPOBrain(
            config=MLPPPOBrainConfig(**config),  # type: ignore[arg-type]
            device=DeviceType.CPU,
        )

    def test_a_configured_scale_reaches_the_topology(self) -> None:
        # It did not before: the brain built a topology that could not perturb, so a
        # `node_perturbation` arm here silently ran the Hebbian rule under the variant's name.
        brain = self._mlp_brain(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
        )
        assert brain.topology.node_noise == pytest.approx(_INITIAL)
        assert brain.topology.plastic_perturbations != []

    def test_the_configured_eligibility_reaches_the_rule(self) -> None:
        brain = self._mlp_brain(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
        )
        assert brain._rule is not None
        assert brain._rule.eligibility == "node_perturbation"

    def test_the_schedule_advances_per_episode(self) -> None:
        brain = self._mlp_brain(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        )
        brain.prepare_episode()
        assert brain.topology.current_node_noise == pytest.approx(_INITIAL)
        first = brain.topology.current_node_noise
        brain.prepare_episode()
        assert brain.topology.current_node_noise < first

    def test_loading_a_policy_restarts_the_schedule(self) -> None:
        # A warm start explores the loaded policy from the initial scale. The connectome gets
        # this through the rule's state reset; this brain's load path does not call it, so
        # without an explicit restart a warm-started arm would resume the saving run's decay.
        from quantumnematode.brain.weights import WeightComponent

        brain = self._mlp_brain(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=_INITIAL,
            plasticity_node_noise_final=_FINAL,
            plasticity_node_noise_anneal_episodes=_EPISODES,
        )
        for _ in range(_EPISODES + 1):
            brain.prepare_episode()
        assert brain.topology.current_node_noise == pytest.approx(_FINAL)

        brain.load_weight_components(
            {"policy": WeightComponent(name="policy", state=brain.actor.state_dict())},
        )
        assert brain.topology.current_node_noise == pytest.approx(_INITIAL)

    def test_the_default_construction_is_unchanged(self) -> None:
        # Wiring these through must not alter any arm that does not ask for them.
        brain = self._mlp_brain()
        assert brain.topology.node_noise == 0.0
        assert brain.topology.plastic_perturbations == []
        assert brain.topology.current_node_noise == 0.0


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
