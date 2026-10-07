"""The body-drive action space, the anatomical readout, and their agreement with the environment.

Covers the continuous-action-policy requirement "The body-drive action space" (the action is 25
bounded numbers; a brain and an environment that disagree are refused) and the connectome-ppo-brain
requirement "The anatomical neuromuscular readout" (the readout is not learned).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._policy import BODY_DRIVE_DIM
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env.body import DRIVE_WIDTH
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.utils.config_loader import SensingConfig, SimulationConfig

_REPO = Path(__file__).resolve().parents[6]
_HARD350 = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)


def _connectome(**overrides: object) -> ConnectomePPOBrain:
    config = ConnectomePPOBrainConfig(
        seed=0,
        action_mode="continuous",
        action_space="body_drive",
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


class TestConfig:
    def test_the_widths_agree(self) -> None:
        assert BODY_DRIVE_DIM == DRIVE_WIDTH == 25

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"action_mode": "discrete"}, "requires action_mode: continuous"),
            ({"signed_speed": True}, "signed_speed is not read"),
            ({"continuous_std_mode": "state_dependent"}, "state_dependent"),
        ],
    )
    def test_refusals(self, overrides: dict[str, object], match: str) -> None:
        settings: dict[str, Any] = {"action_mode": "continuous", "action_space": "body_drive"}
        settings.update(overrides)
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(**settings)

    def test_leaky_dynamics_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dynamics='leaky'"):
            ConnectomePPOBrainConfig(
                action_mode="continuous",
                action_space="body_drive",
                dynamics="leaky",
            )


class TestAgreement:
    def _raw(self) -> dict[str, Any]:
        raw = yaml.safe_load(_HARD350.read_text())
        raw["environment"]["continuous"]["allow_reversal"] = True
        return raw

    def test_a_matched_pair_loads(self) -> None:
        raw = self._raw()
        raw["environment"]["continuous"]["body_model"] = "kinematic"
        raw["brain"]["config"]["action_space"] = "body_drive"
        SimulationConfig.model_validate(raw)

    def test_a_kinematic_body_needs_a_body_drive_brain(self) -> None:
        raw = self._raw()
        raw["environment"]["continuous"]["body_model"] = "kinematic"
        raw["brain"]["config"]["signed_speed"] = True
        with pytest.raises(ValueError, match="needs a body-drive brain"):
            SimulationConfig.model_validate(raw)

    def test_a_body_drive_brain_needs_a_kinematic_body(self) -> None:
        raw = self._raw()
        raw["brain"]["config"]["action_space"] = "body_drive"
        with pytest.raises(ValueError, match="needs a body-drive brain"):
            SimulationConfig.model_validate(raw)

    def test_a_kinematic_body_needs_reversal(self) -> None:
        raw = yaml.safe_load(_HARD350.read_text())
        raw["environment"]["continuous"]["body_model"] = "kinematic"
        raw["brain"]["config"]["action_space"] = "body_drive"
        with pytest.raises(ValueError, match="requires allow_reversal"):
            SimulationConfig.model_validate(raw)


class TestConnectomeReadout:
    def test_the_readout_is_not_learned(self) -> None:
        topo = _connectome().topology
        learnable = {id(p) for p in topo.learnable_parameters}
        assert id(topo.readout) not in learnable
        assert topo.log_std.shape == (BODY_DRIVE_DIM,)

    def test_the_mean_is_the_anatomy_read_from_the_rates(self) -> None:
        topo = _connectome().topology
        h = torch.tanh(torch.randn(topo.n_neurons, generator=torch.Generator().manual_seed(0)))
        mean = topo.body_drive_mean(h)
        assert mean.shape == (BODY_DRIVE_DIM,)
        assert bool((mean.abs() <= 1.0 + 1e-6).all())
        classes = topo._pool_motor_classes(h)
        expected = ((classes[0] + classes[1]) - (classes[2] + classes[3])) / 4.0
        assert mean[-1].item() == pytest.approx(expected.item())
        batched = topo.body_drive_mean(h.unsqueeze(0).repeat(3, 1))
        assert torch.allclose(batched, mean.expand(3, -1))

    def test_the_action_is_25_bounded_numbers(self) -> None:
        brain = _connectome()
        params = BrainParams(food_gradient_strength=0.4, food_gradient_direction=0.2)
        action = brain.run_brain(
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )[0]
        assert action.continuous is not None
        assert len(action.continuous) == BODY_DRIVE_DIM
        assert all(-1.0 <= v <= 1.0 for v in action.continuous)


class TestMlp:
    def test_the_mlp_emits_the_same_drive(self) -> None:
        config = MLPPPOBrainConfig(
            seed=0,
            action_mode="continuous",
            action_space="body_drive",
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )
        brain = MLPPPOBrain(config=config, device=DeviceType.CPU)
        assert brain._action_low.shape == (BODY_DRIVE_DIM,)
        assert brain.log_std.shape == (BODY_DRIVE_DIM,)


class TestThroughTheBody:
    def test_an_episode_runs_and_the_head_moves(self) -> None:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(
                world_size_mm=20.0,
                allow_reversal=True,
                body_model="kinematic",
            ),
            seed=0,
        )
        agent = QuantumNematodeAgent(
            brain=_connectome(rollout_buffer_size=16, num_minibatches=2, num_epochs=2),
            env=env,
            satiety_config=SatietyConfig(initial_satiety=100.0),
            sensing_config=SensingConfig(),
        )
        start = env.agents[agent.agent_id].pos_continuous
        agent.run_episode(RewardConfig(), max_steps=20)
        assert isinstance(agent.env, Continuous2DEnvironment)
        assert agent.agent_id in agent.env.bodies
        assert agent.env.agents[agent.agent_id].pos_continuous != start
