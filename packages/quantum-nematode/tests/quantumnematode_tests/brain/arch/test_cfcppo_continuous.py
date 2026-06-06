"""End-to-end smoke for CfC-PPO continuous-action mode on the continuous-2D env.

Drives the full brain → runner → environment loop in continuous mode through the
AutoNCP motor pool + chunk-based BPTT update, checking it trains without NaN/Inf:
the agent moves, the PPO update runs, and the cfc / head / critic / log-std
parameters stay finite. Covered for both actor-head modes (motor, mlp).
"""

from __future__ import annotations

import math

import pytest
import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import CfCBrainConfig, CfCPPOBrain
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID


def _continuous_agent(actor_head: str = "motor") -> QuantumNematodeAgent:
    """Build a continuous-mode CfC-PPO agent on a small continuous-2D env.

    Parameters
    ----------
    actor_head : str
        Which CfC actor head to use (``"motor"`` or ``"mlp"``).

    Returns
    -------
    QuantumNematodeAgent
        An agent whose CfC brain is in continuous action mode, wired to a small
        `Continuous2DEnvironment` for fast smoke training.
    """
    config = CfCBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        action_mode="continuous",
        actor_head=actor_head,  # type: ignore[arg-type]
        units=16,
        rollout_buffer_size=16,
        bptt_chunk_length=4,
        seed=0,
    )
    brain = CfCPPOBrain(config=config)
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
    )
    satiety_config = SatietyConfig(initial_satiety=100.0)
    return QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)


class TestCfCPPOContinuousSmoke:
    """CfC-PPO drives the continuous-2D substrate without numerical failure."""

    @pytest.mark.parametrize("actor_head", ["motor", "mlp"])
    def test_continuous_head_is_active(self, actor_head: str) -> None:
        """The continuous config selects a 2-D motor pool + learnable log-std."""
        brain = _continuous_agent(actor_head).brain
        assert isinstance(brain, CfCPPOBrain)
        assert brain.continuous
        assert brain.log_std.shape == (2,)

    @pytest.mark.parametrize("actor_head", ["motor", "mlp"])
    def test_trains_without_nan_and_moves(self, actor_head: str) -> None:
        """Several episodes run the BPTT update and keep all parameters finite."""
        agent = _continuous_agent(actor_head)
        brain = agent.brain
        assert isinstance(brain, CfCPPOBrain)
        env = agent.env
        assert isinstance(env, Continuous2DEnvironment)

        start = env.agents[DEFAULT_AGENT_ID].pos_continuous
        reward_config = RewardConfig()
        for _ in range(8):
            agent.run_episode(reward_config, max_steps=20)

        end = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert end is not None
        assert start is not None
        assert end != start
        assert -math.pi <= env.agents[DEFAULT_AGENT_ID].heading_rad <= math.pi

        params = [*brain.cfc.parameters(), *brain.critic.parameters(), brain.log_std]
        for param in params:
            assert torch.isfinite(param).all()

    def test_log_std_persists_across_save_load(self) -> None:
        """The continuous log-std survives a weight save/load round-trip."""
        brain = _continuous_agent().brain
        assert isinstance(brain, CfCPPOBrain)
        with torch.no_grad():
            brain.log_std.copy_(torch.tensor([-0.7, 0.3]))

        components = brain.get_weight_components()
        assert "log_std" in components

        reloaded = _continuous_agent().brain
        assert isinstance(reloaded, CfCPPOBrain)
        reloaded.load_weight_components(components)
        assert torch.allclose(reloaded.log_std, torch.tensor([-0.7, 0.3]))
