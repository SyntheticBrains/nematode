"""End-to-end smoke for LSTM-PPO continuous-action mode on the continuous-2D env.

Drives the full brain → runner → environment loop in continuous mode through the
recurrent buffer + chunk-based BPTT update, checking it trains without NaN/Inf:
the agent moves, the PPO update runs, and the actor / critic / log-std parameters
stay finite.
"""

from __future__ import annotations

import math

import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import LSTMPPOBrain, LSTMPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID


def _continuous_agent() -> QuantumNematodeAgent:
    """Build a continuous-mode LSTM-PPO agent on a small continuous-2D env.

    Returns
    -------
    QuantumNematodeAgent
        An agent whose recurrent brain is in continuous action mode, wired to a
        small `Continuous2DEnvironment` for fast smoke training.
    """
    config = LSTMPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        action_mode="continuous",
        lstm_hidden_dim=16,
        rollout_buffer_size=16,
        bptt_chunk_length=4,
        seed=0,
    )
    brain = LSTMPPOBrain(config=config)
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
    )
    satiety_config = SatietyConfig(initial_satiety=100.0)
    return QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)


class TestLSTMPPOContinuousSmoke:
    """LSTM-PPO drives the continuous-2D substrate without numerical failure."""

    def test_continuous_head_is_active(self) -> None:
        """The continuous config selects the 2-D Gaussian head + learnable log-std."""
        brain = _continuous_agent().brain
        assert isinstance(brain, LSTMPPOBrain)
        assert brain.continuous
        assert brain.log_std.shape == (2,)
        assert brain.actor[-1].out_features == 2  # type: ignore[index]

    def test_trains_without_nan_and_moves(self) -> None:
        """Several episodes run the BPTT update and keep all parameters finite."""
        agent = _continuous_agent()
        brain = agent.brain
        assert isinstance(brain, LSTMPPOBrain)
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

        params = [
            *brain.rnn.parameters(),
            *brain.actor.parameters(),
            *brain.critic.parameters(),
            brain.log_std,
        ]
        for param in params:
            assert torch.isfinite(param).all()

    def test_log_std_persists_across_save_load(self) -> None:
        """The continuous log-std survives a weight save/load round-trip."""
        brain = _continuous_agent().brain
        assert isinstance(brain, LSTMPPOBrain)
        with torch.no_grad():
            brain.log_std.copy_(torch.tensor([-0.7, 0.3]))

        components = brain.get_weight_components()
        assert "log_std" in components

        reloaded = _continuous_agent().brain
        assert isinstance(reloaded, LSTMPPOBrain)
        reloaded.load_weight_components(components)
        assert torch.allclose(reloaded.log_std, torch.tensor([-0.7, 0.3]))
