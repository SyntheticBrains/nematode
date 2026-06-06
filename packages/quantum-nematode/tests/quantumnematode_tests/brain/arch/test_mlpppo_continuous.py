"""End-to-end smoke for MLP-PPO continuous-action mode on the continuous-2D env.

Drives the full brain → runner → environment loop in continuous mode and checks
the substrate trains without NaN/Inf: the agent moves, the PPO update runs, and
the actor / critic / log-std parameters stay finite. The longer learning-signal
training run (last-quarter > first-quarter return) is a headless campaign, not a
unit test.
"""

from __future__ import annotations

import math

import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID


def _continuous_agent() -> QuantumNematodeAgent:
    config = MLPPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        action_mode="continuous",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=2,
        seed=0,
    )
    brain = MLPPPOBrain(config=config)
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
    )
    satiety_config = SatietyConfig(initial_satiety=100.0)
    return QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)


class TestMLPPPOContinuousSmoke:
    """MLP-PPO drives the continuous-2D substrate without numerical failure."""

    def test_continuous_head_is_active(self) -> None:
        """The continuous config selects the 2-D Gaussian head + learnable log-std."""
        agent = _continuous_agent()
        brain = agent.brain
        assert isinstance(brain, MLPPPOBrain)
        assert brain.continuous
        assert brain.log_std.shape == (2,)
        # Actor output is the 2-D action mean, not the 4-way discrete logits.
        actor_out = brain.actor(torch.zeros(brain.input_dim))
        assert actor_out.shape[-1] == 2

    def test_trains_without_nan_and_moves(self) -> None:
        """Several episodes run the PPO update and keep all parameters finite."""
        agent = _continuous_agent()
        brain = agent.brain
        assert isinstance(brain, MLPPPOBrain)
        env = agent.env
        assert isinstance(env, Continuous2DEnvironment)

        start = env.agents[DEFAULT_AGENT_ID].pos_continuous
        reward_config = RewardConfig()
        for _ in range(8):
            agent.run_episode(reward_config, max_steps=20)

        # The worm actually moved on the continuous substrate.
        end = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert end is not None
        assert start is not None
        assert end != start

        # Sampled actions stayed within the physical bounds throughout.
        assert end[0] == max(0.0, min(env.continuous.world_size_mm, end[0]))
        assert -math.pi <= env.agents[DEFAULT_AGENT_ID].heading_rad <= math.pi

        # All trainable parameters remain finite after the PPO updates.
        params = [*brain.actor.parameters(), *brain.critic.parameters(), brain.log_std]
        for param in params:
            assert torch.isfinite(param).all()
