"""Continuous-action mode for the connectome PPO brain on the continuous-2D env.

Covers the §4.3 continuous-output adapter (motor readout → 2-D Gaussian mean +
log-std) and the §4.4 strict-mask invariant: the chemical strict-mask and fixed
gap junctions are upstream of the readout, so they are byte-identical across the
discrete and continuous output modes and remain enforced under continuous PPO.
"""

from __future__ import annotations

import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID

_N_ACTIONS = 4
_CONTINUOUS_DIM = 2


def _brain(*, continuous: bool, seed: int = 0) -> ConnectomePPOBrain:
    config = ConnectomePPOBrainConfig(
        seed=seed,
        action_mode="continuous" if continuous else "discrete",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=2,
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _continuous_agent() -> QuantumNematodeAgent:
    """Build a continuous-mode connectome agent on a small continuous-2D env.

    Returns
    -------
    QuantumNematodeAgent
        An agent whose connectome brain is in continuous action mode.
    """
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
    )
    return QuantumNematodeAgent(
        brain=_brain(continuous=True),
        env=env,
        satiety_config=SatietyConfig(initial_satiety=100.0),
    )


class TestConnectomeContinuousHead:
    """The continuous adapter reshapes only the readout, not the connectome."""

    def test_readout_and_log_std_shapes(self) -> None:
        """Continuous mode: readout maps 4 motor classes → 2-D mean, plus a log-std."""
        brain = _brain(continuous=True)
        assert brain.continuous
        assert brain.topology.readout.shape == (_CONTINUOUS_DIM, _N_ACTIONS)
        assert brain.topology.log_std.shape == (_CONTINUOUS_DIM,)

    def test_discrete_readout_unchanged(self) -> None:
        """Discrete mode keeps the 4x4 readout and has no log-std."""
        brain = _brain(continuous=False)
        assert not brain.continuous
        assert brain.topology.readout.shape == (_N_ACTIONS, _N_ACTIONS)
        assert not hasattr(brain.topology, "log_std")


class TestStrictMaskInvariantAcrossOutputModes:
    """The strict-mask + gap junctions are output-mode-independent (§4.4)."""

    def test_mask_and_gap_byte_identical_across_modes(self) -> None:
        """Same seed, discrete vs continuous → identical m_chem, g_gap, initial w_chem."""
        discrete = _brain(continuous=False, seed=7)
        continuous = _brain(continuous=True, seed=7)
        assert torch.equal(discrete.topology.m_chem, continuous.topology.m_chem)
        assert torch.equal(discrete.topology.g_gap, continuous.topology.g_gap)
        assert torch.equal(discrete.topology.w_chem, continuous.topology.w_chem)

    def test_strict_mask_enforced_after_continuous_update(self) -> None:
        """After continuous PPO updates, non-wild-type chemical edges stay zero."""
        agent = _continuous_agent()
        brain = agent.brain
        assert isinstance(brain, ConnectomePPOBrain)
        not_mask = ~brain.topology.m_chem

        reward_config = RewardConfig()
        for _ in range(8):
            agent.run_episode(reward_config, max_steps=20)

        # Strict mask invariant: edges outside the wild-type mask remain exactly 0.
        assert torch.equal(
            brain.topology.w_chem.detach()[not_mask],
            torch.zeros_like(brain.topology.w_chem.detach()[not_mask]),
        )


class TestConnectomeContinuousSmoke:
    """Connectome drives the continuous-2D substrate without numerical failure."""

    def test_trains_without_nan_and_moves(self) -> None:
        """Several episodes run the PPO update and keep all parameters finite."""
        agent = _continuous_agent()
        brain = agent.brain
        assert isinstance(brain, ConnectomePPOBrain)
        env = agent.env
        assert isinstance(env, Continuous2DEnvironment)

        start = env.agents[DEFAULT_AGENT_ID].pos_continuous
        reward_config = RewardConfig()
        for _ in range(8):
            agent.run_episode(reward_config, max_steps=20)

        end = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert start is not None
        assert end is not None
        assert end != start

        params = [*brain.topology.learnable_parameters, *brain.critic.parameters()]
        for param in params:
            assert torch.isfinite(param).all()

    def test_log_std_persists_in_state_dict(self) -> None:
        """The continuous log-std round-trips through the topology state_dict."""
        brain = _brain(continuous=True)
        with torch.no_grad():
            brain.topology.log_std.copy_(torch.tensor([-0.7, 0.3]))
        state = brain.topology.state_dict()
        assert "log_std" in state

        reloaded = _brain(continuous=True)
        reloaded.topology.load_state_dict(state)
        assert torch.allclose(reloaded.topology.log_std, torch.tensor([-0.7, 0.3]))
