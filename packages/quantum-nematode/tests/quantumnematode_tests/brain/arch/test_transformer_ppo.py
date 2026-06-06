"""Tests for the Transformer (temporal-window self-attention) PPO brain.

Covers registration/wiring, the front-zero-padded window, and end-to-end smokes
(discrete on the grid substrate, continuous on continuous-2D) that train without
NaN/Inf.
"""

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import TransformerPPOBrain, TransformerPPOBrainConfig
from quantumnematode.brain.arch._registry import list_registered_brains
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID

_CONTINUOUS_DIM = 2


def _config(*, continuous: bool) -> TransformerPPOBrainConfig:
    """Build a small TransformerPPOBrainConfig for fast smoke tests.

    Parameters
    ----------
    continuous : bool
        If True, select the continuous (tanh-Gaussian) action head; otherwise
        the discrete (categorical) head.

    Returns
    -------
    TransformerPPOBrainConfig
        A compact config (small window/encoder + small rollout buffer).
    """
    return TransformerPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        action_mode="continuous" if continuous else "discrete",
        window_size=8,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=64,
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=2,
        seed=0,
    )


class TestTransformerWiring:
    """Registration + head shapes."""

    def test_registered(self) -> None:
        """The transformer self-registers under 'transformerppo'."""
        assert "transformerppo" in list_registered_brains()

    def test_discrete_head_shape(self) -> None:
        """Discrete mode: actor outputs the 4 action logits, no log-std."""
        brain = TransformerPPOBrain(config=_config(continuous=False))
        assert not brain.continuous
        assert brain.actor.out_features == 4
        assert not hasattr(brain, "log_std")

    def test_continuous_head_shape(self) -> None:
        """Continuous mode: actor outputs the 2-D mean + a learnable log-std."""
        brain = TransformerPPOBrain(config=_config(continuous=True))
        assert brain.continuous
        assert brain.actor.out_features == _CONTINUOUS_DIM
        assert brain.log_std.shape == (_CONTINUOUS_DIM,)

    def test_indivisible_d_model_rejected(self) -> None:
        """`d_model` not divisible by `nhead` fails fast at config validation."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="divisible by nhead"):
            TransformerPPOBrainConfig(
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
                d_model=64,
                nhead=5,
            )


class TestTemporalWindow:
    """The rolling window is front-zero-padded and reset per episode."""

    def test_window_front_padded(self) -> None:
        """Before the window fills, older positions are zero and the newest is last."""
        brain = TransformerPPOBrain(config=_config(continuous=False))
        feat = np.ones(brain.input_dim, dtype=np.float32)
        window = brain._windowed_input(feat)
        assert window.shape == (brain.window_size, brain.input_dim)
        # Only the last row is the (single) real feature; the rest are zero-padding.
        assert np.allclose(window[-1], 1.0)
        assert np.allclose(window[:-1], 0.0)

    def test_prepare_episode_resets_window(self) -> None:
        """`prepare_episode` clears the temporal window."""
        brain = TransformerPPOBrain(config=_config(continuous=False))
        brain._windowed_input(np.ones(brain.input_dim, dtype=np.float32))
        brain.prepare_episode()
        assert len(brain._window) == 0


class TestTransformerSmoke:
    """End-to-end training smokes on both substrates."""

    def test_discrete_grid_trains_without_nan(self) -> None:
        """Discrete transformer trains on the grid substrate without NaN/Inf."""
        brain = TransformerPPOBrain(config=_config(continuous=False))
        env = DynamicForagingEnvironment(
            grid_size=12,
            foraging=ForagingParams(target_foods_to_collect=3),
        )
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=SatietyConfig(initial_satiety=100.0),
        )
        reward_config = RewardConfig()
        for _ in range(6):
            agent.run_episode(reward_config, max_steps=20)
        for param in brain._trainable_parameters():
            assert torch.isfinite(param).all()

    def test_continuous_2d_trains_and_moves(self) -> None:
        """Continuous transformer trains on continuous-2D; the worm moves, params finite."""
        brain = TransformerPPOBrain(config=_config(continuous=True))
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
            seed=0,
        )
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=SatietyConfig(initial_satiety=100.0),
        )
        start = env.agents[DEFAULT_AGENT_ID].pos_continuous
        reward_config = RewardConfig()
        for _ in range(6):
            agent.run_episode(reward_config, max_steps=20)
        end = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert start is not None
        assert end is not None
        assert end != start
        for param in brain._trainable_parameters():
            assert torch.isfinite(param).all()

    def test_weight_round_trip(self) -> None:
        """Weight components round-trip (incl. continuous log-std)."""
        brain = TransformerPPOBrain(config=_config(continuous=True))
        with torch.no_grad():
            brain.log_std.copy_(torch.tensor([-0.6, 0.4]))
        components = brain.get_weight_components()
        assert {"policy", "value", "log_std"} <= set(components)

        reloaded = TransformerPPOBrain(config=_config(continuous=True))
        reloaded.load_weight_components(components)
        assert torch.allclose(reloaded.log_std, torch.tensor([-0.6, 0.4]))
