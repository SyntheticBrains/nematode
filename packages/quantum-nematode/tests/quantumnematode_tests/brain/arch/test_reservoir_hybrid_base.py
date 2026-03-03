"""Unit tests for the ReservoirHybridBase and shared PPO infrastructure.

Tests the base class config, rollout buffer, and shared methods using QRHBrain
as the concrete subclass.
"""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch._reservoir_hybrid_base import (
    ReservoirHybridBaseConfig,
    _RolloutBuffer,
)
from quantumnematode.brain.arch.qrh import QRHBrain, QRHBrainConfig
from quantumnematode.brain.modules import ModuleName


class TestReservoirHybridBaseConfig:
    """Test shared config defaults and field inheritance."""

    def test_default_config_values(self):
        """Test that base config defaults match expected values."""
        config = ReservoirHybridBaseConfig()

        assert config.readout_hidden_dim == 64
        assert config.readout_num_layers == 2
        assert config.actor_lr == 0.0003
        assert config.critic_lr == 0.0003
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.ppo_clip_epsilon == 0.2
        assert config.ppo_epochs == 4
        assert config.ppo_minibatches == 4
        assert config.ppo_buffer_size == 512
        assert config.entropy_coeff == 0.01
        assert config.value_loss_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.lr_warmup_episodes == 0
        assert config.lr_warmup_start is None
        assert config.lr_decay_episodes is None
        assert config.lr_decay_end is None
        assert config.entropy_coeff_end is None
        assert config.entropy_decay_episodes is None
        assert config.sensory_modules is None
        assert config.seed is None

    def test_custom_config_values(self):
        """Test that custom values are accepted."""
        config = ReservoirHybridBaseConfig(
            readout_hidden_dim=128,
            actor_lr=0.001,
            ppo_buffer_size=1024,
            entropy_coeff=0.05,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )

        assert config.readout_hidden_dim == 128
        assert config.actor_lr == 0.001
        assert config.ppo_buffer_size == 1024
        assert config.entropy_coeff == 0.05
        assert config.sensory_modules == [ModuleName.FOOD_CHEMOTAXIS]

    def test_qrh_config_inherits_base_fields(self):
        """QRHBrainConfig inherits all base fields."""
        config = QRHBrainConfig(actor_lr=0.001)

        assert config.actor_lr == 0.001
        assert config.readout_hidden_dim == 64  # base default
        assert config.num_reservoir_qubits == 8  # QRH-specific default
        assert isinstance(config, ReservoirHybridBaseConfig)


class TestRolloutBuffer:
    """Test the shared rollout buffer."""

    def test_buffer_add_and_length(self):
        """Test adding experiences to the buffer."""
        buf = _RolloutBuffer(buffer_size=10, device=torch.device("cpu"))
        assert len(buf) == 0
        assert not buf.is_full()

        buf.add(
            state=np.array([1.0, 2.0]),
            action=0,
            log_prob=torch.tensor(-0.5),
            value=torch.tensor([0.1]),
            reward=1.0,
            done=False,
        )

        assert len(buf) == 1
        assert buf.position == 1
        assert not buf.is_full()

    def test_buffer_is_full(self):
        """Test buffer full detection."""
        buf = _RolloutBuffer(buffer_size=4, device=torch.device("cpu"))

        for i in range(4):
            buf.add(
                state=np.array([float(i)]),
                action=i % 2,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.1]),
                reward=1.0,
                done=(i == 3),
            )

        assert buf.is_full()
        assert len(buf) == 4

    def test_buffer_reset(self):
        """Test buffer reset clears all data."""
        buf = _RolloutBuffer(buffer_size=10, device=torch.device("cpu"))
        buf.add(
            state=np.array([1.0]),
            action=0,
            log_prob=torch.tensor(-0.5),
            value=torch.tensor([0.1]),
            reward=1.0,
            done=False,
        )

        buf.reset()
        assert len(buf) == 0
        assert buf.position == 0
        assert buf.states == []

    def test_buffer_gae_computation(self):
        """Test GAE computation produces valid tensors."""
        buf = _RolloutBuffer(buffer_size=8, device=torch.device("cpu"))

        for i in range(8):
            buf.add(
                state=np.array([float(i)]),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.1 * i]),
                reward=1.0 if i % 3 == 0 else 0.0,
                done=(i == 7),
            )

        returns, advantages = buf.compute_returns_and_advantages(
            last_value=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert returns.shape == (8,)
        assert advantages.shape == (8,)
        assert not torch.isnan(returns).any()
        assert not torch.isnan(advantages).any()

    def test_buffer_minibatch_generation(self):
        """Test minibatch generation produces correct shapes."""
        buf = _RolloutBuffer(
            buffer_size=8,
            device=torch.device("cpu"),
            rng=np.random.default_rng(42),
        )

        for i in range(8):
            buf.add(
                state=np.array([float(i), float(i) * 2]),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.1]),
                reward=1.0,
                done=False,
            )

        returns = torch.randn(8)
        advantages = torch.randn(8)

        batches = list(buf.get_minibatches(2, returns, advantages))
        assert len(batches) == 2

        for batch in batches:
            assert batch["states"].shape == (4, 2)
            assert batch["actions"].shape == (4,)
            assert batch["old_log_probs"].shape == (4,)
            assert batch["returns"].shape == (4,)
            assert batch["advantages"].shape == (4,)


class TestReservoirHybridBaseViaQRH:
    """Test shared base class behavior through QRHBrain (concrete subclass)."""

    @pytest.fixture
    def brain(self):
        """Create a QRH brain with small config for testing."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        return QRHBrain(config)

    def test_brain_is_reservoir_hybrid_base(self, brain):
        """QRHBrain is an instance of ReservoirHybridBase."""
        from quantumnematode.brain.arch._reservoir_hybrid_base import ReservoirHybridBase

        assert isinstance(brain, ReservoirHybridBase)

    def test_brain_name_attribute(self, brain):
        """Brain name is set to 'QRH'."""
        assert brain._brain_name == "QRH"

    def test_base_attributes_initialized(self, brain):
        """Base class initializes all shared attributes."""
        assert brain.feature_dim > 0
        assert brain.num_actions == 4
        assert brain.input_dim == 2  # legacy mode
        assert brain.sensory_modules is None
        assert brain._episode_count == 0
        assert brain.training is True
        assert brain.current_probabilities is None
        assert brain.last_value is None

    def test_base_networks_present(self, brain):
        """Base class creates actor, critic, and feature_norm."""
        assert brain.actor is not None
        assert brain.critic is not None
        assert brain.feature_norm is not None
        assert brain.optimizer is not None

    def test_base_copy_method(self, brain):
        """Base copy() produces independent clone with same reservoir."""
        from quantumnematode.brain.arch import BrainParams

        # Do a forward pass to mutate weights
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=None,
        )
        brain.run_brain(params, top_only=False, top_randomize=False)

        copy = brain.copy()

        # Same type and feature dim
        assert type(copy) is type(brain)
        assert copy.feature_dim == brain.feature_dim
        assert copy._episode_count == brain._episode_count

        # Independent weights (same values initially)
        for p_orig, p_copy in zip(brain.actor.parameters(), copy.actor.parameters(), strict=True):
            assert torch.allclose(p_orig, p_copy)

        # Mutate copy weights — shouldn't affect original
        with torch.no_grad():
            for p in copy.actor.parameters():
                p.add_(1.0)

        for p_orig, p_copy in zip(brain.actor.parameters(), copy.actor.parameters(), strict=True):
            assert not torch.allclose(p_orig, p_copy)

    def test_base_episode_lifecycle(self, brain):
        """Test prepare_episode and post_process_episode."""
        assert brain._episode_count == 0

        brain.prepare_episode()
        assert brain._pending_state is None
        assert brain.last_value is None

        brain.post_process_episode(episode_success=True)
        assert brain._episode_count == 1
        assert brain._pending_state is None

    def test_base_lr_scheduling_disabled_by_default(self, brain):
        """LR scheduling is disabled when warmup_episodes=0 and decay=None."""
        assert not brain.lr_scheduling_enabled
        assert brain._get_current_lr() == brain.base_lr

    def test_base_lr_scheduling_warmup(self):
        """Test LR warmup progression via base class."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            actor_lr=0.001,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
            seed=42,
        )
        brain = QRHBrain(config)

        assert brain.lr_scheduling_enabled

        # Before any episodes: warmup start
        lr_0 = brain._get_current_lr()
        assert lr_0 == pytest.approx(0.0001, abs=1e-6)

        # Midpoint
        brain._episode_count = 5
        lr_5 = brain._get_current_lr()
        expected = 0.0001 + (0.001 - 0.0001) * 0.5
        assert lr_5 == pytest.approx(expected, abs=1e-6)

        # After warmup
        brain._episode_count = 10
        lr_10 = brain._get_current_lr()
        assert lr_10 == pytest.approx(0.001, abs=1e-6)

    def test_base_entropy_decay_disabled_by_default(self, brain):
        """Entropy decay is disabled when entropy_coeff_end and entropy_decay_episodes are None."""
        assert brain.entropy_coeff_end is None
        assert brain.entropy_decay_episodes is None
        # Returns static entropy_coeff when decay is disabled
        assert brain._get_current_entropy_coeff() == brain.entropy_coeff

    def test_base_entropy_decay_linear(self):
        """Test linear entropy coefficient decay over episodes."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            entropy_coeff=0.02,
            entropy_coeff_end=0.005,
            entropy_decay_episodes=100,
            seed=42,
        )
        brain = QRHBrain(config)

        # At episode 0: should be start value
        assert brain._get_current_entropy_coeff() == pytest.approx(0.02, abs=1e-6)

        # At episode 50 (midpoint): should be halfway
        brain._episode_count = 50
        expected_mid = 0.02 + 0.5 * (0.005 - 0.02)  # 0.0125
        assert brain._get_current_entropy_coeff() == pytest.approx(expected_mid, abs=1e-6)

        # At episode 100 (end): should be end value
        brain._episode_count = 100
        assert brain._get_current_entropy_coeff() == pytest.approx(0.005, abs=1e-6)

        # Beyond decay episodes: should stay at end value
        brain._episode_count = 200
        assert brain._get_current_entropy_coeff() == pytest.approx(0.005, abs=1e-6)

    def test_base_entropy_decay_partial_config(self):
        """Entropy decay requires both end and episodes to be set."""
        # Only entropy_coeff_end set, no decay episodes
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            entropy_coeff=0.02,
            entropy_coeff_end=0.005,
            entropy_decay_episodes=None,
            seed=42,
        )
        brain = QRHBrain(config)
        # Should return static value since decay_episodes is None
        assert brain._get_current_entropy_coeff() == pytest.approx(0.02, abs=1e-6)

    def test_base_action_set_property(self, brain):
        """Test action_set getter and setter."""
        assert len(brain.action_set) == 4

        new_actions = [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        brain.action_set = new_actions
        assert brain.action_set == new_actions

        with pytest.raises(ValueError, match="Cannot set action_set"):
            brain.action_set = [Action.FORWARD, Action.LEFT]  # wrong length

    def test_base_update_memory_noop(self, brain):
        """update_memory is a no-op and doesn't raise."""
        brain.update_memory(reward=1.0)
        brain.update_memory(reward=None)

    def test_deferred_ppo_update_flag_initialized(self, brain):
        """Deferred PPO update flag starts as False."""
        assert brain._deferred_ppo_update is False

    def test_prepare_episode_resets_deferred_flag(self, brain):
        """prepare_episode() clears any pending deferred PPO update."""
        brain._deferred_ppo_update = True
        brain.prepare_episode()
        assert brain._deferred_ppo_update is False

    def test_buffer_full_mid_episode_defers_update(self):
        """Buffer filling mid-episode sets _deferred_ppo_update instead of updating."""
        from quantumnematode.brain.arch import BrainParams

        # Buffer size of 4 so we can fill it quickly
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            ppo_buffer_size=4,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = QRHBrain(config)
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0, agent_direction=None)

        # Fill the buffer without triggering episode_done — this should defer, not update
        for _ in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        assert brain._deferred_ppo_update is True
        # Buffer should still be full (not reset yet)
        assert len(brain.buffer) == brain.config.ppo_buffer_size

    def test_deferred_update_executes_on_next_run_brain(self):
        """Deferred PPO update runs during next run_brain() with correct bootstrap value."""
        from quantumnematode.brain.arch import BrainParams

        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            ppo_buffer_size=4,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = QRHBrain(config)
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0, agent_direction=None)

        # Fill buffer mid-episode to trigger deferred flag
        for _ in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        assert brain._deferred_ppo_update is True

        # Next run_brain() should execute the deferred update and clear the buffer
        brain.run_brain(params, top_only=False, top_randomize=False)

        assert brain._deferred_ppo_update is False
        # Buffer should be reset after the deferred update
        assert len(brain.buffer) == 0

    def test_deferred_update_flushed_when_episode_ends_before_next_run_brain(self):
        """Deferred update is flushed cleanly if the episode ends before the next run_brain()."""
        from quantumnematode.brain.arch import BrainParams

        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            ppo_buffer_size=4,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = QRHBrain(config)
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0, agent_direction=None)

        # Fill buffer mid-episode → deferred flag set, buffer is full
        for _ in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        assert brain._deferred_ppo_update is True

        # Simulate the episode ending before the next run_brain(): call learn(episode_done=True)
        # while the deferred flag is still set. The early-flush guard fires first (flushing
        # the full buffer and resetting it), then the terminal transition is added (1 item).
        # With only 1 item that's below ppo_minibatches=2, no terminal update fires.
        # The key invariants: deferred flag is cleared and there is no buffer overflow.
        brain.learn(params, reward=1.0, episode_done=True)

        assert brain._deferred_ppo_update is False
        # Buffer has exactly 1 item (the terminal transition), not buffer_size+1
        assert len(brain.buffer) == 1

        # A subsequent run_brain() can proceed normally — flag is clear, no deferred flush
        brain.run_brain(params, top_only=False, top_randomize=False)
        assert brain._deferred_ppo_update is False

    def test_episode_done_triggers_immediate_update(self):
        """Episode-end buffer flushes still trigger an immediate (non-deferred) update."""
        from quantumnematode.brain.arch import BrainParams

        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = QRHBrain(config)
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0, agent_direction=None)

        # Run a few steps then end the episode
        for i in range(4):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 3))

        # Immediate update on done — deferred flag should never have been set
        assert brain._deferred_ppo_update is False
        # Buffer should be reset after immediate update
        assert len(brain.buffer) == 0
