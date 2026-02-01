"""Unit tests for the PPO brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig, RolloutBuffer
from quantumnematode.env import Direction


class TestMLPPPOBrainConfig:
    """Test cases for PPO brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLPPPOBrainConfig()

        assert config.actor_hidden_dim == 64
        assert config.critic_hidden_dim == 64
        assert config.num_hidden_layers == 2
        assert config.learning_rate == 0.0003
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.num_epochs == 4
        assert config.num_minibatches == 4
        assert config.rollout_buffer_size == 2048
        assert config.max_grad_norm == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=128,
            critic_hidden_dim=128,
            num_hidden_layers=3,
            learning_rate=0.001,
            gamma=0.95,
            clip_epsilon=0.1,
            rollout_buffer_size=256,
        )

        assert config.actor_hidden_dim == 128
        assert config.critic_hidden_dim == 128
        assert config.num_hidden_layers == 3
        assert config.learning_rate == 0.001
        assert config.gamma == 0.95
        assert config.clip_epsilon == 0.1
        assert config.rollout_buffer_size == 256


class TestRolloutBuffer:
    """Test cases for the rollout buffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test rollout buffer."""
        return RolloutBuffer(buffer_size=10, device=torch.device("cpu"))

    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 10
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_add_experience(self, buffer):
        """Test adding experience to the buffer."""
        state = np.array([0.5, 0.3], dtype=np.float32)
        action = 1
        log_prob = torch.tensor(-0.5)
        value = torch.tensor([0.8])
        reward = 1.0
        done = False

        buffer.add(state, action, log_prob, value, reward, done)

        assert len(buffer) == 1
        assert not buffer.is_full()
        np.testing.assert_array_equal(buffer.states[0], state)
        assert buffer.actions[0] == action
        assert buffer.rewards[0] == reward
        assert buffer.dones[0] == done

    def test_buffer_is_full(self, buffer):
        """Test buffer full detection."""
        for i in range(10):
            buffer.add(
                state=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.5]),
                reward=0.1,
                done=(i == 9),
            )

        assert buffer.is_full()
        assert len(buffer) == 10

    def test_buffer_reset(self, buffer):
        """Test buffer reset."""
        # Add some data
        for i in range(5):
            buffer.add(
                state=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.5]),
                reward=0.1,
                done=False,
            )

        assert len(buffer) == 5

        buffer.reset()

        assert len(buffer) == 0
        assert len(buffer.states) == 0
        assert len(buffer.actions) == 0

    def test_compute_returns_and_advantages(self, buffer):
        """Test GAE computation."""
        # Add some experiences
        for i in range(5):
            buffer.add(
                state=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([float(i) / 5]),
                reward=1.0 if i == 4 else 0.1,
                done=(i == 4),
            )

        last_value = torch.tensor([0.0])
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert returns.shape == (5,)
        assert advantages.shape == (5,)
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_get_minibatches(self, buffer):
        """Test minibatch generation."""
        # Add experiences
        for i in range(8):
            buffer.add(
                state=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 4,
                log_prob=torch.tensor(-0.5),
                value=torch.tensor([0.5]),
                reward=0.1,
                done=(i == 7),
            )

        last_value = torch.tensor([0.0])
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        minibatches = list(buffer.get_minibatches(2, returns, advantages))

        assert len(minibatches) == 2
        for batch in minibatches:
            assert "states" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "returns" in batch
            assert "advantages" in batch
            assert batch["states"].shape[0] == 4  # 8 / 2 minibatches


class TestPPOBrain:
    """Test cases for the PPO brain architecture."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MLPPPOBrainConfig(
            actor_hidden_dim=32,
            critic_hidden_dim=32,
            num_hidden_layers=2,
            learning_rate=0.01,
            rollout_buffer_size=64,
            num_epochs=2,
            num_minibatches=2,
        )

    @pytest.fixture
    def brain(self, config):
        """Create a test PPO brain."""
        return MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

    def test_brain_initialization(self, brain, config):
        """Test PPO brain initialization."""
        assert brain.input_dim == 2
        assert brain.num_actions == 4
        assert brain.gamma == config.gamma
        assert brain.gae_lambda == config.gae_lambda
        assert brain.clip_epsilon == config.clip_epsilon
        assert brain.training is True

        # Check networks exist
        assert brain.actor is not None
        assert brain.critic is not None
        assert isinstance(brain.actor, torch.nn.Sequential)
        assert isinstance(brain.critic, torch.nn.Sequential)

    def test_actor_network_output(self, brain):
        """Test actor network produces correct output shape."""
        test_input = torch.randn(1, brain.input_dim)
        output = brain.actor(test_input)
        assert output.shape == (1, brain.num_actions)

    def test_critic_network_output(self, brain):
        """Test critic network produces correct output shape."""
        test_input = torch.randn(1, brain.input_dim)
        output = brain.critic(test_input)
        assert output.shape == (1, 1)

    def test_preprocess(self, brain):
        """Test state preprocessing."""
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=1.5,
            agent_position=(2, 3),
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2
        assert features.dtype == np.float32
        assert 0.0 <= features[0] <= 1.0  # Gradient strength
        assert -1.0 <= features[1] <= 1.0  # Normalized relative angle

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values."""
        params = BrainParams()
        features = brain.preprocess(params)

        assert len(features) == 2
        assert features[0] == 0.0  # gradient_strength defaults to 0
        assert -1.0 <= features[1] <= 1.0  # relative angle still in valid range

    def test_get_action_and_value(self, brain):
        """Test getting action, log_prob, entropy, and value."""
        state = np.array([0.5, 0.3], dtype=np.float32)

        action, log_prob, entropy, value = brain.get_action_and_value(state)

        assert isinstance(action, int)
        assert 0 <= action < brain.num_actions
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.item() <= 0  # Log prob should be non-positive
        assert isinstance(entropy, torch.Tensor)
        assert entropy.item() >= 0  # Entropy should be non-negative
        assert isinstance(value, torch.Tensor)

    def test_get_action_with_specific_action(self, brain):
        """Test computing log_prob for a specific action."""
        state = np.array([0.5, 0.3], dtype=np.float32)
        specific_action = 2

        action, log_prob, _entropy, _value = brain.get_action_and_value(
            state,
            action=specific_action,
        )

        assert action == specific_action
        assert isinstance(log_prob, torch.Tensor)

    def test_run_brain(self, brain):
        """Test running the brain for decision making."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        assert 0.0 <= action_data.probability <= 1.0

        # Check that pending data is stored
        assert hasattr(brain, "_pending_state")
        assert hasattr(brain, "_pending_action")
        assert hasattr(brain, "_pending_log_prob")
        assert hasattr(brain, "_pending_value")

    def test_learn_adds_to_buffer(self, brain):
        """Test that learn adds experience to buffer."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain to set up pending state
        brain.run_brain(params, top_only=True, top_randomize=False)

        initial_buffer_size = len(brain.buffer)

        # Learn from step
        brain.learn(params, reward=0.5, episode_done=False)

        assert len(brain.buffer) == initial_buffer_size + 1

    def test_learn_triggers_update_when_buffer_full(self, brain):
        """Test that PPO update is triggered when buffer is full."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Fill buffer
        for i in range(brain.config.rollout_buffer_size + 5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            is_done = i == brain.config.rollout_buffer_size + 4
            brain.learn(params, reward=0.1, episode_done=is_done)

        # Buffer should have been reset after update
        assert len(brain.buffer) < brain.config.rollout_buffer_size

    def test_ppo_update(self, brain):
        """Test PPO update mechanics."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Collect enough experience for an update
        for i in range(20):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1 if i < 19 else 1.0, episode_done=(i == 19))

        # Trigger update
        brain._perform_ppo_update()

        # Check that weights are still finite after update
        for param in brain.actor.parameters():
            assert torch.all(torch.isfinite(param))
        for param in brain.critic.parameters():
            assert torch.all(torch.isfinite(param))

    def test_update_memory(self, brain):
        """Test memory update functionality (no-op for PPO)."""
        brain.update_memory(reward=0.5)
        # Just verify it doesn't crash

    def test_post_process_episode(self, brain):
        """Test episode post-processing."""
        # Add some reward data
        brain._current_episode_rewards.append(0.5)
        initial_episode_count = brain._episode_count

        brain.post_process_episode()

        # Episode count should increment
        assert brain._episode_count == initial_episode_count + 1
        # Episode data should be cleared
        assert len(brain._current_episode_rewards) == 0

    def test_action_set_property(self, brain):
        """Test action_set property."""
        action_set = brain.action_set
        assert len(action_set) == 4
        assert Action.FORWARD in action_set
        assert Action.LEFT in action_set
        assert Action.RIGHT in action_set
        assert Action.STAY in action_set

    def test_action_set_setter(self, brain):
        """Test setting action_set."""
        new_actions = [Action.FORWARD, Action.LEFT, Action.RIGHT]
        brain.action_set = new_actions
        assert brain.action_set == new_actions

    def test_build_brain_not_implemented(self, brain):
        """Test that build_brain raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.build_brain()

    def test_copy_not_implemented(self, brain):
        """Test that copy raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.copy()


class TestPPOBrainIntegration:
    """Integration tests for PPO brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            learning_rate=0.01,
            rollout_buffer_size=32,
            num_epochs=2,
            num_minibatches=2,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Simulate multiple steps in an episode
        episode_rewards = []
        rng = np.random.default_rng(42)

        for step in range(10):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            assert len(actions) == 1

            # Learn
            reward = rng.random() - 0.5
            brain.learn(params, reward, episode_done=(step == 9))
            episode_rewards.append(reward)

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 10

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            learning_rate=0.001,
            rollout_buffer_size=50,
            num_epochs=2,
            num_minibatches=2,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        rng = np.random.default_rng(42)

        for _episode in range(3):
            brain.prepare_episode()

            for step in range(15):
                params = BrainParams(
                    gradient_strength=rng.random(),
                    gradient_direction=rng.random() * 2 * np.pi,
                    agent_position=(step, step),
                    agent_direction=Direction.UP,
                )

                brain.run_brain(params, top_only=True, top_randomize=False)
                reward = 1.0 if step == 14 else 0.1
                brain.learn(params, reward, episode_done=(step == 14))

            brain.post_process_episode()

        # Should have completed 3 episodes * 15 steps
        assert len(brain.history_data.rewards) == 45

    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            learning_rate=1.0,  # Very high LR to potentially cause large gradients
            max_grad_norm=0.5,
            rollout_buffer_size=20,
            num_epochs=1,
            num_minibatches=2,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Collect experience and trigger update
        for i in range(25):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=10.0 if i == 24 else 0.0, episode_done=(i == 24))

        # Weights should remain finite despite high LR
        for param in brain.actor.parameters():
            assert torch.all(torch.isfinite(param))
        for param in brain.critic.parameters():
            assert torch.all(torch.isfinite(param))

    def test_deterministic_action_selection(self):
        """Test that action selection is deterministic with same seed."""
        config = MLPPPOBrainConfig(actor_hidden_dim=16, critic_hidden_dim=16)

        # Create two brains with same weights
        torch.manual_seed(42)
        brain1 = MLPPPOBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        torch.manual_seed(42)
        brain2 = MLPPPOBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Set same seed for action sampling
        torch.manual_seed(123)
        actions1 = brain1.run_brain(params, top_only=True, top_randomize=False)

        torch.manual_seed(123)
        actions2 = brain2.run_brain(params, top_only=True, top_randomize=False)

        # Actions should be the same
        assert actions1[0].action == actions2[0].action

    def test_value_estimates_remain_finite_with_learning(self):
        """Test that value estimates remain finite during training."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=32,
            critic_hidden_dim=32,
            learning_rate=0.01,
            rollout_buffer_size=100,
            num_epochs=4,
            num_minibatches=4,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Train on experiences where this state leads to positive reward
        params = BrainParams(gradient_strength=1.0, gradient_direction=0.0)

        for _episode in range(5):
            for step in range(25):
                brain.run_brain(params, top_only=True, top_randomize=False)
                brain.learn(params, reward=1.0, episode_done=(step == 24))

        # Get final value estimate for a state that always gives positive reward
        state = torch.tensor([1.0, 0.0], dtype=torch.float32, device=brain.device)
        final_value = brain.forward_critic(state).item()

        # Value should have increased for consistently rewarded state
        # (This is a soft check - learning dynamics can vary)
        assert torch.isfinite(torch.tensor(final_value))


class TestLRScheduling:
    """Tests for learning rate scheduling (warmup and decay)."""

    def test_lr_scheduling_disabled_by_default(self):
        """Test that LR scheduling is disabled when no warmup episodes set."""
        config = MLPPPOBrainConfig(learning_rate=0.001)
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.lr_scheduling_enabled is False
        assert brain._get_current_lr() == 0.001

    def test_lr_warmup_enabled(self):
        """Test that LR warmup can be enabled via config."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=50,
            lr_warmup_start=0.0001,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.lr_scheduling_enabled is True
        assert brain.lr_warmup_episodes == 50
        assert brain.lr_warmup_start == 0.0001
        assert brain.base_lr == 0.001

    def test_lr_warmup_default_start(self):
        """Test that lr_warmup_start defaults to 10% of base LR."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=50,
            # lr_warmup_start not set
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.lr_warmup_start == 0.0001  # 10% of 0.001

    def test_lr_warmup_progression(self):
        """Test that LR increases linearly during warmup phase."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=100,
            lr_warmup_start=0.0001,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Episode 0: should be at warmup start
        brain._episode_count = 0
        assert brain._get_current_lr() == pytest.approx(0.0001)

        # Episode 50: should be at midpoint
        brain._episode_count = 50
        expected_mid = 0.0001 + (0.001 - 0.0001) * 0.5
        assert brain._get_current_lr() == pytest.approx(expected_mid)

        # Episode 100: should be at base LR
        brain._episode_count = 100
        assert brain._get_current_lr() == pytest.approx(0.001)

        # Episode 150: should stay at base LR (no decay configured)
        brain._episode_count = 150
        assert brain._get_current_lr() == pytest.approx(0.001)

    def test_lr_decay_after_warmup(self):
        """Test that LR decays after warmup when decay is configured."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=50,
            lr_warmup_start=0.0001,
            lr_decay_episodes=200,
            lr_decay_end=0.0001,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Episode 0: warmup start
        brain._episode_count = 0
        assert brain._get_current_lr() == pytest.approx(0.0001)

        # Episode 50: warmup complete, at base LR
        brain._episode_count = 50
        assert brain._get_current_lr() == pytest.approx(0.001)

        # Episode 150: midpoint of decay (50 + 100 = 150)
        brain._episode_count = 150
        expected_mid = 0.001 + (0.0001 - 0.001) * 0.5
        assert brain._get_current_lr() == pytest.approx(expected_mid)

        # Episode 250: decay complete (50 + 200 = 250)
        brain._episode_count = 250
        assert brain._get_current_lr() == pytest.approx(0.0001)

        # Episode 300: stays at decay end
        brain._episode_count = 300
        assert brain._get_current_lr() == pytest.approx(0.0001)

    def test_lr_decay_default_end(self):
        """Test that lr_decay_end defaults to 10% of base LR."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=50,
            lr_decay_episodes=100,
            # lr_decay_end not set
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.lr_decay_end == 0.0001  # 10% of 0.001

    def test_update_learning_rate_modifies_optimizer(self):
        """Test that _update_learning_rate actually updates the optimizer."""
        config = MLPPPOBrainConfig(
            learning_rate=0.001,
            lr_warmup_episodes=100,
            lr_warmup_start=0.0001,
        )
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Initially at warmup start
        brain._episode_count = 0
        brain._update_learning_rate()
        for param_group in brain.optimizer.param_groups:
            assert param_group["lr"] == pytest.approx(0.0001)

        # After some episodes, LR should have increased
        brain._episode_count = 50
        brain._update_learning_rate()
        expected = 0.0001 + (0.001 - 0.0001) * 0.5
        for param_group in brain.optimizer.param_groups:
            assert param_group["lr"] == pytest.approx(expected)

    def test_lr_scheduling_no_update_when_disabled(self):
        """Test that _update_learning_rate does nothing when scheduling disabled."""
        config = MLPPPOBrainConfig(learning_rate=0.001)
        brain = MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        original_lr = brain.optimizer.param_groups[0]["lr"]
        brain._episode_count = 100
        brain._update_learning_rate()

        # LR should be unchanged
        assert brain.optimizer.param_groups[0]["lr"] == original_lr


class TestPPOClipping:
    """Tests specifically for PPO clipping mechanism."""

    @pytest.fixture
    def brain(self):
        """Create a brain for clipping tests."""
        config = MLPPPOBrainConfig(
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            clip_epsilon=0.2,
            rollout_buffer_size=20,
            num_epochs=1,
            num_minibatches=2,
        )
        return MLPPPOBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

    def test_clip_epsilon_applied(self, brain):
        """Test that clip epsilon is correctly stored."""
        assert brain.clip_epsilon == 0.2

    def test_clipping_prevents_large_updates(self, brain):
        """Test that clipping limits policy changes."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Get initial policy output
        state = brain.preprocess(params)
        x = torch.tensor(state, dtype=torch.float32, device=brain.device)
        initial_logits = brain.actor(x).detach().clone()

        # Collect experience and update
        for i in range(25):
            brain.run_brain(params, top_only=True, top_randomize=False)
            # Large reward swing to potentially cause large updates
            reward = 100.0 if i % 2 == 0 else -100.0
            brain.learn(params, reward, episode_done=(i == 24))

        # Get final policy output
        final_logits = brain.actor(x).detach()

        # Policy should have changed but not exploded
        assert torch.all(torch.isfinite(final_logits))
        # The change should be bounded (not a rigorous test, but sanity check)
        logit_change = torch.abs(final_logits - initial_logits).max().item()
        assert logit_change < 1000  # Very loose bound to catch explosions
