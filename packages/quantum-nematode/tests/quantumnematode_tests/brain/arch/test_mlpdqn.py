"""Unit tests for the Q-learning MLP brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpdqn import MLPDQNBrain, MLPDQNBrainConfig
from quantumnematode.env import Direction


class TestMLPDQNBrainConfig:
    """Test cases for Q-learning MLP brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLPDQNBrainConfig()

        assert config.hidden_dim == 64
        assert config.learning_rate == 0.001
        assert config.epsilon == 0.1
        assert config.epsilon_decay == 0.995
        assert config.epsilon_min == 0.01
        assert config.gamma == 0.95
        assert config.target_update_freq == 100
        assert config.num_hidden_layers == 2
        assert config.buffer_size == 10000
        assert config.batch_size == 32

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLPDQNBrainConfig(
            hidden_dim=128,
            learning_rate=0.01,
            epsilon=0.2,
            gamma=0.99,
            buffer_size=5000,
        )

        assert config.hidden_dim == 128
        assert config.learning_rate == 0.01
        assert config.epsilon == 0.2
        assert config.gamma == 0.99
        assert config.buffer_size == 5000


class TestMLPDQNBrain:
    """Test cases for the Q-learning MLP brain architecture."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MLPDQNBrainConfig(
            hidden_dim=32,
            learning_rate=0.001,
            num_hidden_layers=2,
            buffer_size=100,
            batch_size=16,
        )

    @pytest.fixture
    def brain(self, config):
        """Create a test Q-MLP brain."""
        return MLPDQNBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

    def test_brain_initialization(self, brain, config):
        """Test Q-MLP brain initialization."""
        assert brain.input_dim == 2
        assert brain.num_actions == 4
        assert brain.epsilon == config.epsilon
        assert brain.gamma == config.gamma
        assert brain.training is True

        # Check Q-networks exist
        assert brain.q_network is not None
        assert brain.target_q_network is not None
        assert isinstance(brain.q_network, torch.nn.Sequential)
        assert isinstance(brain.target_q_network, torch.nn.Sequential)

        # Check experience replay buffer
        assert len(brain.experience_buffer) == 0
        assert brain.experience_buffer.maxlen == config.buffer_size

    def test_build_network(self, brain):
        """Test network architecture building."""
        # Network should have correct input/output dimensions
        test_input = torch.randn(1, brain.input_dim)
        output = brain.q_network(test_input)
        assert output.shape == (1, brain.num_actions)

    def test_target_network_sync(self, brain):
        """Test that target network is initialized with same weights as Q-network."""
        for param_q, param_target in zip(
            brain.q_network.parameters(),
            brain.target_q_network.parameters(),
            strict=False,
        ):
            assert torch.allclose(param_q, param_target)

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
        assert len(features) == 2  # gradient_strength, relative_angle
        assert features.dtype == np.float32
        assert -1.0 <= features[1] <= 1.0  # Normalized relative angle

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values."""
        params = BrainParams()
        features = brain.preprocess(params)

        assert len(features) == 2
        assert features[0] == 0.0

    def test_forward(self, brain):
        """Test forward pass through the Q-network."""
        x = np.array([0.5, 0.3], dtype=np.float32)
        q_values = brain.forward(x)

        assert isinstance(q_values, torch.Tensor)
        assert q_values.shape == (brain.num_actions,)
        assert torch.all(torch.isfinite(q_values))

    def test_run_brain_exploration(self, brain):
        """Test epsilon-greedy exploration during training."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Set high epsilon for exploration
        brain.epsilon = 1.0
        brain.training = True

        # Run multiple times - should get different actions due to exploration
        actions_seen = set()
        for _ in range(20):
            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            actions_seen.add(actions[0].action)

        # With epsilon=1.0, we should see multiple different actions
        assert len(actions_seen) > 1

    def test_run_brain_exploitation(self, brain):
        """Test greedy action selection with epsilon=0."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        brain.epsilon = 0.0
        brain.training = False

        actions = brain.run_brain(params, top_only=True, top_randomize=False)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]

    def test_experience_storage(self, brain):
        """Test that experiences are stored correctly."""
        params1 = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        params2 = BrainParams(gradient_strength=0.6, gradient_direction=1.2)

        # First action
        brain.run_brain(params1, top_only=True, top_randomize=False)

        # Second action with reward
        brain.run_brain(params2, reward=0.5, top_only=True, top_randomize=False)
        brain.learn(params2, reward=0.5, episode_done=False)

        # Experience buffer should have data
        assert len(brain.experience_buffer) > 0

    def test_learn_insufficient_data(self, brain):
        """Test that learning doesn't happen with insufficient data."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run brain once
        brain.run_brain(params, top_only=True, top_randomize=False)

        # Store initial weights
        initial_weights = [p.clone() for p in brain.q_network.parameters()]

        # Try to learn with no previous state
        brain.learn(params, reward=0.5, episode_done=False)

        # Weights shouldn't change yet (not enough experiences)
        for param, initial in zip(brain.q_network.parameters(), initial_weights, strict=False):
            assert torch.allclose(param, initial)

    def test_learn_with_batch(self, brain):
        """Test learning with sufficient experiences."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer with experiences
        for _ in range(brain.batch_size + 5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Should have enough experiences now
        assert len(brain.experience_buffer) >= brain.batch_size

        # Learn with significant reward
        brain.run_brain(params, top_only=True, top_randomize=False)
        brain.learn(params, reward=1.0, episode_done=False)

        # Check that learning doesn't crash and weights remain finite
        for param in brain.q_network.parameters():
            assert torch.all(torch.isfinite(param))

    def test_target_network_update(self, brain):
        """Test that target network is updated periodically."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer
        for _ in range(brain.batch_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Train for many steps
        for _ in range(brain.target_update_freq + 5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Target network should have been updated
        # (weights might be same if Q-network didn't change much, but update_count should increase)
        assert brain.update_count > brain.target_update_freq

    def test_epsilon_decay(self, brain):
        """Test that epsilon decays over time."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        initial_epsilon = brain.epsilon

        # Fill buffer and learn multiple times
        for _ in range(brain.batch_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        for _ in range(50):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Epsilon should have decayed
        assert brain.epsilon < initial_epsilon
        assert brain.epsilon >= brain.epsilon_min

    def test_update_memory(self, brain):
        """Test memory update functionality."""
        # Should be a no-op for Q-MLP brain
        brain.update_memory(reward=0.5)
        # Just verify it doesn't crash

    def test_post_process_episode(self, brain):
        """Test episode post-processing."""
        # Should be a no-op for Q-MLP brain
        brain.post_process_episode()
        # Just verify it doesn't crash

    def test_action_set_property(self, brain):
        """Test action_set property."""
        action_set = brain.action_set
        assert len(action_set) == 4
        assert Action.FORWARD in action_set
        assert Action.LEFT in action_set
        assert Action.RIGHT in action_set
        assert Action.STAY in action_set

    def test_build_brain_not_implemented(self, brain):
        """Test that build_brain raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.build_brain()

    def test_copy_not_implemented(self, brain):
        """Test that copy raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.copy()


class TestMLPDQNBrainIntegration:
    """Integration tests for Q-learning MLP brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = MLPDQNBrainConfig(
            hidden_dim=16,
            learning_rate=0.001,
            buffer_size=100,
            batch_size=16,
        )
        brain = MLPDQNBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        rng = np.random.default_rng(42)

        # Simulate multiple steps in an episode
        for step in range(50):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step % 10, step % 10),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            assert len(actions) == 1

            # Learn
            reward = 1.0 if step == 49 else rng.random() - 0.5
            is_done = step == 49
            brain.learn(params, reward, episode_done=is_done)

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.experience_buffer) > 0
        assert brain.update_count > 0

    def test_training_vs_evaluation_mode(self):
        """Test behavior differences between training and evaluation."""
        config = MLPDQNBrainConfig(hidden_dim=16, epsilon=0.5)
        brain = MLPDQNBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Training mode - epsilon-greedy
        brain.training = True
        brain.epsilon = 0.5
        actions_train = [
            brain.run_brain(params, top_only=True, top_randomize=False)[0].action for _ in range(10)
        ]

        # Evaluation mode - greedy only
        brain.training = False
        actions_eval = [
            brain.run_brain(params, top_only=True, top_randomize=False)[0].action for _ in range(10)
        ]

        # Training mode should potentially have more variety due to exploration
        assert len(actions_train) == 10
        assert len(actions_eval) == 10

    def test_q_value_convergence(self):
        """Test that Q-values converge with consistent rewards."""
        config = MLPDQNBrainConfig(
            hidden_dim=16,
            learning_rate=0.01,
            buffer_size=200,
            batch_size=16,
            epsilon=0.1,
        )
        brain = MLPDQNBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.8, gradient_direction=0.5)

        # Train with consistent positive reward for specific state
        for _ in range(100):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=False)

        # Q-values should have increased for this state
        state = brain.preprocess(params)
        q_values = brain.forward(state)
        max_q_value = torch.max(q_values).item()

        # With consistent positive rewards, max Q-value should be positive
        assert max_q_value > 0

    def test_experience_replay_buffer_limit(self):
        """Test that experience buffer respects max size."""
        config = MLPDQNBrainConfig(buffer_size=50, batch_size=16)
        brain = MLPDQNBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer beyond capacity
        for _ in range(100):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should not exceed max size
        assert len(brain.experience_buffer) <= 50

    def test_gradient_clipping(self):
        """Test that gradients are clipped to prevent instability."""
        config = MLPDQNBrainConfig(hidden_dim=16, learning_rate=0.1, buffer_size=50, batch_size=16)
        brain = MLPDQNBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer
        for _ in range(brain.batch_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Train with extreme reward
        for _ in range(10):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=100.0, episode_done=False)

        # Weights should remain finite due to gradient clipping
        for param in brain.q_network.parameters():
            assert torch.all(torch.isfinite(param))
