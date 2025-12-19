"""Unit tests for the spiking neural network brain architecture with surrogate gradients."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._spiking_layers import (
    LIFLayer,
    SpikingPolicyNetwork,
    SurrogateGradientSpike,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.spiking import SpikingBrain, SpikingBrainConfig
from quantumnematode.env import Direction


class TestSurrogateGradientSpike:
    """Test cases for the surrogate gradient spike function."""

    def test_forward_pass(self):
        """Test forward pass produces binary spikes."""
        v = torch.tensor([[0.5, 1.2, 0.8, 1.5]])
        v_threshold = 1.0
        alpha = 10.0

        spikes = SurrogateGradientSpike.apply(v, v_threshold, alpha)
        assert isinstance(spikes, torch.Tensor)  # Type hint for pyright

        # Check that spikes are binary
        assert torch.all((spikes == 0) | (spikes == 1))
        # Check correct spike pattern
        assert spikes[0, 0] == 0  # 0.5 < 1.0
        assert spikes[0, 1] == 1  # 1.2 >= 1.0
        assert spikes[0, 2] == 0  # 0.8 < 1.0
        assert spikes[0, 3] == 1  # 1.5 >= 1.0

    def test_gradient_flow(self):
        """Test that gradients flow through the surrogate function."""
        v = torch.tensor([[0.5, 1.2]], requires_grad=True)
        v_threshold = 1.0
        alpha = 10.0

        spikes = SurrogateGradientSpike.apply(v, v_threshold, alpha)
        assert isinstance(spikes, torch.Tensor)  # Type hint for pyright
        loss = spikes.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert v.grad is not None
        assert torch.any(v.grad != 0)


class TestLIFLayer:
    """Test cases for the LIF neuron layer."""

    def test_initialization(self):
        """Test LIF layer initialization."""
        layer = LIFLayer(input_dim=2, output_dim=4, tau_m=20.0, v_threshold=1.0)

        assert layer.tau_m == 20.0
        assert layer.v_threshold == 1.0
        assert layer.v_reset == 0.0
        assert layer.v_rest == 0.0

    def test_forward_single_timestep(self):
        """Test forward pass for single timestep."""
        layer = LIFLayer(input_dim=2, output_dim=4)
        x = torch.randn(1, 2)

        spikes, state = layer(x, state=None)

        # Check output shapes
        assert spikes.shape == (1, 4)
        assert isinstance(state, tuple)
        assert len(state) == 2
        v_membrane, _ = state
        assert v_membrane.shape == (1, 4)

        # Spikes should be binary
        assert torch.all((spikes == 0) | (spikes == 1))

    def test_stateful_dynamics(self):
        """Test that membrane state persists across timesteps."""
        layer = LIFLayer(input_dim=2, output_dim=4, tau_m=20.0)
        x = torch.ones(1, 2) * 0.1  # Small constant input

        # First timestep
        spikes1, state1 = layer(x, state=None)
        v_membrane1, _ = state1

        # Second timestep with previous state
        spikes2, state2 = layer(x, state=state1)
        v_membrane2, _ = state2

        # Membrane potential should change
        assert not torch.allclose(v_membrane1, v_membrane2)


class TestSpikingPolicyNetwork:
    """Test cases for the spiking policy network."""

    def test_initialization(self):
        """Test network initialization."""
        net = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=8,
            output_dim=4,
            num_timesteps=10,
            num_hidden_layers=2,
        )

        # Check architecture
        assert len(net.hidden_layers) == 2
        assert net.num_timesteps == 10

    def test_forward_pass(self):
        """Test forward pass through the network."""
        net = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=8,
            output_dim=4,
            num_timesteps=10,
            num_hidden_layers=2,
        )
        x = torch.randn(1, 2)

        action_logits = net(x)

        # Check output shape
        assert action_logits.shape == (1, 4)
        # Logits should be real-valued
        assert torch.all(torch.isfinite(action_logits))

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        net = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=8,
            output_dim=4,
            num_timesteps=10,
            num_hidden_layers=2,
        )
        x = torch.randn(1, 2, requires_grad=True)

        action_logits = net(x)
        loss = action_logits.sum()
        loss.backward()

        # Check that gradients exist for network parameters
        for param in net.parameters():
            assert param.grad is not None


class TestSpikingBrainConfig:
    """Test cases for spiking brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpikingBrainConfig()

        # Network topology
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 2
        assert config.num_timesteps == 50

        # LIF neuron parameters
        assert config.tau_m == 20.0
        assert config.v_threshold == 1.0
        assert config.v_reset == 0.0
        assert config.v_rest == 0.0

        # Learning parameters
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
        assert config.baseline_alpha == 0.05
        assert config.entropy_beta == 0.01

        # Surrogate gradient
        assert config.surrogate_alpha == 10.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpikingBrainConfig(
            hidden_size=64,
            num_timesteps=30,
            tau_m=10.0,
            learning_rate=0.01,
        )

        assert config.hidden_size == 64
        assert config.num_timesteps == 30
        assert config.tau_m == 10.0
        assert config.learning_rate == 0.01


class TestSpikingBrain:
    """Test cases for the spiking neural network brain."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SpikingBrainConfig(
            hidden_size=8,
            num_timesteps=10,
            num_hidden_layers=2,
            learning_rate=0.01,
        )

    @pytest.fixture
    def brain(self, config):
        """Create a test spiking brain."""
        return SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

    def test_brain_initialization(self, brain, config):
        """Test spiking brain initialization."""
        assert brain.config == config
        assert brain.input_dim == 2
        assert brain.num_actions == 4

        # Check policy network exists
        assert brain.policy is not None
        assert isinstance(brain.policy, SpikingPolicyNetwork)

        # Check optimizer exists
        assert brain.optimizer is not None

    def test_preprocess(self, brain):
        """Test state preprocessing with relative angles."""
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=1.5,
            agent_position=(2, 3),
            agent_direction=Direction.UP,
        )

        state = brain.preprocess(params)
        assert len(state) == 2
        assert 0.0 <= state[0] <= 1.0  # gradient_strength normalized
        assert -1.0 <= state[1] <= 1.0  # relative angle normalized

    def test_preprocess_relative_angle(self, brain):
        """Test that preprocessing computes relative angles correctly."""
        # Agent facing UP (0.5π), gradient pointing RIGHT (0.0)
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,  # RIGHT
            agent_position=(1, 1),
            agent_direction=Direction.UP,  # UP
        )

        state = brain.preprocess(params)
        # Relative angle should be (0.0 - 0.5π) = -0.5π
        # Normalized: -0.5π / π = -0.5
        expected_rel_angle = -0.5
        assert np.isclose(state[1], expected_rel_angle, atol=0.01)

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values."""
        params = BrainParams()

        state = brain.preprocess(params)
        assert len(state) == 2
        assert state[0] == 0.0
        assert state[1] == 0.0

    def test_run_brain(self, brain):
        """Test running the brain for decision making."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=False, top_randomize=False)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        assert 0.0 <= action_data.probability <= 1.0
        assert action_data.state is not None

        # Episode buffers should be populated
        assert len(brain.episode_states) == 1
        assert len(brain.episode_actions) == 1
        assert len(brain.episode_log_probs) == 1

    def test_learn_policy_gradient(self, brain):
        """Test learning with policy gradients."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain a few times to collect experience
        for _ in range(3):
            brain.run_brain(params, top_only=False, top_randomize=False)

        # Get initial parameters
        initial_params = [p.clone() for p in brain.policy.parameters()]

        # Learn with positive rewards (episode done)
        for i in range(3):
            brain.learn(params, reward=1.0, episode_done=(i == 2))

        # Parameters should have changed after learning
        final_params = list(brain.policy.parameters())
        params_changed = any(
            not torch.allclose(init, final)
            for init, final in zip(initial_params, final_params, strict=False)
        )
        assert params_changed

        # Episode buffers should be cleared after episode_done
        assert len(brain.episode_states) == 0
        assert len(brain.episode_actions) == 0
        assert len(brain.episode_log_probs) == 0
        assert len(brain.episode_rewards) == 0

    def test_baseline_update(self, brain):
        """Test baseline update mechanism."""
        initial_baseline = brain.baseline

        # Simulate an episode with positive rewards
        params = BrainParams()
        for _ in range(3):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=5.0, episode_done=False)

        brain.learn(params, reward=5.0, episode_done=True)

        # Baseline should have updated
        assert brain.baseline != initial_baseline

    def test_update_memory(self, brain):
        """Test memory update functionality."""
        reward = 0.75
        brain.update_memory(reward)

        assert brain.latest_data.reward == reward
        assert reward in brain.history_data.rewards

    def test_post_process_episode(self, brain):
        """Test episode post-processing."""
        # Add some episode data
        brain.overfit_detector_current_episode_actions = [Action.FORWARD, Action.LEFT]
        brain.overfit_detector_current_episode_positions = [(1.0, 1.0), (2.0, 1.0)]
        brain.overfit_detector_current_episode_rewards = [0.1, 0.2]

        brain.post_process_episode()

        # Episode data should be cleared
        assert len(brain.overfit_detector_current_episode_actions) == 0
        assert len(brain.overfit_detector_current_episode_positions) == 0
        assert len(brain.overfit_detector_current_episode_rewards) == 0

    def test_copy(self, brain):
        """Test brain copying functionality."""
        # Get initial policy state
        initial_state = brain.policy.state_dict()

        copied_brain = brain.copy()

        # Check that copy has same structure
        assert copied_brain.config.hidden_size == brain.config.hidden_size
        assert copied_brain.input_dim == brain.input_dim
        assert copied_brain.num_actions == brain.num_actions

        # Check that policy parameters are copied
        for key in initial_state:
            assert torch.allclose(
                copied_brain.policy.state_dict()[key],
                initial_state[key],
            )

        # Check that modifying copy doesn't affect original
        for param in copied_brain.policy.parameters():
            param.data.fill_(0.5)

        original_params_unchanged = any(
            not torch.allclose(param.data, torch.full_like(param.data, 0.5))
            for param in brain.policy.parameters()
        )
        assert original_params_unchanged

    def test_action_set_property(self, brain):
        """Test action_set property."""
        action_set = brain.action_set
        assert len(action_set) == 4
        assert Action.FORWARD in action_set
        assert Action.LEFT in action_set
        assert Action.RIGHT in action_set
        assert Action.STAY in action_set


class TestSpikingBrainIntegration:
    """Integration tests for spiking brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = SpikingBrainConfig(hidden_size=8, num_timesteps=10)
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Simulate multiple steps in an episode
        episode_rewards = []
        rng = np.random.default_rng(42)

        for step in range(5):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(float(step), float(step)),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=False, top_randomize=False)
            assert len(actions) == 1

            # Update memory
            reward = rng.random() - 0.5  # Random reward
            brain.update_memory(reward)
            episode_rewards.append(reward)

            # Learn
            is_done = step == 4
            brain.learn(params, reward, episode_done=is_done)

        # End episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 5

    def test_gradient_updates_with_positive_rewards(self):
        """Test that network learns from positive rewards."""
        config = SpikingBrainConfig(hidden_size=8, num_timesteps=10, learning_rate=0.1)
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Get initial parameters
        initial_params = [p.clone() for p in brain.policy.parameters()]

        # Run multiple episodes with positive rewards
        for _ in range(3):
            for step in range(5):
                params = BrainParams(
                    gradient_strength=0.5,
                    gradient_direction=1.0,
                    agent_position=(1.0, 1.0),
                    agent_direction=Direction.UP,
                )
                brain.run_brain(params, top_only=False, top_randomize=False)
                brain.learn(params, reward=1.0, episode_done=(step == 4))

            brain.post_process_episode()

        # Parameters should have changed
        final_params = list(brain.policy.parameters())
        params_changed = any(
            not torch.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_params, final_params, strict=False)
        )
        assert params_changed

    def test_deterministic_behavior_with_seed(self):
        """Test that brain behavior is deterministic when using fixed random seed."""
        config = SpikingBrainConfig(hidden_size=4, num_timesteps=10)

        # Create two identical brains with same seed
        torch.manual_seed(42)
        brain1 = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        torch.manual_seed(42)
        brain2 = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Should have identical initial parameters
        for p1, p2 in zip(brain1.policy.parameters(), brain2.policy.parameters(), strict=False):
            assert torch.allclose(p1, p2)
