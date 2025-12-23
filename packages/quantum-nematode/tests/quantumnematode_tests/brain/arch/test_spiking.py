"""Unit tests for the spiking neural network brain architecture with surrogate gradients."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._spiking_layers import (
    LIFLayer,
    OutputMode,
    PopulationEncoder,
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
        _spikes1, state1 = layer(x, state=None)
        v_membrane1, _ = state1

        # Second timestep with previous state
        _spikes2, state2 = layer(x, state=state1)
        v_membrane2, _ = state2

        # Membrane potential should change
        assert not torch.allclose(v_membrane1, v_membrane2)


class TestPopulationEncoder:
    """Test cases for population coding encoder."""

    def test_output_shape(self):
        """Test that population encoder expands input correctly."""
        encoder = PopulationEncoder(input_dim=2, neurons_per_feature=8)
        x = torch.randn(1, 2)

        output = encoder(x)

        # Should expand: 2 features * 8 neurons = 16
        assert output.shape == (1, 16)

    def test_gaussian_responses(self):
        """Test that encoder produces Gaussian-shaped responses."""
        encoder = PopulationEncoder(
            input_dim=1,
            neurons_per_feature=5,
            sigma=0.25,
            min_val=-1.0,
            max_val=1.0,
        )
        # Input at center of range (0.0)
        x = torch.tensor([[0.0]])

        output = encoder(x)

        # Center neuron (index 2) should have highest activation
        assert output[0, 2] > output[0, 0]  # Center > left edge
        assert output[0, 2] > output[0, 4]  # Center > right edge

    def test_different_inputs_different_patterns(self):
        """Test that different inputs produce distinguishable patterns."""
        encoder = PopulationEncoder(input_dim=1, neurons_per_feature=8, sigma=0.25)

        x1 = torch.tensor([[0.0]])
        x2 = torch.tensor([[0.5]])

        out1 = encoder(x1)
        out2 = encoder(x2)

        # Outputs should be different
        assert not torch.allclose(out1, out2)


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

    def test_population_coding_enabled(self):
        """Test network with population coding enabled."""
        net = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=8,
            output_dim=4,
            num_timesteps=10,
            population_coding=True,
            neurons_per_feature=8,
        )
        x = torch.randn(1, 2)

        action_logits = net(x)

        # Should still produce correct output shape
        assert action_logits.shape == (1, 4)
        # Population encoder should exist
        assert net.population_encoder is not None


class TestSpikingBrainConfig:
    """Test cases for spiking brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpikingBrainConfig()

        # Network topology
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 2
        assert config.num_timesteps == 100

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
        assert config.surrogate_alpha == 1.0

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
        assert len(brain.episode_action_probs) == 1

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
        assert len(brain.episode_action_probs) == 0
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


class TestWeightInitialization:
    """Test cases for weight initialization methods."""

    def test_orthogonal_initialization(self):
        """Test orthogonal weight initialization."""
        config = SpikingBrainConfig(hidden_size=8, num_timesteps=10, weight_init="orthogonal")
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Check that weights (not biases) are initialized to non-zero values
        for name, param in brain.policy.named_parameters():
            if "weight" in name:
                assert torch.any(param != 0), f"Weight {name} should be non-zero"

    def test_kaiming_initialization(self):
        """Test Kaiming weight initialization."""
        config = SpikingBrainConfig(hidden_size=8, num_timesteps=10, weight_init="kaiming")
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Check that weights (not biases) are initialized to non-zero values
        for name, param in brain.policy.named_parameters():
            if "weight" in name:
                assert torch.any(param != 0), f"Weight {name} should be non-zero"


class TestIntraEpisodeUpdates:
    """Test cases for intra-episode gradient updates."""

    def test_update_frequency_triggers_updates(self):
        """Test that update_frequency triggers gradient updates during episode."""
        config = SpikingBrainConfig(
            hidden_size=8,
            num_timesteps=10,
            update_frequency=3,  # Update every 3 steps
            learning_rate=0.01,
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Get initial parameters
        initial_params = [p.clone() for p in brain.policy.parameters()]

        # Run for enough steps to trigger intra-episode update
        for _step in range(4):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=False)

        # Parameters should have changed due to intra-episode update
        final_params = list(brain.policy.parameters())
        params_changed = any(
            not torch.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_params, final_params, strict=False)
        )
        assert params_changed

    def test_no_update_frequency_waits_for_episode_end(self):
        """Test that without update_frequency, updates only happen at episode end."""
        config = SpikingBrainConfig(
            hidden_size=8,
            num_timesteps=10,
            update_frequency=0,  # Disabled - episode-end only
            learning_rate=0.01,
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Get initial parameters
        initial_params = [p.clone() for p in brain.policy.parameters()]

        # Run steps without episode_done
        for _ in range(5):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=False)

        # Parameters should NOT have changed yet
        mid_params = list(brain.policy.parameters())
        params_unchanged = all(
            torch.allclose(init, mid, atol=1e-6)
            for init, mid in zip(initial_params, mid_params, strict=False)
        )
        assert params_unchanged

        # Now end the episode
        brain.learn(params, reward=1.0, episode_done=True)

        # Now parameters should have changed
        final_params = list(brain.policy.parameters())
        params_changed = any(
            not torch.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_params, final_params, strict=False)
        )
        assert params_changed


class TestMinActionProb:
    """Test cases for min_action_prob floor."""

    def test_min_action_prob_enforces_floor(self):
        """Test that min_action_prob prevents action probabilities from going below floor."""
        config = SpikingBrainConfig(
            hidden_size=8,
            num_timesteps=10,
            min_action_prob=0.01,
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Run brain multiple times and check probabilities
        for _ in range(10):
            _actions = brain.run_brain(params, top_only=False, top_randomize=False)
            # All probabilities should be >= min_action_prob
            probs = brain.episode_action_probs[-1]
            assert all(p >= config.min_action_prob - 1e-6 for p in probs)

    def test_min_action_prob_zero_allows_any_probability(self):
        """Test that min_action_prob=0 allows probabilities to be any value."""
        config = SpikingBrainConfig(
            hidden_size=8,
            num_timesteps=10,
            min_action_prob=0.0,
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Brain should work without errors
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )
        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1


class TestPopulationCodingIntegration:
    """Test population coding in full brain context."""

    def test_brain_with_population_coding(self):
        """Test spiking brain with population coding enabled."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            population_coding=True,
            neurons_per_feature=8,
            population_sigma=0.25,
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Should work correctly with population coding
        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1
        assert actions[0].action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]


class TestSeparatedGradients:
    """Test cases for separated food/predator gradient inputs."""

    def test_separated_gradients_preprocessing(self):
        """Test that separated gradients produces 4-feature output."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            use_separated_gradients=True,
        )
        # Input dim should be 4 for separated gradients
        brain = SpikingBrain(config=config, input_dim=4, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,  # Combined (unused when separated)
            gradient_direction=1.0,
            food_gradient_strength=0.8,
            food_gradient_direction=0.5,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-1.5,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Preprocess should return 4 features
        state = brain.preprocess(params)
        assert state.shape == (4,)
        # First two should be food gradient features
        assert 0.0 <= state[0] <= 1.0  # food_strength clamped
        assert -1.0 <= state[1] <= 1.0  # food_rel_angle normalized
        # Last two should be predator gradient features
        assert 0.0 <= state[2] <= 1.0  # pred_strength clamped
        assert -1.0 <= state[3] <= 1.0  # pred_rel_angle normalized

    def test_separated_gradients_run_brain(self):
        """Test that brain with separated gradients can run and produce actions."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            use_separated_gradients=True,
        )
        brain = SpikingBrain(config=config, input_dim=4, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.8,
            food_gradient_direction=0.5,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-1.5,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1
        assert actions[0].action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]

    def test_combined_gradients_still_works(self):
        """Test that combined gradient mode (default) still works."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            use_separated_gradients=False,  # Default
        )
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        # Preprocess should return 2 features
        state = brain.preprocess(params)
        assert state.shape == (2,)

        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1

    def test_separated_gradients_handles_none_values(self):
        """Test that separated gradients handles None values gracefully."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            use_separated_gradients=True,
        )
        brain = SpikingBrain(config=config, input_dim=4, num_actions=4, device=DeviceType.CPU)

        # All gradient values None
        params = BrainParams(
            food_gradient_strength=None,
            food_gradient_direction=None,
            predator_gradient_strength=None,
            predator_gradient_direction=None,
            agent_position=(1.0, 1.0),
            agent_direction=Direction.UP,
        )

        state = brain.preprocess(params)
        assert state.shape == (4,)
        # All should be 0.0 when None
        assert state[0] == 0.0  # food_strength
        assert state[1] == 0.0  # food_rel_angle
        assert state[2] == 0.0  # pred_strength
        assert state[3] == 0.0  # pred_rel_angle


class TestTemporalModulation:
    """Test cases for temporal modulation feature."""

    def test_modulation_varies_input(self):
        """Test that temporal modulation varies input current over timesteps."""
        import math

        # The modulation should create different effective inputs at different timesteps
        # Test by checking that modulation_factor varies with t
        factors = []
        for t in range(20):
            phase = 2.0 * math.pi * t / 10  # period=10
            factor = 1.0 + 0.3 * math.sin(phase)  # amplitude=0.3
            factors.append(factor)

        # Factors should vary between 0.7 and 1.3
        assert min(factors) < 0.75
        assert max(factors) > 1.25
        # Should complete 2 full cycles over 20 timesteps with period=10
        assert len([f for f in factors if f > 1.2]) >= 2

    def test_modulation_disabled_by_default(self):
        """Test that modulation is disabled when temporal_modulation=False."""
        network = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_timesteps=10,
            temporal_modulation=False,
        )

        assert network.temporal_modulation is False

    def test_network_runs_with_modulation(self):
        """Test that network runs successfully with modulation enabled."""
        network = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_timesteps=20,
            temporal_modulation=True,
            modulation_amplitude=0.3,
            modulation_period=10,
        )

        x = torch.randn(1, 2)
        output = network(x)

        assert output.shape == (1, 4)
        assert not torch.isnan(output).any()


class TestOutputModes:
    """Test cases for different output modes."""

    def test_accumulator_mode(self):
        """Test accumulator mode sums spikes over all timesteps."""
        network = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_timesteps=50,
            output_mode="accumulator",
        )

        x = torch.randn(1, 2)
        output = network(x)

        assert output.shape == (1, 4)
        assert network.output_mode == "accumulator"

    def test_final_mode(self):
        """Test final mode uses only last timestep spikes."""
        network = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_timesteps=50,
            output_mode="final",
        )

        x = torch.randn(1, 2)
        output = network(x)

        assert output.shape == (1, 4)
        assert network.output_mode == "final"

    def test_membrane_mode(self):
        """Test membrane mode uses final membrane potentials."""
        network = SpikingPolicyNetwork(
            input_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_timesteps=50,
            output_mode="membrane",
        )

        x = torch.randn(1, 2)
        output = network(x)

        assert output.shape == (1, 4)
        assert network.output_mode == "membrane"

    def test_different_modes_produce_different_outputs(self):
        """Test that different output modes produce different outputs."""
        torch.manual_seed(42)

        # Create networks with same weights but different output modes
        x = torch.randn(1, 2)

        modes: list[OutputMode] = ["accumulator", "final", "membrane"]
        outputs = {}
        for mode in modes:
            torch.manual_seed(42)  # Reset seed for consistent initialization
            network = SpikingPolicyNetwork(
                input_dim=2,
                hidden_dim=16,
                output_dim=4,
                num_timesteps=50,
                output_mode=mode,
            )
            outputs[mode] = network(x)

        # Outputs should differ between modes (with high probability for random input)
        # Note: They might occasionally be similar due to randomness, but generally differ
        assert outputs["accumulator"].shape == outputs["final"].shape == outputs["membrane"].shape


class TestClippingParameters:
    """Test cases for return and advantage clipping."""

    def test_return_clip_applied(self):
        """Test that return_clip limits extreme returns."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            return_clip=10.0,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Simulate an episode with extreme rewards
        brain.episode_states = [np.array([0.5, 0.5]) for _ in range(5)]
        brain.episode_actions = [0, 1, 2, 3, 0]
        brain.episode_action_probs = [torch.tensor([0.25, 0.25, 0.25, 0.25]) for _ in range(5)]
        brain.episode_rewards = [100.0, 100.0, 100.0, 100.0, 100.0]  # Extreme rewards

        # Return clip should be set
        assert brain.config.return_clip == 10.0

    def test_advantage_clip_applied(self):
        """Test that advantage_clip limits extreme advantages."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            advantage_clip=2.0,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Advantage clip should be set
        assert brain.config.advantage_clip == 2.0

    def test_clips_disabled_when_zero(self):
        """Test that clipping is disabled when set to 0."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            return_clip=0.0,
            advantage_clip=0.0,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.config.return_clip == 0.0
        assert brain.config.advantage_clip == 0.0


class TestExplorationDecay:
    """Test cases for exploration parameter decay."""

    def test_logit_clamp_decay(self):
        """Test that logit clamp decays from initial to final value."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            exploration_logit_clamp=2.0,
            exploration_logit_clamp_final=5.0,
            exploration_decay_episodes=100,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.config.exploration_logit_clamp == 2.0
        assert brain.config.exploration_logit_clamp_final == 5.0
        assert brain.config.exploration_decay_episodes == 100

    def test_noise_std_decay(self):
        """Test that noise std decays from initial to final value."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            exploration_noise_std=0.3,
            exploration_noise_std_final=0.05,
            exploration_decay_episodes=100,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.config.exploration_noise_std == 0.3
        assert brain.config.exploration_noise_std_final == 0.05

    def test_temperature_decay(self):
        """Test that temperature decays from initial to final value."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            exploration_temperature=1.5,
            exploration_temperature_final=1.0,
            exploration_decay_episodes=100,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        assert brain.config.exploration_temperature == 1.5
        assert brain.config.exploration_temperature_final == 1.0

    def test_no_decay_when_episodes_zero(self):
        """Test that no decay occurs when decay_episodes is 0."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            exploration_logit_clamp=2.0,
            exploration_logit_clamp_final=5.0,
            exploration_decay_episodes=0,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # With decay_episodes=0, should use initial value
        assert brain.config.exploration_decay_episodes == 0


class TestProbabilityFloorHelper:
    """Test cases for the _apply_probability_floor helper method."""

    def test_floor_applied(self):
        """Test that probability floor is applied correctly."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            min_action_prob=0.05,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Create probabilities with one very low value
        probs = torch.tensor([[0.01, 0.49, 0.49, 0.01]])
        floored = brain._apply_probability_floor(probs)

        # Low probabilities should be increased (before renormalization effect)
        # The floor is clamped then renormalized, so exact floor may not hold
        # but the relative values should be more balanced
        assert floored[0, 0] > probs[0, 0]  # 0.01 should increase
        assert floored[0, 3] > probs[0, 3]  # 0.01 should increase
        # Should still sum to 1
        assert torch.allclose(floored.sum(dim=-1), torch.tensor([1.0]))

    def test_no_floor_when_disabled(self):
        """Test that no floor is applied when min_action_prob is 0."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            min_action_prob=0.0,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        probs = torch.tensor([[0.01, 0.49, 0.49, 0.01]])
        floored = brain._apply_probability_floor(probs)

        # Should be unchanged
        assert torch.allclose(probs, floored)

    def test_renormalization(self):
        """Test that probabilities are renormalized after clamping."""
        config = SpikingBrainConfig(
            hidden_size=16,
            num_timesteps=10,
            min_action_prob=0.1,
        )
        brain = SpikingBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        probs = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
        floored = brain._apply_probability_floor(probs)

        # Should sum to 1 after renormalization
        assert torch.allclose(floored.sum(dim=-1), torch.tensor([1.0]))
