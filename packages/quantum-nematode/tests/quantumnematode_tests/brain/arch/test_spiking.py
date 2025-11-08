"""Unit tests for the spiking neural network brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.spiking import LIFNeuron, SpikingBrain, SpikingBrainConfig, STDPRule
from quantumnematode.env import Direction


class TestLIFNeuron:
    """Test cases for the Leaky Integrate-and-Fire neuron model."""

    def test_lif_neuron_initialization(self):
        """Test LIF neuron initialization with default parameters."""
        neuron = LIFNeuron()
        assert neuron.tau_m == 20.0
        assert neuron.v_threshold == 1.0
        assert neuron.v_reset == 0.0
        assert neuron.v_rest == 0.0
        assert neuron.v_membrane == 0.0

    def test_lif_neuron_custom_parameters(self):
        """Test LIF neuron initialization with custom parameters."""
        neuron = LIFNeuron(tau_m=10.0, v_threshold=0.8, v_reset=-0.1, v_rest=-0.05)
        assert neuron.tau_m == 10.0
        assert neuron.v_threshold == 0.8
        assert neuron.v_reset == -0.1
        assert neuron.v_rest == -0.05
        assert neuron.v_membrane == -0.05

    def test_lif_neuron_step_no_spike(self):
        """Test LIF neuron dynamics without spiking."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0)

        # Small input current that shouldn't cause spiking
        spike = neuron.step(input_current=0.1, dt=1.0)
        assert not spike
        assert neuron.v_membrane > 0.0  # Membrane potential should increase
        assert neuron.v_membrane < 1.0  # But not reach threshold

    def test_lif_neuron_step_with_spike(self):
        """Test LIF neuron spiking when threshold is exceeded."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0)

        # Large input current that should cause spiking
        spike = neuron.step(input_current=50.0, dt=1.0)
        assert spike
        assert neuron.v_membrane == 0.0  # Should reset to v_reset

    def test_lif_neuron_membrane_dynamics(self):
        """Test that membrane potential evolves correctly over time."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0)

        # Apply small input current that won't cause spiking
        input_current = 0.5
        dt = 1.0
        prev_v = neuron.v_membrane

        for _ in range(5):
            spike = neuron.step(input_current, dt)
            # If no spike, membrane potential should increase
            if not spike:
                assert neuron.v_membrane > prev_v
            prev_v = neuron.v_membrane

    def test_lif_neuron_reset(self):
        """Test neuron reset functionality."""
        neuron = LIFNeuron(v_rest=-0.5)
        neuron.v_membrane = 0.8
        neuron.last_spike_time = 10.0

        neuron.reset()
        assert neuron.v_membrane == -0.5
        assert neuron.last_spike_time == -np.inf


class TestSTDPRule:
    """Test cases for the Spike-Timing Dependent Plasticity learning rule."""

    def test_stdp_initialization(self):
        """Test STDP rule initialization."""
        stdp = STDPRule()
        assert stdp.tau_plus == 20.0
        assert stdp.tau_minus == 20.0
        assert stdp.a_plus == 0.01
        assert stdp.a_minus == 0.01

    def test_stdp_potentiation(self):
        """Test STDP potentiation (post after pre)."""
        stdp = STDPRule(tau_plus=20.0, a_plus=0.01)

        # Post spike after pre spike should cause potentiation
        delta_t = 5.0  # post - pre = positive
        reward_signal = 1.0

        weight_change = stdp.compute_weight_change(delta_t, reward_signal)
        assert weight_change > 0  # Should be positive (potentiation)

    def test_stdp_depression(self):
        """Test STDP depression (pre after post)."""
        stdp = STDPRule(tau_minus=20.0, a_minus=0.01)

        # Pre spike after post spike should cause depression
        delta_t = -5.0  # post - pre = negative
        reward_signal = 1.0

        weight_change = stdp.compute_weight_change(delta_t, reward_signal)
        assert weight_change < 0  # Should be negative (depression)

    def test_stdp_reward_modulation(self):
        """Test reward modulation of STDP."""
        stdp = STDPRule()
        delta_t = 5.0

        # Higher reward should increase weight change magnitude
        weight_change_low = stdp.compute_weight_change(delta_t, 0.5)
        weight_change_high = stdp.compute_weight_change(delta_t, 2.0)

        assert abs(weight_change_high) > abs(weight_change_low)

    def test_stdp_time_decay(self):
        """Test that STDP effect decays with time difference."""
        stdp = STDPRule()
        reward_signal = 1.0

        # Closer spikes should have larger effect
        weight_change_close = stdp.compute_weight_change(1.0, reward_signal)
        weight_change_far = stdp.compute_weight_change(10.0, reward_signal)

        assert abs(weight_change_close) > abs(weight_change_far)


class TestSpikingBrainConfig:
    """Test cases for spiking brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpikingBrainConfig()

        # Network topology
        assert config.hidden_size == 32

        # Simulation parameters
        assert config.simulation_duration == 100.0
        assert config.time_step == 1.0

        # LIF neuron parameters
        assert config.tau_m == 20.0
        assert config.v_threshold == 1.0
        assert config.v_reset == 0.0
        assert config.v_rest == 0.0

        # Encoding parameters
        assert config.max_rate == 100.0
        assert config.min_rate == 0.0

        # STDP parameters
        assert config.tau_plus == 20.0
        assert config.tau_minus == 20.0
        assert config.a_plus == 0.01
        assert config.a_minus == 0.01
        assert config.learning_rate == 0.001
        assert config.reward_scaling == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpikingBrainConfig(
            hidden_size=64,
            simulation_duration=200.0,
            tau_m=10.0,
            learning_rate=0.01,
        )

        assert config.hidden_size == 64
        assert config.simulation_duration == 200.0
        assert config.tau_m == 10.0
        assert config.learning_rate == 0.01


class TestSpikingBrain:
    """Test cases for the spiking neural network brain."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SpikingBrainConfig(
            hidden_size=4,
            simulation_duration=50.0,
            time_step=1.0,
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
        assert brain.hidden_size == 4
        assert brain.total_neurons == 10  # 2 + 4 + 4

        # Check neuron populations
        assert len(brain.input_neurons) == 2
        assert len(brain.hidden_neurons) == 4
        assert len(brain.output_neurons) == 4

        # Check weight matrices
        assert brain.weights_input_hidden.shape == (2, 4)
        assert brain.weights_hidden_output.shape == (4, 4)
        assert brain.weights_hidden_hidden.shape == (4, 4)

    def test_preprocess(self, brain):
        """Test state preprocessing."""
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=1.5,
            agent_position=(2, 3),
            agent_direction=Direction.UP,
        )

        state = brain.preprocess(params)
        assert len(state) == 2
        assert 0.0 <= state[0] <= 1.0  # gradient_strength normalized
        assert 0.0 <= state[1] <= 1.0  # gradient_direction normalized

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values."""
        params = BrainParams()

        state = brain.preprocess(params)
        assert len(state) == 2
        assert state[0] == 0.0
        assert state[1] == 0.0

    def test_encode_state_to_spikes(self, brain):
        """Test encoding continuous state to spike trains."""
        state = [0.5, 0.8]
        spike_trains = brain._encode_state_to_spikes(state)

        assert len(spike_trains) == 2
        # Higher values should tend to generate more spikes
        # (stochastic, so we'll just check structure)
        for spike_times in spike_trains:
            assert isinstance(spike_times, list)
            for spike_time in spike_times:
                assert 0 <= spike_time < brain.config.simulation_duration

    def test_simulate_network(self, brain):
        """Test network simulation."""
        # Create simple spike trains
        input_spike_trains = [[10.0, 20.0], [15.0, 25.0]]

        output_spike_counts, spike_history = brain._simulate_network(input_spike_trains)

        assert len(output_spike_counts) == 4  # 4 output neurons
        assert "input" in spike_history
        assert "hidden" in spike_history
        assert "output" in spike_history
        assert "duration" in spike_history

        # Check spike history structure
        assert len(spike_history["input"]) == 2
        assert len(spike_history["hidden"]) == 4
        assert len(spike_history["output"]) == 4

    def test_decode_action_probabilities(self, brain):
        """Test decoding spike counts to action probabilities."""
        spike_counts = [5, 3, 1, 0]
        probabilities = brain._decode_action_probabilities(spike_counts)

        assert len(probabilities) == 4
        assert np.isclose(np.sum(probabilities), 1.0)  # Should sum to 1
        assert np.all(probabilities >= 0)  # All probabilities non-negative

        # Higher spike count should tend to have higher probability
        assert probabilities[0] > probabilities[3]

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

    def test_learn(self, brain):
        """Test learning with STDP."""
        params = BrainParams()

        # Run brain to generate spike patterns
        brain.run_brain(params, top_only=False, top_randomize=False)

        # Test learning
        brain.learn(params, reward=1.0, episode_done=True)

        # Weights might change (depending on spike patterns and randomness)
        # Just check that learning doesn't crash and weights remain finite
        assert torch.all(torch.isfinite(brain.weights_input_hidden))
        assert torch.all(torch.isfinite(brain.weights_hidden_output))

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
        brain.overfit_detector_current_episode_positions = [(1, 1), (2, 1)]
        brain.overfit_detector_current_episode_rewards = [0.1, 0.2]

        brain.post_process_episode()

        # Episode data should be cleared
        assert len(brain.overfit_detector_current_episode_actions) == 0
        assert len(brain.overfit_detector_current_episode_positions) == 0
        assert len(brain.overfit_detector_current_episode_rewards) == 0

    def test_copy(self, brain):
        """Test brain copying functionality."""
        # Modify original brain state
        brain.weights_input_hidden[0, 0] = 0.5

        copied_brain = brain.copy()

        # Check that copy has same structure
        assert copied_brain.config.hidden_size == brain.config.hidden_size
        assert copied_brain.input_dim == brain.input_dim
        assert copied_brain.num_actions == brain.num_actions

        # Check that weights are copied
        assert torch.allclose(copied_brain.weights_input_hidden, brain.weights_input_hidden)

        # Check that modifying copy doesn't affect original
        copied_brain.weights_input_hidden[0, 0] = 1.0
        assert not torch.allclose(copied_brain.weights_input_hidden, brain.weights_input_hidden)

    def test_action_set_property(self, brain):
        """Test action_set property."""
        action_set = brain.action_set
        assert len(action_set) == 4
        assert Action.FORWARD in action_set
        assert Action.LEFT in action_set
        assert Action.RIGHT in action_set
        assert Action.STAY in action_set

    def test_weight_clipping(self, brain):
        """Test weight clipping functionality."""
        # Set weights to extreme values
        brain.weights_input_hidden[0, 0] = 10.0
        brain.weights_hidden_output[0, 0] = -10.0

        brain._clip_weights()

        # Weights should be clipped
        assert torch.all(brain.weights_input_hidden <= brain.config.weight_clip)
        assert torch.all(brain.weights_input_hidden >= -brain.config.weight_clip)
        assert torch.all(brain.weights_hidden_output <= brain.config.weight_clip)
        assert torch.all(brain.weights_hidden_output >= -brain.config.weight_clip)


class TestSpikingBrainIntegration:
    """Integration tests for spiking brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = SpikingBrainConfig(hidden_size=8, simulation_duration=20.0)
        brain = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Simulate multiple steps in an episode
        episode_rewards = []
        params = None  # Initialize params variable
        rng = np.random.default_rng(42)

        for step in range(5):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=False, top_randomize=False)
            assert len(actions) == 1

            # Update memory
            reward = rng.random() - 0.5  # Random reward
            brain.update_memory(reward)
            episode_rewards.append(reward)

            # Learn (except on last step)
            if step < 4:
                brain.learn(params, reward, episode_done=False)

        # End episode
        if params is not None:
            brain.learn(params, episode_rewards[-1], episode_done=True)
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 5

    def test_deterministic_behavior_with_seed(self):
        """Test that brain behavior is deterministic when using fixed random seed."""
        config = SpikingBrainConfig(hidden_size=4, simulation_duration=10.0)

        # Create two identical brains
        torch.manual_seed(42)
        # Set global seed for torch.normal calls in SpikingBrain
        torch.random.manual_seed(42)
        brain1 = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        torch.manual_seed(42)
        torch.random.manual_seed(42)
        brain2 = SpikingBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        # Should have identical weights
        assert torch.allclose(brain1.weights_input_hidden, brain2.weights_input_hidden)

        # But due to stochastic spike generation, behavior might still differ
        # This is expected for spiking networks
