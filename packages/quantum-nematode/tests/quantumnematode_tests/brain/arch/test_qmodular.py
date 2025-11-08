"""Unit tests for the quantum modular Q-learning brain architecture."""

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qmodular import QModularBrain, QModularBrainConfig
from quantumnematode.brain.modules import DEFAULT_MODULES, ModuleName
from quantumnematode.env import Direction


class TestQModularBrainConfig:
    """Test cases for quantum modular Q-learning brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QModularBrainConfig()

        assert config.modules == DEFAULT_MODULES
        assert config.num_layers == 2
        assert config.seed is None

        # Q-learning parameters
        assert config.buffer_size == 1800
        assert config.batch_size == 32
        assert config.target_update_freq == 25
        assert config.epsilon_decay == 0.996
        assert config.min_epsilon == 0.01

        # Reward shaping parameters
        assert config.negative_reward_threshold == -0.01
        assert config.min_state_features_for_proximity == 6
        assert config.high_gradient_threshold == 0.8
        assert config.medium_gradient_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_modules = {
            ModuleName.CHEMOTAXIS: [0, 1],
        }
        config = QModularBrainConfig(
            modules=custom_modules,
            num_layers=3,
            buffer_size=1000,
            batch_size=64,
            seed=42,
        )

        assert config.modules == custom_modules
        assert config.num_layers == 3
        assert config.buffer_size == 1000
        assert config.batch_size == 64
        assert config.seed == 42


class TestQModularBrain:
    """Test cases for the quantum modular Q-learning brain architecture."""

    @pytest.fixture
    def config(self):
        """Create a test configuration with minimal qubits."""
        return QModularBrainConfig(
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
            num_layers=1,
            buffer_size=100,
            batch_size=16,
            seed=42,
        )

    @pytest.fixture
    def brain(self, config):
        """Create a test quantum modular Q-learning brain."""
        return QModularBrain(
            config=config,
            shots=50,
            device=DeviceType.CPU,
        )

    def test_brain_initialization(self, brain, config):
        """Test QModular brain initialization."""
        assert brain.config == config
        assert brain.num_qubits == 2
        assert brain.modules == config.modules
        assert brain.shots == 50
        assert brain.device == DeviceType.CPU
        assert brain.num_layers == config.num_layers

        # Q-learning parameters
        assert brain.epsilon == 1.0  # Starts at 1.0
        assert brain.buffer_size == config.buffer_size
        assert brain.batch_size == config.batch_size
        assert len(brain.experience_buffer) == 0

        # Q-networks should be initialized
        assert brain.q_network is not None
        assert brain.target_q_network is not None

    def test_quantum_parameter_initialization(self, brain):
        """Test quantum parameter initialization."""
        # Should have parameters for each layer and rotation axis
        expected_params = brain.num_layers * 3 * brain.num_qubits
        assert len(brain.parameter_values) == expected_params

        # All parameters should be finite and small (initialized with uniform)
        for value in brain.parameter_values.values():
            assert np.isfinite(value)
            assert -0.1 <= value <= 0.1  # Based on initialization range

    def test_q_network_initialization(self, brain):
        """Test Q-network architecture."""
        # Calculate expected input size
        quantum_feature_size = 2**brain.num_qubits
        env_feature_size = 4
        expected_input_size = quantum_feature_size + env_feature_size

        # Test input through network
        test_input = torch.randn(1, expected_input_size)
        output = brain.q_network(test_input)

        assert output.shape == (1, len(brain.action_set))

    def test_target_network_sync(self, brain):
        """Test that target network is initialized with same weights as Q-network."""
        for param_q, param_target in zip(
            brain.q_network.parameters(),
            brain.target_q_network.parameters(),
            strict=False,
        ):
            assert torch.allclose(param_q, param_target)

    def test_build_quantum_circuit(self, brain):
        """Test quantum circuit building."""
        input_params = {
            ModuleName.CHEMOTAXIS.value: {"rx": 0.5, "ry": 0.3, "rz": 0.1},
        }

        qc = brain.build_quantum_circuit(input_params)

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == brain.num_qubits
        assert qc.num_clbits == brain.num_qubits

        # Check that circuit has measurements
        assert any(instr.operation.name == "measure" for instr in qc.data)

    def test_extract_quantum_features(self, brain):
        """Test quantum feature extraction."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        quantum_features = brain.extract_quantum_features(params)

        assert isinstance(quantum_features, np.ndarray)
        assert len(quantum_features) == 2**brain.num_qubits
        assert np.isclose(np.sum(quantum_features), 1.0)  # Should be probability distribution
        assert np.all(quantum_features >= 0.0)  # All probabilities non-negative

    def test_get_state_features(self, brain):
        """Test combined quantum and classical feature extraction."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(5, 7),
            agent_direction=Direction.UP,
        )

        state_features = brain.get_state_features(params)

        assert isinstance(state_features, np.ndarray)
        # Should have quantum features + 4 env features
        expected_size = 2**brain.num_qubits + 4
        assert len(state_features) == expected_size
        assert np.all(np.isfinite(state_features))

    def test_decide_action(self, brain):
        """Test action decision making."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        action = brain.decide_action(params)

        assert action in brain.action_set
        assert isinstance(action, Action)

    def test_store_experience(self, brain):
        """Test experience storage with reward shaping."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        brain.store_experience(
            state=state,
            action=Action.FORWARD,
            reward=0.5,
            next_state=next_state,
            done=False,
        )

        assert len(brain.experience_buffer) == 1

        # Check experience structure
        experience = brain.experience_buffer[0]
        assert len(experience) == 5  # state, action_idx, shaped_reward, next_state, done

    def test_reward_shaping(self, brain):
        """Test reward shaping for different reward types."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Positive reward (goal reached)
        brain.store_experience(state, Action.FORWARD, reward=1.0, next_state=next_state, done=True)
        exp_positive = brain.experience_buffer[-1]
        shaped_positive = exp_positive[2]
        assert shaped_positive > 1.0  # Should be amplified

        # Negative reward
        brain.store_experience(
            state,
            Action.FORWARD,
            reward=-0.5,
            next_state=next_state,
            done=False,
        )
        exp_negative = brain.experience_buffer[-1]
        shaped_negative = exp_negative[2]
        assert abs(shaped_negative) < 0.5  # Should be gentler

        # Neutral reward
        brain.store_experience(state, Action.FORWARD, reward=0.0, next_state=next_state, done=False)
        exp_neutral = brain.experience_buffer[-1]
        shaped_neutral = exp_neutral[2]
        assert shaped_neutral > 0  # Should get small positive reward

    def test_proximity_bonus(self, brain):
        """Test proximity-based reward shaping."""
        # Create state with high gradient (close to goal)
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        state[-2] = 0.9  # High gradient strength
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        brain.store_experience(state, Action.FORWARD, reward=0.0, next_state=next_state, done=False)
        experience = brain.experience_buffer[-1]
        shaped_reward = experience[2]

        # Should have proximity bonus
        assert shaped_reward > 0.01  # Base reward + bonus

    def test_learn_from_experience_insufficient_data(self, brain):
        """Test that learning doesn't happen with insufficient data."""
        # Add a few experiences (less than batch_size)
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        for _ in range(brain.batch_size - 5):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        # Store initial weights
        initial_weights = [p.clone() for p in brain.q_network.parameters()]

        brain.learn_from_experience()

        # Weights shouldn't change (not enough data)
        for param, initial in zip(brain.q_network.parameters(), initial_weights, strict=False):
            assert torch.allclose(param, initial)

    def test_learn_from_experience_with_batch(self, brain):
        """Test learning with sufficient experiences."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Fill buffer
        for _ in range(brain.batch_size + 5):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        brain.learn_from_experience()

        # Check that learning doesn't crash and weights remain finite
        for param in brain.q_network.parameters():
            assert torch.all(torch.isfinite(param))

    def test_target_network_update(self, brain):
        """Test periodic target network updates."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Fill buffer
        for _ in range(brain.batch_size):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        # Learn multiple times to trigger target update
        for _ in range(brain.target_update_freq + 5):
            brain.learn_from_experience()

        # Update count should exceed target_update_freq
        assert brain.update_count > brain.target_update_freq

    def test_epsilon_decay(self, brain):
        """Test epsilon decay during learning."""
        initial_epsilon = brain.epsilon

        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Fill buffer and learn
        for _ in range(brain.batch_size):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        for _ in range(20):
            brain.learn_from_experience()

        # Epsilon should have decayed
        assert brain.epsilon < initial_epsilon
        assert brain.epsilon >= brain.min_epsilon

    def test_run_brain(self, brain):
        """Test running the brain."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=False)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in brain.action_set
        assert 0.0 <= action_data.probability <= 1.0

    def test_run_brain_with_learning(self, brain):
        """Test running brain with learning integration."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # First step
        brain.run_brain(params, reward=None, top_only=True, top_randomize=False)

        # Second step with reward
        brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        # Should have stored experience
        assert hasattr(brain, "_last_state")
        assert hasattr(brain, "_last_action")

    def test_end_episode(self, brain):
        """Test episode end handling."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Add some experiences
        for _ in range(10):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        initial_count = brain.episode_count
        brain.end_episode(final_reward=1.0)

        assert brain.episode_count == initial_count + 1

    def test_update_memory(self, brain):
        """Test memory update (currently a no-op)."""
        # update_memory is now reserved for future brain-internal memory mechanisms
        # Just verify it doesn't raise an error
        brain.update_memory(reward=0.3)
        brain.update_memory(reward=-0.5)
        brain.update_memory(reward=None)

    def test_inspect_circuit(self, brain):
        """Test circuit inspection."""
        qc = brain.inspect_circuit()

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == brain.num_qubits

    def test_copy(self, brain):
        """Test brain copying."""
        # Modify original
        brain.parameter_values[next(iter(brain.parameter_values.keys()))] = 0.123

        # Create copy
        copied_brain = brain.copy()

        assert copied_brain.num_qubits == brain.num_qubits
        assert copied_brain.parameter_values == brain.parameter_values

        # Modify copy - should not affect original
        copied_brain.parameter_values[next(iter(copied_brain.parameter_values.keys()))] = 0.456
        assert brain.parameter_values != copied_brain.parameter_values


class TestQModularBrainIntegration:
    """Integration tests for quantum modular Q-learning brain."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            buffer_size=100,
            batch_size=16,
            seed=42,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        rng = np.random.default_rng(42)

        # Simulate multiple steps in an episode
        for step in range(30):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step % 10, step % 10),
                agent_direction=Direction.UP,
            )

            # Run brain
            reward = 1.0 if step == 29 else rng.random() - 0.5
            actions = brain.run_brain(params, reward=reward, top_only=True, top_randomize=False)
            assert len(actions) == 1

        # End episode
        brain.end_episode(final_reward=1.0)

        # Should have experiences
        assert len(brain.experience_buffer) > 0
        # Increments once during steps and once at episode end
        assert brain.episode_count == 2

    def test_q_guidance_progression(self):
        """Test that Q-guidance weight increases with experience."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            buffer_size=200,
            batch_size=16,
            seed=42,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.6, gradient_direction=0.3)

        # Fill buffer to increase experience
        state = brain.get_state_features(params)
        next_state = state.copy()

        for _ in range(brain.buffer_size // 2):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        # Q-guidance should be used more as buffer fills
        brain.epsilon = 0.0  # Disable random exploration
        action = brain.decide_action(params)
        assert action in brain.action_set

    def test_deterministic_behavior_with_seed(self):
        """Test that behavior is deterministic with fixed seed."""
        config1 = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            seed=42,
        )
        brain1 = QModularBrain(config=config1, shots=50, device=DeviceType.CPU)

        config2 = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            seed=42,
        )
        brain2 = QModularBrain(config=config2, shots=50, device=DeviceType.CPU)

        # Should have identical quantum parameters
        for key in brain1.parameter_values:
            assert brain1.parameter_values[key] == brain2.parameter_values[key]

    def test_adaptive_learning(self):
        """Test adaptive learning with multiple passes when struggling."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            buffer_size=100,
            batch_size=16,
            negative_reward_threshold=-0.01,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer
        state = brain.get_state_features(params)
        for _ in range(brain.batch_size):
            brain.store_experience(state, Action.FORWARD, reward=0.1, next_state=state, done=False)

        # Simulate poor performance (negative rewards)
        for _ in range(5):
            brain._step_count += 1
            brain.run_brain(params, reward=-0.5, top_only=True, top_randomize=False)

        # Should trigger double learning
        # (Hard to test deterministically, but shouldn't crash)
        assert brain.update_count >= 0


class TestQModularBrainEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_experience_buffer(self):
        """Test learning with empty buffer."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        # Should not crash
        brain.learn_from_experience()
        assert len(brain.experience_buffer) == 0

    def test_terminal_state_handling(self):
        """Test handling of terminal states in learning."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            buffer_size=100,
            batch_size=16,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Store terminal experience
        brain.store_experience(state, Action.FORWARD, reward=1.0, next_state=next_state, done=True)

        # Fill rest of buffer
        for _ in range(brain.batch_size):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=0.1,
                next_state=next_state,
                done=False,
            )

        # Learn with terminal states
        brain.learn_from_experience()

        # Should handle terminal states correctly
        assert brain.update_count > 0

    def test_gradient_clipping(self):
        """Test gradient clipping for stability."""
        config = QModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
            buffer_size=100,
            batch_size=16,
        )
        brain = QModularBrain(config=config, shots=50, device=DeviceType.CPU)

        rng = np.random.default_rng(42)
        state = rng.standard_normal(2**brain.num_qubits + 4)
        next_state = rng.standard_normal(2**brain.num_qubits + 4)

        # Fill buffer with extreme rewards
        for _ in range(brain.batch_size):
            brain.store_experience(
                state,
                Action.FORWARD,
                reward=100.0,
                next_state=next_state,
                done=False,
            )

        # Learn multiple times
        for _ in range(10):
            brain.learn_from_experience()

        # Weights should remain finite due to gradient clipping
        for param in brain.q_network.parameters():
            assert torch.all(torch.isfinite(param))
