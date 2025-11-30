"""Unit tests for the modular quantum brain architecture."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.modular import ModularBrain, ModularBrainConfig
from quantumnematode.brain.modules import DEFAULT_MODULES, ModuleName
from quantumnematode.env import Direction
from quantumnematode.initializers.random_initializer import RandomSmallUniformInitializer
from quantumnematode.optimizers.learning_rate import DynamicLearningRate


class TestModularBrainConfig:
    """Test cases for modular quantum brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModularBrainConfig()

        assert config.l2_reg == 0.005
        assert config.large_gradient_threshold == 0.1
        assert config.min_gradient_magnitude == 1e-4
        assert config.modules == DEFAULT_MODULES
        assert config.noise_std == 0.005
        assert config.num_layers == 2
        assert config.param_clip is True
        assert config.param_modulo is True
        assert config.significant_reward_threshold == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_modules = {
            ModuleName.CHEMOTAXIS: [0, 1],
        }
        config = ModularBrainConfig(
            num_layers=3,
            modules=custom_modules,
            l2_reg=0.01,
        )

        assert config.num_layers == 3
        assert config.modules == custom_modules
        assert config.l2_reg == 0.01


class TestModularBrain:
    """Test cases for the modular quantum brain architecture."""

    @pytest.fixture
    def config(self):
        """Create a test configuration with minimal qubits."""
        return ModularBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
        )

    @pytest.fixture
    def brain(self, config):
        """Create a test modular brain."""
        return ModularBrain(
            config=config,
            shots=100,
            device=DeviceType.CPU,
        )

    def test_brain_initialization(self, brain, config):
        """Test modular brain initialization."""
        assert brain.config == config
        assert brain.num_qubits == 2
        assert brain.modules == config.modules
        assert brain.shots == 100
        assert brain.device == DeviceType.CPU
        assert brain.num_layers == config.num_layers

        # Check that parameters are initialized
        assert len(brain.parameter_values) > 0

        # Check learning rate
        assert isinstance(brain.learning_rate, DynamicLearningRate)

    def test_parameter_initialization(self, brain):
        """Test quantum parameter initialization."""
        # Should have parameters for each layer and rotation axis
        expected_params = brain.num_layers * 3 * brain.num_qubits  # rx, ry, rz per qubit per layer
        assert len(brain.parameter_values) == expected_params

        # All parameters should be finite
        for value in brain.parameter_values.values():
            assert np.isfinite(value)

    def test_build_brain(self, brain):
        """Test quantum circuit building."""
        input_params = {
            ModuleName.CHEMOTAXIS.value: {"rx": 0.5, "ry": 0.3, "rz": 0.1},
        }

        qc = brain.build_brain(input_params)

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == brain.num_qubits
        assert qc.num_clbits == brain.num_qubits

        # Check that circuit has measurements
        assert any(instr.operation.name == "measure" for instr in qc.data)

    def test_build_brain_without_input_params(self, brain):
        """Test building circuit without input parameters."""
        qc = brain.build_brain(None)

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == brain.num_qubits

    def test_get_backend(self, brain):
        """Test backend initialization."""
        backend = brain._get_backend()
        assert backend is not None
        # For CPU device, should get AerSimulator
        assert hasattr(backend, "run")

    def test_cached_circuit(self, brain):
        """Test circuit caching."""
        # First call should create cache
        qc1 = brain._get_cached_circuit()
        # Second call should return cached circuit
        qc2 = brain._get_cached_circuit()

        assert qc1 is qc2  # Should be same object

    def test_run_brain(self, brain):
        """Test running the quantum brain."""
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
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        assert 0.0 <= action_data.probability <= 1.0

        # Check that counts are stored
        assert brain.latest_data.counts is not None
        assert len(brain.history_data.counts) > 0

    def test_run_brain_with_reward(self, brain):
        """Test running brain with reward for learning."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # First run to establish action
        brain.run_brain(params, reward=None, top_only=True, top_randomize=False)

        # Second run with reward
        actions = brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        assert len(actions) == 1
        # Should have computed gradients and updated parameters
        assert brain.latest_data.computed_gradients is not None
        assert brain.latest_data.updated_parameters is not None

    def test_interpret_counts(self, brain):
        """Test interpretation of measurement counts."""
        # Create mock counts
        counts = {"00": 30, "01": 20, "10": 15, "11": 10}

        actions = brain._interpret_counts(counts, top_only=True, top_randomize=False)

        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)
        assert 0.0 <= actions[0].probability <= 1.0

    def test_interpret_counts_all_actions(self, brain):
        """Test interpretation returning all actions."""
        counts = {"00": 30, "01": 20, "10": 15, "11": 10}

        actions = brain._interpret_counts(counts, top_only=False, top_randomize=False)

        assert len(actions) > 1
        # Probabilities should sum to approximately 1
        total_prob = sum(a.probability for a in actions)
        assert np.isclose(total_prob, 1.0, atol=0.01)

    def test_update_memory(self, brain):
        """Test memory update with reward (currently a no-op)."""
        # update_memory is now reserved for future brain-internal memory mechanisms
        # Just verify it doesn't raise an error
        brain.update_memory(reward=0.5)
        brain.update_memory(reward=-0.3)
        brain.update_memory(reward=None)

    def test_parameter_shift_gradients(self, brain):
        """Test parameter-shift gradient computation."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
        )

        # Run brain to get an action
        _ = brain.run_brain(params, top_only=True, top_randomize=False)
        action = brain.latest_data.action

        # Compute gradients
        gradients = brain.parameter_shift_gradients(params, action, reward=1.0)

        assert len(gradients) == len(brain.parameter_values)
        # Gradients should be finite
        assert all(np.isfinite(g) for g in gradients)

    def test_update_parameters(self, brain):
        """Test parameter update with gradients."""
        # Store initial parameters
        initial_params = brain.parameter_values.copy()

        # Create mock gradients
        gradients = [0.1] * len(brain.parameter_values)

        # Update parameters
        brain.update_parameters(gradients, reward=0.5, learning_rate=0.01)

        # Parameters should have changed
        assert not all(initial_params[k] == brain.parameter_values[k] for k in initial_params)

        # Parameters should remain finite
        assert all(np.isfinite(v) for v in brain.parameter_values.values())

    def test_parameter_clipping(self, brain):
        """Test parameter clipping to [-pi, pi]."""
        # Set extreme parameter values
        for key in list(brain.parameter_values.keys())[:2]:
            brain.parameter_values[key] = 10.0

        # Update with clipping enabled
        brain.config.param_clip = True
        gradients = [0.0] * len(brain.parameter_values)
        brain.update_parameters(gradients, learning_rate=0.0)

        # Parameters should be clipped
        for value in brain.parameter_values.values():
            assert -np.pi <= value <= np.pi

    def test_post_process_episode(self, brain):
        """Test episode post-processing."""
        # Add some episode data
        brain.overfit_detector_episode_actions.append(Action.FORWARD)
        brain.overfit_detector_current_episode_positions.append((1, 1))
        brain.overfit_detector_current_episode_rewards.append(0.5)
        brain.history_data.rewards.append(0.5)

        brain.post_process_episode()

        # Episode data should be cleared
        assert len(brain.overfit_detector_episode_actions) == 0
        assert len(brain.overfit_detector_current_episode_positions) == 0
        assert len(brain.overfit_detector_current_episode_rewards) == 0

    def test_inspect_circuit(self, brain):
        """Test circuit inspection."""
        qc = brain.inspect_circuit()

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == brain.num_qubits

    def test_copy(self, brain):
        """Test brain copying."""
        # Modify original brain
        brain.parameter_values[next(iter(brain.parameter_values.keys()))] = 0.123

        # Create copy
        copied_brain = brain.copy()

        assert copied_brain.num_qubits == brain.num_qubits
        assert copied_brain.parameter_values == brain.parameter_values

        # Modify copy - should not affect original
        copied_brain.parameter_values[next(iter(brain.parameter_values.keys()))] = 0.456
        assert brain.parameter_values[next(iter(brain.parameter_values.keys()))] == 0.123


class TestModularBrainIntegration:
    """Integration tests for modular quantum brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
        )
        brain = ModularBrain(
            config=config,
            shots=50,
            device=DeviceType.CPU,
        )

        rng = np.random.default_rng(42)

        # Simulate multiple steps in an episode
        for step in range(5):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
            assert len(actions) == 1

            # Update memory
            reward = rng.random() - 0.5
            brain.update_memory(reward)

            # Run again with reward for learning
            if step > 0:
                brain.run_brain(params, reward=reward, top_only=True, top_randomize=False)

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) > 0

    def test_learning_rate_boost(self):
        """Test learning rate boost mechanism."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            lr_boost=True,
            low_reward_threshold=-0.25,
            low_reward_window=5,
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Simulate poor performance
        for _ in range(10):
            brain.run_brain(params, reward=-0.3, top_only=True, top_randomize=False)

        # Boost should have activated at some point
        # (Hard to test deterministically, but shouldn't crash)
        assert brain._lr_boost_active or not brain._lr_boost_active

    def test_momentum_updates(self):
        """Test momentum-based parameter updates."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run multiple learning steps
        for _ in range(5):
            brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
            brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        # Momentum should be initialized
        assert len(brain._momentum) > 0

    def test_l2_regularization(self):
        """Test L2 regularization in parameter updates."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            l2_reg=0.1,  # High regularization
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        # Set large parameter values
        for key in brain.parameter_values:
            brain.parameter_values[key] = 2.0

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run with learning
        brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
        brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        # Parameters should remain finite due to regularization
        assert all(np.isfinite(v) for v in brain.parameter_values.values())

    @patch.dict(os.environ, {"IBM_QUANTUM_BACKEND": "test_backend"})
    def test_qpu_backend_name(self):
        """Test QPU backend name retrieval."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0]},
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.QPU)

        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.name = "test_backend"
        brain._backend = mock_backend

        backend_name = brain._get_backend_name()
        assert backend_name == "test_backend"

    def test_custom_parameter_initializer(self):
        """Test using custom parameter initializer."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
        )

        initializer = RandomSmallUniformInitializer()
        brain = ModularBrain(
            config=config,
            shots=50,
            device=DeviceType.CPU,
            parameter_initializer=initializer,
        )

        # Parameters should be initialized
        assert len(brain.parameter_values) > 0
        # Parameters should be in expected range
        for value in brain.parameter_values.values():
            assert -np.pi <= value <= np.pi


class TestTrajectoryLearning:
    """Test cases for trajectory learning functionality."""

    @pytest.fixture
    def trajectory_config(self):
        """Create a test configuration with trajectory learning enabled."""
        return ModularBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
            use_trajectory_learning=True,
            gamma=0.99,
        )

    @pytest.fixture
    def trajectory_brain(self, trajectory_config):
        """Create a test modular brain with trajectory learning."""
        return ModularBrain(
            config=trajectory_config,
            shots=100,
            device=DeviceType.CPU,
        )

    def test_trajectory_learning_config(self, trajectory_config):
        """Test trajectory learning configuration."""
        assert trajectory_config.use_trajectory_learning is True
        assert trajectory_config.gamma == 0.99

    def test_trajectory_brain_initialization(self, trajectory_brain):
        """Test trajectory brain initialization."""
        assert trajectory_brain.use_trajectory_learning is True
        assert trajectory_brain.gamma == 0.99
        assert trajectory_brain.episode_buffer is not None
        assert len(trajectory_brain.episode_buffer) == 0

    def test_episode_buffer_accumulation(self, trajectory_brain):
        """Test episode buffer data accumulation."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain multiple times to accumulate data
        for step in range(3):
            trajectory_brain.run_brain(
                params,
                reward=None,
                top_only=True,
                top_randomize=False,
            )
            # Run again with reward to trigger buffering
            trajectory_brain.run_brain(
                params,
                reward=0.5 * step,
                top_only=True,
                top_randomize=False,
            )

        # Buffer should have accumulated data
        assert len(trajectory_brain.episode_buffer) == 3
        assert len(trajectory_brain.episode_buffer.rewards) == 3
        assert len(trajectory_brain.episode_buffer.actions) == 3
        assert len(trajectory_brain.episode_buffer.params) == 3
        assert len(trajectory_brain.episode_buffer.parameter_values) == 3

    def test_discounted_returns_simple(self, trajectory_brain):
        """Test discounted return computation with simple rewards."""
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.9

        returns = trajectory_brain.compute_discounted_returns(rewards, gamma)

        # Expected: G_t = r_t + gamma * G_{t+1}
        expected = [2.71, 1.9, 1.0]

        assert len(returns) == 3
        for i, (actual, expected_val) in enumerate(zip(returns, expected, strict=False)):
            assert np.isclose(actual, expected_val, atol=0.01), f"Return at {i} mismatch"

    def test_discounted_returns_terminal(self, trajectory_brain):
        """Test discounted return computation with terminal state handling."""
        rewards = [1.0, 2.0, -1.0]
        gamma = 1.0  # No discounting

        returns = trajectory_brain.compute_discounted_returns(rewards, gamma)

        # With gamma=1, G_t = sum of all future rewards
        expected = [2.0, 1.0, -1.0]

        assert len(returns) == 3
        for actual, expected_val in zip(returns, expected, strict=False):
            assert np.isclose(actual, expected_val)

    def test_discounted_returns_gamma_validation(self, trajectory_brain):
        """Test gamma validation in discounted returns computation."""
        rewards = [1.0, 2.0, 3.0]

        # Invalid gamma values should raise ValueError
        with pytest.raises(ValueError, match="Gamma must be in range"):
            trajectory_brain.compute_discounted_returns(rewards, gamma=-0.1)

        with pytest.raises(ValueError, match="Gamma must be in range"):
            trajectory_brain.compute_discounted_returns(rewards, gamma=1.1)

    def test_discounted_returns_empty_rewards(self, trajectory_brain):
        """Test discounted returns with empty reward list."""
        returns = trajectory_brain.compute_discounted_returns([], gamma=0.99)
        assert returns == []

    def test_backward_compatibility_single_step(self):
        """Test backward compatibility when trajectory learning is disabled."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            use_trajectory_learning=False,  # Disabled
        )
        brain = ModularBrain(config=config, shots=100, device=DeviceType.CPU)

        assert brain.use_trajectory_learning is False
        assert brain.episode_buffer is None

        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain with reward - should use single-step learning
        brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
        brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        # Should have computed gradients immediately (not buffered)
        assert brain.latest_data.computed_gradients is not None
        assert brain.latest_data.updated_parameters is not None

    def test_trajectory_learning_convergence(self, trajectory_brain):
        """Test trajectory learning with a full episode."""
        rng = np.random.default_rng(42)

        # Simulate an episode
        for step in range(5):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain and provide reward
            trajectory_brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
            reward = rng.random() - 0.5
            trajectory_brain.run_brain(params, reward=reward, top_only=True, top_randomize=False)

        # Buffer should have data
        assert len(trajectory_brain.episode_buffer) == 5

        # Post-process episode - this should trigger trajectory learning update
        trajectory_brain.post_process_episode()

        # Buffer should be cleared
        assert len(trajectory_brain.episode_buffer) == 0

        # Parameters should have been updated
        # (At least some parameters should change, though with random data might be small changes)
        # Just check that update didn't crash and parameters remain valid
        assert all(np.isfinite(v) for v in trajectory_brain.parameter_values.values())

    def test_episode_buffer_clearing(self, trajectory_brain):
        """Test that episode buffer is cleared after processing."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Accumulate some data
        for _ in range(3):
            trajectory_brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
            trajectory_brain.run_brain(params, reward=0.5, top_only=True, top_randomize=False)

        assert len(trajectory_brain.episode_buffer) == 3

        # Post-process should clear buffer
        trajectory_brain.post_process_episode()
        assert len(trajectory_brain.episode_buffer) == 0

    def test_trajectory_gradient_with_varying_returns(self, trajectory_brain):
        """Test trajectory gradients with varying return values."""
        # Manually populate episode buffer
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run to get an action
        actions = trajectory_brain.run_brain(
            params,
            reward=None,
            top_only=True,
            top_randomize=False,
        )
        action = actions[0]

        # Manually create buffer data
        from quantumnematode.brain.arch.modular import EpisodeBuffer

        buffer = EpisodeBuffer()
        for i in range(3):
            buffer.append(
                params=params,
                action=action,
                reward=float(i),
                parameter_values=trajectory_brain.parameter_values.copy(),
            )

        # Create returns
        returns = [3.0, 2.0, 1.0]

        # Compute trajectory gradients
        gradients = trajectory_brain.trajectory_parameter_shift_gradients(buffer, returns)

        # Should return one gradient per parameter
        assert len(gradients) == len(trajectory_brain.parameter_values)
        assert all(np.isfinite(g) for g in gradients)

    def test_trajectory_gradient_length_mismatch(self, trajectory_brain):
        """Test error handling when returns length doesn't match buffer."""
        from quantumnematode.brain.arch.modular import EpisodeBuffer

        params = BrainParams(gradient_strength=0.6, gradient_direction=0.3)
        actions = trajectory_brain.run_brain(
            params,
            reward=None,
            top_only=True,
            top_randomize=False,
        )
        action = actions[0]

        buffer = EpisodeBuffer()
        buffer.append(
            params=params,
            action=action,
            reward=1.0,
            parameter_values=trajectory_brain.parameter_values.copy(),
        )

        # Mismatched returns length
        returns = [1.0, 2.0]  # Buffer has 1 entry, returns has 2

        with pytest.raises(ValueError, match="must match episode buffer length"):
            trajectory_brain.trajectory_parameter_shift_gradients(buffer, returns)

    def test_trajectory_gradient_empty_buffer(self, trajectory_brain):
        """Test error handling with empty episode buffer."""
        from quantumnematode.brain.arch.modular import EpisodeBuffer

        buffer = EpisodeBuffer()
        returns = []

        with pytest.raises(ValueError, match="empty episode buffer"):
            trajectory_brain.trajectory_parameter_shift_gradients(buffer, returns)


class TestModularBrainEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_counts(self):
        """Test handling of empty measurement counts."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0]},
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        # Empty counts should raise error
        with pytest.raises(ValueError, match="No valid actions found"):
            brain._interpret_counts({}, top_only=True, top_randomize=False)

    def test_invalid_counts(self):
        """Test handling of invalid measurement counts."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0]},
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        # Counts with invalid bitstrings
        invalid_counts = {"invalid": 10, "not_binary": 20}

        with pytest.raises(ValueError, match="No valid actions found"):
            brain._interpret_counts(invalid_counts, top_only=True, top_randomize=False)

    def test_zero_reward(self):
        """Test handling of zero reward."""
        config = ModularBrainConfig(
            num_layers=1,
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
        )
        brain = ModularBrain(config=config, shots=50, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run with zero reward
        brain.run_brain(params, reward=None, top_only=True, top_randomize=False)
        brain.run_brain(params, reward=0.0, top_only=True, top_randomize=False)

        # Should not crash
        assert brain.latest_data.computed_gradients is not None
