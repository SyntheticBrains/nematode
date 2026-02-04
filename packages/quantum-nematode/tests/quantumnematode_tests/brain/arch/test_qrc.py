"""Unit tests for the QRC (Quantum Reservoir Computing) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qrc import QRCBrain, QRCBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestQRCBrainConfig:
    """Test cases for QRC brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QRCBrainConfig()

        assert config.num_reservoir_qubits == 8
        assert config.reservoir_depth == 3
        assert config.reservoir_seed == 42
        assert config.readout_hidden == 32
        assert config.readout_type == "mlp"
        assert config.shots == 1024
        assert config.gamma == 0.99
        assert config.learning_rate == 0.001
        assert config.baseline_alpha == 0.05

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QRCBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            reservoir_seed=123,
            readout_hidden=64,
            readout_type="linear",
            shots=512,
            gamma=0.95,
            learning_rate=0.005,
        )

        assert config.num_reservoir_qubits == 4
        assert config.reservoir_depth == 2
        assert config.reservoir_seed == 123
        assert config.readout_hidden == 64
        assert config.readout_type == "linear"
        assert config.shots == 512
        assert config.gamma == 0.95
        assert config.learning_rate == 0.005

    def test_config_sensory_modules_default(self):
        """Test that sensory_modules defaults to None (legacy mode)."""
        config = QRCBrainConfig()
        assert config.sensory_modules is None

    def test_config_with_sensory_modules(self):
        """Test configuration with sensory modules."""
        config = QRCBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )

        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2
        assert ModuleName.FOOD_CHEMOTAXIS in config.sensory_modules
        assert ModuleName.NOCICEPTION in config.sensory_modules

    def test_validation_num_reservoir_qubits(self):
        """Test validation for num_reservoir_qubits."""
        with pytest.raises(ValueError, match="num_reservoir_qubits must be >= 2"):
            QRCBrainConfig(num_reservoir_qubits=1)

    def test_validation_reservoir_depth(self):
        """Test validation for reservoir_depth."""
        with pytest.raises(ValueError, match="reservoir_depth must be >= 1"):
            QRCBrainConfig(reservoir_depth=0)

    def test_validation_readout_hidden(self):
        """Test validation for readout_hidden."""
        with pytest.raises(ValueError, match="readout_hidden must be >= 1"):
            QRCBrainConfig(readout_hidden=0)

    def test_validation_shots(self):
        """Test validation for shots."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QRCBrainConfig(shots=50)

    def test_validation_readout_type(self):
        """Test validation for readout_type."""
        with pytest.raises(ValueError, match="readout_type must be one of"):
            QRCBrainConfig(readout_type="invalid")


class TestQRCBrainReservoirCircuit:
    """Test cases for reservoir circuit construction."""

    @pytest.fixture
    def small_config(self) -> QRCBrainConfig:
        """Create a small test configuration for faster tests."""
        return QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=2,
            reservoir_seed=42,
            readout_hidden=8,
            shots=100,
        )

    @pytest.fixture
    def brain(self, small_config) -> QRCBrain:
        """Create a test QRC brain."""
        return QRCBrain(
            config=small_config,
            num_actions=4,
            device=DeviceType.CPU,
        )

    def test_reservoir_circuit_structure(self, brain):
        """Verify Hadamard + rotation + CZ structure in reservoir circuit."""
        circuit = brain._reservoir_circuit

        # Check circuit has correct number of qubits
        assert circuit.num_qubits == brain.num_qubits

        # Extract gate names
        gate_names = [instruction.operation.name for instruction in circuit.data]

        # Should have Hadamard gates at the beginning
        h_count = gate_names.count("h")
        assert h_count == brain.num_qubits, "Should have H gate on each qubit"

        # Should have rotation gates (rx, ry, rz)
        rx_count = gate_names.count("rx")
        ry_count = gate_names.count("ry")
        rz_count = gate_names.count("rz")

        # Each layer has one rx, ry, rz per qubit
        expected_rotations_per_type = brain.num_qubits * brain.reservoir_depth
        assert rx_count == expected_rotations_per_type
        assert ry_count == expected_rotations_per_type
        assert rz_count == expected_rotations_per_type

        # Should have CZ gates for entanglement
        cz_count = gate_names.count("cz")
        # Each layer has num_qubits CZ gates in circular topology
        expected_cz = brain.num_qubits * brain.reservoir_depth
        assert cz_count == expected_cz

    def test_reservoir_reproducibility(self):
        """Same seed should produce identical reservoir circuits and outputs."""
        config1 = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=2,
            reservoir_seed=42,
            shots=200,
        )
        config2 = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=2,
            reservoir_seed=42,
            shots=200,
        )

        brain1 = QRCBrain(config=config1, num_actions=4, device=DeviceType.CPU)
        brain2 = QRCBrain(config=config2, num_actions=4, device=DeviceType.CPU)

        # Circuits should be identical
        circuit1 = brain1._reservoir_circuit
        circuit2 = brain2._reservoir_circuit

        # Compare gate parameters
        for instr1, instr2 in zip(circuit1.data, circuit2.data, strict=False):
            assert instr1.operation.name == instr2.operation.name
            if hasattr(instr1.operation, "params") and instr1.operation.params:
                for p1, p2 in zip(instr1.operation.params, instr2.operation.params, strict=False):
                    assert np.isclose(float(p1), float(p2))

        # Same input should produce similar outputs (probabilistic, so test multiple times)
        features = np.array([0.5, 0.3], dtype=np.float32)
        state1 = brain1._extract_reservoir_state(features)
        state2 = brain2._extract_reservoir_state(features)

        # States should be similar (probabilistic, allow some tolerance)
        assert state1.shape == state2.shape
        # With same seed and same circuit, outputs should be close
        assert np.allclose(state1, state2, atol=0.2)  # Allow shot noise variation

    def test_different_seeds_different_circuits(self):
        """Different seeds should produce different reservoir circuits."""
        config1 = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=2,
            reservoir_seed=42,
        )
        config2 = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=2,
            reservoir_seed=123,
        )

        brain1 = QRCBrain(config=config1, num_actions=4, device=DeviceType.CPU)
        brain2 = QRCBrain(config=config2, num_actions=4, device=DeviceType.CPU)

        # Compare rotation angles - they should be different
        circuit1 = brain1._reservoir_circuit
        circuit2 = brain2._reservoir_circuit

        angles_different = False
        for instr1, instr2 in zip(circuit1.data, circuit2.data, strict=False):
            if instr1.operation.name in ("rx", "ry", "rz") and not np.isclose(
                float(instr1.operation.params[0]),
                float(instr2.operation.params[0]),
            ):
                angles_different = True
                break

        assert angles_different, "Different seeds should produce different rotation angles"


class TestQRCBrainInputEncoding:
    """Test cases for input encoding."""

    @pytest.fixture
    def brain(self) -> QRCBrain:
        """Create a test QRC brain."""
        config = QRCBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            shots=100,
        )
        return QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_preprocess(self, brain):
        """Test state preprocessing extracts correct features."""
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
        assert features[0] == 0.8  # gradient_strength
        assert -1.0 <= features[1] <= 1.0  # Normalized relative angle

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values defaults correctly."""
        params = BrainParams()
        features = brain.preprocess(params)

        assert len(features) == 2
        assert features[0] == 0.0  # gradient_strength defaults to 0

    def test_input_encoding_angle_calculation(self, brain):
        """Test RY angle calculation for input encoding."""
        features = np.array([0.5, 0.0], dtype=np.float32)

        # Build circuit with input encoding
        circuit = brain._encode_inputs(features)

        # First feature (0.5) should produce RY(0.5 * pi) = RY(1.57...)
        # Check that RY gate exists with correct angle
        ry_gates = [instr for instr in circuit.data if instr.operation.name == "ry"]
        assert len(ry_gates) > 0

        # First RY should encode the first feature
        first_ry = ry_gates[0]
        expected_angle = 0.5 * np.pi
        assert np.isclose(float(first_ry.operation.params[0]), expected_angle, atol=1e-6)


class TestQRCBrainSensoryModules:
    """Test cases for sensory modules feature extraction."""

    def test_brain_with_sensory_modules_input_dim(self):
        """Test that input_dim is computed from sensory modules."""
        config = QRCBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Each module contributes 2 features [strength, angle]
        assert brain.input_dim == 4
        assert brain.sensory_modules is not None
        assert len(brain.sensory_modules) == 2

    def test_preprocess_with_sensory_modules(self):
        """Test preprocessing with sensory modules uses extract_classical_features."""
        config = QRCBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)

        assert isinstance(features, np.ndarray)
        assert len(features) == 4  # 2 modules * 2 features each
        assert features.dtype == np.float32
        # Features should be in expected ranges
        assert 0.0 <= features[0] <= 1.0  # food strength [0, 1]
        assert -1.0 <= features[1] <= 1.0  # food angle [-1, 1]
        assert 0.0 <= features[2] <= 1.0  # predator strength [0, 1]
        assert -1.0 <= features[3] <= 1.0  # predator angle [-1, 1]

    def test_legacy_mode_when_no_sensory_modules(self):
        """Test that brain uses legacy preprocessing when sensory_modules is None."""
        config = QRCBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            shots=100,
            # No sensory_modules - legacy mode
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        assert brain.sensory_modules is None
        assert brain.input_dim == 2  # Legacy mode: gradient_strength + relative_angle

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        features = brain.preprocess(params)

        assert len(features) == 2


class TestQRCBrainReadoutNetwork:
    """Test cases for readout network architecture."""

    def test_mlp_readout_architecture(self):
        """Test MLP readout has correct layer dimensions."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,  # 2^3 = 8 input dim
            readout_hidden=16,
            readout_type="mlp",
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Test forward pass dimensions
        input_dim = 2**3  # 8
        test_input = torch.randn(input_dim)
        output = brain.readout(test_input)

        assert output.shape == (4,)  # num_actions

        # Check MLP structure (Linear -> ReLU -> Linear)
        assert isinstance(brain.readout, torch.nn.Sequential)
        assert len(brain.readout) == 3  # Linear, ReLU, Linear

    def test_linear_readout_architecture(self):
        """Test linear readout has correct layer dimensions."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,  # 2^3 = 8 input dim
            readout_type="linear",
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Test forward pass dimensions
        input_dim = 2**3  # 8
        test_input = torch.randn(input_dim)
        output = brain.readout(test_input)

        assert output.shape == (4,)  # num_actions

        # Linear readout is a single Linear layer
        assert isinstance(brain.readout, torch.nn.Linear)


class TestQRCBrainLearning:
    """Test cases for REINFORCE learning."""

    @pytest.fixture
    def brain(self) -> QRCBrain:
        """Create a test QRC brain."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden=8,
            shots=100,
            learning_rate=0.01,
        )
        return QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

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

        # Check that episode data is being tracked
        assert len(brain.episode_states) > 0
        assert len(brain.episode_actions) > 0
        assert len(brain.episode_log_probs) > 0

    def test_learn(self, brain):
        """Test learning with REINFORCE policy gradient."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain to generate actions
        for _ in range(5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=False)

        # Trigger update at episode end
        brain.learn(params, reward=1.0, episode_done=True)

        # Weights should remain finite
        for param in brain.readout.parameters():
            assert torch.all(torch.isfinite(param))

    def test_episode_buffer_management(self, brain):
        """Test episode buffer is managed correctly."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run multiple steps
        for _ in range(3):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should have data
        assert len(brain.episode_states) > 0
        assert len(brain.episode_rewards) > 0

        # Complete episode
        brain.learn(params, reward=1.0, episode_done=True)

        # Buffer should be cleared
        assert len(brain.episode_states) == 0
        assert len(brain.episode_rewards) == 0
        assert len(brain.episode_actions) == 0
        assert len(brain.episode_log_probs) == 0


class TestQRCBrainCopy:
    """Test cases for brain copying."""

    @pytest.fixture
    def brain(self) -> QRCBrain:
        """Create a test QRC brain."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden=8,
            shots=100,
        )
        return QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_copy_independence(self, brain):
        """Test that copy is independent - modifying copy doesn't affect original."""
        # Get original weights
        original_weights = {name: param.clone() for name, param in brain.readout.named_parameters()}

        # Create copy
        brain_copy = brain.copy()

        # Modify copy's readout weights
        with torch.no_grad():
            for param in brain_copy.readout.parameters():
                param.fill_(999.0)

        # Original should be unchanged
        for name, param in brain.readout.named_parameters():
            assert torch.allclose(param, original_weights[name]), (
                f"Original weights changed for {name}"
            )

    def test_copy_shares_reservoir(self, brain):
        """Test that copy shares the same reservoir circuit."""
        brain_copy = brain.copy()

        # Both should have the same reservoir circuit structure
        # (same gates, same angles - the reservoir is fixed)
        orig_circuit = brain._reservoir_circuit
        copy_circuit = brain_copy._reservoir_circuit

        assert orig_circuit.num_qubits == copy_circuit.num_qubits

        for instr1, instr2 in zip(orig_circuit.data, copy_circuit.data, strict=False):
            assert instr1.operation.name == instr2.operation.name
            if hasattr(instr1.operation, "params") and instr1.operation.params:
                for p1, p2 in zip(instr1.operation.params, instr2.operation.params, strict=False):
                    assert np.isclose(float(p1), float(p2))

    def test_copy_preserves_baseline(self, brain):
        """Test that copy preserves baseline value."""
        brain.baseline = 0.5
        brain_copy = brain.copy()

        assert brain_copy.baseline == brain.baseline


class TestQRCBrainIntegration:
    """Integration tests for QRC brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden=8,
            shots=100,
            learning_rate=0.01,
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Simulate multiple steps in an episode
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
            reward = rng.random() - 0.5  # Random reward
            brain.learn(params, reward, episode_done=(step == 9))

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 10

    def test_baseline_updates(self):
        """Test that baseline is updated during learning."""
        config = QRCBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            shots=100,
            baseline_alpha=0.1,
        )
        brain = QRCBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Initial baseline
        initial_baseline = brain.baseline

        # Run multiple episodes
        for _ in range(5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=True)

        # Baseline should have updated
        assert brain.baseline != initial_baseline
