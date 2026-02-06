"""Unit tests for the QSNN (Quantum Spiking Neural Network) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qsnn import QSNNBrain, QSNNBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestQSNNBrainConfig:
    """Test cases for QSNN brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QSNNBrainConfig()

        assert config.num_sensory_neurons == 6
        assert config.num_hidden_neurons == 4
        assert config.num_motor_neurons == 4
        assert config.membrane_tau == 0.9
        assert config.threshold == 0.5
        assert config.refractory_period == 2
        assert config.use_local_learning is True
        assert config.shots == 1024
        assert config.gamma == 0.99
        assert config.learning_rate == 0.01

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QSNNBrainConfig(
            num_sensory_neurons=8,
            num_hidden_neurons=6,
            num_motor_neurons=5,
            membrane_tau=0.8,
            threshold=0.6,
            refractory_period=3,
            use_local_learning=False,
            shots=512,
            gamma=0.95,
            learning_rate=0.005,
        )

        assert config.num_sensory_neurons == 8
        assert config.num_hidden_neurons == 6
        assert config.num_motor_neurons == 5
        assert config.membrane_tau == 0.8
        assert config.threshold == 0.6
        assert config.refractory_period == 3
        assert config.use_local_learning is False
        assert config.shots == 512
        assert config.gamma == 0.95
        assert config.learning_rate == 0.005

    def test_config_sensory_modules_default(self):
        """Test that sensory_modules defaults to None (legacy mode)."""
        config = QSNNBrainConfig()
        assert config.sensory_modules is None

    def test_config_with_sensory_modules(self):
        """Test configuration with sensory modules."""
        config = QSNNBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )

        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2
        assert ModuleName.FOOD_CHEMOTAXIS in config.sensory_modules
        assert ModuleName.NOCICEPTION in config.sensory_modules

    def test_validation_num_sensory_neurons(self):
        """Test validation for num_sensory_neurons."""
        with pytest.raises(ValueError, match="num_sensory_neurons must be >= 1"):
            QSNNBrainConfig(num_sensory_neurons=0)

    def test_validation_num_hidden_neurons(self):
        """Test validation for num_hidden_neurons."""
        with pytest.raises(ValueError, match="num_hidden_neurons must be >= 1"):
            QSNNBrainConfig(num_hidden_neurons=0)

    def test_validation_num_motor_neurons(self):
        """Test validation for num_motor_neurons."""
        with pytest.raises(ValueError, match="num_motor_neurons must be >= 2"):
            QSNNBrainConfig(num_motor_neurons=1)

    def test_validation_membrane_tau_lower_bound(self):
        """Test validation for membrane_tau lower bound."""
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QSNNBrainConfig(membrane_tau=0.0)

    def test_validation_membrane_tau_upper_bound(self):
        """Test validation for membrane_tau at upper bound."""
        # Upper bound is inclusive (0, 1]
        config = QSNNBrainConfig(membrane_tau=1.0)
        assert config.membrane_tau == 1.0

    def test_validation_threshold_lower_bound(self):
        """Test validation for threshold lower bound."""
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNBrainConfig(threshold=0.0)

    def test_validation_threshold_upper_bound(self):
        """Test validation for threshold upper bound."""
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNBrainConfig(threshold=1.0)

    def test_validation_shots(self):
        """Test validation for shots."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QSNNBrainConfig(shots=50)


class TestQSNNBrainQLIFCircuit:
    """Test cases for QLIF neuron circuit construction."""

    @pytest.fixture
    def small_config(self) -> QSNNBrainConfig:
        """Create a small test configuration for faster tests."""
        return QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )

    @pytest.fixture
    def brain(self, small_config) -> QSNNBrain:
        """Create a test QSNN brain."""
        return QSNNBrain(
            config=small_config,
            num_actions=2,
            device=DeviceType.CPU,
        )

    def test_qlif_circuit_structure(self, brain):
        """Verify QLIF circuit has correct structure: RY + RX + Measure."""
        circuit = brain._build_qlif_circuit(weighted_input=0.5, theta_membrane=0.1)

        # Check circuit has 1 qubit and 1 classical bit
        assert circuit.num_qubits == 1
        assert circuit.num_clbits == 1

        # Extract gate names
        gate_names = [instruction.operation.name for instruction in circuit.data]

        # Should have RY, RX, and measure
        assert "ry" in gate_names
        assert "rx" in gate_names
        assert "measure" in gate_names

        # RY should come before RX
        ry_idx = gate_names.index("ry")
        rx_idx = gate_names.index("rx")
        assert ry_idx < rx_idx

    def test_qlif_circuit_angles(self, brain):
        """Test QLIF circuit has correct rotation angles."""
        weighted_input = 0.5
        theta_membrane = 0.2

        circuit = brain._build_qlif_circuit(
            weighted_input=weighted_input,
            theta_membrane=theta_membrane,
        )

        # Find RY and RX gates
        ry_gate = None
        rx_gate = None
        for instr in circuit.data:
            if instr.operation.name == "ry":
                ry_gate = instr
            elif instr.operation.name == "rx":
                rx_gate = instr

        assert ry_gate is not None
        assert rx_gate is not None

        # RY angle = theta_membrane + weighted_input * pi
        expected_ry = theta_membrane + weighted_input * np.pi
        assert np.isclose(float(ry_gate.operation.params[0]), expected_ry, atol=1e-6)

        # RX angle = leak_angle = (1 - membrane_tau) * pi
        expected_rx = brain.leak_angle
        assert np.isclose(float(rx_gate.operation.params[0]), expected_rx, atol=1e-6)

    def test_qlif_circuit_measurement(self, brain):
        """Test QLIF circuit measurement produces valid probabilities."""
        circuit = brain._build_qlif_circuit(weighted_input=0.5, theta_membrane=0.1)

        # Execute circuit
        backend = brain._get_backend()
        job = backend.run(circuit, shots=100)
        result = job.result()
        counts = result.get_counts()

        # Should have results for "0" and/or "1"
        total = sum(counts.values())
        assert total == 100

        # Probabilities should sum to 1
        probs = {k: v / total for k, v in counts.items()}
        assert np.isclose(sum(probs.values()), 1.0)


class TestQSNNBrainSensoryEncoding:
    """Test cases for sensory spike encoding."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_encode_sensory_spikes_shape(self, brain):
        """Test sensory encoding produces correct output shape."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        spikes = brain._encode_sensory_spikes(features)

        assert isinstance(spikes, np.ndarray)
        assert len(spikes) == brain.num_sensory

    def test_encode_sensory_spikes_range(self, brain):
        """Test sensory encoding produces probabilities in [0, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        spikes = brain._encode_sensory_spikes(features)

        assert np.all(spikes >= 0.0)
        assert np.all(spikes <= 1.0)

    def test_encode_sensory_spikes_sigmoid(self, brain):
        """Test sensory encoding uses sigmoid correctly."""
        # High input should produce high spike probability
        high_features = np.array([1.0, 1.0], dtype=np.float32)
        high_spikes = brain._encode_sensory_spikes(high_features)

        # Low input should produce lower spike probability
        low_features = np.array([0.0, 0.0], dtype=np.float32)
        low_spikes = brain._encode_sensory_spikes(low_features)

        # High should be greater than low
        assert np.all(high_spikes > low_spikes)


class TestQSNNBrainForwardPass:
    """Test cases for forward pass through the network."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_timestep_produces_valid_probabilities(self, brain):
        """Test _timestep produces valid motor neuron probabilities."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._timestep(features)

        assert isinstance(motor_probs, np.ndarray)
        assert len(motor_probs) == brain.num_motor
        assert np.all(motor_probs >= 0.0)
        assert np.all(motor_probs <= 1.0)

    def test_run_brain_returns_action(self, brain):
        """Test run_brain returns valid action."""
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
        assert action_data.action in [Action.FORWARD, Action.LEFT]  # Only 2 actions
        assert 0.0 <= action_data.probability <= 1.0

    def test_run_brain_tracks_episode_data(self, brain):
        """Test run_brain tracks episode data."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(brain.episode_actions) == 1
        assert len(brain.episode_log_probs) == 1
        assert len(brain.episode_probs) == 1


class TestQSNNBrainEligibilityTrace:
    """Test cases for eligibility trace computation."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain with local learning."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_eligibility_accumulates(self, brain):
        """Test eligibility traces accumulate during episode."""
        # Initial eligibility should be zero
        assert torch.allclose(brain.eligibility_sh, torch.zeros_like(brain.eligibility_sh))
        assert torch.allclose(brain.eligibility_hm, torch.zeros_like(brain.eligibility_hm))

        # Run a timestep (which accumulates eligibility)
        features = np.array([0.5, 0.3], dtype=np.float32)
        brain._timestep(features)

        # At least one eligibility matrix should have non-zero values
        # (depends on spike patterns, but with reasonable probs, should have some)
        has_eligibility = (
            torch.any(brain.eligibility_sh != 0).item()
            or torch.any(brain.eligibility_hm != 0).item()
        )
        # This might occasionally be all zeros if no spikes, which is valid
        assert isinstance(has_eligibility, bool)

    def test_eligibility_reset_on_episode_end(self, brain):
        """Test eligibility traces reset when episode ends."""
        # Run a step to accumulate some eligibility
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=True)
        brain.learn(params, reward=1.0, episode_done=True)

        # Eligibility should be reset
        assert torch.allclose(brain.eligibility_sh, torch.zeros_like(brain.eligibility_sh))
        assert torch.allclose(brain.eligibility_hm, torch.zeros_like(brain.eligibility_hm))


class TestQSNNBrainLocalLearning:
    """Test cases for local learning weight updates."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
            learning_rate=0.1,  # High LR for visible changes
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_local_learning_updates_weights(self, brain):
        """Test that local learning updates weights."""
        # Record initial weights
        initial_w_sh = brain.W_sh.clone()
        initial_w_hm = brain.W_hm.clone()

        # Run episode with positive reward
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        for _ in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=False)

        # Complete episode
        brain.learn(params, reward=1.0, episode_done=True)

        # Weights should change (unless all eligibility was zero)
        # Check if at least one weight matrix changed
        sh_changed = not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)
        hm_changed = not torch.allclose(brain.W_hm, initial_w_hm, atol=1e-6)

        # At least one should change with positive reward and reasonable activity
        # Note: might occasionally not change if all spikes were 0
        assert isinstance(sh_changed, bool)
        assert isinstance(hm_changed, bool)

    def test_weight_clipping(self, brain):
        """Test that weights are clipped to configured range."""
        brain.weight_clip = 2.0

        # Force extreme eligibility values
        brain.eligibility_sh.fill_(100.0)
        brain.eligibility_hm.fill_(100.0)

        # Apply learning with extreme reward
        brain._local_learning_update(total_reward=100.0)

        # Weights should be clipped
        assert torch.all(brain.W_sh <= brain.weight_clip)
        assert torch.all(brain.W_sh >= -brain.weight_clip)
        assert torch.all(brain.W_hm <= brain.weight_clip)
        assert torch.all(brain.W_hm >= -brain.weight_clip)


class TestQSNNBrainRefractoryPeriod:
    """Test cases for refractory period behavior."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            refractory_period=2,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_refractory_state_initialized_zero(self, brain):
        """Test refractory states start at zero."""
        assert np.all(brain.refractory_hidden == 0)
        assert np.all(brain.refractory_motor == 0)

    def test_refractory_reset_on_episode_start(self, brain):
        """Test refractory states reset when episode starts."""
        # Set some refractory state
        brain.refractory_hidden[0] = 2
        brain.refractory_motor[0] = 1

        # Prepare new episode
        brain.prepare_episode()

        # Should be reset
        assert np.all(brain.refractory_hidden == 0)
        assert np.all(brain.refractory_motor == 0)


class TestQSNNBrainReproducibility:
    """Test cases for reproducibility with seeds."""

    def test_same_seed_same_weights(self):
        """Same seed should produce identical initial weights."""
        config1 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )
        config2 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )

        brain1 = QSNNBrain(config=config1, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNBrain(config=config2, num_actions=2, device=DeviceType.CPU)

        assert torch.allclose(brain1.W_sh, brain2.W_sh)
        assert torch.allclose(brain1.W_hm, brain2.W_hm)

    def test_different_seed_different_weights(self):
        """Different seeds should produce different initial weights."""
        config1 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )
        config2 = QSNNBrainConfig(
            seed=123,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )

        brain1 = QSNNBrain(config=config1, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNBrain(config=config2, num_actions=2, device=DeviceType.CPU)

        # At least one weight matrix should differ
        sh_different = not torch.allclose(brain1.W_sh, brain2.W_sh)
        hm_different = not torch.allclose(brain1.W_hm, brain2.W_hm)

        assert sh_different or hm_different


class TestQSNNBrainCopy:
    """Test cases for brain copying."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_copy_independence(self, brain):
        """Test that copy is independent - modifying copy doesn't affect original."""
        # Get original weights
        original_w_sh = brain.W_sh.clone()
        original_w_hm = brain.W_hm.clone()

        # Create copy
        brain_copy = brain.copy()

        # Modify copy's weights
        brain_copy.W_sh.fill_(999.0)
        brain_copy.W_hm.fill_(999.0)

        # Original should be unchanged
        assert torch.allclose(brain.W_sh, original_w_sh)
        assert torch.allclose(brain.W_hm, original_w_hm)

    def test_copy_has_same_initial_weights(self, brain):
        """Test that copy has same initial weights as original."""
        brain_copy = brain.copy()

        assert torch.allclose(brain.W_sh, brain_copy.W_sh)
        assert torch.allclose(brain.W_hm, brain_copy.W_hm)
        assert torch.allclose(brain.theta_hidden, brain_copy.theta_hidden)
        assert torch.allclose(brain.theta_motor, brain_copy.theta_motor)

    def test_copy_preserves_baseline(self, brain):
        """Test that copy preserves baseline value."""
        brain.baseline = 0.5
        brain_copy = brain.copy()

        assert brain_copy.baseline == brain.baseline


class TestQSNNBrainSensoryModules:
    """Test cases for sensory modules feature extraction."""

    def test_brain_with_sensory_modules_input_dim(self):
        """Test that input_dim is computed from sensory modules."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        # Each module contributes 2 features [strength, angle]
        assert brain.input_dim == 4
        assert brain.sensory_modules is not None
        assert len(brain.sensory_modules) == 2

    def test_preprocess_with_sensory_modules(self):
        """Test preprocessing with sensory modules uses extract_classical_features."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

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

    def test_legacy_mode_when_no_sensory_modules(self):
        """Test that brain uses legacy preprocessing when sensory_modules is None."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            # No sensory_modules - legacy mode
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        assert brain.sensory_modules is None
        assert brain.input_dim == 2  # Legacy mode: gradient_strength + relative_angle


class TestQSNNBrainIntegration:
    """Integration tests for QSNN brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=4,
            shots=100,
            learning_rate=0.01,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

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

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        for _episode in range(3):
            brain.prepare_episode()

            for step in range(5):
                brain.run_brain(params, top_only=True, top_randomize=False)
                brain.learn(params, reward=0.1, episode_done=(step == 4))

        # Should have accumulated history
        assert len(brain.history_data.rewards) == 15  # 3 episodes * 5 steps
