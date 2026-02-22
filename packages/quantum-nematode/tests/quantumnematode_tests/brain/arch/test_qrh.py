"""Unit tests for the QRH (Quantum Reservoir Hybrid) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qrh import (
    QRHBrain,
    QRHBrainConfig,
    _compute_feature_dim,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestQRHBrainConfig:
    """Test cases for QRH brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QRHBrainConfig()

        assert config.num_reservoir_qubits == 8
        assert config.reservoir_depth == 3
        assert config.reservoir_seed == 42
        assert config.shots == 1024
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
        assert config.use_random_topology is False
        assert config.sensory_modules is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            reservoir_seed=123,
            readout_hidden_dim=32,
            readout_num_layers=1,
            actor_lr=0.001,
            critic_lr=0.0005,
            gamma=0.95,
            ppo_buffer_size=256,
            use_random_topology=True,
        )

        assert config.num_reservoir_qubits == 4
        assert config.reservoir_depth == 2
        assert config.reservoir_seed == 123
        assert config.readout_hidden_dim == 32
        assert config.readout_num_layers == 1
        assert config.actor_lr == 0.001
        assert config.critic_lr == 0.0005
        assert config.gamma == 0.95
        assert config.ppo_buffer_size == 256
        assert config.use_random_topology is True

    def test_validation_num_reservoir_qubits(self):
        """Validate num_reservoir_qubits >= 2."""
        with pytest.raises(ValueError, match="num_reservoir_qubits must be >= 2"):
            QRHBrainConfig(num_reservoir_qubits=1)

    def test_validation_reservoir_depth(self):
        """Validate reservoir_depth >= 1."""
        with pytest.raises(ValueError, match="reservoir_depth must be >= 1"):
            QRHBrainConfig(reservoir_depth=0)

    def test_validation_shots(self):
        """Validate shots >= 100."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QRHBrainConfig(shots=50)

    def test_validation_readout_hidden_dim(self):
        """Validate readout_hidden_dim >= 1."""
        with pytest.raises(ValueError, match="readout_hidden_dim must be >= 1"):
            QRHBrainConfig(readout_hidden_dim=0)

    def test_config_with_sensory_modules(self):
        """Test configuration with sensory modules."""
        config = QRHBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2


class TestQRHReservoirCircuit:
    """Test cases for reservoir circuit construction."""

    @pytest.fixture
    def small_config(self) -> QRHBrainConfig:
        """Create a small test configuration for faster tests."""
        return QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            reservoir_seed=42,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
        )

    @pytest.fixture
    def brain(self, small_config) -> QRHBrain:
        """Create a test QRH brain."""
        return QRHBrain(config=small_config, num_actions=4, device=DeviceType.CPU)

    def test_cz_gates_present(self, brain):
        """Structured reservoir should contain CZ gates from gap junctions."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        from qiskit import QuantumCircuit

        # Build circuit manually to inspect gates
        qc = QuantumCircuit(brain.num_qubits)
        for q in range(brain.num_qubits):
            qc.h(q)
        for i, feature in enumerate(features):
            angle = float(feature) * np.pi
            for q in range(brain.num_qubits):
                if i % 2 == 0:
                    qc.ry(angle, q)
                else:
                    qc.rz(angle, q)
        brain._build_structured_reservoir(qc)

        gate_names = [instr.operation.name for instr in qc.data]
        assert "cz" in gate_names, "Structured reservoir should have CZ gates"

    def test_fixed_rotations_present(self, brain):
        """Structured reservoir should contain fixed RY/RZ from chemical synapses."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(brain.num_qubits)
        brain._build_structured_reservoir(qc)

        gate_names = [instr.operation.name for instr in qc.data]
        assert "ry" in gate_names, "Should have RY from chemical synapses"
        assert "rz" in gate_names, "Should have RZ from chemical synapses"

    def test_seed_reproducibility(self):
        """Same seed should produce identical feature vectors."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=2,
            reservoir_seed=42,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain1 = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        brain2 = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        features = np.array([0.5, 0.3], dtype=np.float32)
        result1 = brain1._get_reservoir_features(features)
        result2 = brain2._get_reservoir_features(features)

        np.testing.assert_array_equal(result1, result2)

    def test_structured_vs_random_topology(self):
        """Structured and random topologies should produce different features."""
        structured_config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=2,
            reservoir_seed=42,
            readout_hidden_dim=8,
            readout_num_layers=1,
            use_random_topology=False,
        )
        random_config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=2,
            reservoir_seed=42,
            readout_hidden_dim=8,
            readout_num_layers=1,
            use_random_topology=True,
        )
        structured_brain = QRHBrain(
            config=structured_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        random_brain = QRHBrain(
            config=random_config,
            num_actions=4,
            device=DeviceType.CPU,
        )

        features = np.array([0.5, 0.3], dtype=np.float32)
        structured_out = structured_brain._get_reservoir_features(features)
        random_out = random_brain._get_reservoir_features(features)

        # Should produce different features (extremely unlikely to match)
        assert not np.allclose(structured_out, random_out, atol=1e-6)


class TestQRHFeatureExtraction:
    """Test cases for Z-expectation and ZZ-correlation feature extraction."""

    @pytest.fixture
    def brain(self) -> QRHBrain:
        """Create a test QRH brain with 4 qubits."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_feature_dimension(self, brain):
        """Feature dimension should be N + N(N-1)/2."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # 4 qubits: 4 + 4*3/2 = 4 + 6 = 10
        expected_dim = _compute_feature_dim(4)
        assert expected_dim == 10
        assert result.shape == (expected_dim,)

    def test_feature_dimension_8_qubits(self):
        """Feature dimension for 8 qubits should be 36."""
        assert _compute_feature_dim(8) == 36

    def test_z_expectations_range(self, brain):
        """Z-expectations should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # First N values are Z-expectations
        z_expectations = result[: brain.num_qubits]
        assert np.all(z_expectations >= -1.0 - 1e-8)
        assert np.all(z_expectations <= 1.0 + 1e-8)

    def test_zz_correlations_range(self, brain):
        """ZZ-correlations should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # Values after first N are ZZ-correlations
        zz_correlations = result[brain.num_qubits :]
        assert np.all(zz_correlations >= -1.0 - 1e-8)
        assert np.all(zz_correlations <= 1.0 + 1e-8)

    def test_input_sensitivity(self):
        """Different inputs should produce different features (non-degeneracy)."""
        # Use deeper reservoir for better discriminative power
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=3,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        features_a = np.array([0.1, 0.2], dtype=np.float32)
        features_b = np.array([0.9, -0.8], dtype=np.float32)

        result_a = brain._get_reservoir_features(features_a)
        result_b = brain._get_reservoir_features(features_b)

        assert not np.allclose(result_a, result_b, atol=1e-4)

    def test_determinism(self, brain):
        """Same input should always produce same features (statevector is exact)."""
        features = np.array([0.5, 0.3], dtype=np.float32)

        result1 = brain._get_reservoir_features(features)
        result2 = brain._get_reservoir_features(features)

        np.testing.assert_array_equal(result1, result2)


class TestQRHBrainReadout:
    """Test cases for actor and critic readout networks."""

    @pytest.fixture
    def brain(self) -> QRHBrain:
        """Create a test QRH brain."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            readout_hidden_dim=16,
            readout_num_layers=2,
        )
        return QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_actor_output_shape(self, brain):
        """Actor should produce num_actions logits."""
        feature_dim = _compute_feature_dim(4)
        x = torch.randn(feature_dim)
        output = brain.actor(x)
        assert output.shape == (4,)

    def test_critic_output_shape(self, brain):
        """Critic should produce a single value scalar."""
        feature_dim = _compute_feature_dim(4)
        x = torch.randn(feature_dim)
        output = brain.critic(x)
        assert output.shape == (1,)

    def test_actor_is_sequential(self, brain):
        """Actor should be an nn.Sequential (MLP readout)."""
        assert isinstance(brain.actor, torch.nn.Sequential)

    def test_critic_is_sequential(self, brain):
        """Critic should be an nn.Sequential (MLP readout)."""
        assert isinstance(brain.critic, torch.nn.Sequential)


class TestQRHBrainLearning:
    """Test cases for PPO learning."""

    @pytest.fixture
    def brain(self) -> QRHBrain:
        """Create a test QRH brain with small buffer for faster tests."""
        config = QRHBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            ppo_minibatches=2,
            ppo_epochs=2,
        )
        return QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_run_brain_returns_valid_action(self, brain):
        """run_brain should return valid ActionData."""
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

    def test_ppo_update_changes_weights(self, brain):
        """PPO update should modify readout weights."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        # Get initial weights
        initial_actor_weights = {
            name: param.clone() for name, param in brain.actor.named_parameters()
        }

        # Fill buffer and trigger update
        for step in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(step == brain.config.ppo_buffer_size - 1))

        # Weights should have changed
        weights_changed = False
        for name, param in brain.actor.named_parameters():
            if not torch.allclose(param, initial_actor_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "PPO update should change actor weights"

    def test_buffer_management(self, brain):
        """Buffer should reset after PPO update."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer
        for _step in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should be empty after update
        assert len(brain.buffer) == 0

    def test_full_episode_workflow(self):
        """Test a complete episode with QRH brain."""
        config = QRHBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=16,
            ppo_minibatches=2,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        rng = np.random.default_rng(42)

        brain.prepare_episode()

        for step in range(10):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            assert len(actions) == 1

            reward = rng.random() - 0.5
            brain.learn(params, reward, episode_done=(step == 9))

        brain.post_process_episode()

        assert len(brain.history_data.rewards) == 10
        assert len(brain.history_data.actions) == 10


class TestQRHBrainCopy:
    """Test cases for brain copying."""

    @pytest.fixture
    def brain(self) -> QRHBrain:
        """Create a test QRH brain."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_copy_independence(self, brain):
        """Modifying copy should not affect original."""
        original_weights = {name: param.clone() for name, param in brain.actor.named_parameters()}

        brain_copy = brain.copy()

        # Modify copy's weights
        with torch.no_grad():
            for param in brain_copy.actor.parameters():
                param.fill_(999.0)

        # Original should be unchanged
        for name, param in brain.actor.named_parameters():
            assert torch.allclose(param, original_weights[name]), (
                f"Original weights changed for {name}"
            )

    def test_copy_shares_reservoir_topology(self, brain):
        """Copy should produce identical reservoir features for same input."""
        brain_copy = brain.copy()

        features = np.array([0.5, 0.3], dtype=np.float32)
        orig_result = brain._get_reservoir_features(features)
        copy_result = brain_copy._get_reservoir_features(features)

        np.testing.assert_array_equal(orig_result, copy_result)

    def test_copy_preserves_topology_setting(self, brain):
        """Copy should preserve use_random_topology setting."""
        brain_copy = brain.copy()
        assert brain_copy.use_random_topology == brain.use_random_topology


class TestQRHBrainSensoryModules:
    """Test cases for sensory module integration."""

    def test_unified_mode_dimensions(self):
        """Unified mode should compute input_dim from sensory modules."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Each module contributes 2 features [strength, angle]
        assert brain.input_dim == 4
        assert brain.sensory_modules is not None

    def test_legacy_fallback(self):
        """Legacy mode (no sensory_modules) should use 2 features."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        assert brain.sensory_modules is None
        assert brain.input_dim == 2

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        features = brain.preprocess(params)
        assert len(features) == 2

    def test_preprocess_with_sensory_modules(self):
        """Preprocessing with sensory modules should use extract_classical_features."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)
        assert isinstance(features, np.ndarray)
        assert len(features) == 4
        assert features.dtype == np.float32
