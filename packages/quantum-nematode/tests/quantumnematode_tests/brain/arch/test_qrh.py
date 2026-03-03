"""Unit tests for the QRH (Quantum Reservoir Hybrid) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qrh import (
    SENSORY_QUBITS,
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
        assert config.num_sensory_qubits is None
        assert config.lr_warmup_episodes == 0
        assert config.lr_warmup_start is None
        assert config.lr_decay_episodes is None
        assert config.lr_decay_end is None

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
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(brain.num_qubits)
        brain._build_structured_reservoir(qc)

        gate_names = [instr.operation.name for instr in qc.data]
        assert "cz" in gate_names, "Structured reservoir should have CZ gates"

    def test_controlled_rotations_present(self, brain):
        """Structured reservoir should contain CRY/CRZ from chemical synapses."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(brain.num_qubits)
        brain._build_structured_reservoir(qc)

        gate_names = [instr.operation.name for instr in qc.data]
        assert "cry" in gate_names, "Should have CRY from chemical synapses"
        assert "crz" in gate_names, "Should have CRZ from chemical synapses"

    def test_per_qubit_encoding(self, brain):
        """Input encoding should only target sensory qubits, not all qubits."""
        features = np.array([0.8, -0.6], dtype=np.float32)
        zero_features = np.array([0.0, 0.0], dtype=np.float32)

        result_active = brain._get_reservoir_features(features)
        result_zero = brain._get_reservoir_features(zero_features)

        # Non-zero input should produce different features than zero input
        assert not np.allclose(result_active, result_zero, atol=1e-4), (
            "Per-qubit encoding should produce input-sensitive features"
        )

    def test_sensory_qubits_default(self):
        """Default sensory qubits should be [0, 1] for legacy 2-feature mode."""
        config = QRHBrainConfig(
            num_reservoir_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.sensory_qubits == [0, 1]
        # Module constant preserved for backward compatibility
        assert SENSORY_QUBITS == [0, 1]

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
    """Test cases for X/Y/Z-expectation and ZZ-correlation feature extraction."""

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
        """Feature dimension should be 3N + N(N-1)/2."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # 4 qubits: 3*4 + 4*3/2 = 12 + 6 = 18
        expected_dim = _compute_feature_dim(4)
        assert expected_dim == 18
        assert result.shape == (expected_dim,)

    def test_feature_dimension_8_qubits(self):
        """Feature dimension for 8 qubits should be 52."""
        assert _compute_feature_dim(8) == 52

    def test_xyz_expectations_range(self, brain):
        """X, Y, Z expectations should all be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        x_expectations = result[:n]
        y_expectations = result[n : 2 * n]
        z_expectations = result[2 * n : 3 * n]

        for name, exp in [("X", x_expectations), ("Y", y_expectations), ("Z", z_expectations)]:
            assert np.all(exp >= -1.0 - 1e-6), f"{name} expectations below -1"
            assert np.all(exp <= 1.0 + 1e-6), f"{name} expectations above 1"

    def test_zz_correlations_range(self, brain):
        """ZZ-correlations should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # ZZ correlations start after 3N expectations
        zz_correlations = result[3 * brain.num_qubits :]
        assert np.all(zz_correlations >= -1.0 - 1e-6)
        assert np.all(zz_correlations <= 1.0 + 1e-6)

    def test_xy_expectations_nontrivial(self, brain):
        """X and Y expectations should not all be zero for non-trivial states."""
        features = np.array([0.8, -0.6], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        x_expectations = result[:n]
        y_expectations = result[n : 2 * n]

        # At least some X or Y expectations should be non-zero
        assert not (
            np.allclose(x_expectations, 0, atol=1e-6) and np.allclose(y_expectations, 0, atol=1e-6)
        ), "X and Y expectations should capture phase information"

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
        """Buffer resets after mid-episode PPO update, which is deferred to next run_brain()."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Fill buffer mid-episode — update is deferred until next run_brain()
        for _step in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer is full and deferred update is pending
        assert brain._deferred_ppo_update is True
        assert len(brain.buffer) == brain.config.ppo_buffer_size

        # The next run_brain() executes the deferred update and resets the buffer
        brain.run_brain(params, top_only=True, top_randomize=False)
        assert brain._deferred_ppo_update is False
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

    def test_copy_preserves_episode_count(self, brain):
        """Copy should preserve episode counter."""
        brain._episode_count = 5
        brain_copy = brain.copy()
        assert brain_copy._episode_count == 5


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


class TestQRHEpisodeBoundaries:
    """Test cases for episode lifecycle and state management."""

    @pytest.fixture
    def brain(self) -> QRHBrain:
        """Create a test QRH brain for episode boundary tests."""
        config = QRHBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=16,
            ppo_minibatches=2,
        )
        return QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_prepare_episode_clears_pending_state(self, brain):
        """prepare_episode() should clear all pending state."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=False)

        assert brain._pending_state is not None

        brain.prepare_episode()

        assert brain._pending_state is None
        assert brain._pending_action is None
        assert brain._pending_log_prob is None
        assert brain._pending_value is None
        assert brain.last_value is None

    def test_post_process_episode_clears_pending_state(self, brain):
        """post_process_episode() should clear pending state and increment counter."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=False)

        initial_count = brain._episode_count
        brain.post_process_episode()

        assert brain._episode_count == initial_count + 1
        assert brain._pending_state is None
        assert brain._pending_action is None

    def test_multi_episode_no_cross_contamination(self, brain):
        """Running multiple episodes should not leak state between them."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        for _ep in range(3):
            brain.prepare_episode()

            for step in range(5):
                brain.run_brain(params, top_only=True, top_randomize=False)
                brain.learn(params, reward=0.1, episode_done=(step == 4))

            brain.post_process_episode()

        assert brain._episode_count == 3
        assert brain._pending_state is None

    def test_action_set_setter_validation(self, brain):
        """Setting action_set with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot set action_set of length"):
            brain.action_set = [Action.FORWARD, Action.LEFT]  # Only 2, brain expects 4

    def test_preprocess_with_none_values(self, brain):
        """Preprocessing with None gradient values should not crash."""
        params = BrainParams()
        features = brain.preprocess(params)
        assert len(features) == 2
        assert np.isfinite(features).all()


class TestQRHLRScheduling:
    """Test cases for learning rate scheduling."""

    def test_lr_warmup_config(self):
        """Test LR warmup config fields round-trip."""
        config = QRHBrainConfig(
            lr_warmup_episodes=30,
            lr_warmup_start=0.00005,
            lr_decay_episodes=100,
            lr_decay_end=0.00001,
        )
        assert config.lr_warmup_episodes == 30
        assert config.lr_warmup_start == 0.00005
        assert config.lr_decay_episodes == 100
        assert config.lr_decay_end == 0.00001

    def test_lr_no_scheduling_by_default(self):
        """LR should stay constant when warmup=0 and no decay."""
        config = QRHBrainConfig(num_reservoir_qubits=2, actor_lr=0.001)
        brain = QRHBrain(config=config, num_actions=4)
        assert not brain.lr_scheduling_enabled
        assert brain._get_current_lr() == 0.001

        # After several episodes, LR should remain unchanged
        for _ in range(10):
            brain.post_process_episode()
        assert brain._get_current_lr() == 0.001

    def test_lr_warmup_progression(self):
        """LR should ramp from warmup_start to base_lr over warmup_episodes."""
        config = QRHBrainConfig(
            num_reservoir_qubits=2,
            actor_lr=0.001,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
        )
        brain = QRHBrain(config=config, num_actions=4)
        assert brain.lr_scheduling_enabled

        # Episode 0: should be at warmup_start
        assert brain._get_current_lr() == pytest.approx(0.0001)

        # Episode 5 (midpoint): should be halfway between warmup_start and base_lr
        brain._episode_count = 5
        assert brain._get_current_lr() == pytest.approx(0.00055)

        # Episode 10 (end of warmup): should be at base_lr
        brain._episode_count = 10
        assert brain._get_current_lr() == pytest.approx(0.001)

        # Episode 20 (past warmup, no decay): should stay at base_lr
        brain._episode_count = 20
        assert brain._get_current_lr() == pytest.approx(0.001)

    def test_lr_warmup_then_decay(self):
        """LR should warmup then decay when both are configured."""
        config = QRHBrainConfig(
            num_reservoir_qubits=2,
            actor_lr=0.001,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
            lr_decay_episodes=20,
            lr_decay_end=0.0002,
        )
        brain = QRHBrain(config=config, num_actions=4)

        # Episode 0: warmup start
        assert brain._get_current_lr() == pytest.approx(0.0001)

        # Episode 10: at base_lr (end of warmup, start of decay)
        brain._episode_count = 10
        assert brain._get_current_lr() == pytest.approx(0.001)

        # Episode 20: halfway through decay
        brain._episode_count = 20
        expected = 0.001 + (0.0002 - 0.001) * 0.5  # 0.0006
        assert brain._get_current_lr() == pytest.approx(expected)

        # Episode 30: at decay end
        brain._episode_count = 30
        assert brain._get_current_lr() == pytest.approx(0.0002)

        # Episode 50: stays at decay end
        brain._episode_count = 50
        assert brain._get_current_lr() == pytest.approx(0.0002)

    def test_lr_scheduling_in_post_process(self):
        """post_process_episode() should update optimizer LR."""
        config = QRHBrainConfig(
            num_reservoir_qubits=2,
            actor_lr=0.001,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
        )
        brain = QRHBrain(config=config, num_actions=4)

        # Initial LR should be warmup_start (episode 0)
        # Note: optimizer starts at base_lr, first post_process sets to ep 1 LR
        brain.post_process_episode()  # episode_count becomes 1
        expected_lr = 0.0001 + (0.001 - 0.0001) * (1 / 10)  # 0.00019
        actual_lr = brain.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_lr)

    def test_lr_copy_preserves_scheduling(self):
        """copy() should preserve LR scheduling config and episode count."""
        config = QRHBrainConfig(
            num_reservoir_qubits=2,
            actor_lr=0.001,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
        )
        brain = QRHBrain(config=config, num_actions=4)

        # Advance a few episodes
        for _ in range(5):
            brain.post_process_episode()

        brain_copy = brain.copy()
        assert brain_copy.lr_scheduling_enabled
        assert brain_copy.lr_warmup_episodes == 10
        assert brain_copy.lr_warmup_start == 0.0001
        assert brain_copy._episode_count == 5


class TestQRHSensoryQubits:
    """Test cases for configurable sensory qubit count."""

    def test_sensory_qubits_auto_from_input_dim(self):
        """With 4 input features, should auto-compute 4 sensory qubits."""
        config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.input_dim == 4
        assert brain.sensory_qubits == [0, 1, 2, 3]

    def test_sensory_qubits_explicit_override(self):
        """Explicit num_sensory_qubits should override auto-computation."""
        config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            num_sensory_qubits=3,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.sensory_qubits == [0, 1, 2]

    def test_sensory_qubits_validation_exceeds_total(self):
        """num_sensory_qubits > num_reservoir_qubits should raise ValueError."""
        with pytest.raises(
            ValueError,
            match=r"num_sensory_qubits.*must be.*<= num_reservoir_qubits",
        ):
            QRHBrainConfig(num_reservoir_qubits=4, num_sensory_qubits=5)

    def test_sensory_qubits_validation_zero(self):
        """num_sensory_qubits < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_sensory_qubits must be >= 1"):
            QRHBrainConfig(num_reservoir_qubits=4, num_sensory_qubits=0)

    def test_sensory_qubits_capped_at_num_qubits(self):
        """Auto-computed sensory qubits should not exceed total qubit count."""
        config = QRHBrainConfig(
            num_reservoir_qubits=3,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        # input_dim=4 but only 3 qubits, so capped at 3
        assert brain.sensory_qubits == [0, 1, 2]

    def test_four_features_no_wrapping(self):
        """With 4 sensory qubits, food-only and predator-only inputs differ."""
        config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=2,
            readout_hidden_dim=8,
            readout_num_layers=1,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Food strong, predator absent vs food absent, predator strong
        food_features = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pred_features = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        food_result = brain._get_reservoir_features(food_features)
        pred_result = brain._get_reservoir_features(pred_features)

        assert not np.allclose(food_result, pred_result, atol=1e-4), (
            "Food-only and predator-only inputs should produce distinct reservoir "
            "features when sensory qubits are separated"
        )

    def test_copy_preserves_sensory_qubits(self):
        """Copy should preserve sensory qubit configuration."""
        config = QRHBrainConfig(
            num_reservoir_qubits=8,
            reservoir_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            num_sensory_qubits=4,
        )
        brain = QRHBrain(config=config, num_actions=4, device=DeviceType.CPU)
        brain_copy = brain.copy()
        assert brain_copy.sensory_qubits == [0, 1, 2, 3]
        assert brain_copy._get_current_lr() == pytest.approx(brain._get_current_lr())
