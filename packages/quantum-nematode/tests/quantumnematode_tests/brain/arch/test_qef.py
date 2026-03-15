"""Unit tests for the QEF (Quantum Entangled Features) brain architecture."""

from typing import Literal

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qef import (
    MODALITY_PAIRED_CZ,
    QEFBrain,
    QEFBrainConfig,
    _compute_feature_dim,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestQEFBrainConfig:
    """Test cases for QEF brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QEFBrainConfig()

        assert config.num_qubits == 8
        assert config.circuit_depth == 2
        assert config.circuit_seed == 42
        assert config.entanglement_topology == "modality_paired"
        assert config.entanglement_enabled is True
        assert config.trainable_entanglement is False
        assert config.readout_hidden_dim == 64
        assert config.readout_num_layers == 2
        assert config.sensory_modules is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=3,
            circuit_seed=123,
            entanglement_topology="ring",
            entanglement_enabled=False,
            readout_hidden_dim=32,
            readout_num_layers=1,
            actor_lr=0.001,
        )

        assert config.num_qubits == 4
        assert config.circuit_depth == 3
        assert config.circuit_seed == 123
        assert config.entanglement_topology == "ring"
        assert config.entanglement_enabled is False
        assert config.readout_hidden_dim == 32

    def test_validation_num_qubits(self):
        """Validate num_qubits >= 2."""
        with pytest.raises(ValueError, match="num_qubits must be >= 2"):
            QEFBrainConfig(num_qubits=1)

    def test_validation_circuit_depth(self):
        """Validate circuit_depth >= 1."""
        with pytest.raises(ValueError, match="circuit_depth must be >= 1"):
            QEFBrainConfig(circuit_depth=0)

    def test_trainable_entanglement_raises(self):
        """Trainable entanglement should raise NotImplementedError."""
        config = QEFBrainConfig(trainable_entanglement=True)
        with pytest.raises(NotImplementedError, match="Trainable entanglement"):
            QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_config_with_sensory_modules(self):
        """Test configuration with sensory modules."""
        config = QEFBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2


class TestQEFFeatureExtraction:
    """Test cases for Z + ZZ + cos/sin feature extraction."""

    @pytest.fixture
    def brain(self) -> QEFBrain:
        """Create a test QEF brain with 4 qubits."""
        config = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_feature_dimension_4_qubits(self, brain):
        """Feature dimension should be 3N + N(N-1)/2 for 4 qubits."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # 4 qubits: 3*4 + 4*3/2 = 12 + 6 = 18
        expected_dim = _compute_feature_dim(4)
        assert expected_dim == 18
        assert result.shape == (expected_dim,)

    def test_feature_dimension_8_qubits(self):
        """Feature dimension for 8 qubits should be 52."""
        assert _compute_feature_dim(8) == 52

    def test_z_expectations_range(self, brain):
        """Z expectations should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        z_expectations = result[:n]
        assert np.all(z_expectations >= -1.0 - 1e-6), "Z expectations below -1"
        assert np.all(z_expectations <= 1.0 + 1e-6), "Z expectations above 1"

    def test_zz_correlations_range(self, brain):
        """ZZ-correlations should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        num_zz = n * (n - 1) // 2
        zz_correlations = result[n : n + num_zz]
        assert np.all(zz_correlations >= -1.0 - 1e-6)
        assert np.all(zz_correlations <= 1.0 + 1e-6)

    def test_cos_sin_range(self, brain):
        """cos/sin features should be in [-1, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        num_zz = n * (n - 1) // 2
        cos_sin_features = result[n + num_zz :]
        assert len(cos_sin_features) == 2 * n
        assert np.all(cos_sin_features >= -1.0 - 1e-6)
        assert np.all(cos_sin_features <= 1.0 + 1e-6)

    def test_feature_vector_ordering(self, brain):
        """Feature vector should be ordered: [z, zz, cos_z, sin_z]."""
        features = np.array([0.8, -0.6], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        n = brain.num_qubits
        num_zz = n * (n - 1) // 2

        # Verify total length matches expected ordering
        assert len(result) == n + num_zz + 2 * n

        # Extract segments
        z_part = result[:n]
        _ = result[n : n + num_zz]  # zz_part — validates segment exists
        cos_part = result[n + num_zz : n + num_zz + n]
        sin_part = result[n + num_zz + n :]

        # cos/sin should be consistent with z expectations
        np.testing.assert_allclose(cos_part, np.cos(z_part), atol=1e-6)
        np.testing.assert_allclose(sin_part, np.sin(z_part), atol=1e-6)

    def test_determinism(self, brain):
        """Same input should always produce identical features."""
        features = np.array([0.5, 0.3], dtype=np.float32)

        result1 = brain._get_reservoir_features(features)
        result2 = brain._get_reservoir_features(features)

        np.testing.assert_array_equal(result1, result2)

    def test_input_sensitivity(self):
        """Different inputs should produce different features."""
        config = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=2,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

        features_a = np.array([0.1, 0.2], dtype=np.float32)
        features_b = np.array([0.9, -0.8], dtype=np.float32)

        result_a = brain._get_reservoir_features(features_a)
        result_b = brain._get_reservoir_features(features_b)

        assert not np.allclose(result_a, result_b, atol=1e-4)

    def test_feature_wrapping_more_features_than_qubits(self):
        """Features should wrap when input_dim > num_qubits."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # 5 features on 3 qubits: features 3,4 wrap to qubits 0,1
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        expected_dim = _compute_feature_dim(3)
        assert result.shape == (expected_dim,)

        # Should differ from using only the first 3 features (no wrapping)
        features_truncated = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result_truncated = brain._get_reservoir_features(features_truncated)
        assert not np.allclose(result, result_truncated, atol=1e-6)

    def test_minimum_qubits(self):
        """Minimum qubit count (2) should work correctly."""
        config = QEFBrainConfig(
            num_qubits=2,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

        features = np.array([0.5], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        # 2 qubits: 3*2 + 2*1/2 = 6 + 1 = 7
        assert _compute_feature_dim(2) == 7
        assert result.shape == (7,)

    def test_data_reuploading_depth_effect(self):
        """Deeper circuits should produce different features than depth=1."""
        config_d1 = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        config_d3 = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=3,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain_d1 = QEFBrain(config=config_d1, num_actions=4, device=DeviceType.CPU)
        brain_d3 = QEFBrain(config=config_d3, num_actions=4, device=DeviceType.CPU)

        features = np.array([0.5, 0.3], dtype=np.float32)
        result_d1 = brain_d1._get_reservoir_features(features)
        result_d3 = brain_d3._get_reservoir_features(features)

        assert result_d1.shape == result_d3.shape
        assert not np.allclose(result_d1, result_d3, atol=1e-6)

    def test_sparse_encoding_same_dimension(self):
        """Sparse encoding should produce the same feature dimension as uniform."""
        uniform = QEFBrain(
            config=QEFBrainConfig(
                num_qubits=8,
                circuit_depth=1,
                encoding_mode="uniform",
                readout_hidden_dim=8,
                readout_num_layers=1,
            ),
            num_actions=4,
            device=DeviceType.CPU,
        )
        sparse = QEFBrain(
            config=QEFBrainConfig(
                num_qubits=8,
                circuit_depth=1,
                encoding_mode="sparse",
                readout_hidden_dim=8,
                readout_num_layers=1,
            ),
            num_actions=4,
            device=DeviceType.CPU,
        )

        features = np.array([0.5, 0.3], dtype=np.float32)
        uniform_out = uniform._get_reservoir_features(features)
        sparse_out = sparse._get_reservoir_features(features)

        assert uniform_out.shape == sparse_out.shape

    def test_sparse_vs_uniform_differ(self):
        """Sparse and uniform encoding should produce different features."""
        uniform = QEFBrain(
            config=QEFBrainConfig(
                num_qubits=8,
                circuit_depth=2,
                encoding_mode="uniform",
                readout_hidden_dim=8,
                readout_num_layers=1,
            ),
            num_actions=4,
            device=DeviceType.CPU,
        )
        sparse = QEFBrain(
            config=QEFBrainConfig(
                num_qubits=8,
                circuit_depth=2,
                encoding_mode="sparse",
                readout_hidden_dim=8,
                readout_num_layers=1,
            ),
            num_actions=4,
            device=DeviceType.CPU,
        )

        features = np.array([0.5, 0.3], dtype=np.float32)
        uniform_out = uniform._get_reservoir_features(features)
        sparse_out = sparse._get_reservoir_features(features)

        assert not np.allclose(uniform_out, sparse_out, atol=1e-6)

    def test_sparse_encoding_default_is_uniform(self):
        """Default encoding_mode should be 'uniform'."""
        config = QEFBrainConfig()
        assert config.encoding_mode == "uniform"


class TestQEFGateAndFeatureModes:
    """Test cases for gate_mode (cz vs cry_crz) and feature_mode (z_cossin vs xyz)."""

    def _make_brain(
        self,
        gate_mode: Literal["cz", "cry_crz"] = "cz",
        feature_mode: Literal["z_cossin", "xyz"] = "z_cossin",
        **kwargs,
    ) -> QEFBrain:
        config = QEFBrainConfig(
            num_qubits=kwargs.get("num_qubits", 8),
            circuit_depth=kwargs.get("circuit_depth", 2),
            circuit_seed=kwargs.get("circuit_seed", 42),
            entanglement_topology=kwargs.get("topology", "modality_paired"),
            entanglement_enabled=kwargs.get("entanglement_enabled", True),
            gate_mode=gate_mode,
            feature_mode=feature_mode,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_default_modes(self):
        """Default gate_mode and feature_mode should be cz and z_cossin."""
        config = QEFBrainConfig()
        assert config.gate_mode == "cz"
        assert config.feature_mode == "z_cossin"

    def test_cry_crz_differs_from_cz(self):
        """CRY/CRZ gates should produce different features than CZ-only."""
        cz_brain = self._make_brain(gate_mode="cz")
        cry_brain = self._make_brain(gate_mode="cry_crz")

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        cz_out = cz_brain._get_reservoir_features(features)
        cry_out = cry_brain._get_reservoir_features(features)

        assert cz_out.shape == cry_out.shape
        assert not np.allclose(cz_out, cry_out, atol=1e-6)

    def test_xyz_differs_from_z_cossin(self):
        """XYZ features should differ from Z+cos/sin features."""
        cossin_brain = self._make_brain(feature_mode="z_cossin")
        xyz_brain = self._make_brain(feature_mode="xyz")

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        cossin_out = cossin_brain._get_reservoir_features(features)
        xyz_out = xyz_brain._get_reservoir_features(features)

        # Same dimension (3N + N(N-1)/2 for both)
        assert cossin_out.shape == xyz_out.shape
        assert not np.allclose(cossin_out, xyz_out, atol=1e-6)

    def test_xyz_feature_ranges(self):
        """XYZ features should all be in [-1, 1]."""
        brain = self._make_brain(feature_mode="xyz")
        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        result = brain._get_reservoir_features(features)

        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_cry_crz_with_xyz_combined(self):
        """Combined CRY/CRZ + XYZ mode should differ from all other combinations."""
        original = self._make_brain(gate_mode="cz", feature_mode="z_cossin")
        combined = self._make_brain(gate_mode="cry_crz", feature_mode="xyz")

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        orig_out = original._get_reservoir_features(features)
        comb_out = combined._get_reservoir_features(features)

        assert orig_out.shape == comb_out.shape
        assert not np.allclose(orig_out, comb_out, atol=1e-6)

    def test_cry_crz_separable_equals_cz_separable(self):
        """With entanglement disabled, gate_mode should not matter (no gates applied)."""
        cz_sep = self._make_brain(gate_mode="cz", entanglement_enabled=False)
        cry_sep = self._make_brain(gate_mode="cry_crz", entanglement_enabled=False)

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        cz_out = cz_sep._get_reservoir_features(features)
        cry_out = cry_sep._get_reservoir_features(features)

        np.testing.assert_array_almost_equal(cz_out, cry_out)

    def test_cry_crz_deterministic(self):
        """Same seed should produce identical CRY/CRZ angles."""
        brain1 = self._make_brain(gate_mode="cry_crz", circuit_seed=42)
        brain2 = self._make_brain(gate_mode="cry_crz", circuit_seed=42)

        features = np.array([0.5, 0.3], dtype=np.float32)
        out1 = brain1._get_reservoir_features(features)
        out2 = brain2._get_reservoir_features(features)

        np.testing.assert_array_equal(out1, out2)


class TestQEFTopology:
    """Test cases for entanglement topology variations."""

    def _make_brain(
        self,
        topology: Literal["modality_paired", "ring", "random"],
        *,
        enabled: bool = True,
        seed: int = 42,
    ) -> QEFBrain:
        """Create a QEF brain with the given topology settings."""
        config = QEFBrainConfig(
            num_qubits=8,
            circuit_depth=2,
            circuit_seed=seed,
            entanglement_topology=topology,
            entanglement_enabled=enabled,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_modality_paired_cz_pairs(self):
        """Modality-paired topology should have the expected cross-modal CZ pairs."""
        brain = self._make_brain("modality_paired")
        expected_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
        assert brain._cz_pairs == expected_pairs
        assert brain._cz_pairs == MODALITY_PAIRED_CZ

    def test_modality_paired_filters_small_qubit_count(self):
        """Modality-paired topology should filter pairs exceeding qubit count."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            entanglement_topology="modality_paired",
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        # Only (0,2) survives — (1,3), (4,6), (5,7) all have indices >= 3
        assert brain._cz_pairs == [(0, 2)]

    def test_different_topologies_produce_different_features(self):
        """Modality-paired, ring, and random topologies should produce different features."""
        mp_brain = self._make_brain("modality_paired")
        ring_brain = self._make_brain("ring")
        random_brain = self._make_brain("random")

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)

        mp_out = mp_brain._get_reservoir_features(features)
        ring_out = ring_brain._get_reservoir_features(features)
        random_out = random_brain._get_reservoir_features(features)

        assert not np.allclose(mp_out, ring_out, atol=1e-6)
        assert not np.allclose(mp_out, random_out, atol=1e-6)
        assert not np.allclose(ring_out, random_out, atol=1e-6)

    def test_separable_vs_entangled_differ(self):
        """Entangled and separable should produce different features."""
        entangled = self._make_brain("modality_paired", enabled=True)
        separable = self._make_brain("modality_paired", enabled=False)

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)

        entangled_out = entangled._get_reservoir_features(features)
        separable_out = separable._get_reservoir_features(features)

        assert not np.allclose(entangled_out, separable_out, atol=1e-6)

    def test_separable_same_dimension(self):
        """Separable mode should produce same feature dimension as entangled."""
        entangled = self._make_brain("modality_paired", enabled=True)
        separable = self._make_brain("modality_paired", enabled=False)

        features = np.array([0.5, 0.3], dtype=np.float32)

        entangled_out = entangled._get_reservoir_features(features)
        separable_out = separable._get_reservoir_features(features)

        assert entangled_out.shape == separable_out.shape

    def test_unused_qubits_participate_in_entanglement(self):
        """Unencoded qubits should still participate in entanglement via ZZ correlations."""
        # 2 features on 8 qubits: 6 qubits unencoded
        entangled = self._make_brain("ring", enabled=True)
        separable = self._make_brain("ring", enabled=False)

        features = np.array([0.5, 0.3], dtype=np.float32)
        entangled_result = entangled._get_reservoir_features(features)
        separable_result = separable._get_reservoir_features(features)

        n = entangled.num_qubits
        num_zz = n * (n - 1) // 2

        # ZZ correlations should differ between entangled and separable,
        # demonstrating that unencoded qubits participate via CZ gates
        entangled_zz = entangled_result[n : n + num_zz]
        separable_zz = separable_result[n : n + num_zz]

        assert not np.allclose(entangled_zz, separable_zz, atol=1e-6), (
            "Entanglement should affect ZZ correlations involving unencoded qubits"
        )

    def test_random_topology_reproducibility(self):
        """Same seed should produce identical random topology."""
        brain1 = self._make_brain("random", seed=42)
        brain2 = self._make_brain("random", seed=42)

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        result1 = brain1._get_reservoir_features(features)
        result2 = brain2._get_reservoir_features(features)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_random_topology(self):
        """Different seeds should produce different random topologies."""
        brain1 = self._make_brain("random", seed=42)
        brain2 = self._make_brain("random", seed=99)

        features = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        result1 = brain1._get_reservoir_features(features)
        result2 = brain2._get_reservoir_features(features)

        assert not np.allclose(result1, result2, atol=1e-6)


class TestQEFBrainReadout:
    """Test cases for actor and critic readout networks."""

    @pytest.fixture
    def brain(self) -> QEFBrain:
        """Create a test QEF brain."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=16,
            readout_num_layers=2,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

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

    def test_run_brain_returns_valid_action(self):
        """run_brain should return valid ActionData."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

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


class TestQEFBrainLearning:
    """Test cases for PPO learning."""

    @pytest.fixture
    def brain(self) -> QEFBrain:
        """Create a test QEF brain with small buffer for faster tests."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            ppo_minibatches=2,
            ppo_epochs=2,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_ppo_update_changes_weights(self, brain):
        """PPO update should modify readout weights."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        initial_actor_weights = {
            name: param.clone() for name, param in brain.actor.named_parameters()
        }

        # Fill buffer and trigger update
        for step in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(step == brain.config.ppo_buffer_size - 1))

        weights_changed = False
        for name, param in brain.actor.named_parameters():
            if not torch.allclose(param, initial_actor_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "PPO update should change actor weights"

    def test_buffer_management(self, brain):
        """Buffer resets after mid-episode PPO update."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        for _step in range(brain.config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        assert brain._deferred_ppo_update is True
        assert len(brain.buffer) == brain.config.ppo_buffer_size

        brain.run_brain(params, top_only=True, top_randomize=False)
        assert brain._deferred_ppo_update is False
        assert len(brain.buffer) == 0


class TestQEFHybridInput:
    """Test cases for hybrid input mode (raw sensory + quantum features)."""

    def _make_brain(self, *, hybrid: bool, num_qubits: int = 4) -> QEFBrain:
        config = QEFBrainConfig(
            num_qubits=num_qubits,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            hybrid_input=hybrid,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_config_default_false(self):
        """hybrid_input should default to False."""
        config = QEFBrainConfig()
        assert config.hybrid_input is False

    def test_feature_dim_non_hybrid(self):
        """Non-hybrid feature dim should be quantum-only (3N + N(N-1)/2)."""
        brain = self._make_brain(hybrid=False, num_qubits=4)
        assert brain.feature_dim == _compute_feature_dim(4)  # 18

    def test_feature_dim_hybrid(self):
        """Hybrid feature dim should be input_dim + quantum_dim."""
        brain = self._make_brain(hybrid=True, num_qubits=4)
        quantum_dim = _compute_feature_dim(4)  # 18
        expected = brain.input_dim + quantum_dim  # 2 + 18 = 20
        assert brain.feature_dim == expected

    def test_feature_dim_hybrid_with_sensory_modules(self):
        """Hybrid with 3 sensory modules should have 7 + 52 = 59 features."""
        config = QEFBrainConfig(
            num_qubits=8,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            hybrid_input=True,
            sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.THERMOTAXIS,
            ],
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.input_dim == 7
        assert brain.feature_dim == 7 + 52  # 59

    def test_hybrid_output_contains_raw_features(self):
        """Hybrid output should start with raw sensory features."""
        brain = self._make_brain(hybrid=True, num_qubits=4)
        raw_features = np.array([0.5, 0.3], dtype=np.float32)
        result = brain._get_reservoir_features(raw_features)

        # First input_dim elements should be the raw features
        np.testing.assert_array_equal(result[:2], raw_features)

    def test_hybrid_output_contains_quantum_features(self):
        """Hybrid output should contain quantum features after raw features."""
        hybrid_brain = self._make_brain(hybrid=True, num_qubits=4)
        non_hybrid_brain = self._make_brain(hybrid=False, num_qubits=4)

        raw_features = np.array([0.5, 0.3], dtype=np.float32)
        hybrid_result = hybrid_brain._get_reservoir_features(raw_features)
        quantum_result = non_hybrid_brain._get_reservoir_features(raw_features)

        # Quantum portion of hybrid should match non-hybrid output
        np.testing.assert_array_equal(hybrid_result[2:], quantum_result)

    def test_hybrid_vs_non_hybrid_different_dim(self):
        """Hybrid and non-hybrid should produce different-sized feature vectors."""
        hybrid_brain = self._make_brain(hybrid=True, num_qubits=4)
        non_hybrid_brain = self._make_brain(hybrid=False, num_qubits=4)

        raw_features = np.array([0.5, 0.3], dtype=np.float32)
        hybrid_result = hybrid_brain._get_reservoir_features(raw_features)
        non_hybrid_result = non_hybrid_brain._get_reservoir_features(raw_features)

        assert len(hybrid_result) == len(non_hybrid_result) + 2

    def test_hybrid_actor_critic_accept_correct_dim(self):
        """Actor and critic networks should accept hybrid feature dimension."""
        brain = self._make_brain(hybrid=True, num_qubits=4)
        x = torch.randn(brain.feature_dim)
        logits = brain.actor(x)
        value = brain.critic(x)
        assert logits.shape == (4,)
        assert value.shape == (1,)

    def test_hybrid_run_brain(self):
        """run_brain should work with hybrid input."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)

    def test_hybrid_copy_preserves_setting(self):
        """Copy should preserve hybrid_input setting."""
        brain = self._make_brain(hybrid=True, num_qubits=4)
        brain_copy = brain.copy()
        assert brain_copy.hybrid_input is True
        assert brain_copy.feature_dim == brain.feature_dim


class TestQEFFeatureGating:
    """Test cases for learnable feature gating."""

    def test_config_default_false(self):
        """feature_gating should default to 'none'."""
        config = QEFBrainConfig()
        assert config.feature_gating == "none"

    def test_gating_changes_feature_dim(self):
        """Feature gating should not change feature dimension."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="static",
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.feature_dim == _compute_feature_dim(4)

    def test_gating_with_hybrid(self):
        """Feature gating + hybrid should work together."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="static",
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.feature_dim == brain.input_dim + _compute_feature_dim(4)
        assert hasattr(brain, "gate_weights")

    def test_gating_run_brain(self):
        """run_brain should work with feature gating."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            feature_gating="static",
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )
        actions = brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(actions) == 1

    def test_gating_ppo_update(self):
        """PPO update should work and update gate weights with feature gating."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            ppo_minibatches=2,
            ppo_epochs=2,
            feature_gating="static",
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        initial_gates = brain.gate_weights.clone()

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        for step in range(config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(step == config.ppo_buffer_size - 1))

        assert not torch.allclose(brain.gate_weights, initial_gates), (
            "Gate weights should change during PPO update"
        )


class TestQEFContextGating:
    """Test cases for context-dependent feature gating."""

    def test_context_gating_creates_network(self):
        """Context gating should create a gate_network."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="context",
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert hasattr(brain, "gate_network")

    def test_context_gate_output_shape(self):
        """Context gate should preserve feature_dim shape."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="context",
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        x = torch.randn(brain.feature_dim)
        gated = brain._apply_feature_gating(x)
        assert gated.shape == x.shape

    def test_context_gate_input_dependent(self):
        """Different raw inputs should produce different gates."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="context",
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        # Use same quantum features but different raw features
        quantum_part = torch.randn(brain.feature_dim - brain._raw_input_dim)
        x1 = torch.cat([torch.tensor([1.0, 0.0]), quantum_part])
        x2 = torch.cat([torch.tensor([0.0, 1.0]), quantum_part])
        gated1 = brain._apply_feature_gating(x1)
        gated2 = brain._apply_feature_gating(x2)
        # Same quantum features but different raw → different gates → different output
        assert not torch.allclose(gated1[brain._raw_input_dim :], gated2[brain._raw_input_dim :])

    def test_context_gating_run_brain(self):
        """run_brain should work with context gating."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            feature_gating="context",
            hybrid_input=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )
        actions = brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(actions) == 1

    def test_context_gating_with_sensory_modules(self):
        """Context gating should work with sensory modules (7-dim raw input)."""
        config = QEFBrainConfig(
            num_qubits=8,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            feature_gating="context",
            hybrid_input=True,
            sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.THERMOTAXIS,
            ],
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain._raw_input_dim == 7
        assert brain.feature_dim == 7 + 52  # 59
        x = torch.randn(brain.feature_dim)
        gated = brain._apply_feature_gating(x)
        assert gated.shape == (59,)


class TestQEFSeparateCritic:
    """Test cases for separate critic input."""

    def test_config_default_false(self):
        """separate_critic should default to False."""
        config = QEFBrainConfig()
        assert config.separate_critic is False

    def test_requires_hybrid_input(self):
        """separate_critic should require hybrid_input."""
        with pytest.raises(ValueError, match="separate_critic requires hybrid_input"):
            QEFBrainConfig(separate_critic=True, hybrid_input=False)

    def test_separate_critic_network_dim(self):
        """Separate critic should have raw input dim, not feature dim."""
        config = QEFBrainConfig(
            num_qubits=4,
            readout_hidden_dim=8,
            readout_num_layers=1,
            hybrid_input=True,
            separate_critic=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        # Critic input should be raw_input_dim (2 for legacy)
        x_raw = torch.randn(brain._raw_input_dim)
        value = brain.critic(x_raw)
        assert value.shape == (1,)

    def test_separate_critic_run_brain(self):
        """run_brain should work with separate critic."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            hybrid_input=True,
            separate_critic=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )
        actions = brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(actions) == 1

    def test_separate_critic_ppo_update(self):
        """PPO update should work with separate critic."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            ppo_minibatches=2,
            ppo_epochs=2,
            hybrid_input=True,
            separate_critic=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        initial_critic_weights = {
            name: param.clone() for name, param in brain.critic.named_parameters()
        }

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        for step in range(config.ppo_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(step == config.ppo_buffer_size - 1))

        weights_changed = any(
            not torch.allclose(param, initial_critic_weights[name])
            for name, param in brain.critic.named_parameters()
        )
        assert weights_changed, "Separate critic weights should change during PPO update"

    def test_all_features_combined(self):
        """Hybrid + gating + separate critic should all work together."""
        config = QEFBrainConfig(
            num_qubits=3,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
            ppo_buffer_size=8,
            hybrid_input=True,
            feature_gating="static",
            separate_critic=True,
        )
        brain = QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )
        actions = brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(actions) == 1


class TestQEFBrainCopy:
    """Test cases for brain copying."""

    @pytest.fixture
    def brain(self) -> QEFBrain:
        """Create a test QEF brain."""
        config = QEFBrainConfig(
            num_qubits=4,
            circuit_depth=1,
            readout_hidden_dim=8,
            readout_num_layers=1,
        )
        return QEFBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_copy_independence(self, brain):
        """Modifying copy should not affect original."""
        original_weights = {name: param.clone() for name, param in brain.actor.named_parameters()}

        brain_copy = brain.copy()

        with torch.no_grad():
            for param in brain_copy.actor.parameters():
                param.fill_(999.0)

        for name, param in brain.actor.named_parameters():
            assert torch.allclose(param, original_weights[name]), (
                f"Original weights changed for {name}"
            )

    def test_copy_shares_topology(self, brain):
        """Copy should produce identical features for same input."""
        brain_copy = brain.copy()

        features = np.array([0.5, 0.3], dtype=np.float32)
        orig_result = brain._get_reservoir_features(features)
        copy_result = brain_copy._get_reservoir_features(features)

        np.testing.assert_array_equal(orig_result, copy_result)

    def test_copy_preserves_config(self, brain):
        """Copy should preserve topology and config settings."""
        brain_copy = brain.copy()
        assert brain_copy.entanglement_topology == brain.entanglement_topology
        assert brain_copy.entanglement_enabled == brain.entanglement_enabled
        assert brain_copy.num_qubits == brain.num_qubits
        assert brain_copy.circuit_depth == brain.circuit_depth
