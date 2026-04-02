"""Unit tests for the CRH (Classical Reservoir Hybrid) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._reservoir_hybrid_base import ReservoirHybridBase
from quantumnematode.brain.arch.crh import CRHBrain, CRHBrainConfig, InputEncoding
from quantumnematode.brain.modules import ModuleName

# =============================================================================
# TestCRHBrainConfig
# =============================================================================


class TestCRHBrainConfig:
    """Test CRH-specific config defaults, custom values, and validators."""

    def test_default_config_values(self):
        """Test that default CRH config values match expected values."""
        config = CRHBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )

        assert config.num_reservoir_neurons == 10
        assert config.reservoir_depth == 3
        assert config.reservoir_seed == 42
        assert config.spectral_radius == 0.9
        assert config.input_connectivity == "sparse"
        assert config.input_scale == 1.0
        assert config.feature_channels == ["raw", "cos_sin", "pairwise"]
        assert config.num_sensory_neurons is None
        assert config.input_encoding == "linear"

        # Inherited base defaults
        assert config.readout_hidden_dim == 64
        assert config.actor_lr == 0.0003
        assert config.ppo_buffer_size == 512
        assert config.sensory_modules == [ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION]

    def test_custom_config_values(self):
        """Test that custom values are accepted."""
        config = CRHBrainConfig(
            num_reservoir_neurons=20,
            reservoir_depth=5,
            reservoir_seed=123,
            spectral_radius=0.95,
            input_connectivity="dense",
            input_scale=0.5,
            feature_channels=["raw", "squared"],
            num_sensory_neurons=8,
            readout_hidden_dim=128,
            actor_lr=0.001,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )

        assert config.num_reservoir_neurons == 20
        assert config.reservoir_depth == 5
        assert config.reservoir_seed == 123
        assert config.spectral_radius == 0.95
        assert config.input_connectivity == "dense"
        assert config.input_scale == 0.5
        assert config.feature_channels == ["raw", "squared"]
        assert config.num_sensory_neurons == 8
        assert config.readout_hidden_dim == 128
        assert config.actor_lr == 0.001

    def test_inherits_from_base_config(self):
        """CRHBrainConfig inherits from ReservoirHybridBaseConfig."""
        from quantumnematode.brain.arch._reservoir_hybrid_base import ReservoirHybridBaseConfig

        config = CRHBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert isinstance(config, ReservoirHybridBaseConfig)

    def test_validate_feature_channels_non_empty(self):
        """feature_channels must be non-empty."""
        with pytest.raises(ValueError, match="non-empty"):
            CRHBrainConfig(
                feature_channels=[],
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_input_connectivity_values(self):
        """input_connectivity must be 'sparse' or 'dense'."""
        with pytest.raises(ValueError, match=r"sparse.*dense"):
            CRHBrainConfig(
                input_connectivity="invalid",
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_spectral_radius_positive(self):
        """spectral_radius must be > 0."""
        with pytest.raises(ValueError, match="spectral_radius"):
            CRHBrainConfig(
                spectral_radius=0.0,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

        with pytest.raises(ValueError, match="spectral_radius"):
            CRHBrainConfig(
                spectral_radius=-0.5,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_input_scale_positive(self):
        """input_scale must be > 0."""
        with pytest.raises(ValueError, match="input_scale"):
            CRHBrainConfig(
                input_scale=0.0,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

        with pytest.raises(ValueError, match="input_scale"):
            CRHBrainConfig(
                input_scale=-0.5,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_num_reservoir_neurons_minimum(self):
        """num_reservoir_neurons must be >= 2."""
        with pytest.raises(ValueError, match="num_reservoir_neurons"):
            CRHBrainConfig(
                num_reservoir_neurons=1,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_num_sensory_neurons_bounds(self):
        """num_sensory_neurons must be >= 1 and <= num_reservoir_neurons."""
        with pytest.raises(ValueError, match="num_sensory_neurons"):
            CRHBrainConfig(
                num_reservoir_neurons=10,
                num_sensory_neurons=0,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

        with pytest.raises(ValueError, match="num_sensory_neurons"):
            CRHBrainConfig(
                num_reservoir_neurons=10,
                num_sensory_neurons=11,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_reservoir_depth_minimum(self):
        """reservoir_depth must be >= 1."""
        with pytest.raises(ValueError, match="reservoir_depth"):
            CRHBrainConfig(
                reservoir_depth=0,
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )

    def test_validate_input_encoding_linear(self):
        """input_encoding defaults to 'linear'."""
        config = CRHBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.input_encoding == "linear"

    def test_validate_input_encoding_trig(self):
        """input_encoding accepts 'trig'."""
        config = CRHBrainConfig(
            input_encoding="trig",
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.input_encoding == "trig"

    def test_validate_input_encoding_invalid(self):
        """input_encoding rejects invalid values."""
        with pytest.raises(ValueError, match="input_encoding"):
            CRHBrainConfig(
                input_encoding="invalid",  # type: ignore[arg-type]
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            )


# =============================================================================
# TestCRHReservoir
# =============================================================================


class TestCRHReservoir:
    """Test ESN reservoir construction: shapes, spectral radius, reproducibility."""

    @pytest.fixture
    def brain(self):
        """Create a CRH brain with small config for testing."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            reservoir_seed=42,
            spectral_radius=0.9,
            input_connectivity="sparse",
            num_sensory_neurons=2,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        return CRHBrain(config)

    def test_w_in_shape(self, brain):
        """W_in has shape (num_neurons, input_dim)."""
        assert brain.W_in.shape == (6, 4)

    def test_w_res_shape(self, brain):
        """W_res has shape (num_neurons, num_neurons)."""
        assert brain.W_res.shape == (6, 6)

    def test_spectral_radius_scaling(self, brain):
        """W_res largest eigenvalue magnitude equals configured spectral_radius."""
        eigenvalues = np.linalg.eigvals(brain.W_res)
        max_eigval = np.max(np.abs(eigenvalues))
        assert max_eigval == pytest.approx(0.9, abs=1e-6)

    def test_seed_reproducibility(self):
        """Two brains with same seed produce identical matrices."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            reservoir_seed=99,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain1 = CRHBrain(config)

        config2 = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            reservoir_seed=99,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain2 = CRHBrain(config2)

        np.testing.assert_array_equal(brain1.W_in, brain2.W_in)
        np.testing.assert_array_equal(brain1.W_res, brain2.W_res)

    def test_different_seeds_different_matrices(self):
        """Different reservoir seeds produce different matrices."""
        config1 = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_seed=42,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        config2 = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_seed=99,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain1 = CRHBrain(config1)
        brain2 = CRHBrain(config2)

        assert not np.array_equal(brain1.W_in, brain2.W_in)
        assert not np.array_equal(brain1.W_res, brain2.W_res)

    def test_sparse_connectivity_zeros_non_sensory(self, brain):
        """Sparse mode zeros out W_in rows beyond num_sensory_neurons."""
        # Rows 0-1 (sensory) should have non-zero entries
        assert np.any(brain.W_in[:2] != 0)
        # Rows 2-5 (non-sensory) should be all zero
        np.testing.assert_array_equal(brain.W_in[2:], 0.0)

    def test_dense_connectivity_all_rows_nonzero(self):
        """Dense mode: all W_in rows have non-zero entries."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            reservoir_seed=42,
            input_connectivity="dense",
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain = CRHBrain(config)

        for i in range(6):
            assert np.any(brain.W_in[i] != 0), f"Row {i} should be non-zero in dense mode"

    def test_w_in_scale(self):
        """W_in values are bounded by input_scale."""
        config = CRHBrainConfig(
            num_reservoir_neurons=10,
            input_connectivity="dense",
            input_scale=0.5,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain = CRHBrain(config)

        assert np.all(brain.W_in >= -0.5)
        assert np.all(brain.W_in <= 0.5)


# =============================================================================
# TestCRHFeatureExtraction
# =============================================================================


class TestCRHFeatureExtraction:
    """Test feature channel extraction: dimensions, ranges, ablation matching."""

    def _make_brain(self, channels, num_neurons=10, **kwargs):
        """Create a CRH brain with specific channels."""
        config = CRHBrainConfig(
            num_reservoir_neurons=num_neurons,
            reservoir_depth=2,
            reservoir_seed=42,
            feature_channels=channels,
            sensory_modules=kwargs.pop(
                "sensory_modules",
                [ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ),
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
            **kwargs,
        )
        return CRHBrain(config)

    def test_raw_channel_dimension(self):
        """Raw channel produces N features."""
        brain = self._make_brain(["raw"], num_neurons=8)
        assert brain.feature_dim == 8

    def test_cos_sin_channel_dimension(self):
        """Cos/sin channel produces 2N features."""
        brain = self._make_brain(["cos_sin"], num_neurons=8)
        assert brain.feature_dim == 16

    def test_squared_channel_dimension(self):
        """Squared channel produces N features."""
        brain = self._make_brain(["squared"], num_neurons=8)
        assert brain.feature_dim == 8

    def test_pairwise_channel_dimension(self):
        """Pairwise channel produces N(N-1)/2 features."""
        brain = self._make_brain(["pairwise"], num_neurons=8)
        assert brain.feature_dim == 28  # 8*7/2

    def test_combined_channels_dimension(self):
        """Combined channels sum dimensions correctly."""
        brain = self._make_brain(["raw", "cos_sin", "squared", "pairwise"], num_neurons=8)
        expected = 8 + 16 + 8 + 28  # 60
        assert brain.feature_dim == expected

    def test_ablation_mode_75_features(self):
        """Ablation mode: [raw, cos_sin, pairwise] with N=10 = 75 features."""
        brain = self._make_brain(["raw", "cos_sin", "pairwise"], num_neurons=10)
        assert brain.feature_dim == 75  # 10 + 20 + 45

    def test_raw_range(self):
        """Raw features are in [-1, 1] (tanh output)."""
        brain = self._make_brain(["raw"], num_neurons=6)
        features = brain._get_reservoir_features(np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32))
        assert features.shape == (6,)
        assert np.all(features >= -1.0)
        assert np.all(features <= 1.0)

    def test_cos_sin_range(self):
        """Cos/sin features are in [-1, 1]."""
        brain = self._make_brain(["cos_sin"], num_neurons=6)
        features = brain._get_reservoir_features(np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32))
        assert features.shape == (12,)  # 2*6
        assert np.all(features >= -1.0)
        assert np.all(features <= 1.0)

    def test_squared_range(self):
        """Squared features are in [0, 1]."""
        brain = self._make_brain(["squared"], num_neurons=6)
        features = brain._get_reservoir_features(np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32))
        assert features.shape == (6,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_pairwise_range(self):
        """Pairwise features are in [-1, 1] (product of tanh outputs)."""
        brain = self._make_brain(["pairwise"], num_neurons=6)
        features = brain._get_reservoir_features(np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32))
        assert features.shape == (15,)  # 6*5/2
        assert np.all(features >= -1.0)
        assert np.all(features <= 1.0)

    def test_feature_determinism(self):
        """Same input produces same features (stateless reservoir)."""
        brain = self._make_brain(["raw", "cos_sin", "pairwise"], num_neurons=6)
        x = np.array([0.7, -0.2, 0.1, 0.4], dtype=np.float32)
        f1 = brain._get_reservoir_features(x)
        f2 = brain._get_reservoir_features(x)
        np.testing.assert_array_equal(f1, f2)


# =============================================================================
# TestCRHBrainReadout
# =============================================================================


class TestCRHBrainReadout:
    """Test that actor/critic output shapes are correct (inherited from base)."""

    @pytest.fixture
    def brain(self):
        """Create a CRH brain for readout testing."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            feature_channels=["raw", "cos_sin"],
            readout_hidden_dim=32,
            readout_num_layers=2,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        return CRHBrain(config)

    def test_actor_output_shape(self, brain):
        """Actor produces logits for num_actions."""
        x = torch.randn(brain.feature_dim)
        x = brain.feature_norm(x)
        logits = brain.actor(x)
        assert logits.shape == (4,)

    def test_critic_output_shape(self, brain):
        """Critic produces single scalar value."""
        x = torch.randn(brain.feature_dim)
        x = brain.feature_norm(x)
        value = brain.critic(x)
        assert value.shape == (1,)


# =============================================================================
# TestCRHBrainLearning
# =============================================================================


class TestCRHBrainLearning:
    """Test run_brain(), PPO update, buffer management, full episode workflow."""

    @pytest.fixture
    def brain(self):
        """Create a CRH brain with small buffer for fast PPO updates."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ppo_buffer_size=8,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        return CRHBrain(config)

    def test_run_brain_returns_action_data(self, brain):
        """run_brain() returns a list with valid ActionData."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=1.0,
            agent_direction=None,
        )
        result = brain.run_brain(params, top_only=False, top_randomize=False)

        assert len(result) == 1
        action_data = result[0]
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        assert 0.0 <= action_data.probability <= 1.0

    def test_ppo_update_changes_weights(self, brain):
        """PPO update modifies actor weights."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=1.0,
            agent_direction=None,
        )

        # Record initial weights
        initial_weights = [p.clone() for p in brain.actor.parameters()]

        # Fill buffer and trigger PPO update
        for i in range(8):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 7))

        # Weights should have changed
        changed = any(
            not torch.allclose(p_init, p_new)
            for p_init, p_new in zip(initial_weights, brain.actor.parameters(), strict=True)
        )
        assert changed, "PPO update should modify actor weights"

    def test_buffer_management(self, brain):
        """Buffer fills and resets after PPO update."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=1.0,
            agent_direction=None,
        )

        for i in range(8):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 7))

        # Buffer should be reset after PPO update
        assert len(brain.buffer) == 0

    def test_full_episode_workflow(self, brain):
        """Full episode: prepare -> run/learn loop -> post_process."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=1.0,
            agent_direction=None,
        )

        brain.prepare_episode()
        assert brain._episode_count == 0

        for _ in range(4):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=False)

        brain.post_process_episode(episode_success=True)
        assert brain._episode_count == 1


# =============================================================================
# TestCRHBrainCopy
# =============================================================================


class TestCRHBrainCopy:
    """Test copy independence, shared W_in/W_res values, independent readout."""

    @pytest.fixture
    def brain(self):
        """Create a CRH brain for copy testing."""
        config = CRHBrainConfig(
            num_reservoir_neurons=6,
            reservoir_depth=2,
            feature_channels=["raw", "cos_sin"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        return CRHBrain(config)

    def test_copy_type_and_feature_dim(self, brain):
        """Copy is same type with same feature_dim."""
        copy = brain.copy()
        assert type(copy) is CRHBrain
        assert copy.feature_dim == brain.feature_dim
        assert copy.num_neurons == brain.num_neurons
        assert copy._episode_count == brain._episode_count

    def test_copy_has_same_reservoir_matrices(self, brain):
        """Copy has identical W_in and W_res (regenerated from same seed)."""
        copy = brain.copy()
        np.testing.assert_array_equal(copy.W_in, brain.W_in)
        np.testing.assert_array_equal(copy.W_res, brain.W_res)

    def test_copy_has_same_readout_weights(self, brain):
        """Copy initially has identical readout weights."""
        copy = brain.copy()
        for p_orig, p_copy in zip(
            brain.actor.parameters(),
            copy.actor.parameters(),
            strict=True,
        ):
            assert torch.allclose(p_orig, p_copy)

    def test_copy_readout_independence(self, brain):
        """Mutating copy's readout doesn't affect original."""
        copy = brain.copy()

        with torch.no_grad():
            for p in copy.actor.parameters():
                p.add_(1.0)

        for p_orig, p_copy in zip(
            brain.actor.parameters(),
            copy.actor.parameters(),
            strict=True,
        ):
            assert not torch.allclose(p_orig, p_copy)

    def test_copy_is_reservoir_hybrid_base(self, brain):
        """Copy is an instance of ReservoirHybridBase."""
        copy = brain.copy()
        assert isinstance(copy, ReservoirHybridBase)


# =============================================================================
# TestCRHBrainSensoryModules
# =============================================================================


class TestCRHBrainSensoryModules:
    """Test unified sensory mode dimensions."""

    def test_unified_mode_dimensions(self):
        """Unified sensory modules correctly set input_dim."""
        config = CRHBrainConfig(
            num_reservoir_neurons=10,
            feature_channels=["raw"],
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = CRHBrain(config)

        # food_chemotaxis: 2 features, nociception: 2 features -> 4
        assert brain.input_dim == 4
        assert brain.W_in.shape == (10, 4)

    def test_triple_objective_modules(self):
        """Triple-objective sensory modules match expected dimensions."""
        config = CRHBrainConfig(
            num_reservoir_neurons=10,
            feature_channels=["raw", "cos_sin", "pairwise"],
            sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.THERMOTAXIS,
            ],
            num_sensory_neurons=7,
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
        )
        brain = CRHBrain(config)

        # food_chemotaxis: 2, nociception: 2, thermotaxis: 3 -> 7
        assert brain.input_dim == 7
        assert brain.num_sensory == 7
        assert brain.feature_dim == 75  # 10 + 20 + 45
        assert brain.W_in.shape == (10, 7)


# =============================================================================
# TestCRHTrigEncoding
# =============================================================================


class TestCRHTrigEncoding:
    """Test trigonometric input encoding for Domingo confound control."""

    def _make_brain(self, input_encoding: InputEncoding = "trig", num_neurons: int = 10, **kwargs):
        """Create a CRH brain with specified encoding."""
        config = CRHBrainConfig(
            num_reservoir_neurons=num_neurons,
            reservoir_depth=2,
            reservoir_seed=42,
            feature_channels=["raw", "cos_sin", "pairwise"],
            input_encoding=input_encoding,
            sensory_modules=kwargs.pop(
                "sensory_modules",
                [ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            ),
            ppo_buffer_size=16,
            ppo_minibatches=2,
            ppo_epochs=1,
            seed=42,
            **kwargs,
        )
        return CRHBrain(config)

    def test_trig_encoding_w_in_shape(self):
        """Trig encoding doubles W_in column dimension."""
        brain = self._make_brain("trig")
        # input_dim=4 (food_chemotaxis + nociception), trig doubles to 8
        assert brain.W_in.shape == (10, 8)
        assert brain.w_in_dim == 8

    def test_trig_encoding_w_in_shape_with_modules(self):
        """Trig encoding doubles W_in with sensory modules (7 features -> 14)."""
        brain = self._make_brain(
            "trig",
            sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.THERMOTAXIS,
            ],
            num_sensory_neurons=7,
        )
        assert brain.input_dim == 7
        assert brain.w_in_dim == 14
        assert brain.W_in.shape == (10, 14)

    def test_trig_encoding_feature_dim_unchanged(self):
        """Trig encoding doesn't change output feature dimension (still 75)."""
        brain_linear = self._make_brain("linear")
        brain_trig = self._make_brain("trig")
        assert brain_linear.feature_dim == brain_trig.feature_dim == 75

    def test_trig_encoding_output_shape(self):
        """_get_reservoir_features produces same-length features regardless of encoding."""
        brain = self._make_brain("trig")
        features = brain._get_reservoir_features(np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32))
        assert features.shape == (75,)

    def test_trig_encoding_different_from_linear(self):
        """Trig encoding produces different features from linear for same input."""
        brain_linear = self._make_brain("linear")
        brain_trig = self._make_brain("trig")
        x = np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32)
        f_linear = brain_linear._get_reservoir_features(x)
        f_trig = brain_trig._get_reservoir_features(x)
        assert not np.allclose(f_linear, f_trig)

    def test_trig_encoding_deterministic(self):
        """Same input produces same output (stateless)."""
        brain = self._make_brain("trig")
        x = np.array([0.7, -0.2, 0.1, 0.4], dtype=np.float32)
        f1 = brain._get_reservoir_features(x)
        f2 = brain._get_reservoir_features(x)
        np.testing.assert_array_equal(f1, f2)

    def test_encode_input_trig_math(self):
        """Verify _encode_input produces correct sin/cos expansion."""
        brain = self._make_brain("trig")
        x = np.array([0.5, -0.3], dtype=np.float64)
        encoded = brain._encode_input(x)

        expected = np.array(
            [
                np.sin(0.5 * np.pi),  # sin(0.5π) ≈ 1.0
                np.sin(-0.3 * np.pi),  # sin(-0.3π) ≈ -0.809
                np.cos(0.5 * np.pi),  # cos(0.5π) ≈ 0.0
                np.cos(-0.3 * np.pi),  # cos(-0.3π) ≈ 0.588
            ],
        )
        np.testing.assert_allclose(encoded, expected, atol=1e-10)

    def test_encode_input_linear_passthrough(self):
        """Linear encoding is identity."""
        brain = self._make_brain("linear")
        x = np.array([0.5, -0.3], dtype=np.float64)
        encoded = brain._encode_input(x)
        np.testing.assert_array_equal(encoded, x)

    def test_linear_w_in_shape_unchanged(self):
        """Linear encoding preserves original W_in shape."""
        brain = self._make_brain("linear")
        assert brain.W_in.shape == (10, 4)
        assert brain.w_in_dim == 4
