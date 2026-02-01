"""Tests for MLPPPOBrain unified feature extraction integration."""

import numpy as np
import pytest
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestPPOBrainLegacyMode:
    """Test MLPPPOBrain in legacy mode (backward compatibility)."""

    def test_legacy_mode_default_input_dim(self):
        """Test that legacy mode defaults to 2 features."""
        config = MLPPPOBrainConfig()
        brain = MLPPPOBrain(config=config)

        assert brain.input_dim == 2
        assert brain.sensory_modules is None

    def test_legacy_mode_explicit_input_dim(self):
        """Test that explicit input_dim is respected in legacy mode."""
        config = MLPPPOBrainConfig()
        brain = MLPPPOBrain(config=config, input_dim=5)

        assert brain.input_dim == 5
        assert brain.sensory_modules is None

    def test_legacy_preprocess_output_shape(self):
        """Test that legacy preprocessing returns 2 features."""
        config = MLPPPOBrainConfig()
        brain = MLPPPOBrain(config=config)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)

        assert features.shape == (2,)
        assert features.dtype == np.float32

    def test_legacy_preprocess_values(self):
        """Test that legacy preprocessing computes correct values."""
        config = MLPPPOBrainConfig()
        brain = MLPPPOBrain(config=config)

        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=np.pi / 2,  # Pointing up
            agent_direction=Direction.UP,  # Facing up (aligned)
        )

        features = brain.preprocess(params)

        # First feature: gradient_strength
        assert features[0] == pytest.approx(0.8)
        # Second feature: rel_angle_norm (should be ~0 when aligned)
        assert abs(features[1]) < 0.1


class TestPPOBrainUnifiedMode:
    """Test MLPPPOBrain in unified sensory mode with classical feature extraction."""

    def test_unified_mode_auto_computes_input_dim(self):
        """Test that input_dim is auto-computed from modules."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config)

        # 2 modules * 2 features each = 4 (classical extraction: [strength, angle])
        assert brain.input_dim == 4
        assert brain.sensory_modules == [ModuleName.CHEMOTAXIS, ModuleName.NOCICEPTION]

    def test_unified_mode_single_module(self):
        """Test unified mode with single module."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.CHEMOTAXIS],
        )
        brain = MLPPPOBrain(config=config)

        # 1 module * 2 features = 2 (classical: [strength, angle])
        assert brain.input_dim == 2

    def test_unified_mode_overrides_explicit_input_dim(self):
        """Test that sensory_modules overrides explicit input_dim."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.CHEMOTAXIS],
        )
        # Explicitly pass wrong input_dim - should be overridden
        brain = MLPPPOBrain(config=config, input_dim=10)

        assert brain.input_dim == 2  # Overridden by modules (1 * 2 features)

    def test_unified_preprocess_output_shape(self):
        """Test that unified preprocessing returns correct shape."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        )
        brain = MLPPPOBrain(config=config)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)

        # 2 modules * 2 features = 4 (classical: [strength, angle] per module)
        assert features.shape == (4,)
        assert features.dtype == np.float32

    def test_unified_preprocess_uses_modules(self):
        """Test that unified preprocessing uses specified modules."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config)

        params = BrainParams(
            predator_gradient_strength=0.9,
            predator_gradient_direction=np.pi,
            agent_direction=Direction.DOWN,
        )

        features = brain.preprocess(params)

        # Should have nociception features (2: [strength, angle])
        assert features.shape == (2,)
        # First feature is strength (should be 0.9 for predator strength)
        assert features[0] == pytest.approx(0.9)

    def test_unified_preprocess_semantic_ranges(self):
        """Test that classical features preserve semantic ranges."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )
        brain = MLPPPOBrain(config=config)

        # No food signal - strength should be 0, not -1
        params_no_food = BrainParams(
            food_gradient_strength=0.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features_no_food = brain.preprocess(params_no_food)
        assert features_no_food[0] == pytest.approx(0.0)  # strength = 0 means no signal

        # Strong food signal - strength should be near 1
        params_with_food = BrainParams(
            food_gradient_strength=0.9,
            food_gradient_direction=np.pi / 2,  # Directly ahead (UP)
            agent_direction=Direction.UP,
        )
        features_with_food = brain.preprocess(params_with_food)
        assert features_with_food[0] == pytest.approx(0.9)  # strength preserved
        assert abs(features_with_food[1]) < 0.1  # angle ~0 when aligned


class TestPPOBrainRunWithUnifiedFeatures:
    """Test MLPPPOBrain execution with unified features."""

    def test_run_brain_with_unified_features(self):
        """Test that run_brain works with unified feature extraction."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config, num_actions=4)

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            food_gradient_strength=0.3,
            predator_gradient_strength=0.2,
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=False)

        assert len(actions) == 1
        assert actions[0].action is not None
        assert 0 <= actions[0].probability <= 1

    def test_run_brain_legacy_mode_still_works(self):
        """Test that run_brain still works in legacy mode."""
        config = MLPPPOBrainConfig()
        brain = MLPPPOBrain(config=config, input_dim=2, num_actions=4)

        params = BrainParams(
            gradient_strength=0.7,
            gradient_direction=np.pi / 4,
            agent_direction=Direction.RIGHT,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=False)

        assert len(actions) == 1
        assert actions[0].action is not None


class TestPPOBrainWithScientificModuleNames:
    """Test MLPPPOBrain with scientific module names."""

    def test_food_chemotaxis_module(self):
        """Test using FOOD_CHEMOTAXIS (scientific name)."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )
        brain = MLPPPOBrain(config=config)

        # 1 module * 2 features = 2 (classical: [strength, angle])
        assert brain.input_dim == 2

        params = BrainParams(
            food_gradient_strength=0.6,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)
        assert features.shape == (2,)
        # First feature is food strength
        assert features[0] == pytest.approx(0.6)

    def test_nociception_module(self):
        """Test using NOCICEPTION (scientific name)."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config)

        # 1 module * 2 features = 2
        assert brain.input_dim == 2

    def test_mechanosensation_module(self):
        """Test using MECHANOSENSATION module."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.MECHANOSENSATION],
        )
        brain = MLPPPOBrain(config=config)

        # 1 module * 2 features = 2 (classical: [boundary, predator] for mechano)
        assert brain.input_dim == 2

        params = BrainParams(
            boundary_contact=True,
            predator_contact=False,
        )

        features = brain.preprocess(params)
        assert features.shape == (2,)
        # First feature (strength=boundary) should be 1.0 for contact
        assert features[0] == pytest.approx(1.0)
        # Second feature (angle=predator) should be 0.0 for no contact
        assert features[1] == pytest.approx(0.0)

    def test_multi_sensory_config(self):
        """Test with multiple sensory modules for multi-objective scenarios."""
        config = MLPPPOBrainConfig(
            sensory_modules=[
                ModuleName.CHEMOTAXIS,
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.MECHANOSENSATION,
            ],
        )
        brain = MLPPPOBrain(config=config)

        # 4 modules * 2 features = 8 (classical extraction)
        assert brain.input_dim == 8
