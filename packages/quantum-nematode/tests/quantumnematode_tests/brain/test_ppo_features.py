"""Tests for MLPPPOBrain feature extraction integration."""

import numpy as np
import pytest
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestPPOBrainUnifiedMode:
    """Test MLPPPOBrain in unified sensory mode with classical feature extraction."""

    def test_unified_mode_auto_computes_input_dim(self):
        """Test that input_dim is auto-computed from modules."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config)

        # 2 modules * 2 features each = 4 (classical extraction: [strength, angle])
        assert brain.input_dim == 4
        assert brain.sensory_modules == [ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION]

    def test_unified_mode_single_module(self):
        """Test unified mode with single module."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )
        brain = MLPPPOBrain(config=config)

        # 1 module * 2 features = 2 (classical: [strength, angle])
        assert brain.input_dim == 2

    def test_unified_preprocess_output_shape(self):
        """Test that unified preprocessing returns correct shape."""
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        )
        brain = MLPPPOBrain(config=config)

        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=0.0,
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
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = MLPPPOBrain(config=config, num_actions=4)

        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=0.0,
            predator_gradient_strength=0.2,
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=False)

        assert len(actions) == 1
        assert actions[0].action is not None
        assert 0 <= actions[0].probability <= 1


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
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.MECHANOSENSATION,
                ModuleName.PROPRIOCEPTION,
            ],
        )
        brain = MLPPPOBrain(config=config)

        # 4 modules * 2 features = 8 (classical extraction)
        assert brain.input_dim == 8
