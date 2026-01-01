"""Tests for the unified feature extraction layer."""

import numpy as np
import pytest
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.features import (
    extract_flat_features,
    extract_sensory_features,
    get_feature_dimension,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestExtractSensoryFeatures:
    """Test the extract_sensory_features function."""

    def test_extract_all_modules_by_default(self):
        """Test that all modules are extracted when none specified."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = extract_sensory_features(params)

        # Should have features for all modules in the extractor registry
        assert len(features) > 0

        # Check that each feature is a numpy array with 3 elements
        for feature_array in features.values():
            assert isinstance(feature_array, np.ndarray)
            assert feature_array.shape == (3,)
            assert feature_array.dtype == np.float32

    def test_extract_specific_modules(self):
        """Test extracting only specific modules."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        modules = [ModuleName.CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
        features = extract_sensory_features(params, modules)

        assert len(features) == 2
        assert "chemotaxis" in features
        assert "proprioception" in features

    def test_chemotaxis_features_values(self):
        """Test that chemotaxis features have expected values."""
        params = BrainParams(
            gradient_strength=1.0,  # Max strength
            gradient_direction=np.pi / 2,  # Pointing up
            agent_direction=Direction.UP,  # Facing up (aligned)
        )
        features = extract_sensory_features(params, [ModuleName.CHEMOTAXIS])

        chemotaxis = features["chemotaxis"]
        # RX should be scaled gradient strength
        assert chemotaxis[0] == pytest.approx(np.pi / 2, rel=0.01)  # 1.0 * π - π/2 = π/2
        # RY should be near 0 (aligned with direction)
        assert abs(chemotaxis[1]) < 0.1

    def test_proprioception_features_values(self):
        """Test that proprioception features encode direction correctly."""
        for direction, expected_rz in [
            (Direction.UP, 0.0),
            (Direction.DOWN, np.pi),
            (Direction.LEFT, np.pi / 2),
            (Direction.RIGHT, -np.pi / 2),
        ]:
            params = BrainParams(agent_direction=direction)
            features = extract_sensory_features(params, [ModuleName.PROPRIOCEPTION])

            prop = features["proprioception"]
            assert prop[2] == pytest.approx(expected_rz, rel=0.01)

    def test_empty_params(self):
        """Test with default BrainParams (all None values)."""
        params = BrainParams()
        features = extract_sensory_features(params, [ModuleName.CHEMOTAXIS])

        # Should still return features, just with default values
        assert "chemotaxis" in features
        assert features["chemotaxis"].shape == (3,)


class TestExtractFlatFeatures:
    """Test the extract_flat_features function."""

    def test_flat_features_shape(self):
        """Test that flat features have correct shape."""
        params = BrainParams(agent_direction=Direction.UP)
        modules = [ModuleName.CHEMOTAXIS, ModuleName.PROPRIOCEPTION]

        features = extract_flat_features(params, modules)

        # 2 modules * 3 rotations each = 6 features
        assert features.shape == (6,)
        assert features.dtype == np.float32

    def test_flat_features_consistent_order(self):
        """Test that flat features maintain consistent ordering."""
        params = BrainParams(
            gradient_strength=0.5,
            agent_direction=Direction.LEFT,
        )
        modules = [ModuleName.CHEMOTAXIS, ModuleName.PROPRIOCEPTION]

        # Extract multiple times and verify same order
        features1 = extract_flat_features(params, modules)
        features2 = extract_flat_features(params, modules)

        np.testing.assert_array_equal(features1, features2)

    def test_flat_features_empty_modules(self):
        """Test with empty module list."""
        params = BrainParams()
        features = extract_flat_features(params, [])

        assert features.shape == (0,)


class TestGetFeatureDimension:
    """Test the get_feature_dimension function."""

    def test_dimension_two_modules(self):
        """Test dimension for two modules."""
        modules = [ModuleName.CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
        dim = get_feature_dimension(modules)

        assert dim == 6  # 2 modules * 3 rotations

    def test_dimension_single_module(self):
        """Test dimension for single module."""
        modules = [ModuleName.CHEMOTAXIS]
        dim = get_feature_dimension(modules)

        assert dim == 3

    def test_dimension_empty(self):
        """Test dimension for empty module list."""
        dim = get_feature_dimension([])
        assert dim == 0


class TestBackwardCompatibility:
    """Test backward compatibility with legacy module names."""

    def test_appetitive_alias_works(self):
        """Test that APPETITIVE still works as alias for FOOD_CHEMOTAXIS."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )

        # Both should return features
        appetitive = extract_sensory_features(params, [ModuleName.APPETITIVE])
        food_chem = extract_sensory_features(params, [ModuleName.FOOD_CHEMOTAXIS])

        # Values should be equal (same underlying function)
        np.testing.assert_array_almost_equal(
            appetitive["appetitive"],
            food_chem["food_chemotaxis"],
        )

    def test_aversive_alias_works(self):
        """Test that AVERSIVE still works as alias for NOCICEPTION."""
        params = BrainParams(
            predator_gradient_strength=0.5,
            predator_gradient_direction=np.pi,
            agent_direction=Direction.UP,
        )

        aversive = extract_sensory_features(params, [ModuleName.AVERSIVE])
        nociception = extract_sensory_features(params, [ModuleName.NOCICEPTION])

        # Values should be equal (same underlying function)
        np.testing.assert_array_almost_equal(
            aversive["aversive"],
            nociception["nociception"],
        )

    def test_oxygen_alias_works(self):
        """Test that OXYGEN still works as alias for AEROTAXIS."""
        params = BrainParams()

        oxygen = extract_sensory_features(params, [ModuleName.OXYGEN])
        aerotaxis = extract_sensory_features(params, [ModuleName.AEROTAXIS])

        # Both are placeholders returning zeros
        np.testing.assert_array_almost_equal(
            oxygen["oxygen"],
            aerotaxis["aerotaxis"],
        )


class TestScientificNames:
    """Test the new scientific module names."""

    def test_food_chemotaxis_features(self):
        """Test FOOD_CHEMOTAXIS module."""
        params = BrainParams(
            food_gradient_strength=1.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )

        features = extract_sensory_features(params, [ModuleName.FOOD_CHEMOTAXIS])

        assert "food_chemotaxis" in features
        assert features["food_chemotaxis"].shape == (3,)

    def test_nociception_features(self):
        """Test NOCICEPTION module."""
        params = BrainParams(
            predator_gradient_strength=0.8,
            predator_gradient_direction=np.pi,
            agent_direction=Direction.UP,
        )

        features = extract_sensory_features(params, [ModuleName.NOCICEPTION])

        assert "nociception" in features
        assert features["nociception"].shape == (3,)

    def test_aerotaxis_placeholder(self):
        """Test AEROTAXIS module returns zeros (placeholder)."""
        params = BrainParams()

        features = extract_sensory_features(params, [ModuleName.AEROTAXIS])

        assert "aerotaxis" in features
        np.testing.assert_array_equal(
            features["aerotaxis"],
            np.zeros(3, dtype=np.float32),
        )
