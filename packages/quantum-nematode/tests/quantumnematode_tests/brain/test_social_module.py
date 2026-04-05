"""Tests for the social proximity sensory module (Phase 4 multi-agent)."""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.brain.arch._brain import BrainParams
from quantumnematode.brain.modules import (
    SENSORY_MODULES,
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)


class TestSocialProximityModule:
    """Tests for the social_proximity sensory module."""

    def test_registered(self) -> None:
        """Test that social_proximity is registered in SENSORY_MODULES."""
        assert ModuleName.SOCIAL_PROXIMITY in SENSORY_MODULES

    def test_classical_dim_is_one(self) -> None:
        """Test that social_proximity contributes exactly 1 dimension."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        assert module.classical_dim == 1

    def test_zero_nearby_agents(self) -> None:
        """Test that strength is 0.0 when no nearby agents."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=0)
        features = module.to_classical(params)
        assert len(features) == 1
        assert features[0] == pytest.approx(0.0)

    def test_none_nearby_agents(self) -> None:
        """Test that strength is 0.0 when nearby_agents_count is None (single-agent)."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams()
        features = module.to_classical(params)
        assert len(features) == 1
        assert features[0] == pytest.approx(0.0)

    def test_five_nearby_agents(self) -> None:
        """Test normalized count: 5 agents -> 0.5 strength."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=5)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.5)

    def test_ten_nearby_agents(self) -> None:
        """Test normalized count: 10 agents -> 1.0 strength."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=10)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(1.0)

    def test_clamped_above_ten(self) -> None:
        """Test that counts above 10 are clamped to 1.0."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=15)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(1.0)

    def test_one_nearby_agent(self) -> None:
        """Test normalized count: 1 agent -> 0.1 strength."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=1)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.1)

    def test_negative_count_clamped_to_zero(self) -> None:
        """Regression: negative nearby_agents_count is clamped to 0."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=-1)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)

    def test_quantum_transform_produces_valid_angles(self) -> None:
        """Test that quantum transform produces angles in valid range."""
        module = SENSORY_MODULES[ModuleName.SOCIAL_PROXIMITY]
        params = BrainParams(nearby_agents_count=5)
        quantum = module.to_quantum(params)
        assert quantum.shape == (3,)
        # All angles should be in [-pi/2, pi/2]
        assert all(abs(a) <= np.pi / 2 + 1e-6 for a in quantum)

    def test_feature_dimension_contribution(self) -> None:
        """Test that get_classical_feature_dimension counts social_proximity as 1."""
        dim = get_classical_feature_dimension([ModuleName.SOCIAL_PROXIMITY])
        assert dim == 1

    def test_combined_with_other_modules(self) -> None:
        """Test social_proximity combined with food_chemotaxis in feature extraction."""
        modules = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.SOCIAL_PROXIMITY]
        dim = get_classical_feature_dimension(modules)
        # food_chemotaxis=2 + social_proximity=1
        assert dim == 3

        params = BrainParams(
            food_gradient_strength=0.8,
            food_gradient_direction=1.0,
            nearby_agents_count=3,
        )
        features = extract_classical_features(params, modules)
        assert len(features) == 3


class TestBrainParamsNearbyAgentsCount:
    """Tests for the nearby_agents_count field in BrainParams."""

    def test_default_is_none(self) -> None:
        """Test that nearby_agents_count defaults to None."""
        params = BrainParams()
        assert params.nearby_agents_count is None

    def test_set_count(self) -> None:
        """Test setting nearby_agents_count."""
        params = BrainParams(nearby_agents_count=3)
        assert params.nearby_agents_count == 3

    def test_backward_compatible(self) -> None:
        """Test that existing BrainParams fields still work with new field."""
        params = BrainParams(
            food_gradient_strength=0.5,
            temperature=20.0,
            nearby_agents_count=2,
        )
        assert params.food_gradient_strength == 0.5
        assert params.temperature == 20.0
        assert params.nearby_agents_count == 2
