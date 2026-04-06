"""Tests for pheromone sensing modules."""

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


class TestPheromoneModuleRegistration:
    """Tests for pheromone module registration."""

    def test_pheromone_food_registered(self) -> None:
        """Test PHEROMONE_FOOD is in SENSORY_MODULES."""
        assert ModuleName.PHEROMONE_FOOD in SENSORY_MODULES

    def test_pheromone_alarm_registered(self) -> None:
        """Test PHEROMONE_ALARM is in SENSORY_MODULES."""
        assert ModuleName.PHEROMONE_ALARM in SENSORY_MODULES

    def test_pheromone_food_temporal_registered(self) -> None:
        """Test PHEROMONE_FOOD_TEMPORAL is in SENSORY_MODULES."""
        assert ModuleName.PHEROMONE_FOOD_TEMPORAL in SENSORY_MODULES

    def test_pheromone_alarm_temporal_registered(self) -> None:
        """Test PHEROMONE_ALARM_TEMPORAL is in SENSORY_MODULES."""
        assert ModuleName.PHEROMONE_ALARM_TEMPORAL in SENSORY_MODULES

    def test_all_classical_dim_is_two(self) -> None:
        """Test all pheromone modules have classical_dim=2."""
        for name in (
            ModuleName.PHEROMONE_FOOD,
            ModuleName.PHEROMONE_ALARM,
            ModuleName.PHEROMONE_FOOD_TEMPORAL,
            ModuleName.PHEROMONE_ALARM_TEMPORAL,
        ):
            assert SENSORY_MODULES[name].classical_dim == 2


class TestPheromoneOracleModules:
    """Tests for oracle pheromone modules."""

    def test_food_gradient_strength(self) -> None:
        """Test food pheromone gradient strength is extracted."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_FOOD]
        params = BrainParams(pheromone_food_gradient_strength=0.7)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.7)

    def test_food_no_data_returns_zero(self) -> None:
        """Test food pheromone with no data returns zeros."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_FOOD]
        params = BrainParams()
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)
        assert features[1] == pytest.approx(0.0)

    def test_alarm_gradient_strength(self) -> None:
        """Test alarm pheromone gradient strength is extracted."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_ALARM]
        params = BrainParams(pheromone_alarm_gradient_strength=0.5)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.5)

    def test_alarm_no_data_returns_zero(self) -> None:
        """Test alarm pheromone with no data returns zeros."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_ALARM]
        params = BrainParams()
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)


class TestPheromoneTemporalModules:
    """Tests for temporal pheromone modules."""

    def test_food_temporal_concentration(self) -> None:
        """Test food pheromone temporal concentration is extracted."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_FOOD_TEMPORAL]
        params = BrainParams(pheromone_food_concentration=0.8)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.8)

    def test_food_temporal_derivative(self) -> None:
        """Test food pheromone temporal derivative is scaled via tanh."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_FOOD_TEMPORAL]
        params = BrainParams(
            pheromone_food_concentration=0.5,
            pheromone_food_dconcentration_dt=0.1,
            derivative_scale=50.0,
        )
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.5)
        # angle = tanh(0.1 * 50) = tanh(5) ≈ 1.0
        assert features[1] > 0.99

    def test_alarm_temporal_no_data(self) -> None:
        """Test alarm pheromone temporal with no data returns zeros."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_ALARM_TEMPORAL]
        params = BrainParams()
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)
        assert features[1] == pytest.approx(0.0)

    def test_alarm_temporal_concentration(self) -> None:
        """Test alarm pheromone temporal concentration is extracted."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_ALARM_TEMPORAL]
        params = BrainParams(pheromone_alarm_concentration=0.6)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.6)


class TestPheromoneQuantumTransform:
    """Tests for quantum feature transforms."""

    def test_food_quantum_valid_angles(self) -> None:
        """Test food oracle quantum transform produces valid rotation angles."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_FOOD]
        params = BrainParams(
            pheromone_food_gradient_strength=0.5,
            pheromone_food_gradient_direction=1.0,
        )
        quantum = module.to_quantum(params)
        assert quantum.shape == (3,)
        assert all(abs(a) <= np.pi / 2 + 1e-6 for a in quantum)

    def test_alarm_temporal_quantum_valid_angles(self) -> None:
        """Test alarm temporal quantum transform produces valid rotation angles."""
        module = SENSORY_MODULES[ModuleName.PHEROMONE_ALARM_TEMPORAL]
        params = BrainParams(pheromone_alarm_concentration=0.3)
        quantum = module.to_quantum(params)
        assert quantum.shape == (3,)
        assert all(abs(a) <= np.pi / 2 + 1e-6 for a in quantum)


class TestPheromoneFeatureDimension:
    """Tests for feature dimension computation."""

    def test_single_pheromone_module_dim(self) -> None:
        """Test single pheromone module contributes 2 dimensions."""
        assert get_classical_feature_dimension([ModuleName.PHEROMONE_FOOD]) == 2

    def test_combined_food_and_pheromone(self) -> None:
        """Test food_chemotaxis + pheromone_food = 4 dimensions."""
        modules = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PHEROMONE_FOOD]
        assert get_classical_feature_dimension(modules) == 4

    def test_all_pheromone_modules(self) -> None:
        """Test both pheromone oracle modules = 4 dimensions."""
        modules = [
            ModuleName.PHEROMONE_FOOD,
            ModuleName.PHEROMONE_ALARM,
        ]
        assert get_classical_feature_dimension(modules) == 4

    def test_combined_extraction(self) -> None:
        """Test feature extraction with food + pheromone modules."""
        modules = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PHEROMONE_FOOD]
        params = BrainParams(
            food_gradient_strength=0.8,
            food_gradient_direction=1.0,
            pheromone_food_gradient_strength=0.3,
            pheromone_food_gradient_direction=0.5,
        )
        features = extract_classical_features(params, modules)
        assert len(features) == 4
