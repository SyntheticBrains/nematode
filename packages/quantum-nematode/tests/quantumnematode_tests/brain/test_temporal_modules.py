"""Tests for temporal sensing sensory modules."""

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


class TestFoodChemotaxisTemporal:
    """Tests for the food_chemotaxis_temporal module."""

    def test_registered(self) -> None:
        """Test that food_chemotaxis_temporal is registered in SENSORY_MODULES."""
        assert ModuleName.FOOD_CHEMOTAXIS_TEMPORAL in SENSORY_MODULES

    def test_strength_from_food_concentration(self) -> None:
        """Test that strength feature equals the food concentration value."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams(food_concentration=0.7)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.7)

    def test_strength_zero_when_none(self) -> None:
        """Test that strength is zero when food concentration is not set."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams()
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)

    def test_angle_from_derivative(self) -> None:
        """Test that angle feature is tanh of scaled food concentration derivative."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams(food_concentration=0.5, food_dconcentration_dt=0.3)
        features = module.to_classical(params)
        # angle = tanh(0.3 * derivative_scale), default scale=50.0
        assert features[1] == pytest.approx(np.tanh(0.3 * 50.0))

    def test_angle_zero_when_no_derivative(self) -> None:
        """Test that angle is zero when no derivative is provided."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams(food_concentration=0.5)
        features = module.to_classical(params)
        assert features[1] == pytest.approx(0.0)

    def test_angle_clamped_by_tanh(self) -> None:
        """Test that angle is clamped to [-1, 1] by the tanh function."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams(food_concentration=0.5, food_dconcentration_dt=100.0)
        features = module.to_classical(params)
        assert -1.0 <= features[1] <= 1.0

    def test_classical_dim_is_2(self) -> None:
        """Test that the classical feature dimension is 2."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        assert module.classical_dim == 2

    def test_quantum_output_shape(self) -> None:
        """Test that quantum output has shape (3,) with values in valid range."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_TEMPORAL]
        params = BrainParams(food_concentration=0.5, food_dconcentration_dt=0.1)
        q = module.to_quantum(params)
        assert q.shape == (3,)
        assert all(abs(v) <= np.pi / 2 + 0.01 for v in q)


class TestNociceptionTemporal:
    """Tests for the nociception_temporal module."""

    def test_registered(self) -> None:
        """Test that nociception_temporal is registered in SENSORY_MODULES."""
        assert ModuleName.NOCICEPTION_TEMPORAL in SENSORY_MODULES

    def test_strength_from_predator_concentration(self) -> None:
        """Test that strength feature equals the predator concentration value."""
        module = SENSORY_MODULES[ModuleName.NOCICEPTION_TEMPORAL]
        params = BrainParams(predator_concentration=0.4)
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.4)

    def test_angle_from_derivative(self) -> None:
        """Test that angle feature is tanh of scaled predator concentration derivative."""
        module = SENSORY_MODULES[ModuleName.NOCICEPTION_TEMPORAL]
        params = BrainParams(predator_concentration=0.4, predator_dconcentration_dt=-0.2)
        features = module.to_classical(params)
        assert features[1] == pytest.approx(np.tanh(-0.2 * 50.0))

    def test_none_fields_produce_zeros(self) -> None:
        """Test that unset fields produce zero-valued features."""
        module = SENSORY_MODULES[ModuleName.NOCICEPTION_TEMPORAL]
        params = BrainParams()
        features = module.to_classical(params)
        assert features[0] == pytest.approx(0.0)
        assert features[1] == pytest.approx(0.0)


class TestThermotaxisTemporal:
    """Tests for the thermotaxis_temporal module."""

    def test_registered(self) -> None:
        """Test that thermotaxis_temporal is registered in SENSORY_MODULES."""
        assert ModuleName.THERMOTAXIS_TEMPORAL in SENSORY_MODULES

    def test_classical_dim_is_3(self) -> None:
        """Test that the classical feature dimension is 3."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_TEMPORAL]
        assert module.classical_dim == 3

    def test_strength_from_temp_deviation(self) -> None:
        """Test that strength is computed from temperature deviation."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_TEMPORAL]
        params = BrainParams(temperature=25.0, cultivation_temperature=20.0)
        features = module.to_classical(params)
        # deviation = (25 - 20) / 15 = 0.333
        assert features[0] == pytest.approx(5.0 / 15.0)

    def test_angle_from_temperature_ddt(self) -> None:
        """Test that angle feature is tanh of scaled temperature time derivative."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_TEMPORAL]
        params = BrainParams(temperature=20.0, cultivation_temperature=20.0, temperature_ddt=0.5)
        features = module.to_classical(params)
        assert features[1] == pytest.approx(np.tanh(0.5 * 50.0))

    def test_binary_equals_strength(self) -> None:
        """Test that the binary feature equals the strength feature."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_TEMPORAL]
        params = BrainParams(temperature=25.0, cultivation_temperature=20.0)
        features = module.to_classical(params)
        assert features[2] == pytest.approx(features[0])

    def test_disabled_returns_zeros(self) -> None:
        """Test that disabled thermotaxis returns all-zero features."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_TEMPORAL]
        params = BrainParams()
        features = module.to_classical(params)
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))


class TestSTAMModule:
    """Tests for the STAM sensory module."""

    def test_registered(self) -> None:
        """Test that STAM is registered in SENSORY_MODULES."""
        assert ModuleName.STAM in SENSORY_MODULES

    def test_classical_dim_is_9(self) -> None:
        """Test that the STAM classical feature dimension is 9."""
        module = SENSORY_MODULES[ModuleName.STAM]
        assert module.classical_dim == 9

    def test_classical_output_from_stam_state(self) -> None:
        """Test that classical output matches the provided STAM state values."""
        module = SENSORY_MODULES[ModuleName.STAM]
        stam_state = tuple(float(i) * 0.1 for i in range(9))
        params = BrainParams(stam_state=stam_state)
        features = module.to_classical(params)
        assert features.shape == (9,)
        for i in range(9):
            assert features[i] == pytest.approx(float(i) * 0.1)

    def test_classical_output_zeros_when_none(self) -> None:
        """Test that classical output is zeros when STAM state is not set."""
        module = SENSORY_MODULES[ModuleName.STAM]
        params = BrainParams()
        features = module.to_classical(params)
        assert features.shape == (9,)
        np.testing.assert_array_equal(features, np.zeros(9, dtype=np.float32))

    def test_quantum_output_shape(self) -> None:
        """Test that quantum output has shape (3,)."""
        module = SENSORY_MODULES[ModuleName.STAM]
        stam_state = tuple(float(i) * 0.1 for i in range(9))
        params = BrainParams(stam_state=stam_state)
        q = module.to_quantum(params)
        assert q.shape == (3,)

    def test_quantum_output_zeros_when_none(self) -> None:
        """Test that quantum output is zeros when STAM state is not set."""
        module = SENSORY_MODULES[ModuleName.STAM]
        params = BrainParams()
        q = module.to_quantum(params)
        assert q.shape == (3,)
        np.testing.assert_array_equal(q, np.zeros(3, dtype=np.float32))


class TestTemporalModuleIntegration:
    """Integration tests for temporal modules with extract_classical_features."""

    def test_extract_with_temporal_modules(self) -> None:
        """Test that extract_classical_features works with temporal modules."""
        params = BrainParams(
            food_concentration=0.6,
            food_dconcentration_dt=0.1,
            predator_concentration=0.3,
        )
        modules = [ModuleName.FOOD_CHEMOTAXIS_TEMPORAL, ModuleName.NOCICEPTION_TEMPORAL]
        features = extract_classical_features(params, modules)
        # 2 modules x 2 features each = 4
        assert features.shape == (4,)

    def test_extract_with_stam_module(self) -> None:
        """Test that extract_classical_features includes STAM features."""
        stam_state = tuple(float(i) * 0.1 for i in range(9))
        params = BrainParams(food_concentration=0.5, stam_state=stam_state)
        modules = [ModuleName.FOOD_CHEMOTAXIS_TEMPORAL, ModuleName.STAM]
        features = extract_classical_features(params, modules)
        # food_chemotaxis_temporal (2) + stam (9) = 11
        assert features.shape == (11,)

    def test_feature_dimension_with_temporal(self) -> None:
        """Test that total feature dimension sums correctly across temporal modules."""
        modules = [
            ModuleName.FOOD_CHEMOTAXIS_TEMPORAL,
            ModuleName.NOCICEPTION_TEMPORAL,
            ModuleName.THERMOTAXIS_TEMPORAL,
            ModuleName.STAM,
        ]
        dim = get_classical_feature_dimension(modules)
        # 2 + 2 + 3 + 9 = 16
        assert dim == 16
