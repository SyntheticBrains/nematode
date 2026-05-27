"""Tests for the six biology-driven predator sensor modules.

Covers the two-channel split:

- ``predator_mechanosensation_{oracle,temporal,klinotaxis}`` — graded
  contact intensity + anterior/posterior/lateral zone (ASH/ALM/AVM/PLM).
- ``predator_chemosensation_{oracle,temporal,klinotaxis}`` — distal
  sulfolipid concentration + lateral gradient + dC/dt (ASH + ASI per
  Liu et al. 2018, *Nat. Commun.*).

For each module:

- output shape matches the declared classical_dim
- field-to-feature mapping is correct (intensity/concentration → strength;
  zone or derivative → angle; klinotaxis variants emit a third feature)
- agent-direction-relative orientation works as expected

Patterns mirror ``tests/quantumnematode_tests/brain/test_modules.py`` and
``test_klinotaxis_modules.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.modules import (
    SENSORY_MODULES,
    ModuleName,
)
from quantumnematode.env import Direction
from quantumnematode.env.env import ContactZone


def _make_params(**overrides: object) -> BrainParams:
    """Build a minimal BrainParams with sensible defaults for predator tests."""
    defaults: dict[str, object] = {
        "agent_direction": Direction.UP,
    }
    defaults.update(overrides)
    return BrainParams(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Mechanosensation channel
# =============================================================================


class TestPredatorMechanosensationOracle:
    module = SENSORY_MODULES[ModuleName.PREDATOR_MECHANOSENSATION_ORACLE]

    def test_oracle_classical_dim_is_two(self) -> None:
        assert self.module.classical_dim == 2

    def test_oracle_extracts_intensity_as_strength(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.6,
            predator_contact_zone=ContactZone.ANTERIOR,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.6)

    def test_oracle_anterior_zone_maps_to_positive_angle(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.5,
            predator_contact_zone=ContactZone.ANTERIOR,
        )
        features = self.module.extract(params)
        assert features.angle == pytest.approx(1.0)

    def test_oracle_posterior_zone_maps_to_negative_angle(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.5,
            predator_contact_zone=ContactZone.POSTERIOR,
        )
        features = self.module.extract(params)
        assert features.angle == pytest.approx(-1.0)

    def test_oracle_lateral_and_none_zone_map_to_zero_angle(self) -> None:
        for zone in (ContactZone.LATERAL, ContactZone.NONE):
            params = _make_params(
                predator_contact_intensity=0.5,
                predator_contact_zone=zone,
            )
            features = self.module.extract(params)
            assert features.angle == pytest.approx(0.0), (
                f"zone={zone}: expected angle=0.0, got {features.angle}"
            )

    def test_oracle_missing_fields_default_to_zero(self) -> None:
        params = _make_params()  # No predator-related fields populated.
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.0)
        assert features.angle == pytest.approx(0.0)


class TestPredatorMechanosensationTemporal:
    module = SENSORY_MODULES[ModuleName.PREDATOR_MECHANOSENSATION_TEMPORAL]

    def test_temporal_classical_dim_is_two(self) -> None:
        assert self.module.classical_dim == 2

    def test_temporal_extracts_intensity_and_derivative(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.7,
            predator_mechano_dintensity_dt=0.01,
            derivative_scale=50.0,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.7)
        # tanh(0.01 * 50) = tanh(0.5) ≈ 0.462
        assert features.angle == pytest.approx(np.tanh(0.5))

    def test_temporal_zero_derivative_under_steady_intensity(self) -> None:
        # Habituated case: graded intensity persists but derivative is zero.
        params = _make_params(
            predator_contact_intensity=0.7,
            predator_mechano_dintensity_dt=0.0,
        )
        features = self.module.extract(params)
        assert features.angle == pytest.approx(0.0)


class TestPredatorMechanosensationKlinotaxis:
    module = SENSORY_MODULES[ModuleName.PREDATOR_MECHANOSENSATION_KLINOTAXIS]

    def test_klinotaxis_classical_dim_is_three(self) -> None:
        assert self.module.classical_dim == 3

    def test_klinotaxis_extracts_intensity_zone_and_derivative(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.8,
            predator_contact_zone=ContactZone.ANTERIOR,
            predator_mechano_dintensity_dt=0.02,
            derivative_scale=50.0,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.8)
        assert features.angle == pytest.approx(1.0)  # ANTERIOR → +1.0
        # tanh(0.02 * 50) = tanh(1.0) ≈ 0.762
        assert features.binary == pytest.approx(np.tanh(1.0))


# =============================================================================
# Chemosensation channel
# =============================================================================


class TestPredatorChemosensationOracle:
    module = SENSORY_MODULES[ModuleName.PREDATOR_CHEMOSENSATION_ORACLE]

    def test_oracle_classical_dim_is_two(self) -> None:
        assert self.module.classical_dim == 2

    def test_oracle_extracts_distal_concentration_as_strength(self) -> None:
        params = _make_params(
            predator_distal_concentration=0.42,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.42)

    def test_oracle_missing_distal_field_returns_zero_strength(self) -> None:
        params = _make_params()  # No predator-related fields populated.
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.0)


class TestPredatorChemosensationTemporal:
    module = SENSORY_MODULES[ModuleName.PREDATOR_CHEMOSENSATION_TEMPORAL]

    def test_temporal_classical_dim_is_two(self) -> None:
        assert self.module.classical_dim == 2

    def test_temporal_extracts_distal_concentration_and_derivative(self) -> None:
        params = _make_params(
            predator_distal_concentration=0.55,
            predator_distal_dconcentration_dt=0.01,
            derivative_scale=50.0,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.55)
        assert features.angle == pytest.approx(np.tanh(0.5))


class TestPredatorChemosensationKlinotaxis:
    module = SENSORY_MODULES[ModuleName.PREDATOR_CHEMOSENSATION_KLINOTAXIS]

    def test_klinotaxis_classical_dim_is_three(self) -> None:
        assert self.module.classical_dim == 3

    def test_klinotaxis_extracts_concentration_lateral_and_derivative(self) -> None:
        params = _make_params(
            predator_distal_concentration=0.65,
            predator_lateral_gradient=0.02,  # right - left from head-sweep
            predator_distal_dconcentration_dt=0.005,
            lateral_scale=50.0,
            derivative_scale=50.0,
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.65)
        # angle = tanh(0.02 * 50) = tanh(1.0) ≈ 0.762
        assert features.angle == pytest.approx(np.tanh(1.0))
        # binary = tanh(0.005 * 50) = tanh(0.25) ≈ 0.245
        assert features.binary == pytest.approx(np.tanh(0.25))


# =============================================================================
# Ablation modules — sparse-signal mechanosensation variant
# =============================================================================


class TestPredatorMechanosensationKlinotaxisSparseFix:
    module = SENSORY_MODULES[ModuleName.PREDATOR_MECHANOSENSATION_KLINOTAXIS_SPARSE_FIX]

    def test_classical_dim_is_three(self) -> None:
        assert self.module.classical_dim == 3

    def test_in_contact_emits_intensity_as_strength(self) -> None:
        """When in contact (zone != NONE), behaves identically to the canonical channel."""
        params = _make_params(
            predator_contact_intensity=0.8,
            predator_contact_zone=ContactZone.ANTERIOR,
            predator_distal_concentration=0.3,  # Ignored when in contact.
            predator_mechano_dintensity_dt=0.02,
            derivative_scale=50.0,
        )
        features = self.module.extract(params)
        # In-contact: strength = predator_contact_intensity (NOT distal).
        assert features.strength == pytest.approx(0.8)
        assert features.angle == pytest.approx(1.0)  # ANTERIOR → +1.0
        assert features.binary == pytest.approx(np.tanh(1.0))

    def test_off_contact_substitutes_distal_concentration_for_strength(self) -> None:
        """When ContactZone.NONE, strength falls back to distal concentration."""
        params = _make_params(
            predator_contact_intensity=0.0,  # Off-contact.
            predator_contact_zone=ContactZone.NONE,
            predator_distal_concentration=0.42,  # Should appear as strength.
            predator_mechano_dintensity_dt=0.0,
        )
        features = self.module.extract(params)
        # Off-contact: strength = distal concentration (canonical channel would emit 0.0).
        assert features.strength == pytest.approx(0.42)
        # Angle still encodes the zone; NONE → 0.0 (matches canonical).
        assert features.angle == pytest.approx(0.0)

    def test_off_contact_with_zero_distal_emits_zero_strength(self) -> None:
        """When neither in-contact nor near a predator, strength is 0 (matches canonical)."""
        params = _make_params()  # All fields default to None / 0.
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.0)
        assert features.angle == pytest.approx(0.0)

    def test_posterior_contact_emits_intensity_not_distal(self) -> None:
        """Contact zone POSTERIOR also routes to canonical intensity."""
        params = _make_params(
            predator_contact_intensity=0.5,
            predator_contact_zone=ContactZone.POSTERIOR,
            predator_distal_concentration=0.9,  # Ignored — in contact.
        )
        features = self.module.extract(params)
        assert features.strength == pytest.approx(0.5)
        assert features.angle == pytest.approx(-1.0)  # POSTERIOR → -1.0


# =============================================================================
# Ablation modules — composite single-channel predator biology
# =============================================================================


class TestPredatorBiologyKlinotaxisComposite:
    module = SENSORY_MODULES[ModuleName.PREDATOR_BIOLOGY_KLINOTAXIS]

    def test_classical_dim_is_four(self) -> None:
        """Composite collapses the canonical 3+3=6 to a single 4-dim block."""
        assert self.module.classical_dim == 4

    def test_to_classical_emits_all_four_fields(self) -> None:
        params = _make_params(
            predator_contact_intensity=0.8,
            predator_contact_zone=ContactZone.ANTERIOR,
            predator_distal_concentration=0.42,
            predator_distal_dconcentration_dt=0.02,
            derivative_scale=50.0,
        )
        features = self.module.to_classical(params)
        assert features.shape == (4,)
        assert features[0] == pytest.approx(0.8)  # intensity
        assert features[1] == pytest.approx(1.0)  # zone_as_angle (ANTERIOR)
        assert features[2] == pytest.approx(0.42)  # distal_concentration
        # tanh(0.02 * 50) = tanh(1.0)
        assert features[3] == pytest.approx(np.tanh(1.0))

    def test_no_predator_signal_emits_zero_vector(self) -> None:
        params = _make_params()  # All predator fields default to None.
        features = self.module.to_classical(params)
        assert features.shape == (4,)
        np.testing.assert_array_equal(features, np.zeros(4, dtype=np.float32))

    def test_off_contact_with_distal_signal_emits_only_distal_components(self) -> None:
        """Composite distinguishes in-contact from near-but-not-in-contact."""
        params = _make_params(
            predator_contact_intensity=0.0,
            predator_contact_zone=ContactZone.NONE,
            predator_distal_concentration=0.55,
            predator_distal_dconcentration_dt=0.0,
        )
        features = self.module.to_classical(params)
        assert features[0] == pytest.approx(0.0)  # intensity off
        assert features[1] == pytest.approx(0.0)  # zone NONE → 0
        assert features[2] == pytest.approx(0.55)  # distal_concentration on
        assert features[3] == pytest.approx(0.0)  # no derivative

    def test_to_quantum_compresses_to_three_floats(self) -> None:
        """Quantum encoding drops the derivative; emits 3-float gate angles."""
        params = _make_params(
            predator_contact_intensity=0.5,
            predator_contact_zone=ContactZone.POSTERIOR,
            predator_distal_concentration=0.8,
            predator_distal_dconcentration_dt=0.02,  # Dropped by to_quantum.
        )
        quantum = self.module.to_quantum(params)
        assert quantum.shape == (3,)
        # Rescaled [0,1] → [-π/2, π/2]:
        assert quantum[0] == pytest.approx(0.5 * np.pi - np.pi / 2)  # intensity
        assert quantum[1] == pytest.approx(0.8 * np.pi - np.pi / 2)  # distal
        # zone POSTERIOR = -1.0 → -π/2
        assert quantum[2] == pytest.approx(-1.0 * np.pi / 2)


# =============================================================================
# Registry presence
# =============================================================================


class TestRegistryPresence:
    """All eight predator-sensor ModuleName entries are registered in SENSORY_MODULES."""

    @pytest.mark.parametrize(
        "module_name",
        [
            ModuleName.PREDATOR_MECHANOSENSATION_ORACLE,
            ModuleName.PREDATOR_MECHANOSENSATION_TEMPORAL,
            ModuleName.PREDATOR_MECHANOSENSATION_KLINOTAXIS,
            ModuleName.PREDATOR_MECHANOSENSATION_KLINOTAXIS_SPARSE_FIX,
            ModuleName.PREDATOR_CHEMOSENSATION_ORACLE,
            ModuleName.PREDATOR_CHEMOSENSATION_TEMPORAL,
            ModuleName.PREDATOR_CHEMOSENSATION_KLINOTAXIS,
            ModuleName.PREDATOR_BIOLOGY_KLINOTAXIS,
        ],
    )
    def test_module_registered(self, module_name: ModuleName) -> None:
        assert module_name in SENSORY_MODULES
        assert SENSORY_MODULES[module_name].name == module_name
