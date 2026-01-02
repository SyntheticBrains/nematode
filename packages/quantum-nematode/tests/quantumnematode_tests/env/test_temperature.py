"""Tests for the temperature field module."""

import numpy as np
import pytest
from quantumnematode.env.temperature import (
    TemperatureField,
    TemperatureZone,
    TemperatureZoneThresholds,
)


class TestTemperatureField:
    """Test TemperatureField class."""

    def test_base_temperature(self):
        """Test that base temperature is returned at origin with no gradient."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_strength=0.0,
        )
        # With no gradient, all positions should have base temperature
        assert field.get_temperature((0, 0)) == pytest.approx(20.0)
        assert field.get_temperature((10, 10)) == pytest.approx(20.0)

    def test_linear_gradient_rightward(self):
        """Test linear gradient increasing to the right."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_direction=0.0,  # Increases to the right
            gradient_strength=0.5,  # 0.5°C per cell
        )
        # Temperature should increase with x
        temp_at_0 = field.get_temperature((0, 0))
        temp_at_10 = field.get_temperature((10, 0))
        temp_at_20 = field.get_temperature((19, 0))

        assert temp_at_10 > temp_at_0
        assert temp_at_20 > temp_at_10
        assert temp_at_10 - temp_at_0 == pytest.approx(5.0)  # 10 cells * 0.5°C

    def test_linear_gradient_upward(self):
        """Test linear gradient increasing upward (positive y)."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_direction=np.pi / 2,  # Increases upward
            gradient_strength=1.0,
        )
        # Temperature should increase with y
        assert field.get_temperature((0, 10)) > field.get_temperature((0, 0))
        assert field.get_temperature((0, 10)) - field.get_temperature((0, 0)) == pytest.approx(10.0)

    def test_hot_spot(self):
        """Test hot spot contribution."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_strength=0.0,
            hot_spots=[(10, 10, 5.0)],  # +5°C at center
            spot_decay_constant=5.0,
        )
        # At hot spot center
        temp_at_center = field.get_temperature((10, 10))
        assert temp_at_center == pytest.approx(25.0)  # 20 + 5

        # Away from hot spot (should be lower but above base due to decay)
        temp_away = field.get_temperature((15, 10))
        assert 20.0 < temp_away < 25.0

    def test_cold_spot(self):
        """Test cold spot contribution."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_strength=0.0,
            cold_spots=[(10, 10, 5.0)],  # -5°C at center
            spot_decay_constant=5.0,
        )
        # At cold spot center
        temp_at_center = field.get_temperature((10, 10))
        assert temp_at_center == pytest.approx(15.0)  # 20 - 5

        # Away from cold spot (should be higher but below base due to decay)
        temp_away = field.get_temperature((15, 10))
        assert 15.0 < temp_away < 20.0


class TestTemperatureGradient:
    """Test temperature gradient computation."""

    def test_gradient_direction_rightward(self):
        """Test gradient points toward increasing temperature (right)."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_direction=0.0,  # Increases to the right
            gradient_strength=1.0,
        )
        magnitude, direction = field.get_gradient_polar((10, 10))

        # Should point to the right (angle 0)
        assert direction == pytest.approx(0.0, abs=0.1)
        assert magnitude > 0

    def test_gradient_magnitude_with_strength(self):
        """Test gradient magnitude scales with gradient strength."""
        field_weak = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_direction=0.0,
            gradient_strength=0.5,
        )
        field_strong = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_direction=0.0,
            gradient_strength=2.0,
        )
        mag_weak, _ = field_weak.get_gradient_polar((10, 10))
        mag_strong, _ = field_strong.get_gradient_polar((10, 10))

        assert mag_strong > mag_weak

    def test_gradient_no_field(self):
        """Test gradient is zero with no temperature variation."""
        field = TemperatureField(
            grid_size=20,
            base_temperature=20.0,
            gradient_strength=0.0,
        )
        magnitude, direction = field.get_gradient_polar((10, 10))

        assert magnitude == pytest.approx(0.0)


class TestTemperatureZones:
    """Test temperature zone classification."""

    def test_comfort_zone(self):
        """Test comfort zone classification."""
        field = TemperatureField(grid_size=20)
        thresholds = TemperatureZoneThresholds(
            comfort_delta=5.0,
            discomfort_delta=10.0,
            danger_delta=15.0,
        )

        # At cultivation temperature
        zone = field.get_zone(20.0, 20.0, thresholds)
        assert zone == TemperatureZone.COMFORT

        # Within comfort range
        zone = field.get_zone(24.0, 20.0, thresholds)
        assert zone == TemperatureZone.COMFORT

        zone = field.get_zone(16.0, 20.0, thresholds)
        assert zone == TemperatureZone.COMFORT

    def test_discomfort_zones(self):
        """Test discomfort zone classification."""
        field = TemperatureField(grid_size=20)
        thresholds = TemperatureZoneThresholds()

        # Hot discomfort (Tc + 5 to Tc + 10)
        zone = field.get_zone(27.0, 20.0, thresholds)
        assert zone == TemperatureZone.DISCOMFORT_HOT

        # Cold discomfort (Tc - 10 to Tc - 5)
        zone = field.get_zone(12.0, 20.0, thresholds)
        assert zone == TemperatureZone.DISCOMFORT_COLD

    def test_danger_zones(self):
        """Test danger zone classification."""
        field = TemperatureField(grid_size=20)
        thresholds = TemperatureZoneThresholds()

        # Hot danger (Tc + 10 to Tc + 15)
        zone = field.get_zone(32.0, 20.0, thresholds)
        assert zone == TemperatureZone.DANGER_HOT

        # Cold danger (Tc - 15 to Tc - 10)
        zone = field.get_zone(7.0, 20.0, thresholds)
        assert zone == TemperatureZone.DANGER_COLD

    def test_lethal_zones(self):
        """Test lethal zone classification."""
        field = TemperatureField(grid_size=20)
        thresholds = TemperatureZoneThresholds()

        # Lethal hot (> Tc + 15)
        zone = field.get_zone(40.0, 20.0, thresholds)
        assert zone == TemperatureZone.LETHAL_HOT

        # Lethal cold (< Tc - 15)
        zone = field.get_zone(2.0, 20.0, thresholds)
        assert zone == TemperatureZone.LETHAL_COLD
