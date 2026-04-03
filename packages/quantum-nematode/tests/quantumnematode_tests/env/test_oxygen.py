"""Tests for the oxygen field module."""

import numpy as np
import pytest
from quantumnematode.env.oxygen import (
    OxygenField,
    OxygenZone,
    OxygenZoneThresholds,
)


class TestOxygenField:
    """Test OxygenField class."""

    def test_base_oxygen_at_center(self):
        """Test that base oxygen is returned at grid center."""
        field = OxygenField(grid_size=100, base_oxygen=10.0)
        assert field.get_oxygen((50, 50)) == pytest.approx(10.0)

    def test_linear_gradient_east(self):
        """Test linear gradient increasing to the east (direction=0)."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_direction=0.0,
            gradient_strength=0.1,
        )
        center = field.get_oxygen((50, 50))
        east = field.get_oxygen((80, 50))
        west = field.get_oxygen((20, 50))
        assert east > center
        assert west < center

    def test_high_oxygen_spots(self):
        """Test that high oxygen spots increase O2 at their center."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_strength=0.0,
            high_oxygen_spots=[(75, 50, 8.0)],
            spot_decay_constant=5.0,
        )
        o2_at_spot = field.get_oxygen((75, 50))
        o2_at_distance = field.get_oxygen((50, 50))
        assert o2_at_spot > o2_at_distance

    def test_low_oxygen_spots(self):
        """Test that low oxygen spots decrease O2 at their center."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_strength=0.0,
            low_oxygen_spots=[(25, 25, 6.0)],
            spot_decay_constant=5.0,
        )
        o2_at_spot = field.get_oxygen((25, 25))
        o2_at_distance = field.get_oxygen((75, 75))
        assert o2_at_spot < o2_at_distance

    def test_value_clamping_max(self):
        """Test that oxygen values are clamped to 21.0 maximum."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=20.0,
            gradient_direction=0.0,
            gradient_strength=1.0,  # Extreme gradient
        )
        # Far east should be clamped to 21.0
        o2 = field.get_oxygen((99, 50))
        assert o2 <= 21.0

    def test_value_clamping_min(self):
        """Test that oxygen values are clamped to 0.0 minimum."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=2.0,
            gradient_strength=0.0,
            low_oxygen_spots=[(50, 50, 20.0)],
            spot_decay_constant=5.0,
        )
        o2 = field.get_oxygen((50, 50))
        assert o2 >= 0.0


class TestOxygenZones:
    """Test oxygen zone classification."""

    def test_comfort_zone(self):
        """Test comfort zone classification at midpoint."""
        field = OxygenField(grid_size=20)
        zone = field.get_zone(8.5)
        assert zone == OxygenZone.COMFORT

    def test_comfort_zone_boundaries(self):
        """Test comfort zone at lower and upper boundaries."""
        field = OxygenField(grid_size=20)
        zone_lower = field.get_zone(5.0)
        zone_upper = field.get_zone(12.0)
        assert zone_lower == OxygenZone.COMFORT
        assert zone_upper == OxygenZone.COMFORT

    def test_danger_hypoxia(self):
        """Test danger hypoxia zone classification."""
        field = OxygenField(grid_size=20)
        zone = field.get_zone(3.0)
        assert zone == OxygenZone.DANGER_HYPOXIA

    def test_danger_hyperoxia(self):
        """Test danger hyperoxia zone classification."""
        field = OxygenField(grid_size=20)
        zone = field.get_zone(15.0)
        assert zone == OxygenZone.DANGER_HYPEROXIA

    def test_lethal_hypoxia(self):
        """Test lethal hypoxia zone classification."""
        field = OxygenField(grid_size=20)
        zone = field.get_zone(1.0)
        assert zone == OxygenZone.LETHAL_HYPOXIA

    def test_lethal_hyperoxia(self):
        """Test lethal hyperoxia zone classification."""
        field = OxygenField(grid_size=20)
        zone = field.get_zone(19.0)
        assert zone == OxygenZone.LETHAL_HYPEROXIA

    def test_custom_thresholds(self):
        """Test that custom thresholds change zone boundaries."""
        field = OxygenField(grid_size=20)
        thresholds = OxygenZoneThresholds(
            lethal_hypoxia_upper=1.0,
            danger_hypoxia_upper=3.0,
            comfort_lower=3.0,
            comfort_upper=10.0,
            danger_hyperoxia_upper=15.0,
        )
        # 2.0 is danger_hypoxia with defaults, but comfort with custom thresholds
        # (since custom danger_hypoxia_upper=3.0, 2.0 < 3.0 → still danger_hypoxia)
        # Use a value that changes zone: 4.0 is danger_hypoxia with defaults (< 5.0),
        # but comfort with custom thresholds (>= 3.0 and <= 10.0)
        zone = field.get_zone(4.0, thresholds)
        assert zone == OxygenZone.COMFORT


class TestOxygenGradient:
    """Test oxygen gradient computation."""

    def test_gradient_direction(self):
        """Test gradient points east with eastward gradient field."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_direction=0.0,
            gradient_strength=0.1,
        )
        magnitude, direction = field.get_gradient_polar((50, 50))
        # Should point east (angle ~0)
        assert direction == pytest.approx(0.0, abs=0.1)
        assert magnitude > 0

    def test_gradient_magnitude(self):
        """Test gradient has positive magnitude in gradient field."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_direction=0.0,
            gradient_strength=0.1,
        )
        magnitude, _ = field.get_gradient_polar((50, 50))
        assert magnitude > 0

    def test_gradient_zero_at_uniform(self):
        """Test gradient is approximately zero with no gradient or spots."""
        field = OxygenField(
            grid_size=100,
            base_oxygen=10.0,
            gradient_strength=0.0,
        )
        magnitude, _ = field.get_gradient_polar((50, 50))
        assert magnitude == pytest.approx(0.0)
