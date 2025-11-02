"""Tests for the satiety management system."""

import pytest
from quantumnematode.agent import SatietyConfig
from quantumnematode.satiety import SatietyManager


class TestSatietyManagerInitialization:
    """Test satiety manager initialization."""

    def test_initialize_with_default_config(self):
        """Test initialization with default config."""
        config = SatietyConfig()
        manager = SatietyManager(config)

        assert manager.max_satiety == config.initial_satiety
        assert manager.decay_rate == config.satiety_decay_rate
        assert manager.current_satiety == config.initial_satiety

    def test_initialize_with_custom_config(self):
        """Test initialization with custom config values."""
        config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=0.5,
        )
        manager = SatietyManager(config)

        assert manager.max_satiety == 100.0
        assert manager.decay_rate == 0.5
        assert manager.current_satiety == 100.0


class TestSatietyDecay:
    """Test satiety decay mechanics."""

    def test_decay_reduces_satiety(self):
        """Test that decay reduces satiety by decay_rate."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=1.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety()

        assert new_satiety == 99.0
        assert manager.current_satiety == 99.0

    def test_decay_multiple_times(self):
        """Test multiple decay operations."""
        config = SatietyConfig(initial_satiety=10.0, satiety_decay_rate=2.0)
        manager = SatietyManager(config)

        manager.decay_satiety()
        assert manager.current_satiety == 8.0

        manager.decay_satiety()
        assert manager.current_satiety == 6.0

        manager.decay_satiety()
        assert manager.current_satiety == 4.0

    def test_decay_cannot_go_below_zero(self):
        """Test that satiety is clamped at 0.0."""
        config = SatietyConfig(initial_satiety=1.0, satiety_decay_rate=5.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety()

        assert new_satiety == 0.0
        assert manager.current_satiety == 0.0

    def test_decay_at_zero_stays_zero(self):
        """Test that decay at zero satiety stays at zero."""
        config = SatietyConfig(initial_satiety=0.0, satiety_decay_rate=1.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety()

        assert new_satiety == 0.0
        assert manager.current_satiety == 0.0


class TestSatietyRestoration:
    """Test satiety restoration mechanics."""

    def test_restore_increases_satiety(self):
        """Test that restore increases satiety by given amount."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 50.0  # Manually set to test restoration

        new_satiety = manager.restore_satiety(25.0)

        assert new_satiety == 75.0
        assert manager.current_satiety == 75.0

    def test_restore_to_full_satiety(self):
        """Test restoring to maximum satiety."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 10.0

        new_satiety = manager.restore_satiety(100.0)

        assert new_satiety == 100.0
        assert manager.current_satiety == 100.0

    def test_restore_cannot_exceed_max(self):
        """Test that satiety cannot exceed max_satiety."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 95.0

        new_satiety = manager.restore_satiety(10.0)

        assert new_satiety == 100.0
        assert manager.current_satiety == 100.0

    def test_restore_from_zero(self):
        """Test restoring satiety from starvation."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 0.0

        new_satiety = manager.restore_satiety(50.0)

        assert new_satiety == 50.0
        assert manager.current_satiety == 50.0


class TestStarvationDetection:
    """Test starvation detection."""

    def test_not_starved_at_max(self):
        """Test that agent is not starved at maximum satiety."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)

        assert not manager.is_starved()

    def test_not_starved_above_zero(self):
        """Test that agent is not starved above zero."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 0.1

        assert not manager.is_starved()

    def test_starved_at_zero(self):
        """Test that agent is starved at exactly zero."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 0.0

        assert manager.is_starved()

    def test_starved_after_excessive_decay(self):
        """Test starvation after decay reduces satiety to zero."""
        config = SatietyConfig(initial_satiety=5.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        manager.decay_satiety()

        assert manager.is_starved()


class TestSatietyReset:
    """Test satiety reset functionality."""

    def test_reset_restores_to_initial(self):
        """Test that reset restores satiety to initial level."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 25.0

        manager.reset()

        assert manager.current_satiety == 100.0

    def test_reset_from_zero(self):
        """Test reset from starvation."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)
        manager._current_satiety = 0.0

        manager.reset()

        assert manager.current_satiety == 100.0
        assert not manager.is_starved()


class TestSatietyPropertyAccess:
    """Test read-only property access."""

    def test_current_satiety_readonly(self):
        """Test that current_satiety cannot be set directly."""
        config = SatietyConfig(initial_satiety=100.0)
        manager = SatietyManager(config)

        # Attempting to set the property should raise AttributeError
        with pytest.raises(AttributeError):
            manager.current_satiety = 50.0  # type: ignore[misc]

    def test_current_satiety_reflects_internal_state(self):
        """Test that property reflects internal state changes."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        assert manager.current_satiety == 100.0

        manager.decay_satiety()
        assert manager.current_satiety == 90.0

        manager.restore_satiety(5.0)
        assert manager.current_satiety == 95.0
