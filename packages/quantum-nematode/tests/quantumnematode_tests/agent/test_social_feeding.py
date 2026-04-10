"""Tests for social feeding mechanics (satiety decay reduction near conspecifics)."""

from __future__ import annotations

import pytest
from quantumnematode.agent import SatietyConfig
from quantumnematode.agent.satiety import SatietyManager
from quantumnematode.env.env import SocialFeedingParams


class TestSatietyDecayMultiplier:
    """Tests for SatietyManager.decay_satiety() multiplier parameter."""

    def test_default_multiplier_unchanged(self) -> None:
        """Default multiplier (1.0) preserves original decay behavior."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=1.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety()
        assert new_satiety == 99.0

    def test_explicit_multiplier_one(self) -> None:
        """Explicit multiplier=1.0 is same as no argument."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=1.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety(multiplier=1.0)
        assert new_satiety == 99.0

    def test_reduced_decay_with_social_feeding(self) -> None:
        """Multiplier < 1.0 reduces decay (social feeding near conspecifics)."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety(multiplier=0.7)
        assert new_satiety == pytest.approx(93.0)  # 100 - 10 * 0.7 = 93

    def test_increased_decay_with_crowding_penalty(self) -> None:
        """Multiplier > 1.0 increases decay (solitary phenotype crowding penalty)."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety(multiplier=1.5)
        assert new_satiety == pytest.approx(85.0)  # 100 - 10 * 1.5 = 85

    def test_zero_multiplier_no_decay(self) -> None:
        """Multiplier=0.0 prevents all decay."""
        config = SatietyConfig(initial_satiety=100.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety(multiplier=0.0)
        assert new_satiety == 100.0

    def test_multiplier_cannot_go_below_zero(self) -> None:
        """Decay with multiplier still clamps at zero."""
        config = SatietyConfig(initial_satiety=5.0, satiety_decay_rate=10.0)
        manager = SatietyManager(config)

        new_satiety = manager.decay_satiety(multiplier=0.7)
        assert new_satiety == 0.0  # 5 - 10 * 0.7 = -2 -> clamped to 0

    def test_multiple_reduced_decay_steps(self) -> None:
        """Multiple steps with reduced decay extend survival."""
        config = SatietyConfig(initial_satiety=200.0, satiety_decay_rate=1.0)
        normal = SatietyManager(config)
        social = SatietyManager(SatietyConfig(initial_satiety=200.0, satiety_decay_rate=1.0))

        for _ in range(200):
            normal.decay_satiety(multiplier=1.0)
            social.decay_satiety(multiplier=0.7)

        assert normal.current_satiety == 0.0  # 200 - 200*1.0 = 0
        assert social.current_satiety == pytest.approx(60.0)  # 200 - 200*0.7 = 60


class TestSocialFeedingParams:
    """Tests for SocialFeedingParams dataclass."""

    def test_defaults(self) -> None:
        """Default params have social feeding disabled."""
        params = SocialFeedingParams()
        assert params.enabled is False
        assert params.decay_reduction == 0.7
        assert params.solitary_decay == 1.0

    def test_custom_values(self) -> None:
        """Custom params are stored correctly."""
        params = SocialFeedingParams(
            enabled=True,
            decay_reduction=0.5,
            solitary_decay=1.3,
        )
        assert params.enabled is True
        assert params.decay_reduction == 0.5
        assert params.solitary_decay == 1.3
