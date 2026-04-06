"""Tests for pheromone field system."""

from __future__ import annotations

import math

import pytest
from quantumnematode.env.pheromone import (
    HALF_LIFE_PRUNE_FACTOR,
    PheromoneField,
    PheromoneSource,
    PheromoneType,
)


class TestPheromoneField:
    """Tests for PheromoneField core functionality."""

    def test_empty_field_returns_zero(self) -> None:
        """Empty field has zero concentration everywhere."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=50.0,
            max_sources=100,
        )
        assert field.get_concentration((10, 10), current_step=0) == 0.0
        assert field.num_sources == 0

    def test_single_source_at_origin(self) -> None:
        """Source at query position gives maximum concentration."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=50.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        conc = field.get_concentration((10, 10), current_step=0)
        assert conc > 0.5  # tanh(1.0) ≈ 0.76

    def test_spatial_decay_with_distance(self) -> None:
        """Concentration decreases with Manhattan distance."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=1000.0,  # Long half-life to isolate spatial decay
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        near = field.get_concentration((12, 10), current_step=0)  # distance=2
        far = field.get_concentration((20, 10), current_step=0)  # distance=10
        assert near > far > 0

    def test_temporal_decay_with_age(self) -> None:
        """Concentration decreases as source ages."""
        field = PheromoneField(
            spatial_decay_constant=1000.0,  # Large to isolate temporal decay
            temporal_half_life=50.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.ALARM,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        fresh = field.get_concentration((10, 10), current_step=0)
        aged = field.get_concentration((10, 10), current_step=50)  # 1 half-life
        very_old = field.get_concentration((10, 10), current_step=250)  # 5 half-lives
        assert fresh > aged > very_old > 0

    def test_half_life_halves_raw_contribution(self) -> None:
        """At one half-life, the raw temporal factor is 0.5."""
        field = PheromoneField(
            spatial_decay_constant=1000.0,
            temporal_half_life=50.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        # At step 50 (1 half-life), raw contribution ≈ 1.0 * exp(0) * 0.5 = 0.5
        # tanh(0.5) ≈ 0.462
        conc = field.get_concentration((10, 10), current_step=50)
        assert conc == pytest.approx(math.tanh(0.5), abs=0.01)

    def test_multiple_sources_superpose(self) -> None:
        """Multiple sources add their contributions."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=1000.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(8, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        single = field.get_concentration((10, 10), current_step=0)

        field.add_source(
            PheromoneSource(
                position=(12, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_1",
            ),
        )
        double = field.get_concentration((10, 10), current_step=0)
        assert double > single

    def test_gradient_points_toward_source(self) -> None:
        """Gradient vector points from low to high concentration."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=1000.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(20, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        dx, dy = field.get_gradient((10, 10), current_step=0)
        # Source is to the right (x=20), so dx should be positive
        assert dx > 0
        # Source is at same y, so dy should be near 0
        assert abs(dy) < abs(dx)

    def test_gradient_polar_magnitude_and_direction(self) -> None:
        """Polar gradient gives magnitude and direction toward source."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=1000.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(20, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        mag, direction = field.get_gradient_polar((10, 10), current_step=0)
        assert mag > 0
        # Direction should be ~0 radians (pointing right toward x=20)
        assert abs(direction) < 0.5

    def test_pruning_removes_expired_sources(self) -> None:
        """Sources older than max_age are pruned."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=10.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        assert field.num_sources == 1

        max_age = int(10.0 * HALF_LIFE_PRUNE_FACTOR)
        removed = field.prune(current_step=max_age + 1)
        assert removed == 1
        assert field.num_sources == 0

    def test_pruning_keeps_fresh_sources(self) -> None:
        """Fresh sources survive pruning."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=10.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=100,
                emitter_id="agent_0",
            ),
        )
        removed = field.prune(current_step=105)
        assert removed == 0
        assert field.num_sources == 1

    def test_max_sources_enforced(self) -> None:
        """Oldest sources removed when max_sources exceeded."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=1000.0,
            max_sources=3,
        )
        for i in range(5):
            field.add_source(
                PheromoneSource(
                    position=(i, 0),
                    pheromone_type=PheromoneType.FOOD_MARKING,
                    strength=1.0,
                    emission_step=i,
                    emitter_id=f"agent_{i}",
                ),
            )
        assert field.num_sources == 3
        # Oldest (step 0, 1) should be gone; newest (step 2, 3, 4) remain
        assert field.sources[0].emission_step == 2

    def test_clear_removes_all(self) -> None:
        """Clear removes all sources."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=50.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.FOOD_MARKING,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        field.clear()
        assert field.num_sources == 0

    def test_invalid_params_raise(self) -> None:
        """Invalid constructor params raise ValueError."""
        with pytest.raises(ValueError, match="spatial_decay_constant"):
            PheromoneField(spatial_decay_constant=0, temporal_half_life=50, max_sources=100)
        with pytest.raises(ValueError, match="temporal_half_life"):
            PheromoneField(spatial_decay_constant=8, temporal_half_life=0, max_sources=100)
        with pytest.raises(ValueError, match="max_sources"):
            PheromoneField(spatial_decay_constant=8, temporal_half_life=50, max_sources=0)

    def test_pheromone_types(self) -> None:
        """Both pheromone types work."""
        field = PheromoneField(
            spatial_decay_constant=8.0,
            temporal_half_life=50.0,
            max_sources=100,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.ALARM,
                strength=2.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        assert field.get_concentration((10, 10), current_step=0) > 0
