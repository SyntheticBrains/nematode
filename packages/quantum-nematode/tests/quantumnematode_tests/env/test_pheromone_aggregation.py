"""Tests for aggregation pheromone infrastructure."""

from __future__ import annotations

import pytest
from quantumnematode.env.env import (
    DynamicForagingEnvironment,
    ForagingParams,
    HealthParams,
    PheromoneParams,
    PheromoneTypeConfig,
)
from quantumnematode.env.pheromone import PheromoneField, PheromoneSource, PheromoneType


class TestAggregationPheromoneType:
    """Tests for AGGREGATION enum value."""

    def test_aggregation_type_exists(self) -> None:
        """AGGREGATION is a valid PheromoneType."""
        assert PheromoneType.AGGREGATION == "aggregation"

    def test_all_three_types(self) -> None:
        """All three pheromone types coexist."""
        types = list(PheromoneType)
        assert len(types) == 3
        assert PheromoneType.FOOD_MARKING in types
        assert PheromoneType.ALARM in types
        assert PheromoneType.AGGREGATION in types


class TestAggregationPheromoneField:
    """Tests for aggregation-specific PheromoneField behavior."""

    def test_aggregation_source_creation(self) -> None:
        """Aggregation pheromone sources can be added to a field."""
        field = PheromoneField(
            spatial_decay_constant=10.0,
            temporal_half_life=10.0,
            max_sources=200,
        )
        source = PheromoneSource(
            position=(10, 10),
            pheromone_type=PheromoneType.AGGREGATION,
            strength=0.5,
            emission_step=0,
            emitter_id="agent_0",
        )
        field.add_source(source)
        assert field.num_sources == 1

    def test_continuous_emission_accumulates(self) -> None:
        """Multiple sources from continuous emission create detectable signal."""
        field = PheromoneField(
            spatial_decay_constant=10.0,
            temporal_half_life=10.0,
            max_sources=200,
        )
        # Simulate 5 agents emitting for 3 steps
        for step in range(3):
            for agent_idx in range(5):
                field.add_source(
                    PheromoneSource(
                        position=(10 + agent_idx, 10),
                        pheromone_type=PheromoneType.AGGREGATION,
                        strength=0.5,
                        emission_step=step,
                        emitter_id=f"agent_{agent_idx}",
                    ),
                )
        assert field.num_sources == 15
        conc = field.get_concentration((12, 10), current_step=2)
        assert conc > 0.0

    def test_short_half_life_decays_quickly(self) -> None:
        """Aggregation pheromone with short half-life decays fast."""
        field = PheromoneField(
            spatial_decay_constant=10.0,
            temporal_half_life=10.0,
            max_sources=200,
        )
        field.add_source(
            PheromoneSource(
                position=(10, 10),
                pheromone_type=PheromoneType.AGGREGATION,
                strength=1.0,
                emission_step=0,
                emitter_id="agent_0",
            ),
        )
        conc_fresh = field.get_concentration((10, 10), current_step=0)
        conc_aged = field.get_concentration((10, 10), current_step=10)
        # After one half-life, raw signal halves but tanh compresses nonlinearly.
        # tanh(1.0) ≈ 0.76, tanh(0.5) ≈ 0.46 — aged should be meaningfully less.
        assert conc_aged < conc_fresh * 0.7

    def test_gradient_points_toward_cluster(self) -> None:
        """Gradient from multiple sources points toward the cluster."""
        field = PheromoneField(
            spatial_decay_constant=10.0,
            temporal_half_life=50.0,
            max_sources=200,
        )
        # Place several sources at (15, 10)
        for i in range(5):
            field.add_source(
                PheromoneSource(
                    position=(15, 10),
                    pheromone_type=PheromoneType.AGGREGATION,
                    strength=0.5,
                    emission_step=0,
                    emitter_id=f"agent_{i}",
                ),
            )
        # Query gradient at (10, 10) — should point toward (15, 10) = positive x
        dx, dy = field.get_gradient((10, 10), current_step=0)
        assert dx > 0  # Points toward cluster (higher x)

    def test_max_sources_enforced(self) -> None:
        """max_sources limit prevents unbounded growth from continuous emission."""
        field = PheromoneField(
            spatial_decay_constant=10.0,
            temporal_half_life=10.0,
            max_sources=50,
        )
        for step in range(20):
            for agent_idx in range(5):
                field.add_source(
                    PheromoneSource(
                        position=(10, 10),
                        pheromone_type=PheromoneType.AGGREGATION,
                        strength=0.5,
                        emission_step=step,
                        emitter_id=f"agent_{agent_idx}",
                    ),
                )
        # 20 steps * 5 agents = 100 sources, but max is 50
        assert field.num_sources == 50


class TestAggregationEnvironmentIntegration:
    """Tests for aggregation pheromone environment integration."""

    def _make_env(
        self,
        *,
        aggregation: bool = True,
    ) -> DynamicForagingEnvironment:
        """Create test env with optional aggregation pheromone."""
        agg_config = (
            PheromoneTypeConfig(
                emission_strength=0.5,
                spatial_decay_constant=10.0,
                temporal_half_life=10.0,
                max_sources=200,
            )
            if aggregation
            else None
        )
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            seed=42,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            health=HealthParams(),
            pheromones=PheromoneParams(enabled=True, aggregation=agg_config),
        )

    def test_aggregation_field_created(self) -> None:
        """Aggregation field is created when config present."""
        env = self._make_env(aggregation=True)
        assert env.pheromone_field_aggregation is not None

    def test_aggregation_field_none_when_not_configured(self) -> None:
        """Aggregation field is None when aggregation config absent."""
        env = self._make_env(aggregation=False)
        assert env.pheromone_field_aggregation is None

    def test_emit_aggregation_pheromone(self) -> None:
        """Emission adds a source to the aggregation field."""
        env = self._make_env()
        env.emit_aggregation_pheromone((10, 10), current_step=0, emitter_id="agent_0")
        assert env.pheromone_field_aggregation is not None
        assert env.pheromone_field_aggregation.num_sources == 1

    def test_emit_noop_when_not_configured(self) -> None:
        """Emission is a no-op when aggregation not configured."""
        env = self._make_env(aggregation=False)
        env.emit_aggregation_pheromone((10, 10), current_step=0, emitter_id="agent_0")
        assert env.pheromone_field_aggregation is None

    def test_concentration_at_emission_site(self) -> None:
        """Concentration is positive at emission position."""
        env = self._make_env()
        env.emit_aggregation_pheromone((10, 10), current_step=0, emitter_id="agent_0")
        conc = env.get_pheromone_aggregation_concentration((10, 10), current_step=0)
        assert conc > 0.0

    def test_concentration_zero_when_not_configured(self) -> None:
        """Concentration returns 0 when aggregation not configured."""
        env = self._make_env(aggregation=False)
        conc = env.get_pheromone_aggregation_concentration((10, 10), current_step=0)
        assert conc == 0.0

    def test_gradient_none_when_not_configured(self) -> None:
        """Gradient returns None when aggregation not configured."""
        env = self._make_env(aggregation=False)
        result = env.get_pheromone_aggregation_gradient((10, 10), current_step=0)
        assert result is None

    def test_update_prunes_aggregation_field(self) -> None:
        """update_pheromone_fields prunes expired aggregation sources."""
        env = self._make_env()
        env.emit_aggregation_pheromone((10, 10), current_step=0, emitter_id="agent_0")
        assert env.pheromone_field_aggregation is not None
        assert env.pheromone_field_aggregation.num_sources == 1
        # Prune well past max age (half_life=10, max_age=50)
        env.update_pheromone_fields(current_step=100)
        assert env.pheromone_field_aggregation.num_sources == 0

    def test_copy_preserves_aggregation_field(self) -> None:
        """Environment copy creates a fresh aggregation field."""
        env = self._make_env()
        env_copy = env.copy()
        assert env_copy.pheromone_field_aggregation is not None
        # Fields are independent (fresh field on copy, no shared sources)
        assert env_copy.pheromone_field_aggregation.num_sources == 0
