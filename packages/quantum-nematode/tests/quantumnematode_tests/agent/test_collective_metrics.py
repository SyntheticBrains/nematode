"""Tests for collective behavior metrics (aggregation index, alarm evasion, food sharing)."""

from __future__ import annotations

import pytest
from quantumnematode.agent.multi_agent import (
    ALARM_EVASION_THRESHOLD,
    FOOD_SHARING_LOOKBACK_STEPS,
    _compute_aggregation_index,
)


class TestAggregationIndex:
    """Tests for _compute_aggregation_index()."""

    def test_single_agent_returns_zero(self) -> None:
        """Aggregation index is 0 with fewer than 2 agents."""
        assert _compute_aggregation_index([(10, 10)], grid_size=20) == 0.0

    def test_empty_returns_zero(self) -> None:
        """Aggregation index is 0 with no agents."""
        assert _compute_aggregation_index([], grid_size=20) == 0.0

    def test_colocated_agents_return_one(self) -> None:
        """Agents at the same position have aggregation index 1.0."""
        positions = [(10, 10), (10, 10), (10, 10)]
        assert _compute_aggregation_index(positions, grid_size=20) == 1.0

    def test_maximally_dispersed(self) -> None:
        """Agents at opposite corners have low aggregation index."""
        positions = [(0, 0), (19, 19)]
        index = _compute_aggregation_index(positions, grid_size=20)
        # Manhattan distance = 38, max = 2*(20-1) = 38
        # proximity = 1 - 38/38 = 0
        assert index == 0.0

    def test_moderate_proximity(self) -> None:
        """Agents at moderate distance have intermediate index."""
        positions = [(10, 10), (15, 10)]
        index = _compute_aggregation_index(positions, grid_size=20)
        # Manhattan dist = 5, max = 38, proximity = 1 - 5/38 ≈ 0.87
        assert 0.5 < index < 1.0

    def test_multiple_agents_averaged(self) -> None:
        """Index is mean proximity across all pairs."""
        # 3 agents: (0,0), (0,10), (10,0)
        # Distances: 10, 10, 20. Max = 38.
        # Proximities: 1-10/38, 1-10/38, 1-20/38
        positions = [(0, 0), (0, 10), (10, 0)]
        index = _compute_aggregation_index(positions, grid_size=20)
        expected = ((1 - 10 / 38) + (1 - 10 / 38) + (1 - 20 / 38)) / 3
        assert index == pytest.approx(expected)

    def test_grid_size_one_returns_one(self) -> None:
        """Grid size 1 (max_dist=0) returns 1.0 for any 2+ agents."""
        positions = [(0, 0), (0, 0)]
        assert _compute_aggregation_index(positions, grid_size=1) == 1.0


class TestAlarmEvasionConstants:
    """Tests for alarm evasion threshold constant."""

    def test_threshold_value(self) -> None:
        """ALARM_EVASION_THRESHOLD is 0.1."""
        assert ALARM_EVASION_THRESHOLD == 0.1

    def test_lookback_value(self) -> None:
        """FOOD_SHARING_LOOKBACK_STEPS is 20."""
        assert FOOD_SHARING_LOOKBACK_STEPS == 20


class TestMultiAgentEpisodeResultFields:
    """Tests for new collective metric fields on MultiAgentEpisodeResult."""

    def test_default_values(self) -> None:
        """New metric fields default to zero."""
        from quantumnematode.agent.multi_agent import MultiAgentEpisodeResult

        result = MultiAgentEpisodeResult(
            agent_results={},
            total_food_collected=0,
            per_agent_food={},
            food_competition_events=0,
            proximity_events=0,
            agents_alive_at_end=0,
            mean_agent_success=0.0,
            food_gini_coefficient=0.0,
        )
        assert result.social_feeding_events == 0
        assert result.aggregation_index == 0.0
        assert result.alarm_evasion_events == 0
        assert result.food_sharing_events == 0

    def test_custom_values(self) -> None:
        """New metric fields accept custom values."""
        from quantumnematode.agent.multi_agent import MultiAgentEpisodeResult

        result = MultiAgentEpisodeResult(
            agent_results={},
            total_food_collected=10,
            per_agent_food={},
            food_competition_events=2,
            proximity_events=50,
            agents_alive_at_end=3,
            mean_agent_success=0.6,
            food_gini_coefficient=0.3,
            social_feeding_events=100,
            aggregation_index=0.75,
            alarm_evasion_events=5,
            food_sharing_events=3,
        )
        assert result.social_feeding_events == 100
        assert result.aggregation_index == 0.75
        assert result.alarm_evasion_events == 5
        assert result.food_sharing_events == 3


class TestAgentPhenotypeValidation:
    """Tests for agent phenotype validation in MultiAgentSimulation."""

    def test_invalid_phenotype_rejected(self) -> None:
        """Invalid phenotype values raise ValueError."""
        from quantumnematode.agent.agent import QuantumNematodeAgent, SatietyConfig
        from quantumnematode.agent.multi_agent import MultiAgentSimulation
        from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
        from quantumnematode.brain.modules import ModuleName
        from quantumnematode.env import DynamicForagingEnvironment, ForagingParams, HealthParams

        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            seed=42,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            health=HealthParams(),
        )
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            num_hidden_layers=1,
        )
        brain = MLPPPOBrain(config=config, num_actions=4)
        env.add_agent("agent_0")
        agent = QuantumNematodeAgent(
            brain=brain, env=env, agent_id="agent_0",
            satiety_config=SatietyConfig(initial_satiety=200.0),
        )

        with pytest.raises(ValueError, match="invalid social_phenotype"):
            MultiAgentSimulation(
                env=env,
                agents=[agent],
                agent_phenotypes={"agent_0": "aggressive"},
            )
