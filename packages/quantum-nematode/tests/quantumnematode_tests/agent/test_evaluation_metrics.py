"""Tests for Phase 4 evaluation metrics (territorial_index, alarm_response_rate)."""

from __future__ import annotations

import pytest
from quantumnematode.agent.multi_agent import (
    ALARM_RESPONSE_WINDOW,
    MultiAgentEpisodeResult,
    _compute_territorial_index,
)


class TestTerritorialIndex:
    """Tests for _compute_territorial_index()."""

    def test_empty_returns_zero(self) -> None:
        """No agents with food → 0.0."""
        assert _compute_territorial_index({}) == 0.0

    def test_single_agent_returns_zero(self) -> None:
        """Only one agent with food → 0.0 (no comparison possible)."""
        positions = {"agent_0": [(5, 5), (6, 6), (7, 7)]}
        assert _compute_territorial_index(positions) == 0.0

    def test_agents_no_food_returns_zero(self) -> None:
        """Multiple agents but none collected food → 0.0."""
        positions = {"agent_0": [], "agent_1": []}
        assert _compute_territorial_index(positions) == 0.0

    def test_identical_foraging_returns_zero(self) -> None:
        """All agents forage in the same tight spot → spreads equal → Gini = 0."""
        positions = {
            "agent_0": [(10, 10)],
            "agent_1": [(10, 10)],
        }
        # Both have spread = 0, Gini of [0, 0] = 0
        assert _compute_territorial_index(positions) == 0.0

    def test_different_spreads_positive(self) -> None:
        """One agent forages tightly, another ranges widely → positive Gini."""
        positions = {
            "agent_0": [(10, 10), (10, 10), (10, 10)],  # tight cluster, spread ≈ 0
            "agent_1": [(0, 0), (19, 19), (10, 10)],  # wide range, spread > 0
        }
        index = _compute_territorial_index(positions)
        assert index > 0.0

    def test_single_food_per_agent_zero_spread(self) -> None:
        """Each agent collected exactly 1 food → all spreads = 0 → Gini = 0."""
        positions = {
            "agent_0": [(5, 5)],
            "agent_1": [(15, 15)],
        }
        assert _compute_territorial_index(positions) == 0.0

    def test_mixed_food_counts(self) -> None:
        """Agent with no food excluded, agents with food compared."""
        positions = {
            "agent_0": [(5, 5), (6, 6)],
            "agent_1": [],  # no food — excluded
            "agent_2": [(15, 15), (15, 16)],
        }
        # Both have similar small spreads → low Gini
        index = _compute_territorial_index(positions)
        assert index >= 0.0
        assert index <= 1.0

    def test_result_bounded(self) -> None:
        """Index always in [0, 1]."""
        positions = {
            "agent_0": [(0, 0), (1, 0), (0, 1)],
            "agent_1": [(0, 0), (19, 19)],
            "agent_2": [(10, 10)],
        }
        index = _compute_territorial_index(positions)
        assert 0.0 <= index <= 1.0


class TestAlarmResponseConstants:
    """Tests for alarm response constants."""

    def test_alarm_response_window(self) -> None:
        """ALARM_RESPONSE_WINDOW is 5."""
        assert ALARM_RESPONSE_WINDOW == 5


class TestNewResultFields:
    """Tests for new fields on MultiAgentEpisodeResult."""

    def test_default_values(self) -> None:
        """New metric fields default to zero."""
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
        assert result.territorial_index == 0.0
        assert result.alarm_response_rate == 0.0

    def test_custom_values(self) -> None:
        """New metric fields accept custom values."""
        result = MultiAgentEpisodeResult(
            agent_results={},
            total_food_collected=0,
            per_agent_food={},
            food_competition_events=0,
            proximity_events=0,
            agents_alive_at_end=0,
            mean_agent_success=0.0,
            food_gini_coefficient=0.0,
            territorial_index=0.65,
            alarm_response_rate=0.4,
        )
        assert result.territorial_index == 0.65
        assert result.alarm_response_rate == pytest.approx(0.4)
