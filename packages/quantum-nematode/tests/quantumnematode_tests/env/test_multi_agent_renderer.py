"""Tests for multi-agent rendering support."""

from __future__ import annotations

import pytest
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams


class TestGetViewportBoundsFor:
    """Tests for get_viewport_bounds_for() with multiple agents."""

    @pytest.fixture
    def env(self) -> DynamicForagingEnvironment:
        """Create a multi-agent environment."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            seed=42,
        )
        env.add_agent("agent_0", position=(5, 5), max_body_length=3)
        env.add_agent("agent_1", position=(20, 20), max_body_length=3)
        return env

    def test_viewport_centered_on_specified_agent(self, env: DynamicForagingEnvironment) -> None:
        """Test that viewport is centered on the requested agent."""
        bounds = env.get_viewport_bounds_for("agent_0")
        min_x, min_y, max_x, max_y = bounds
        # Agent at (5, 5), viewport 11x11 → center should be near agent
        assert min_x <= 5 <= max_x - 1
        assert min_y <= 5 <= max_y - 1

    def test_different_agents_different_viewports(
        self, env: DynamicForagingEnvironment
    ) -> None:
        """Test that different agents produce different viewport bounds."""
        bounds_0 = env.get_viewport_bounds_for("agent_0")
        bounds_1 = env.get_viewport_bounds_for("agent_1")
        assert bounds_0 != bounds_1

    def test_viewport_clamps_to_grid(self) -> None:
        """Test that viewport clamps to grid boundaries."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            foraging=ForagingParams(foods_on_grid=1, target_foods_to_collect=1),
            seed=42,
        )
        env.add_agent("corner", position=(0, 0), max_body_length=3)
        min_x, min_y, _, _ = env.get_viewport_bounds_for("corner")
        assert min_x >= 0
        assert min_y >= 0

    def test_viewport_clamps_to_grid_upper(self) -> None:
        """Test that viewport clamps to upper grid boundaries."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            foraging=ForagingParams(foods_on_grid=1, target_foods_to_collect=1),
            seed=42,
        )
        env.add_agent("far_corner", position=(29, 29), max_body_length=3)
        _, _, max_x, max_y = env.get_viewport_bounds_for("far_corner")
        assert max_x <= 30
        assert max_y <= 30

    def test_default_delegates_to_for(self, env: DynamicForagingEnvironment) -> None:
        """Test that get_viewport_bounds() delegates to get_viewport_bounds_for(DEFAULT)."""
        default_bounds = env.get_viewport_bounds()
        for_bounds = env.get_viewport_bounds_for("default")
        assert default_bounds == for_bounds

    def test_invalid_agent_raises(self, env: DynamicForagingEnvironment) -> None:
        """Test that requesting bounds for a nonexistent agent raises KeyError."""
        with pytest.raises(KeyError):
            env.get_viewport_bounds_for("nonexistent")
