"""Tests for multi-agent rendering support."""

from __future__ import annotations

from typing import Any

import pytest
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams

# Check if pygame is importable
_pg: Any = None
_has_pygame = False

try:
    import pygame as _pygame_mod

    _pg = _pygame_mod
    _has_pygame = True
except ImportError:
    pass

requires_pygame = pytest.mark.skipif(
    not _has_pygame,
    reason="pygame not installed",
)


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


@requires_pygame
class TestTintedSprites:
    """Tests for tinted sprite generation."""

    def test_tinted_head_sprites_keys(self) -> None:
        """Test that tinted head sprites return all 4 directions."""
        from quantumnematode.env.sprites import create_tinted_head_sprites

        sprites = create_tinted_head_sprites(_pg, (100, 150, 220))
        assert set(sprites.keys()) == {"head_up", "head_down", "head_left", "head_right"}

    def test_tinted_head_sprites_size(self) -> None:
        """Test that tinted head sprites are CELL_SIZE x CELL_SIZE."""
        from quantumnematode.env.sprites import CELL_SIZE, create_tinted_head_sprites

        sprites = create_tinted_head_sprites(_pg, (100, 150, 220))
        for name, surf in sprites.items():
            assert surf.get_width() == CELL_SIZE, f"{name} width"
            assert surf.get_height() == CELL_SIZE, f"{name} height"

    def test_tinted_head_sprites_have_alpha(self) -> None:
        """Test that tinted head sprites have SRCALPHA."""
        from quantumnematode.env.sprites import create_tinted_head_sprites

        sprites = create_tinted_head_sprites(_pg, (100, 150, 220))
        for name, surf in sprites.items():
            assert surf.get_flags() & _pg.SRCALPHA, f"{name} should have SRCALPHA"

    def test_dead_agent_overlay_size(self) -> None:
        """Test that dead agent overlay is CELL_SIZE x CELL_SIZE with alpha."""
        from quantumnematode.env.sprites import CELL_SIZE, create_dead_agent_overlay

        surf = create_dead_agent_overlay(_pg)
        assert surf.get_width() == CELL_SIZE
        assert surf.get_height() == CELL_SIZE
        assert surf.get_flags() & _pg.SRCALPHA

    def test_agent_color_palette_length(self) -> None:
        """Test that the palette has exactly 8 colors."""
        from quantumnematode.env.sprites import AGENT_COLOR_PALETTE

        assert len(AGENT_COLOR_PALETTE) == 8

    def test_tint_body_colors(self) -> None:
        """Test that tint_body_colors returns 3-tuple of RGB tuples."""
        from quantumnematode.env.sprites import tint_body_colors

        bc, oc, hc = tint_body_colors((100, 150, 220))
        assert len(bc) == 3
        assert len(oc) == 3
        assert len(hc) == 3
        assert all(0 <= c <= 255 for c in bc)
        assert all(0 <= c <= 255 for c in oc)
        assert all(0 <= c <= 255 for c in hc)


class TestAgentRenderState:
    """Tests for AgentRenderState dataclass."""

    def test_construction(self) -> None:
        """Test that AgentRenderState can be constructed with all fields."""
        from quantumnematode.env.pygame_renderer import AgentRenderState

        state = AgentRenderState(
            agent_id="agent_0",
            position=(5, 10),
            body=[(4, 10), (3, 10)],
            direction="up",
            alive=True,
            hp=80.0,
            max_hp=100.0,
            foods_collected=3,
            satiety=50.0,
            max_satiety=100.0,
            color_index=1,
        )
        assert state.agent_id == "agent_0"
        assert state.position == (5, 10)
        assert state.direction == "up"
        assert state.alive is True
        assert state.color_index == 1

    def test_frozen(self) -> None:
        """Test that AgentRenderState is immutable."""
        from quantumnematode.env.pygame_renderer import AgentRenderState

        state = AgentRenderState(
            agent_id="a",
            position=(0, 0),
            body=[],
            direction="up",
            alive=True,
            hp=100.0,
            max_hp=100.0,
            foods_collected=0,
            satiety=100.0,
            max_satiety=100.0,
            color_index=0,
        )
        with pytest.raises(AttributeError):
            state.agent_id = "b"  # type: ignore[misc]


class TestMultiAgentSimulationRendering:
    """Tests for MultiAgentSimulation rendering integration."""

    def test_render_step_builds_agent_states(self) -> None:
        """Test that _render_step builds AgentRenderState from env state."""
        from unittest.mock import MagicMock, patch

        from quantumnematode.agent import QuantumNematodeAgent, SatietyConfig
        from quantumnematode.agent.multi_agent import MultiAgentSimulation
        from quantumnematode.brain.arch.qvarcircuit import (
            QVarCircuitBrain,
            QVarCircuitBrainConfig,
        )
        from quantumnematode.brain.modules import ModuleName
        from quantumnematode.env import DynamicForagingEnvironment, ForagingParams

        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            seed=42,
        )
        brain_config = QVarCircuitBrainConfig(
            modules={ModuleName.FOOD_CHEMOTAXIS: [0, 1]}, num_layers=1
        )

        agents = []
        for i in range(2):
            aid = f"agent_{i}"
            env.add_agent(aid, position=None, max_body_length=3)
            brain = QVarCircuitBrain(config=brain_config, shots=10)
            agent = QuantumNematodeAgent(
                brain=brain, env=env, agent_id=aid,
                satiety_config=SatietyConfig(),
            )
            agents.append(agent)

        mock_renderer = MagicMock()
        mock_renderer.closed = False
        mock_renderer.render_multi_agent_frame.return_value = "agent_0"

        sim = MultiAgentSimulation(env=env, agents=agents, renderer=mock_renderer)
        sim._per_agent_food = {"agent_0": 2, "agent_1": 1}

        sim._render_step(current_step=5, step=5, max_steps=100)

        mock_renderer.render_multi_agent_frame.assert_called_once()
        call_kwargs = mock_renderer.render_multi_agent_frame.call_args
        agents_arg = call_kwargs.kwargs.get("agents") or call_kwargs[1].get("agents")
        # If positional
        if agents_arg is None:
            agents_arg = call_kwargs[0][1]
        assert len(agents_arg) == 2
        assert agents_arg[0].agent_id == "agent_0"
        assert agents_arg[0].foods_collected == 2
        assert agents_arg[1].agent_id == "agent_1"
        assert agents_arg[1].color_index == 1

    def test_renderer_closed_sets_flag(self) -> None:
        """Test that renderer_closed flag is set when renderer reports closed."""
        from unittest.mock import MagicMock

        from quantumnematode.agent.multi_agent import MultiAgentSimulation

        sim = MultiAgentSimulation.__new__(MultiAgentSimulation)
        sim._renderer_closed = False
        sim._followed_agent_id = ""
        sim._per_agent_food = {}

        mock_renderer = MagicMock()
        mock_renderer.closed = True
        mock_renderer.render_multi_agent_frame.return_value = ""
        sim.renderer = mock_renderer
        sim.env = MagicMock()
        sim.env.agents = {}
        sim.agents = []

        sim._render_step(0, 0, 100)
        assert sim.renderer_closed is True
