"""Tests for the Pixel (Pygame) theme: sprites, renderer, and theme integration.

Pygame-dependent tests are skipped when pygame is not available or when
no display is available (e.g. CI environments).
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from quantumnematode.env.theme import THEME_SYMBOLS, Theme

# Check if pygame is importable and probe for display
_pg: Any = None
_has_pygame = False
_has_display = False

try:
    import pygame as _pygame_mod

    _pg = _pygame_mod
    _has_pygame = True
except ImportError:
    pass

if _has_pygame and _pg is not None:
    try:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        _pg.init()
        _pg.display.set_mode((1, 1))
        _pg.quit()
        _has_display = True
    except Exception:  # noqa: BLE001, S110
        pass

requires_pygame = pytest.mark.skipif(
    not _has_pygame,
    reason="pygame not installed",
)
requires_display = pytest.mark.skipif(
    not _has_display,
    reason="No display available for pygame",
)


# ---------------------------------------------------------------------------
# Theme integration tests (no pygame needed)
# ---------------------------------------------------------------------------


class TestPixelThemeIntegration:
    """Tests that the PIXEL theme is properly registered."""

    def test_pixel_in_theme_enum(self) -> None:
        """Verify PIXEL is in the Theme enum."""
        assert hasattr(Theme, "PIXEL")
        assert Theme.PIXEL.value == "pixel"

    def test_pixel_has_theme_symbols(self) -> None:
        """Verify PIXEL has a ThemeSymbolSet entry."""
        assert Theme.PIXEL in THEME_SYMBOLS

    def test_default_theme_is_pixel(self) -> None:
        """Verify DEFAULT_THEME is PIXEL."""
        from quantumnematode.env.theme import DEFAULT_THEME

        assert DEFAULT_THEME == Theme.PIXEL


# ---------------------------------------------------------------------------
# Sprite tests (need pygame but not a real display)
# ---------------------------------------------------------------------------


@requires_pygame
class TestSprites:
    """Tests for procedural sprite generation."""

    def test_create_sprites_returns_expected_keys(self) -> None:
        """Verify all expected sprite keys are returned."""
        from quantumnematode.env.sprites import create_sprites

        sprites = create_sprites(_pg)
        expected = {
            "empty",
            "food",
            "predator_random",
            "predator_stationary",
            "predator_pursuit",
            "head_up",
            "head_down",
            "head_left",
            "head_right",
        }
        assert expected.issubset(set(sprites.keys()))

    def test_sprite_surfaces_are_correct_size(self) -> None:
        """Verify all sprites are CELL_SIZE x CELL_SIZE."""
        from quantumnematode.env.sprites import CELL_SIZE, create_sprites

        sprites = create_sprites(_pg)
        for name, surf in sprites.items():
            assert surf.get_width() == CELL_SIZE, f"{name} width mismatch"
            assert surf.get_height() == CELL_SIZE, f"{name} height mismatch"

    def test_entity_sprites_have_alpha(self) -> None:
        """Entity sprites (not soil) should use SRCALPHA for zone transparency."""
        from quantumnematode.env.sprites import create_sprites

        sprites = create_sprites(_pg)
        alpha_sprites = [
            "food",
            "head_up",
            "predator_random",
            "predator_stationary",
            "predator_pursuit",
        ]
        for name in alpha_sprites:
            surf = sprites[name]
            assert surf.get_flags() & _pg.SRCALPHA, f"{name} should have SRCALPHA"

    def test_create_zone_overlay(self) -> None:
        """Verify zone overlays are SRCALPHA and correct size."""
        from quantumnematode.env.sprites import CELL_SIZE, create_zone_overlay

        for zone_name in (
            "lethal_cold",
            "danger_cold",
            "discomfort_cold",
            "comfort",
            "discomfort_hot",
            "danger_hot",
            "lethal_hot",
            "toxic",
        ):
            surf = create_zone_overlay(_pg, zone_name)
            assert surf.get_width() == CELL_SIZE
            assert surf.get_height() == CELL_SIZE
            assert surf.get_flags() & _pg.SRCALPHA

    def test_draw_body_segment_runs(self) -> None:
        """draw_body_segment should not raise."""
        from quantumnematode.env.sprites import CELL_SIZE, draw_body_segment

        surf = _pg.Surface((CELL_SIZE * 3, CELL_SIZE * 3), _pg.SRCALPHA)
        draw_body_segment(
            _pg,
            surf,
            CELL_SIZE,
            CELL_SIZE,
            connect_up=True,
            connect_right=True,
            is_tail=False,
        )
        # Tail variant
        draw_body_segment(
            _pg,
            surf,
            CELL_SIZE,
            CELL_SIZE,
            is_tail=True,
        )


# ---------------------------------------------------------------------------
# Renderer tests (need a display)
# ---------------------------------------------------------------------------


@requires_display
class TestPygameRenderer:
    """Tests for the PygameRenderer class."""

    def _make_renderer(self) -> Any:
        from quantumnematode.env.pygame_renderer import PygameRenderer

        return PygameRenderer(viewport_size=(5, 5), cell_size=16)

    def test_init_and_close(self) -> None:
        """Verify renderer opens and closes cleanly."""
        renderer = self._make_renderer()
        assert not renderer.closed
        renderer.close()
        assert renderer.closed

    def test_close_is_idempotent(self) -> None:
        """Verify calling close twice does not raise."""
        renderer = self._make_renderer()
        renderer.close()
        renderer.close()  # should not raise
        assert renderer.closed

    def test_render_frame_after_close_is_noop(self) -> None:
        """render_frame should silently return if window is closed."""
        renderer = self._make_renderer()
        renderer.close()
        # Should not raise
        renderer.render_frame(MagicMock())


# ---------------------------------------------------------------------------
# Status bar session text parsing
# ---------------------------------------------------------------------------


@requires_pygame
class TestStatusBarSessionText:
    """Test that session_text is parsed into status bar lines."""

    def test_session_text_lines_parsed(self) -> None:
        """Verify render_frame accepts session_text parameter."""
        from quantumnematode.env.pygame_renderer import PygameRenderer

        sig = inspect.signature(PygameRenderer.render_frame)
        assert "session_text" in sig.parameters
