"""Tests for the continuous-2D substrate renderer and offline figures.

Headless: the renderer creates a real pygame display, so these set the SDL dummy
video driver at import time and skip when pygame is unavailable (mirroring the
grid renderer's test posture).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import pytest

# Headless display before pygame initialises a video backend.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

if TYPE_CHECKING:
    from pathlib import Path

# Warm import: importing the env package first triggers a circular import; the
# brain package (imported by the agent path) breaks the cycle.
import quantumnematode.brain  # noqa: F401
from quantumnematode.agent.adaptive_sensor import AdaptiveChemosensor
from quantumnematode.agent.agent import _continuous_lateral_offsets
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.theme import Theme

_has_pygame = False
try:
    import pygame as _pygame_mod  # noqa: F401

    _has_pygame = True
except ImportError:
    pass

requires_pygame = pytest.mark.skipif(not _has_pygame, reason="pygame not installed")


def _make_state(**overrides: Any) -> Any:
    """Build a ContinuousRenderState with sensible defaults for smoke tests."""
    from quantumnematode.env.pygame_renderer import ContinuousRenderState

    defaults: dict[str, Any] = {
        "pos": (10.4, 20.7),
        "heading_rad": 0.5,
        "left_sample": (10.0, 21.0),
        "right_sample": (11.0, 20.0),
        "sweep": 0.5,
        "adaptive_background": 0.2,
        "adaptive_readout": 0.05,
        "adaptive_mode": "contrast",
        "step": 3,
        "max_steps": 100,
        "foods_collected": 1,
        "target_foods": 5,
        "health": 90.0,
        "max_health": 100.0,
        "satiety": 50.0,
        "max_satiety": 100.0,
        "in_danger": False,
        "temperature": None,
        "zone_name": None,
        "oxygen": None,
        "oxygen_zone_name": None,
    }
    defaults.update(overrides)
    return ContinuousRenderState(**defaults)


@requires_pygame
class TestWorldToPixel:
    """The world→pixel mapping: origin, corner, centre, y-inversion, zoom scaling."""

    def test_origin_maps_to_bottom_left(self) -> None:
        """World origin (0, 0) maps to the bottom-left pixel (y inverted)."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            assert renderer._world_to_pixel(0.0, 0.0) == (0, 300)
        finally:
            renderer.close()

    def test_top_right_corner(self) -> None:
        """World (world, world) maps to the top-right pixel."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            assert renderer._world_to_pixel(30.0, 30.0) == (300, 0)
        finally:
            renderer.close()

    def test_centre(self) -> None:
        """World centre maps to the arena centre pixel."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            assert renderer._world_to_pixel(15.0, 15.0) == (150, 150)
        finally:
            renderer.close()

    def test_y_inversion(self) -> None:
        """Increasing world-y maps to decreasing screen-y."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            _, low_y = renderer._world_to_pixel(15.0, 5.0)
            _, high_y = renderer._world_to_pixel(15.0, 25.0)
            assert high_y < low_y
        finally:
            renderer.close()

    def test_zoom_scaling(self) -> None:
        """A larger pixels_per_mm scales pixel coordinates proportionally."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        r10 = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        r20 = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=20.0)
        try:
            px10, _ = r10._world_to_pixel(10.0, 0.0)
            px20, _ = r20._world_to_pixel(10.0, 0.0)
            assert px20 == 2 * px10
        finally:
            r10.close()
            r20.close()

    def test_window_spans_whole_arena(self) -> None:
        """The drawing surface spans the arena plus the status bar (no scroll)."""
        from quantumnematode.env.pygame_renderer import STATUS_BAR_HEIGHT, Continuous2DRenderer

        renderer = Continuous2DRenderer(world_size_mm=50.0, pixels_per_mm=12.0)
        try:
            width, height = renderer._screen.get_size()
            assert width == 600
            assert height >= 600 + STATUS_BAR_HEIGHT  # status bar may grow to fit lines
        finally:
            renderer.close()


@requires_pygame
class TestFollowCamera:
    """The agent-following camera toggle: centring, edge clamping, default off."""

    def _state_at(self, pos: tuple[float, float]) -> Any:
        return _make_state(pos=pos)

    def test_default_is_full_arena(self) -> None:
        """The camera starts in the full-arena view (worm at its true pixel)."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=50.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = Continuous2DRenderer(world_size_mm=50.0, pixels_per_mm=12.0)
        try:
            renderer.render_frame(env, self._state_at((10.0, 10.0)))
            assert not renderer._camera_following
            assert renderer._world_to_pixel(10.0, 10.0) == (120, 480)  # 10*12, (50-10)*12
        finally:
            renderer.close()

    def test_following_centres_worm(self) -> None:
        """With follow on and the worm interior, it maps to the viewport centre."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=50.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = Continuous2DRenderer(world_size_mm=50.0, pixels_per_mm=12.0)
        try:
            renderer._follow_enabled = True
            renderer.render_frame(env, self._state_at((25.0, 25.0)))
            assert renderer._camera_following
            cx, cy = renderer._world_to_pixel(25.0, 25.0)
            centre = renderer._arena_px // 2
            assert abs(cx - centre) <= 1
            assert abs(cy - centre) <= 1
        finally:
            renderer.close()

    def test_following_clamps_at_edge(self) -> None:
        """Near a plate corner the camera clamps (worm moves off-centre, no out-of-plate)."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=50.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = Continuous2DRenderer(world_size_mm=50.0, pixels_per_mm=12.0)
        try:
            renderer._follow_enabled = True
            renderer.render_frame(env, self._state_at((0.5, 0.5)))
            # Clamped to the corner: the plate origin maps exactly to the
            # bottom-left pixel (camera did NOT scroll past 0 — without clamping
            # cam_x0 would be negative and the origin would land at positive x).
            assert renderer._world_to_pixel(0.0, 0.0) == (0, renderer._arena_px)
            # The worm is therefore off-centre (pulled toward the corner).
            cx, _ = renderer._world_to_pixel(0.5, 0.5)
            assert cx < renderer._arena_px // 2
        finally:
            renderer.close()

    def test_following_with_heatmap_crops_without_error(self) -> None:
        """Follow + heatmap exercises the subsurface crop/scale path at a plate edge."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=50.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = Continuous2DRenderer(world_size_mm=50.0, pixels_per_mm=12.0)
        try:
            renderer._follow_enabled = True
            assert renderer._heatmap_enabled  # on by default → crop path runs
            # Interior, then a corner (the crop rect clamps hardest at the edge).
            for pos in ((25.0, 25.0), (0.5, 0.5), (49.5, 49.5)):
                renderer.render_frame(env, self._state_at(pos))
            assert renderer._camera_following
        finally:
            renderer.close()


@requires_pygame
class TestHeatmapCacheKey:
    """The heatmap cache invalidation: dynamic pheromone vs static thermal/oxygen."""

    def _renderer(self) -> Any:
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        return Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=12.0)

    def test_pheromone_key_varies_with_step(self) -> None:
        """The time-varying pheromone field rebuilds each step (key includes step)."""
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=30.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = self._renderer()
        try:
            key_a = renderer._heatmap_cache_key(env, "pheromone", 5)
            key_b = renderer._heatmap_cache_key(env, "pheromone", 6)
            assert key_a != key_b
        finally:
            renderer.close()

    def test_static_field_key_is_step_invariant(self) -> None:
        """Static fields (temperature) keep one key across steps so the cache holds."""
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=30.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        renderer = self._renderer()
        try:
            key_a = renderer._heatmap_cache_key(env, "temperature", 5)
            key_b = renderer._heatmap_cache_key(env, "temperature", 6)
            assert key_a == key_b
        finally:
            renderer.close()


class TestSnapshotGeometry:
    """The klinotaxis sample geometry and adaptive accessor the snapshot relies on."""

    def test_lateral_offsets_match_perpendicular(self) -> None:
        """`_continuous_lateral_offsets` samples perpendicular to heading at `sweep`."""
        pos = (10.0, 10.0)
        heading = 0.0  # facing +x → perpendicular is ±y
        sweep = 2.0
        left, right = _continuous_lateral_offsets(pos, heading, sweep, grid_size=50)
        # perp_x = -sin(0) = 0, perp_y = cos(0) = 1 → left is +y, right is -y.
        assert left == pytest.approx((10.0, 12.0))
        assert right == pytest.approx((10.0, 8.0))

    def test_lateral_offsets_rotate_with_heading(self) -> None:
        """At a 90° heading the sweep axis rotates to the x-axis."""
        pos = (10.0, 10.0)
        left, right = _continuous_lateral_offsets(
            pos,
            math.pi / 2,
            sweep=2.0,
            grid_size=50,
        )
        # heading +y → perp_x = -1, perp_y = 0 → left is -x, right is +x.
        assert left == pytest.approx((8.0, 10.0))
        assert right == pytest.approx((12.0, 10.0))

    def test_adaptive_background_accessor(self) -> None:
        """`AdaptiveChemosensor.background` exposes the integrator background."""
        sensor = AdaptiveChemosensor(readout="contrast", alpha=0.5)
        assert sensor.background == 0.0  # before any sample
        sensor.adapt(concentration=0.8, derivative=0.1)
        assert sensor.background == pytest.approx(0.8)  # seeded on first sample
        assert sensor.last_readout is not None

    def test_adaptive_reset_clears_state(self) -> None:
        """Resetting clears both the background and the cached readout."""
        sensor = AdaptiveChemosensor(readout="contrast", alpha=0.5)
        sensor.adapt(concentration=0.8, derivative=0.1)
        sensor.reset()
        assert sensor.background == 0.0
        assert sensor.last_readout is None


@requires_pygame
class TestSmokeRender:
    """Render one frame against a staged continuous env without raising."""

    def _env(self) -> Continuous2DEnvironment:
        return Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=30.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )

    def test_render_frame_runs(self) -> None:
        """A full frame renders and the screen has the expected dimensions."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = self._env()
        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            renderer.render_frame(env, _make_state())
            assert renderer._screen.get_width() == 300
            assert not renderer.closed
        finally:
            renderer.close()

    def test_narrow_window_wraps_status_and_grows(self) -> None:
        """A narrow window wraps long status lines and grows its height to fit them."""
        from quantumnematode.env.pygame_renderer import (
            STATUS_BAR_HEIGHT,
            Continuous2DRenderer,
        )

        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=20.0),
            theme=Theme.PIXEL_CONTINUOUS,
            seed=1,
        )
        # 20 mm @ 12 px/mm = 240 px wide — too narrow for the adaptive + hint lines.
        renderer = Continuous2DRenderer(world_size_mm=20.0, pixels_per_mm=12.0)
        try:
            renderer.render_frame(env, _make_state(adaptive_mode="fold_change"))
            _, height = renderer._screen.get_size()
            # The status region exceeds the base height → lines wrapped onto extra rows.
            assert height - renderer._arena_px > STATUS_BAR_HEIGHT
        finally:
            renderer.close()

    def test_render_with_quiver_enabled(self) -> None:
        """The gradient quiver overlay renders without raising."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = self._env()
        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            renderer._quiver_enabled = True
            renderer.render_frame(env, _make_state())
        finally:
            renderer.close()

    def test_adaptive_disabled_omits_readout(self) -> None:
        """A disabled adaptive sensor renders with no error (readout omitted)."""
        from quantumnematode.env.pygame_renderer import Continuous2DRenderer

        env = self._env()
        renderer = Continuous2DRenderer(world_size_mm=30.0, pixels_per_mm=10.0)
        try:
            state = _make_state(
                adaptive_mode=None,
                adaptive_background=None,
                adaptive_readout=None,
            )
            renderer.render_frame(env, state)
        finally:
            renderer.close()


class TestOfflineFigures:
    """Offline matplotlib figures generate non-empty PNGs headlessly."""

    def _env(self) -> Continuous2DEnvironment:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=30.0),
            seed=3,
        )
        env.foods = [(8.0, 22.0), (24.0, 9.0)]  # type: ignore[assignment]
        return env

    def test_plot_trajectory(self, tmp_path: Path) -> None:
        """The trajectory figure writes a non-empty PNG."""
        from quantumnematode.report import continuous_figures as cf

        out = tmp_path / "traj.png"
        cf.plot_trajectory(
            [(5.0, 5.0), (8.0, 10.0), (12.0, 18.0)],
            out,
            world_size_mm=30.0,
            foods=[(8.0, 22.0)],
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_field_heatmap(self, tmp_path: Path) -> None:
        """The field-heatmap figure writes a non-empty PNG."""
        from quantumnematode.report import continuous_figures as cf

        out = tmp_path / "heatmap.png"
        cf.plot_field_heatmap(self._env(), out, resolution=40)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_gradient_quiver(self, tmp_path: Path) -> None:
        """The gradient-quiver figure writes a non-empty PNG."""
        from quantumnematode.report import continuous_figures as cf

        out = tmp_path / "quiver.png"
        cf.plot_gradient_quiver(self._env(), out, resolution=8)
        assert out.exists()
        assert out.stat().st_size > 0
