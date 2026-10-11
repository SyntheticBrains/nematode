"""The kinematic body, lawns, internal state and frame recording in the continuous renderer.

Covers the continuous-fidelity-renderer requirements "Worm rendering with continuous heading" (the
kinematic body is drawn from its state; a reversal is marked), "Lawn rendering" (grazing shows),
"Internal-state status lines" (lines appear only where they apply), "Posture overlay" (the overlay
toggles) and "Run frame recording" (frames are saved; headless refuses recording), and the
patchy-lawns requirement "The lawn model's scope" through the renderer that now draws lawns.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import quantumnematode.brain  # noqa: F401  # breaks the env package's import cycle
from quantumnematode.env.body import N_SEGMENTS, BodyParams, midline, new_body
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import ForagingParams
from quantumnematode.env.lawns import LawnParams
from quantumnematode.env.theme import Theme

_REPO = Path(__file__).resolve().parents[5]

_has_pygame = False
try:
    import pygame  # noqa: F401

    _has_pygame = True
except ImportError:
    pass
requires_pygame = pytest.mark.skipif(not _has_pygame, reason="pygame not installed")


def _state(**overrides: Any) -> Any:
    from quantumnematode.env.pygame_renderer import ContinuousRenderState

    defaults: dict[str, Any] = {
        "pos": (10.0, 10.0),
        "heading_rad": 0.0,
        "left_sample": None,
        "right_sample": None,
        "sweep": 0.5,
        "adaptive_background": None,
        "adaptive_readout": None,
        "adaptive_mode": None,
        "step": 3,
        "max_steps": 100,
        "foods_collected": 0,
        "target_foods": 0,
        "health": 100.0,
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


class TestMidline:
    def test_a_straight_body_lies_behind_its_head(self) -> None:
        params = BodyParams(body_length_mm=1.2)
        body = new_body(5.0, 6.0, heading=math.pi / 2)
        body.curvature = np.zeros(N_SEGMENTS)
        points = midline(body, params)
        assert points.shape == (N_SEGMENTS + 1, 2)
        assert points[0] == pytest.approx(np.asarray(body.head))
        steps = np.hypot(*np.diff(points, axis=0).T)
        assert steps == pytest.approx(np.full(N_SEGMENTS, 1.2 / N_SEGMENTS))
        tail_direction = points[-1] - points[0]
        frame = (math.cos(body.frame_angle), math.sin(body.frame_angle))
        assert np.dot(tail_direction, frame) == pytest.approx(-1.2)

    def test_the_environment_exposes_the_body_only_once_it_moves(self) -> None:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=20.0, body_model="kinematic"),
            seed=1,
            theme=Theme.HEADLESS,
        )
        assert env.body_midline() is None
        env.move_agent_body(np.zeros(25))
        points = env.body_midline()
        assert points is not None
        assert points == pytest.approx(midline(env.bodies["default"], env._body.params))
        assert env.body_reversed() is False
        curvature = env.body_curvature()
        assert curvature is not None
        assert curvature.shape == (N_SEGMENTS,)


@pytest.fixture
def renderer() -> Any:
    """Open a small continuous renderer under the dummy driver."""
    from quantumnematode.env.pygame_renderer import Continuous2DRenderer

    r = Continuous2DRenderer(world_size_mm=20.0, pixels_per_mm=10.0)
    yield r
    r.close()


def _lawn_env() -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0),
        foraging=ForagingParams(food_model="lawns", lawns=LawnParams(count=1, radius_mm=3.0)),
        seed=2,
        theme=Theme.PIXEL_CONTINUOUS,
    )


@requires_pygame
class TestRendering:
    def test_the_kinematic_body_is_drawn_through_its_joints(self, renderer: Any) -> None:
        """Scenario "The kinematic body is drawn from its state": its pixels sit on the joints."""
        renderer._heatmap_enabled = False
        joints = tuple((10.0 - 0.1 * i, 10.0) for i in range(N_SEGMENTS + 1))
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        renderer.render_frame(env, _state(body_midline=joints, body_reversed=False))
        from quantumnematode.env.pygame_renderer import WORM_BODY_COLOR

        mid = renderer._world_to_pixel(*joints[N_SEGMENTS // 2])
        assert tuple(renderer._screen.get_at(mid))[:3] == WORM_BODY_COLOR

    def test_a_reversal_is_marked(self, renderer: Any) -> None:
        """Scenario "A reversal is marked": the reversal tint, and a status line."""
        from quantumnematode.env.pygame_renderer import WORM_REVERSAL_COLOR

        renderer._heatmap_enabled = False
        joints = tuple((10.0 - 0.1 * i, 10.0) for i in range(N_SEGMENTS + 1))
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        state = _state(body_midline=joints, body_reversed=True)
        renderer.render_frame(env, state)
        mid = renderer._world_to_pixel(*joints[N_SEGMENTS // 2])
        assert tuple(renderer._screen.get_at(mid))[:3] == WORM_REVERSAL_COLOR
        text = " ".join(t for t, _ in renderer._build_status_lines(state, None))
        assert "Body: reversing" in text

    def test_grazing_shows(self, renderer: Any) -> None:
        """Scenario "Grazing shows": a grazed cell is drawn closer to the soil."""
        renderer._heatmap_enabled = False
        env = _lawn_env()
        field = env.lawn_field
        assert field is not None
        far = _state(pos=(1.0, 1.0))
        # A cell in the lawn's interior, clear of its edge outline.
        inner = int(np.argmin(np.hypot(*(field.cells - field.centres[0]).T)))
        cell = renderer._world_to_pixel(*field.cells[inner])
        renderer.render_frame(env, far)
        full = np.array(tuple(renderer._screen.get_at(cell))[:3], dtype=float)
        field.density[inner] = 0.1
        renderer.render_frame(env, far)
        grazed = np.array(tuple(renderer._screen.get_at(cell))[:3], dtype=float)
        from quantumnematode.env.sprites import SOIL_COLOR

        soil = np.array(SOIL_COLOR, dtype=float)
        assert np.linalg.norm(grazed - soil) < np.linalg.norm(full - soil)

    def test_lines_appear_only_where_they_apply(self, renderer: Any) -> None:
        """Scenario "Lines appear only where they apply"."""
        plain = [t for t, _ in renderer._build_status_lines(_state(), None)]
        assert not any(t.startswith(("Intake", "Serotonin", "Body:")) for t in plain)
        rich = _state(intake=0.02, on_lawn=True, modulators=(0.3, 0.1))
        text = " ".join(t for t, _ in renderer._build_status_lines(rich, None))
        assert "Intake: 0.0200" in text
        assert "on a lawn" in text
        assert "Serotonin: 0.300" in text

    def test_the_posture_overlay_toggles(self, renderer: Any) -> None:
        """Scenario "The overlay toggles": P draws the panel, P again removes it."""
        import pygame as pg
        from quantumnematode.env.pygame_renderer import _POSTURE_PANEL, POSTURE_PANEL_BG

        renderer._heatmap_enabled = False
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        joints = tuple((10.0 - 0.1 * i, 10.0) for i in range(N_SEGMENTS + 1))
        state = _state(body_midline=joints, body_curvature=(1.0,) * N_SEGMENTS)
        corner = (renderer._arena_px - _POSTURE_PANEL[0], 8)
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_p))
        renderer.render_frame(env, state)
        assert tuple(renderer._screen.get_at(corner))[:3] == POSTURE_PANEL_BG
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_p))
        renderer.render_frame(env, state)
        assert tuple(renderer._screen.get_at(corner))[:3] != POSTURE_PANEL_BG


@requires_pygame
def test_frames_are_saved(tmp_path: Path) -> None:
    """Scenario "Frames are saved": four steps at every-2 write two PNGs named by run and step."""
    from quantumnematode.agent import QuantumNematodeAgent, RewardConfig
    from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
    from quantumnematode.brain.arch.dtypes import DeviceType
    from quantumnematode.env.pygame_renderer import FrameRecorder

    env = _lawn_env()
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(seed=0, action_mode="continuous"),
        device=DeviceType.CPU,
    )
    agent = QuantumNematodeAgent(brain=brain, env=env, theme=Theme.PIXEL_CONTINUOUS)
    agent.frame_recorder = FrameRecorder(tmp_path, every=2, run=3)
    agent.run_episode(RewardConfig(), max_steps=4)
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["run0003-step00002.png", "run0003-step00004.png"]
    agent._continuous_renderer.close()


def test_headless_refuses_recording(tmp_path: Path) -> None:
    """Scenario "Headless refuses recording"."""
    result = subprocess.run(  # noqa: S603 - fixed arguments
        [
            sys.executable,
            str(_REPO / "scripts" / "run_simulation.py"),
            "--theme",
            "headless",
            "--record-frames",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 2
    assert "headless renders none" in result.stderr
