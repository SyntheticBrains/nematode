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
        rng = np.random.default_rng(4)
        for _ in range(6):
            env.move_agent_body(rng.uniform(-1.0, 1.0, 25))
        points = env.body_midline()
        assert points is not None
        body = env.bodies["default"]
        assert points[0] == pytest.approx(np.asarray(body.head))
        # The body's heading runs from its middle segment's midpoint to its head: the midline is
        # the physics'.
        middle = (points[N_SEGMENTS // 2] + points[N_SEGMENTS // 2 + 1]) / 2
        middle_to_head = points[0] - middle
        assert math.atan2(middle_to_head[1], middle_to_head[0]) == pytest.approx(
            env._body.heading(body),
        )
        assert env.body_reversed() in (True, False)
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


def _arc() -> tuple[tuple[float, float], ...]:
    """Build a curved midline: joints on a quarter circle of radius 1 mm behind (10, 10)."""
    angles = np.linspace(0.0, math.pi / 2, N_SEGMENTS + 1)
    return tuple((10.0 - math.sin(a), 9.0 + math.cos(a)) for a in angles)


@requires_pygame
class TestRendering:
    def test_the_kinematic_body_is_drawn_through_its_joints(self, renderer: Any) -> None:
        """Scenario "The kinematic body is drawn from its state": a curved body, on its joints."""
        from quantumnematode.env.pygame_renderer import WORM_BODY_COLOR

        renderer._heatmap_enabled = False
        joints = _arc()
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        renderer.render_frame(env, _state(body_midline=joints, body_reversed=False))
        for index in (4, 8):
            pixel = renderer._world_to_pixel(*joints[index])
            assert tuple(renderer._screen.get_at(pixel))[:3] == WORM_BODY_COLOR
        # A straight head-to-tail stroke would pass here; the curved body does not.
        chord = renderer._world_to_pixel(
            (joints[0][0] + joints[-1][0]) / 2,
            (joints[0][1] + joints[-1][1]) / 2,
        )
        assert tuple(renderer._screen.get_at(chord))[:3] != WORM_BODY_COLOR

    def test_a_reversal_is_marked(self, renderer: Any) -> None:
        """Scenario "A reversal is marked": the reversal tint, the tail marker, a status line."""
        from quantumnematode.env.pygame_renderer import WORM_REVERSAL_COLOR, WORM_TAIL_MARKER_COLOR

        renderer._heatmap_enabled = False
        joints = tuple((10.0 - 0.1 * i, 10.0) for i in range(N_SEGMENTS + 1))
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        state = _state(body_midline=joints, body_reversed=True)
        renderer.render_frame(env, state)
        mid = renderer._world_to_pixel(*joints[N_SEGMENTS // 2])
        assert tuple(renderer._screen.get_at(mid))[:3] == WORM_REVERSAL_COLOR
        tail_x, tail_y = renderer._world_to_pixel(*joints[-1])
        beyond = (tail_x - 3, tail_y)  # the marker points tail-ward, here toward -x
        assert tuple(renderer._screen.get_at(beyond))[:3] == WORM_TAIL_MARKER_COLOR
        text = " ".join(t for t, _ in renderer._build_status_lines(state, None))
        assert "Body: reversing" in text
        forward = _state(body_midline=joints, body_reversed=False)
        assert "Body: forward" in " ".join(
            t for t, _ in renderer._build_status_lines(forward, None)
        )

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


@pytest.mark.smoke
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


class TestSnapshotExtras:
    """The agent fills the snapshot's optional fields from the run, and leaves out the rest."""

    @staticmethod
    def _agent(env: Continuous2DEnvironment, **brain: Any) -> Any:
        from quantumnematode.agent import QuantumNematodeAgent
        from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
        from quantumnematode.brain.arch.dtypes import DeviceType

        config = ConnectomePPOBrainConfig(seed=0, action_mode="continuous", **brain)
        return QuantumNematodeAgent(
            brain=ConnectomePPOBrain(config=config, device=DeviceType.CPU),
            env=env,
            theme=Theme.HEADLESS,
        )

    def test_a_point_worm_on_point_food_adds_nothing(self) -> None:
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        assert self._agent(env)._continuous_render_extras((5.0, 5.0)) == {}

    def test_the_kinematic_body_reaches_the_snapshot(self) -> None:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=20.0, body_model="kinematic"),
            seed=1,
        )
        env.move_agent_body(np.full(25, 0.3))
        extras = self._agent(env)._continuous_render_extras((5.0, 5.0))
        expected = env.body_midline()
        assert expected is not None
        assert np.array(extras["body_midline"]) == pytest.approx(expected)
        assert len(extras["body_curvature"]) == N_SEGMENTS
        assert extras["body_reversed"] is env.body_reversed()

    def test_lawns_and_modulators_reach_the_snapshot(self) -> None:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=20.0),
            foraging=ForagingParams(food_model="lawns", lawns=LawnParams(count=1, radius_mm=3.0)),
            seed=2,
        )
        field = env.lawn_field
        assert field is not None
        agent = self._agent(env, modulators=True)
        centre = (float(field.centres[0][0]), float(field.centres[0][1]))
        extras = agent._continuous_render_extras(centre)
        assert extras["on_lawn"] is True
        assert extras["intake"] == 0.0
        assert extras["modulators"] == agent.brain.modulator_concentrations
