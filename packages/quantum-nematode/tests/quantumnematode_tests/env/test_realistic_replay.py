"""Recording kinematic-body runs and replaying them as a worm on a plate.

Covers the realistic-replay requirements "Recording a kinematic-body run for replay" (a run is
recorded at sub-step resolution; recording does not change the run; a point-worm run is refused),
"Replaying in worm time" (interpolation between sub-steps; speed scales the clock) and "A realistic,
honest worm" (the tube passes through the joints; the width profile is elliptical with open tips;
the label names the model and speed).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import quantumnematode.brain  # noqa: F401  # breaks the env package's import cycle
from quantumnematode.env.realistic_renderer import (
    advance,
    catmull_rom,
    interpolate,
    label_text,
    width_profile,
)
from quantumnematode.report.replay import ReplayRecorder, load_replay

_REPO = Path(__file__).resolve().parents[5]
_DEMO = (
    _REPO
    / "configs/scenarios/foraging"
    / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_body_internal_frozen.yml"
)
_POINT = (
    _REPO
    / "configs/scenarios/foraging"
    / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_frozen.yml"
)
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts"))


def _run(steps: int, recorder: ReplayRecorder | None, seed: int = 4) -> Any:
    """Run the body-on-lawns demo for ``steps`` steps, optionally recorded; return the agent."""
    import body_kinematics_eval as harness  # pyright: ignore[reportMissingImports]
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.env.theme import Theme
    from quantumnematode.utils.config_loader import (
        configure_environment,
        configure_reward,
        configure_satiety,
        create_env_from_config,
    )
    from quantumnematode.utils.seeding import set_global_seed

    set_global_seed(seed)
    brain, config, sensing = harness.build_brain(_DEMO, seed)
    env = create_env_from_config(
        configure_environment(config),
        seed=seed,
        max_body_length=1,
        theme=Theme.HEADLESS,
    )
    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing,
    )
    if recorder is not None:
        agent.replay_recorder = recorder
        recorder.begin_run(env, 1)  # type: ignore[arg-type]
    agent.run_episode(configure_reward(config), max_steps=steps)
    return agent


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record a 3-step run of the body-on-lawns demo."""
    recorder = ReplayRecorder(tmp_path_factory.mktemp("replay"), prefix="t-")
    _run(3, recorder)
    return recorder.end_run()


class TestRecording:
    def test_a_run_is_recorded_at_sub_step_resolution(self, recording: Path) -> None:
        """Scenario "A run is recorded at sub-step resolution"."""
        replay = load_replay(recording)
        assert replay.substep_curvature.shape == (60, 12)
        assert replay.substep_head.shape == (60, 2)
        assert len(replay.step_intake) == 3
        assert replay.step_lawn_density is not None
        assert replay.lawn_cells is not None
        assert replay.step_lawn_density.shape == (3, len(replay.lawn_cells))
        assert recording.name == "t-run0001.npz"

    def test_recording_does_not_change_the_run(self, tmp_path: Path) -> None:
        """Scenario "Recording does not change the run"."""
        plain = _run(5, None)
        recorder = ReplayRecorder(tmp_path)
        recorded = _run(5, recorder)
        recorder.end_run()
        assert plain.env.bodies["default"].head == pytest.approx(
            recorded.env.bodies["default"].head,
        )
        assert plain._episode_tracker.intake == recorded._episode_tracker.intake

    def test_a_point_worm_is_refused(self, tmp_path: Path) -> None:
        from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams

        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0), seed=1)
        with pytest.raises(ValueError, match="kinematic body"):
            ReplayRecorder(tmp_path).begin_run(env, 1)

    @pytest.mark.smoke
    def test_the_cli_refuses_a_point_worm(self, tmp_path: Path) -> None:
        """Scenario "A point-worm run is refused"."""
        result = subprocess.run(  # noqa: S603 - fixed arguments
            [
                sys.executable,
                str(_REPO / "scripts" / "run_simulation.py"),
                "--config",
                str(_POINT),
                "--runs",
                "1",
                "--theme",
                "headless",
                "--record-replay",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert result.returncode == 2
        assert "kinematic body" in result.stderr


class TestGeometryAndTime:
    def test_the_tube_passes_through_the_joints(self) -> None:
        """Scenario "The tube passes through the joints"."""
        joints = np.column_stack([np.linspace(0, 1.2, 13), np.sin(np.linspace(0, 3, 13))])
        curve = catmull_rom(joints, subdivisions=8)
        assert curve[::8] == pytest.approx(joints)

    def test_the_width_profile_is_elliptical_with_open_tips(self) -> None:
        """Scenario "The width profile is elliptical with open tips"."""
        profile = width_profile(97)
        assert profile[48] == pytest.approx(1.0)
        assert profile == pytest.approx(profile[::-1])
        assert profile[0] > 0.0
        assert np.all(np.diff(profile[:49]) > 0)

    def test_interpolation_between_sub_steps(self, recording: Path) -> None:
        """Scenario "Interpolation between sub-steps"."""
        replay = load_replay(recording)
        t0, t1 = replay.substep_time[10], replay.substep_time[11]
        pose = interpolate(replay, (t0 + t1) / 2)
        assert pose.head == pytest.approx((replay.substep_head[10] + replay.substep_head[11]) / 2)
        assert pose.curvature == pytest.approx(
            (replay.substep_curvature[10] + replay.substep_curvature[11]) / 2,
        )

    def test_speed_scales_the_clock(self) -> None:
        """Scenario "Speed scales the clock"."""
        assert advance(0.0, 0.05, 10.0, 100.0) == pytest.approx(0.5)
        assert advance(0.0, 5.0, 10.0, 100.0) == pytest.approx(1.0)  # the wall gap is capped
        assert advance(99.9, 0.05, 60.0, 100.0) == pytest.approx(100.0)

    def test_the_label_names_the_model_and_speed(self) -> None:
        """Scenario "The label names the model and speed"."""
        text = label_text(10.0, 12.5)
        assert "kinematic body" in text
        assert "10×" in text  # noqa: RUF001


class TestDrawing:
    def test_a_frame_renders_headless(self, recording: Path) -> None:
        import pygame
        from quantumnematode.env.realistic_renderer import RealisticRenderer

        pygame.display.init()
        surface = pygame.Surface((240, 240))
        renderer = RealisticRenderer(load_replay(recording), 240, texture_px_per_mm=10.0)
        renderer.draw(surface, 5.0, 10.0)
        renderer.follow = False
        renderer.draw(surface, 10.0, 60.0)
        assert surface.get_size() == (240, 240)

    def test_the_keys_drive_the_controls(self, recording: Path) -> None:
        import pygame
        import replay_viewer as viewer  # pyright: ignore[reportMissingImports]
        from quantumnematode.env.realistic_renderer import RealisticRenderer

        pygame.display.init()
        renderer = RealisticRenderer(load_replay(recording), 120, texture_px_per_mm=5.0)
        controls = viewer.Controls()
        viewer.apply_key(pygame.K_3, pygame, controls, renderer, 10.0, 0.25)
        assert controls.speed == 60.0
        viewer.apply_key(pygame.K_SPACE, pygame, controls, renderer, 10.0, 0.25)
        viewer.apply_key(pygame.K_RIGHT, pygame, controls, renderer, 10.0, 0.25)
        assert controls.paused
        assert controls.clock == pytest.approx(0.25)
        viewer.apply_key(pygame.K_c, pygame, controls, renderer, 10.0, 0.25)
        assert renderer.follow is False
        before = renderer.view_body_lengths
        viewer.apply_key(pygame.K_EQUALS, pygame, controls, renderer, 10.0, 0.25)
        assert renderer.view_body_lengths < before
        assert replace(controls, clock=0.0).clock == 0.0
