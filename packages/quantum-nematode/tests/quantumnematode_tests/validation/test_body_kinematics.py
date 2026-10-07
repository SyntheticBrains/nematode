"""The kinematic instruments read a body's undulation, wavelength, speed and reversals.

Covers the body-kinematics requirement "Kinematic instruments" (frequency by band crossing,
wavelength by crossing delays, speed in body lengths per second, reversal fraction, wall
exclusion) and the environment's posture capture and configurable sub-step count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from quantumnematode.env.body import DRIVE_WIDTH, N_SEGMENTS
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.validation.body_kinematics import (
    Kinematics,
    band_crossings,
    in_bands,
    measure,
)

_STEP_S = 5.0
_SUBSTEPS = 40


def _travelling_wave(  # noqa: PLR0913 - the wave and the path it rides on
    *,
    frequency: float,
    wavelength: float,
    n_steps: int,
    direction: float = 1.0,
    start: tuple[float, float] = (50.0, 50.0),
    velocity: tuple[float, float] = (0.1, 0.0),
) -> list[dict[str, Any]]:
    """Return an episode whose curvature is ``sin(2 pi (f t - s / lambda))`` along the body."""
    s = np.arange(N_SEGMENTS) / N_SEGMENTS
    episode = []
    for step in range(n_steps):
        substeps = []
        for k in range(1, _SUBSTEPS + 1):
            t = (step + k / _SUBSTEPS) * _STEP_S
            curvature = 2.0 * np.sin(2.0 * np.pi * (frequency * t - s / wavelength))
            head = np.array(start) + np.array(velocity) * t
            substeps.append((t, curvature, head))
        drive = np.r_[np.zeros(DRIVE_WIDTH - 1), direction]
        episode.append({"drive": drive, "substeps": substeps})
    return episode


def _measure(episodes: list[list[dict[str, Any]]], world: float = 100.0) -> Kinematics:
    return measure(
        episodes,
        world_size_mm=world,
        body_length_mm=1.0,
        step_seconds=_STEP_S,
        reversal_threshold=0.5,
    )


class TestInstruments:
    @pytest.mark.parametrize(("frequency", "wavelength"), [(0.3, 0.65), (0.4, 0.5), (0.25, 0.8)])
    def test_a_known_wave_is_recovered(self, frequency: float, wavelength: float) -> None:
        k = _measure([_travelling_wave(frequency=frequency, wavelength=wavelength, n_steps=20)])
        assert k.frequency_hz == pytest.approx(frequency, rel=0.03)
        assert k.wavelength_bl == pytest.approx(wavelength, rel=0.05)

    def test_speed_is_in_body_lengths_per_second(self) -> None:
        k = _measure([_travelling_wave(frequency=0.3, wavelength=0.65, n_steps=10)])
        assert k.speed_bl_per_s == pytest.approx(0.1)
        assert in_bands(k) == {"frequency": True, "wavelength": True, "speed": False}

    def test_jitter_inside_the_band_is_not_an_undulation(self) -> None:
        jitter = 0.2 * np.sin(np.linspace(0.0, 40.0 * np.pi, 400))
        assert band_crossings(jitter) == 0
        assert band_crossings(np.sin(np.linspace(0.0, 4.0 * np.pi, 400))) == 3

    def test_reversals_are_counted_and_left_out_of_the_wave(self) -> None:
        forward = _travelling_wave(frequency=0.3, wavelength=0.65, n_steps=10)
        backward = _travelling_wave(frequency=0.3, wavelength=0.65, n_steps=10, direction=-1.0)
        k = _measure([forward[:5] + backward[:5] + forward[5:]])
        assert k.reversal_fraction == pytest.approx(5 / 15)
        assert k.frequency_hz == pytest.approx(0.3, rel=0.1)

    def test_steps_near_a_wall_are_excluded(self) -> None:
        episode = _travelling_wave(
            frequency=0.3,
            wavelength=0.65,
            n_steps=10,
            start=(0.5, 50.0),
            velocity=(0.0, 0.0),
        )
        k = _measure([episode])
        assert k.steps_near_wall == 10
        assert k.steps_used == 0
        assert k.speed_bl_per_s is None
        assert k.frequency_hz is None
        assert in_bands(k)["speed"] is None


class TestCapture:
    def _env(self, substeps: int = 20) -> Continuous2DEnvironment:
        return Continuous2DEnvironment(
            continuous=Continuous2DParams(
                world_size_mm=1000.0,
                allow_reversal=True,
                body_model="kinematic",
                body_substeps=substeps,
            ),
            seed=0,
        )

    def test_capture_is_off_by_default(self) -> None:
        env = self._env()
        env.move_agent_body(np.r_[np.zeros(DRIVE_WIDTH - 1), 1.0])
        assert env.posture_log is None

    def test_capture_records_each_substep_without_changing_motion(self) -> None:
        drive = np.r_[np.zeros(DRIVE_WIDTH - 1), 1.0]
        plain, captured = self._env(), self._env()
        captured.posture_log = []
        for _ in range(5):
            plain.move_agent_body(drive)
            captured.move_agent_body(drive)
        assert len(captured.posture_log) == 5
        assert len(captured.posture_log[0]["substeps"]) == 20  # type: ignore[arg-type]
        assert plain.agents["default"].pos_continuous == captured.agents["default"].pos_continuous

    def test_the_substep_count_is_configurable(self) -> None:
        env = self._env(substeps=40)
        env.posture_log = []
        env.move_agent_body(np.r_[np.zeros(DRIVE_WIDTH - 1), 1.0])
        assert len(env.posture_log[0]["substeps"]) == 40  # type: ignore[arg-type]

    def test_the_body_crawls_inside_the_bands_it_was_built_for(self) -> None:
        env = self._env()
        env.posture_log = []
        for _ in range(40):
            env.move_agent_body(np.r_[np.zeros(DRIVE_WIDTH - 1), 1.0])
        k = measure(
            [env.posture_log],
            world_size_mm=1000.0,
            body_length_mm=1.0,
            step_seconds=_STEP_S,
            reversal_threshold=0.5,
        )
        bands = in_bands(k)
        assert bands["frequency"]
        assert bands["wavelength"]
