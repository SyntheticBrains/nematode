"""Kinematic instruments for the segmented body: undulation frequency, wavelength, speed, reversals.

Each instrument reads a captured episode, a list of steps that each carry the drive the body
received, its sub-steps' ``(time, curvature, head)`` and whether it ran tail-to-head. Steps whose
head lies within a margin of a wall are excluded, since a body pressed against an edge is clamped
there and its kinematics describe the wall, not the crawl.

* **Undulation frequency** counts the mid-body curvature's crossings of its own mean, a crossing
  counting only after the curvature has left a band around that mean, so jitter near the mean is not
  read as an undulation. Two crossings make one cycle.
* **Wavelength** follows the wave down the body: each segment crosses its mean upward a fixed delay
  after the segment in front of it, so the wave's speed along the body is one segment's length over
  that delay, and the wavelength is that speed over the frequency.
* **Speed** is the head's displacement per step over the step's duration, in body lengths per
  second, from each episode's second step on.
* **Reversal fraction** is the share of steps the body ran tail-to-head.

Frequency and wavelength are read only on **undulating** steps: forward-running, clear of the walls,
and with every segment's wave amplitude at least ``AMPLITUDE_FLOOR`` of the peak. A segment whose
wave the drive silences holds still at its steering offset and crosses its mean only when the drive
changes between steps, which the crossing delays would read as a wave. Real-worm analyses likewise
read forward runs. Each unbroken stretch of undulating steps is measured on its own, so no reading
joins two pieces of signal. The share of steps that qualify is reported beside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.env.body import MIN_WAVE_AMPLITUDE, wave_amplitude

if TYPE_CHECKING:
    from collections.abc import Sequence

AMPLITUDE_FLOOR = 0.25
"""The least wave amplitude, as a share of the peak, every segment needs for a step to undulate."""

BAND_KAPPA_L = 0.31
"""Half-width of the band a curvature must leave before a mean crossing counts, in kappa * L."""

# The adopted kinematic bands for crawling on agar.
FREQUENCY_BAND_HZ = (0.2, 0.45)
WAVELENGTH_BAND_BL = (0.5, 0.8)
SPEED_BAND_BL_PER_S = (0.12, 0.3)


@dataclass(frozen=True)
class Kinematics:
    """One evaluation's kinematic readings; ``None`` where too little motion was captured."""

    frequency_hz: float | None
    wavelength_bl: float | None
    speed_bl_per_s: float | None
    reversal_fraction: float
    steps_used: int
    steps_near_wall: int
    steps_undulating: int


def _near_wall(head: np.ndarray, world_size_mm: float, margin_mm: float) -> bool:
    return bool(np.any(head < margin_mm) or np.any(head > world_size_mm - margin_mm))


def band_crossings(signal: np.ndarray, band: float = BAND_KAPPA_L) -> int:
    """Count crossings of the signal's mean, each counted only after leaving ``mean +- band``."""
    centred = signal - signal.mean()
    count, armed, side = 0, False, 0
    for value in centred:
        if abs(value) > band:
            new_side = 1 if value > 0 else -1
            if armed and new_side != side:
                count += 1
            armed, side = True, new_side
    return count


def _upward_crossing_times(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    centred = signal - signal.mean()
    idx = np.nonzero((centred[:-1] < 0) & (centred[1:] >= 0))[0]
    # Linear interpolation of the crossing instant between the bracketing samples.
    fraction = -centred[idx] / (centred[idx + 1] - centred[idx])
    return times[idx] + fraction * (times[idx + 1] - times[idx])


def wave_speed(times: np.ndarray, curvature: np.ndarray, segment_length_bl: float) -> float | None:
    """Return the wave's speed along the body, in body lengths per second, from crossing delays.

    Each segment's delay is read against the segment in front of it, a fraction of a period, and the
    delays are summed down the body. Reading every segment against the head instead would wrap past
    a full period wherever the wave needs longer than one to reach that segment.
    """
    cumulative = [0.0]
    for segment in range(1, curvature.shape[1]):
        front = _upward_crossing_times(times, curvature[:, segment - 1])
        back = _upward_crossing_times(times, curvature[:, segment])
        delays = [float(later[0] - t0) for t0 in front if (later := back[back >= t0]).size]
        if not delays:
            return None
        cumulative.append(cumulative[-1] + float(np.median(delays)))
    positions = np.arange(curvature.shape[1]) * segment_length_bl
    slope = float(np.polyfit(positions, cumulative, 1)[0])  # seconds per body length
    return None if slope <= 0 else 1.0 / slope


def measure(  # noqa: PLR0913 - an episode and the geometry it was captured in
    episodes: Sequence[Sequence[dict[str, Any]]],
    *,
    world_size_mm: float,
    body_length_mm: float,
    step_seconds: float,
    reversal_threshold: float,
    wall_margin_mm: float = 1.0,
    min_wave_amplitude: float = MIN_WAVE_AMPLITUDE,
) -> Kinematics:
    """Read the instruments over one or more captured episodes.

    ``min_wave_amplitude`` is the body's floor on a segment's wave, so the amplitude each step is
    gated on is the one the body used. Under a floor at or above ``AMPLITUDE_FLOOR`` every forward
    step undulates.
    """
    freq_counts, freq_time = 0, 0.0
    speeds: list[float] = []
    wave_speeds: list[float] = []
    reversals, total, near_wall, undulating = 0, 0, 0, 0
    stretches: list[list[dict[str, Any]]] = []
    for episode in episodes:
        stretch: list[dict[str, Any]] = []
        previous_head: np.ndarray | None = None
        for step in episode:
            heads = np.array([s[2] for s in step["substeps"]])
            total += 1
            # Clipped as the body clips it, so the amplitude read is the one the body used.
            drive = np.clip(np.asarray(step["drive"], dtype=float), -1.0, 1.0)
            # The executed direction where the capture has it; the requested one otherwise.
            reversed_wave = bool(step.get("reversed", drive[-1] < -reversal_threshold))
            reversals += int(reversed_wave)
            start, previous_head = previous_head, heads[-1]
            if _near_wall(heads[-1], world_size_mm, wall_margin_mm) or (
                start is not None and _near_wall(start, world_size_mm, wall_margin_mm)
            ):
                near_wall += 1
                stretches.append(stretch)
                stretch = []
                continue
            # An episode's first step has no recorded pose before it, so it carries no speed.
            if start is not None:
                displacement = float(np.linalg.norm(heads[-1] - start))
                speeds.append(displacement / body_length_mm / step_seconds)
            amplitude = wave_amplitude(drive, min_wave_amplitude)
            if reversed_wave or amplitude.min() < AMPLITUDE_FLOOR:
                stretches.append(stretch)
                stretch = []
            else:
                stretch.append(step)
                undulating += 1
        stretches.append(stretch)
    for stretch in stretches:
        if not stretch:
            continue
        substeps = [s for step in stretch for s in step["substeps"]]
        times = np.array([s[0] for s in substeps])
        curvature = np.array([s[1] for s in substeps])
        freq_counts += band_crossings(curvature[:, curvature.shape[1] // 2])
        freq_time += float(times[-1] - times[0])
        speed = wave_speed(times, curvature, 1.0 / curvature.shape[1])
        if speed is not None:
            wave_speeds.append(speed)
    frequency = (freq_counts / 2.0) / freq_time if freq_time > 0 and freq_counts else None
    wave = float(np.mean(wave_speeds)) if wave_speeds else None
    wavelength = wave / frequency if (wave is not None and frequency) else None
    return Kinematics(
        frequency_hz=frequency,
        wavelength_bl=wavelength,
        speed_bl_per_s=float(np.mean(speeds)) if speeds else None,
        reversal_fraction=reversals / total if total else 0.0,
        steps_used=total - near_wall,
        steps_near_wall=near_wall,
        steps_undulating=undulating,
    )


def in_bands(kinematics: Kinematics) -> dict[str, bool | None]:
    """Report each instrument against its adopted band; ``None`` where it could not be read."""

    def within(value: float | None, band: tuple[float, float]) -> bool | None:
        return None if value is None else band[0] <= value <= band[1]

    return {
        "frequency": within(kinematics.frequency_hz, FREQUENCY_BAND_HZ),
        "wavelength": within(kinematics.wavelength_bl, WAVELENGTH_BAND_BL),
        "speed": within(kinematics.speed_bl_per_s, SPEED_BAND_BL_PER_S),
    }
