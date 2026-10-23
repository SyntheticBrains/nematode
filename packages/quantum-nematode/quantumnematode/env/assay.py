"""Assay protocols: an episode as a sequence of phases, each with its food and the worm's placement.

A phase lasts a fixed number of worm-seconds. At its start the environment turns the lawns on,
freshly placed, or off (a bare plate), and places the worm:

- ``keep``: where it is;
- ``lawn_centre``: on the first lawn's centre;
- ``off_lawn``: at a uniform point on bare agar, clear of every lawn and of the wall;
- ``arena_centre``: at the arena's centre, as a transfer to a fresh plate.

Every placement but ``keep`` draws a uniform heading. The worm's satiety and its controller's state
carry across phases, as the animal's do across a transfer. One phase is the assay phase, the one a
reference records, with an analysis window that starts some seconds into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# How far the step grid may sit from a phase boundary and still count as on it.
_BOUNDARY_TOLERANCE_S = 1e-6
# A drawn off-lawn start keeps at least this far from the wall.
OFF_LAWN_WALL_MARGIN_MM = 1.0


@dataclass(frozen=True)
class AssayPhase:
    """One phase: its name, duration, food and where the worm is put at its start."""

    name: str
    duration_s: float
    food: str  # "lawns" or "none"
    place: str  # "keep", "lawn_centre", "off_lawn" or "arena_centre"


@dataclass(frozen=True)
class AssayParams:
    """A protocol: its phases in order, the assay phase, and where its analysis starts within it."""

    name: str
    phases: tuple[AssayPhase, ...]
    assay_phase: str
    analysis_from_s: float = 0.0

    def phase_starts_s(self) -> tuple[float, ...]:
        """Return each phase's start, in worm-seconds from the episode's start."""
        starts: list[float] = []
        elapsed = 0.0
        for phase in self.phases:
            starts.append(elapsed)
            elapsed += phase.duration_s
        return tuple(starts)

    def total_s(self) -> float:
        """Return the protocol's length in worm-seconds."""
        return sum(phase.duration_s for phase in self.phases)

    def phase_at(self, time_s: float) -> int:
        """Return the index of the phase running at ``time_s`` worm-seconds into the episode."""
        index = 0
        for i, start in enumerate(self.phase_starts_s()):
            if time_s + _BOUNDARY_TOLERANCE_S >= start:
                index = i
        return index

    def record(self, step_seconds: float) -> dict[str, object]:
        """Return the protocol's timing in steps, for a behaviour capture's file."""
        starts = self.phase_starts_s()
        names = [phase.name for phase in self.phases]
        index = names.index(self.assay_phase)
        assay_start = round(starts[index] / step_seconds)
        assay_steps = round(self.phases[index].duration_s / step_seconds)
        return {
            "name": self.name,
            "phases": [
                {
                    "name": phase.name,
                    "start_step": round(start / step_seconds),
                    "steps": round(phase.duration_s / step_seconds),
                    "food": phase.food,
                }
                for phase, start in zip(self.phases, starts, strict=True)
            ],
            "assay_phase": self.assay_phase,
            "analysis_steps": [
                assay_start + round(self.analysis_from_s / step_seconds),
                assay_start + assay_steps,
            ],
        }


def project_into_circle(
    x: float,
    y: float,
    world_size_mm: float,
) -> tuple[float, float]:
    """Return ``(x, y)`` moved radially onto the circular arena's wall if it lies outside it.

    The circle has diameter ``world_size_mm`` and its centre at the square's centre.
    """
    centre = radius = world_size_mm / 2.0
    dx, dy = x - centre, y - centre
    distance = math.hypot(dx, dy)
    if distance <= radius:
        return x, y
    scale = radius / distance
    return centre + dx * scale, centre + dy * scale
