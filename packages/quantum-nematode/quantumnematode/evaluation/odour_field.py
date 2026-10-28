"""A measured odour field for an assay off food, read through the worm's usual odour senses.

The batched evaluator's lawn odour is a Fick-smoothed lawn density. A chemotaxis assay has no lawn:
its odour is a measured landscape. :class:`OdourField` holds one on its plate, and the evaluator
reads it wherever no lawn is out, through the same saturating map lawn odour takes,
``tanh(C / C_max)``, so concentration, the head-sweep difference and dC/dt follow from it as they
do from a lawn. Positions are in the plate's frame; the measured field covers a rectangle inside
the plate and holds its edge value beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumnematode.benchmark.references import load_chen_tracks, sample_field

CHEN_PLATE_MM = 100.0
CHEN_START_WINDOW_S = 30
CHEN_START_STREAM = 2038
GRADIENT_STEP_MM = 0.15


@dataclass(frozen=True)
class OdourField:
    """A gridded odour field: ``field[y, x]`` on cells of ``cell_mm``, from ``origin_mm``."""

    field: np.ndarray
    cell_mm: float
    origin_mm: tuple[float, float]
    c_max: float

    @property
    def extent_mm(self) -> tuple[float, float]:
        """Return the measured rectangle's width and height."""
        rows, cols = self.field.shape
        return (cols - 1) * self.cell_mm, (rows - 1) * self.cell_mm

    def to_field_frame(self, points: np.ndarray) -> np.ndarray:
        """Return plate positions in the measured field's own frame."""
        return np.asarray(points, dtype=float) - np.asarray(self.origin_mm)

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Return which plate positions lie inside the measured rectangle."""
        p = self.to_field_frame(points)
        width, height = self.extent_mm
        return (p[:, 0] >= 0) & (p[:, 0] <= width) & (p[:, 1] >= 0) & (p[:, 1] <= height)

    def concentration(self, points: np.ndarray) -> np.ndarray:
        """Return the measured concentration at plate positions, in the field's own units."""
        p = self.to_field_frame(points)
        return sample_field(self.field, self.cell_mm, p[:, 0], p[:, 1])

    def odour(self, points: np.ndarray) -> np.ndarray:
        """Return the odour the worm senses at plate positions: ``tanh(C / C_max)``."""
        return np.tanh(self.concentration(points) / self.c_max)

    def gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the sensed odour's gradient direction and tanh'd strength."""
        p = np.asarray(points, dtype=float)
        h = GRADIENT_STEP_MM
        ex, ey = np.array([h, 0.0]), np.array([0.0, h])
        gx = (self.odour(p + ex) - self.odour(p - ex)) / (2 * h)
        gy = (self.odour(p + ey) - self.odour(p - ey)) / (2 * h)
        magnitude = np.hypot(gx, gy)
        return np.where(magnitude > 0, np.arctan2(gy, gx), 0.0), np.tanh(magnitude)


def chen_field() -> OdourField:
    """Return Chen 2025's measured butanone field, centred on its 100 mm square plate."""
    chen = load_chen_tracks()
    rows, cols = chen.field.shape
    width, height = (cols - 1) * chen.field_cell_mm, (rows - 1) * chen.field_cell_mm
    origin = ((CHEN_PLATE_MM - width) / 2.0, (CHEN_PLATE_MM - height) / 2.0)
    return OdourField(
        field=chen.field,
        cell_mm=chen.field_cell_mm,
        origin_mm=origin,
        c_max=float(np.nanmax(chen.field)),
    )


def chen_starts(seeds: list[int], field: OdourField) -> np.ndarray:
    """Return each animal's starting position on the plate, drawn from Chen's real starts.

    The pool is the first positions of standard-protocol fragments that begin within the first
    :data:`CHEN_START_WINDOW_S` seconds; each animal draws one from its own stream.
    """
    chen = load_chen_tracks()
    standard = {r for r, p in enumerate(chen.recording_protocol) if p == "standard"}
    pool = np.array(
        [
            (x[0], y[0])
            for r, first, x, y in chen.fragments()
            if r in standard and first < CHEN_START_WINDOW_S and np.isfinite(x[0])
        ],
    )
    out = np.empty((len(seeds), 2))
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng([int(seed), CHEN_START_STREAM])
        out[i] = pool[rng.integers(len(pool))] + np.asarray(field.origin_mm)
    return out
