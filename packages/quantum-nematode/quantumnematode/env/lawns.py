"""Bacterial lawns: disc patches of food with a density grid, an odour field and intake.

A lawn is a disc of bacteria. Its interior is a grid of cells about a body length wide, each
holding a density that starts at 1 and falls as the worm eats from it, so a lawn can be grazed out
in one place and stay full in another. Its edge is where the grid ends: beyond it there is odour
but no food.

* **Odour.** Each cell contributes its remaining density times its share of the lawn's area, times
  the food field's per-source kernel at its distance. A full lawn therefore smells like one point
  source of unit strength, and a grazed region smells weaker, so the gradient inside a lawn points
  toward what is left.
* **Intake.** A worm inside a lawn eats a fixed fraction of the density of the cell it is on, and
  the cell loses it. What the worm gains is that amount times the lawn's quality. Intake does not
  depend on the worm's speed.
* **Quality** is a lawn's nutritional value per unit eaten. It does not change the odour: a worm
  learns it by eating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MAX_PLACEMENT_ATTEMPTS = 2_000
# A greedy placement can strand itself with room left for none; it then starts afresh.
MAX_PLACEMENT_RESTARTS = 50


@dataclass(frozen=True)
class LawnParams:
    """The lawn food model's geometry, intake and nutrition."""

    count: int = 3
    radius_mm: float = 2.5
    min_separation_mm: float = 2.0
    cell_mm: float = 1.0
    quality: tuple[float, float] = (1.0, 1.0)
    intake_fraction: float = 0.1
    reward_per_intake: float = 1.0
    satiety_per_intake: float = 0.1
    regrowth_per_step: float = 0.0
    start_clearance_mm: float = 2.0
    wall_clearance_mm: float = 1.0


@dataclass(frozen=True)
class Intake:
    """One step's eating: the density removed, its value after quality, and its lawn."""

    amount: float
    value: float
    lawn: int | None


NO_INTAKE = Intake(amount=0.0, value=0.0, lawn=None)


def _disc_cells(centre: np.ndarray, radius: float, cell: float) -> np.ndarray:
    """Return the centres of the grid cells, ``cell`` wide and centred on the disc, inside it."""
    half = math.floor(radius / cell)
    offsets = np.arange(-half, half + 1) * cell
    grid = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
    inside = np.hypot(grid[:, 0], grid[:, 1]) <= radius
    return centre + grid[inside]


class LawnField:
    """Every lawn's cells as flat arrays, with the odour field, intake and regrowth over them."""

    def __init__(
        self,
        centres: np.ndarray,
        radius_mm: float,
        cell_mm: float,
        qualities: np.ndarray,
    ) -> None:
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 2)
        self.radius_mm = float(radius_mm)
        self.cell_mm = float(cell_mm)
        self.qualities = np.asarray(qualities, dtype=float)
        cells = [_disc_cells(c, self.radius_mm, self.cell_mm) for c in self.centres]
        self.cells = np.vstack(cells) if cells else np.zeros((0, 2))
        self.lawn_of = np.concatenate(
            [np.full(len(c), i, dtype=int) for i, c in enumerate(cells)] or [np.zeros(0, int)],
        )
        # Each cell's share of its lawn's area: equal cells, so one over the lawn's cell count.
        counts = np.array([len(c) for c in cells], dtype=float)
        self.weights = 1.0 / counts[self.lawn_of] if len(self.lawn_of) else np.zeros(0)
        self.density = np.ones(len(self.cells))

    @classmethod
    def place(
        cls,
        params: LawnParams,
        rng: np.random.Generator,
        *,
        world_size_mm: float,
        start: Sequence[tuple[float, float]],
    ) -> LawnField:
        """Place ``params.count`` lawns inside the arena, apart from each other and the start.

        Each lawn's edge lies at least ``wall_clearance_mm`` from the walls,
        ``min_separation_mm`` from every other lawn's edge and ``start_clearance_mm`` from every
        start position. Qualities are drawn uniformly from ``params.quality``.

        Raises
        ------
        ValueError
            If the lawns cannot be placed: the arena is too small for them.
        """
        r = params.radius_mm
        lo, hi = r + params.wall_clearance_mm, world_size_mm - r - params.wall_clearance_mm
        if hi < lo:
            msg = f"a lawn of radius {r} mm does not fit a {world_size_mm} mm arena"
            raise ValueError(msg)
        centres: list[np.ndarray] = []
        for _ in range(MAX_PLACEMENT_RESTARTS):
            centres = []
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                if len(centres) == params.count:
                    break
                candidate = rng.uniform(lo, hi, size=2)
                clear_of_lawns = all(
                    np.hypot(*(candidate - c)) >= 2 * r + params.min_separation_mm for c in centres
                )
                clear_of_start = all(
                    np.hypot(candidate[0] - x, candidate[1] - y) >= r + params.start_clearance_mm
                    for x, y in start
                )
                if clear_of_lawns and clear_of_start:
                    centres.append(candidate)
            if len(centres) == params.count:
                break
        if len(centres) < params.count:
            msg = (
                f"placed {len(centres)} of {params.count} lawns of radius {r} mm in a "
                f"{world_size_mm} mm arena; the arena is too small for them"
            )
            raise ValueError(msg)
        q_lo, q_hi = params.quality
        qualities = (
            rng.uniform(q_lo, q_hi, size=params.count)
            if q_hi > q_lo
            else np.full(
                params.count,
                q_lo,
            )
        )
        return cls(np.array(centres), r, params.cell_mm, qualities)

    def copy(self) -> LawnField:
        """Return an independent copy, densities included."""
        new = LawnField.__new__(LawnField)
        new.centres = self.centres.copy()
        new.radius_mm = self.radius_mm
        new.cell_mm = self.cell_mm
        new.qualities = self.qualities.copy()
        new.cells = self.cells.copy()
        new.lawn_of = self.lawn_of.copy()
        new.weights = self.weights.copy()
        new.density = self.density.copy()
        return new

    def _offsets(self, position: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = self.cells[:, 0] - float(position[0])
        dy = self.cells[:, 1] - float(position[1])
        return dx, dy, np.hypot(dx, dy)

    def concentration(
        self,
        position: Sequence[float],
        kernel: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Return the lawns' raw odour at ``position``: density, area share and kernel per cell."""
        if not len(self.cells):
            return 0.0
        _, _, distance = self._offsets(position)
        return float(np.sum(self.density * self.weights * kernel(distance)))

    def gradient(
        self,
        position: Sequence[float],
        kernel: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[float, float]:
        """Return the lawns' odour vector at ``position``, toward each cell by its contribution."""
        if not len(self.cells):
            return 0.0, 0.0
        dx, dy, distance = self._offsets(position)
        away = distance > 0
        strength = self.density[away] * self.weights[away] * kernel(distance[away])
        return (
            float(np.sum(strength * dx[away] / distance[away])),
            float(np.sum(strength * dy[away] / distance[away])),
        )

    def lawn_at(self, position: Sequence[float]) -> int | None:
        """Return the index of the lawn whose disc contains ``position``, or None."""
        if not len(self.centres):
            return None
        inside = np.hypot(*(self.centres - np.asarray(position, dtype=float)).T) <= self.radius_mm
        hits = np.flatnonzero(inside)
        return int(hits[0]) if len(hits) else None

    def cell_at(self, position: Sequence[float]) -> int | None:
        """Return the flat index of the cell under ``position``, or None off every lawn."""
        lawn = self.lawn_at(position)
        if lawn is None:
            return None
        members = np.flatnonzero(self.lawn_of == lawn)
        _, _, distance = self._offsets(position)
        return int(members[np.argmin(distance[members])])

    def eat(self, position: Sequence[float], fraction: float) -> Intake:
        """Eat ``fraction`` of the density of the cell under ``position``; the cell loses it."""
        cell = self.cell_at(position)
        if cell is None:
            return NO_INTAKE
        amount = float(fraction * self.density[cell])
        self.density[cell] -= amount
        lawn = int(self.lawn_of[cell])
        return Intake(amount=amount, value=amount * float(self.qualities[lawn]), lawn=lawn)

    def regrow(self, per_step: float) -> None:
        """Regrow every cell's density by ``per_step``, up to 1."""
        if per_step > 0:
            np.minimum(self.density + per_step, 1.0, out=self.density)

    def remaining(self) -> np.ndarray:
        """Return each lawn's remaining food as a fraction of its full amount."""
        return np.bincount(
            self.lawn_of,
            weights=self.density * self.weights,
            minlength=len(
                self.centres,
            ),
        )
