"""Pheromone field system for multi-agent chemical communication.

Models pheromone emission, diffusion (via point-source exponential decay),
and sensing for biologically grounded agent-agent communication.

C. elegans uses ascaroside pheromones detected by ASK, ADL, and ASI
chemosensory neurons for social behaviors including aggregation on
bacterial lawns and alarm signaling from injured conspecifics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# Scaling factor for tanh normalization of concentration values
PHEROMONE_SCALING_TANH_FACTOR = 1.0

# Number of half-lives after which sources are pruned (~3% remaining strength)
HALF_LIFE_PRUNE_FACTOR = 5


class PheromoneType(StrEnum):
    """Types of pheromone chemicals."""

    FOOD_MARKING = "food_marking"
    ALARM = "alarm"


@dataclass
class PheromoneSource:
    """A single pheromone emission event.

    Attributes
    ----------
    position : tuple[int, int]
        Grid position where pheromone was emitted.
    pheromone_type : PheromoneType
        Type of pheromone emitted.
    strength : float
        Emission strength (amplitude of the point source).
    emission_step : int
        Step number when the pheromone was emitted.
    emitter_id : str
        Agent ID of the emitter.
    """

    position: tuple[int, int]
    pheromone_type: PheromoneType
    strength: float
    emission_step: int
    emitter_id: str


class PheromoneField:
    """Dynamic chemical field using point-source exponential decay.

    Models pheromone concentration as the superposition of contributions from
    individual emission events. Each source decays spatially (with distance)
    and temporally (with age). Same math as food gradients — O(S) per query
    where S = number of active sources.

    Concentration at position P at step T:
        C(P, T) = tanh(Σ strength_i * exp(-dist_i / spatial_decay)
                       * exp(-age_i * ln(2) / half_life))

    Parameters
    ----------
    spatial_decay_constant : float
        Controls how quickly concentration decreases with Manhattan distance.
    temporal_half_life : float
        Number of steps for concentration to halve (temporal decay).
    max_sources : int
        Maximum number of active sources. Oldest pruned when exceeded.
    """

    def __init__(
        self,
        spatial_decay_constant: float,
        temporal_half_life: float,
        max_sources: int,
    ) -> None:
        if spatial_decay_constant <= 0:
            msg = f"spatial_decay_constant must be > 0, got {spatial_decay_constant}"
            raise ValueError(msg)
        if temporal_half_life <= 0:
            msg = f"temporal_half_life must be > 0, got {temporal_half_life}"
            raise ValueError(msg)
        if max_sources <= 0:
            msg = f"max_sources must be > 0, got {max_sources}"
            raise ValueError(msg)

        self.spatial_decay_constant = spatial_decay_constant
        self.temporal_half_life = temporal_half_life
        self.max_sources = max_sources
        self._max_age = int(temporal_half_life * HALF_LIFE_PRUNE_FACTOR)
        self._sources: list[PheromoneSource] = []

    @property
    def sources(self) -> list[PheromoneSource]:
        """Active pheromone sources (read-only view)."""
        return list(self._sources)

    @property
    def num_sources(self) -> int:
        """Number of active sources."""
        return len(self._sources)

    def add_source(self, source: PheromoneSource) -> None:
        """Add a pheromone emission source.

        If the source count exceeds max_sources, the oldest source is removed.
        """
        self._sources.append(source)
        while len(self._sources) > self.max_sources:
            self._sources.pop(0)

    def get_concentration(
        self,
        position: tuple[int, int],
        current_step: int,
    ) -> float:
        """Compute pheromone concentration at a position.

        Superposition of all active sources with spatial and temporal decay.
        Result is tanh-normalized to [0, 1].

        Parameters
        ----------
        position : tuple[int, int]
            Grid position to query.
        current_step : int
            Current simulation step (for temporal decay calculation).

        Returns
        -------
        float
            Pheromone concentration in [0, 1].
        """
        if not self._sources:
            return 0.0

        total = 0.0
        ln2 = math.log(2)
        for source in self._sources:
            distance = abs(position[0] - source.position[0]) + abs(
                position[1] - source.position[1],
            )
            age = max(0, current_step - source.emission_step)
            spatial_factor = math.exp(-distance / self.spatial_decay_constant)
            temporal_factor = math.exp(-age * ln2 / self.temporal_half_life)
            total += source.strength * spatial_factor * temporal_factor

        return float(np.tanh(total * PHEROMONE_SCALING_TANH_FACTOR))

    def get_gradient(
        self,
        position: tuple[int, int],
        current_step: int,
    ) -> tuple[float, float]:
        """Compute pheromone gradient via central differences.

        Parameters
        ----------
        position : tuple[int, int]
            Grid position to query.
        current_step : int
            Current simulation step.

        Returns
        -------
        tuple[float, float]
            Gradient vector (dx, dy).
        """
        x, y = position
        cx_plus = self.get_concentration((x + 1, y), current_step)
        cx_minus = self.get_concentration((x - 1, y), current_step)
        cy_plus = self.get_concentration((x, y + 1), current_step)
        cy_minus = self.get_concentration((x, y - 1), current_step)
        return ((cx_plus - cx_minus) / 2.0, (cy_plus - cy_minus) / 2.0)

    def get_gradient_polar(
        self,
        position: tuple[int, int],
        current_step: int,
    ) -> tuple[float, float]:
        """Compute pheromone gradient in polar coordinates.

        Parameters
        ----------
        position : tuple[int, int]
            Grid position to query.
        current_step : int
            Current simulation step.

        Returns
        -------
        tuple[float, float]
            (magnitude, direction_radians). Magnitude is tanh-normalized.
            Direction is 0 when magnitude is 0.
        """
        dx, dy = self.get_gradient(position, current_step)
        magnitude_raw = math.sqrt(dx**2 + dy**2)
        magnitude = float(np.tanh(magnitude_raw * PHEROMONE_SCALING_TANH_FACTOR))
        direction = math.atan2(dy, dx) if magnitude_raw > 0 else 0.0
        return (magnitude, direction)

    def prune(self, current_step: int) -> int:
        """Remove expired pheromone sources.

        Parameters
        ----------
        current_step : int
            Current simulation step.

        Returns
        -------
        int
            Number of sources removed.
        """
        original_count = len(self._sources)
        self._sources = [
            s for s in self._sources if (current_step - s.emission_step) <= self._max_age
        ]
        return original_count - len(self._sources)

    def clear(self) -> None:
        """Remove all sources."""
        self._sources.clear()
