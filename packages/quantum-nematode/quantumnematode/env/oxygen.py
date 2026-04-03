"""
Oxygen field module for aerotaxis simulation.

This module provides spatial oxygen concentration distributions for simulating
C. elegans aerotaxis behavior. Real worms prefer moderate oxygen levels (5-12%)
and avoid both hypoxic and hyperoxic conditions.

Oxygen-sensing neurons:
- URX, AQR, PQR: detect hyperoxia (>12% O2)
- BAG: detect hypoxia (<5% O2)

The implementation supports:
- Linear oxygen gradients (configurable direction and strength)
- High-oxygen spots (ventilation/surface points) and low-oxygen spots (bacterial sinks)
- Asymmetric oxygen zone classification (5 zones with absolute thresholds)
- Gradient vector computation for navigation
- O2 values clamped to [0.0, 21.0] (atmospheric maximum)

References:
- Cheung BH, et al. (2005). Cell 123(1):157-171
- Zimmer M, et al. (2009). Neuron 61(6):865-879
- Gray JM, et al. (2004). Nature 430(6997):317-322
"""

from enum import Enum

import numpy as np
from pydantic.dataclasses import dataclass

from quantumnematode.dtypes import GradientPolar, GradientVector, GridPosition, OxygenSpot

# Atmospheric oxygen maximum (%)
MAX_OXYGEN = 21.0
MIN_OXYGEN = 0.0


class OxygenZone(Enum):
    """
    Oxygen zones for C. elegans aerotaxis simulation.

    Unlike temperature zones (symmetric around cultivation temperature),
    oxygen zones use absolute percentage thresholds reflecting the biological
    asymmetry between hypoxia and hyperoxia responses.

    Zone thresholds (configurable, defaults):
    - LETHAL_HYPOXIA: < 2% O2 (anaerobic, lethal)
    - DANGER_HYPOXIA: 2-5% O2 (BAG neuron activation)
    - COMFORT: 5-12% O2 (preferred range, URX/BAG quiescent)
    - DANGER_HYPEROXIA: 12-17% O2 (URX/AQR/PQR neuron activation)
    - LETHAL_HYPEROXIA: > 17% O2 (oxidative stress)
    """

    LETHAL_HYPOXIA = "lethal_hypoxia"
    DANGER_HYPOXIA = "danger_hypoxia"
    COMFORT = "comfort"
    DANGER_HYPEROXIA = "danger_hyperoxia"
    LETHAL_HYPEROXIA = "lethal_hyperoxia"


@dataclass
class OxygenZoneThresholds:
    """
    Configurable thresholds for oxygen zone boundaries.

    All thresholds are absolute O2 percentages (not relative to a reference).
    This differs from temperature zones which use deltas from cultivation temperature.
    """

    lethal_hypoxia_upper: float = 2.0  # Below this is lethal hypoxia
    danger_hypoxia_upper: float = 5.0  # Below this is danger hypoxia
    comfort_lower: float = 5.0  # Lower bound of comfort zone
    comfort_upper: float = 12.0  # Upper bound of comfort zone
    danger_hyperoxia_upper: float = 17.0  # Above this is lethal hyperoxia


@dataclass
class OxygenField:
    """
    Defines spatial oxygen concentration distribution for aerotaxis simulation.

    The oxygen at any position is computed as:
    O2(x, y) = base_oxygen
               + gradient contribution (linear, center-relative)
               + high oxygen spot contributions (exponential falloff)
               - low oxygen spot contributions (exponential falloff)

    Values are clamped to [0.0, 21.0] (atmospheric maximum).

    The gradient is computed relative to the grid center, so the
    base_oxygen is at the center of the grid. This ensures agents
    spawning at the center start at the base oxygen level.

    Attributes
    ----------
    grid_size : int
        Size of the simulation grid.
    base_oxygen : float
        Base O2 percentage at the grid center (spawn point).
        Default is 10.0%, center of the C. elegans comfort range (5-12%).
    gradient_direction : float
        Direction of increasing oxygen in radians.
        0 = oxygen increases to the right (+x direction).
    gradient_strength : float
        O2 percentage change per cell. Default 0.1%/cell.
    high_oxygen_spots : list[tuple[int, int, float]]
        List of (x, y, intensity) tuples for localized high-oxygen areas.
        Represents ventilation points or surface exposure.
    low_oxygen_spots : list[tuple[int, int, float]]
        List of (x, y, intensity) tuples for localized low-oxygen areas.
        Represents bacterial consumption sinks.
    spot_decay_constant : float
        Decay constant for spot exponential falloff.
    """

    grid_size: int
    base_oxygen: float = 10.0
    gradient_direction: float = 0.0
    gradient_strength: float = 0.1
    high_oxygen_spots: list[OxygenSpot] | None = None
    low_oxygen_spots: list[OxygenSpot] | None = None
    spot_decay_constant: float = 5.0

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.high_oxygen_spots is None:
            self.high_oxygen_spots = []
        if self.low_oxygen_spots is None:
            self.low_oxygen_spots = []

    def get_oxygen(self, position: GridPosition) -> float:
        """
        Compute oxygen concentration at a given position.

        The gradient is computed relative to the grid center, so the
        base_oxygen is at the center. Values are clamped to [0.0, 21.0].

        Parameters
        ----------
        position : GridPosition
            (x, y) position to query.

        Returns
        -------
        float
            O2 percentage at the given position, clamped to [0.0, 21.0].
        """
        x, y = position

        # Compute position relative to grid center
        center = self.grid_size / 2.0
        rel_x = x - center
        rel_y = y - center

        # Linear gradient contribution (center-relative)
        gradient_component = rel_x * np.cos(self.gradient_direction) + rel_y * np.sin(
            self.gradient_direction,
        )
        o2 = self.base_oxygen + gradient_component * self.gradient_strength

        # High oxygen spot contributions (additive, exponential falloff)
        for hx, hy, intensity in self.high_oxygen_spots or []:
            distance = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
            o2 += intensity * np.exp(-distance / self.spot_decay_constant)

        # Low oxygen spot contributions (subtractive, exponential falloff)
        for lx, ly, intensity in self.low_oxygen_spots or []:
            distance = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
            o2 -= intensity * np.exp(-distance / self.spot_decay_constant)

        return float(np.clip(o2, MIN_OXYGEN, MAX_OXYGEN))

    def get_gradient(self, position: GridPosition) -> GradientVector:
        """
        Compute oxygen gradient vector at a given position.

        Uses central difference approximation:
        ∂O2/∂x ≈ (O2(x+1, y) - O2(x-1, y)) / 2
        ∂O2/∂y ≈ (O2(x, y+1) - O2(x, y-1)) / 2

        Parameters
        ----------
        position : GridPosition
            (x, y) position to query gradient at.

        Returns
        -------
        GradientVector
            (dx, dy) gradient vector pointing toward increasing oxygen.
        """
        x, y = position

        # Central difference for x component
        x_plus = min(x + 1, self.grid_size - 1)
        x_minus = max(x - 1, 0)
        dx = (self.get_oxygen((x_plus, y)) - self.get_oxygen((x_minus, y))) / 2.0

        # Central difference for y component
        y_plus = min(y + 1, self.grid_size - 1)
        y_minus = max(y - 1, 0)
        dy = (self.get_oxygen((x, y_plus)) - self.get_oxygen((x, y_minus))) / 2.0

        return float(dx), float(dy)

    def get_gradient_polar(self, position: GridPosition) -> GradientPolar:
        """
        Compute oxygen gradient in polar coordinates.

        Parameters
        ----------
        position : GridPosition
            (x, y) position to query gradient at.

        Returns
        -------
        GradientPolar
            (magnitude, direction) where:
            - magnitude: gradient strength in O2 % per cell
            - direction: angle in radians pointing toward increasing oxygen
        """
        dx, dy = self.get_gradient(position)
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx) if magnitude > 0 else 0.0
        return float(magnitude), float(direction)

    def get_zone(  # noqa: PLR0911
        self,
        oxygen: float,
        thresholds: OxygenZoneThresholds | None = None,
    ) -> OxygenZone:
        """
        Classify an oxygen concentration into a zone.

        Uses absolute percentage thresholds (not relative to a reference).

        Parameters
        ----------
        oxygen : float
            Current O2 percentage.
        thresholds : OxygenZoneThresholds | None
            Zone boundary thresholds. Uses defaults if None.

        Returns
        -------
        OxygenZone
            The zone classification for this oxygen level.
        """
        if thresholds is None:
            thresholds = OxygenZoneThresholds()

        if oxygen < thresholds.lethal_hypoxia_upper:
            return OxygenZone.LETHAL_HYPOXIA
        if oxygen < thresholds.danger_hypoxia_upper:
            return OxygenZone.DANGER_HYPOXIA
        if oxygen > thresholds.danger_hyperoxia_upper:
            return OxygenZone.LETHAL_HYPEROXIA
        if oxygen > thresholds.comfort_upper:
            return OxygenZone.DANGER_HYPEROXIA

        return OxygenZone.COMFORT
