"""
Temperature field module for thermotaxis simulation.

This module provides spatial temperature distributions for simulating
C. elegans thermotaxis behavior. Real worms navigate toward their
cultivation temperature (Tc) using AFD thermosensory neurons.

The implementation supports:
- Linear temperature gradients (configurable direction and strength)
- Hot spots and cold spots (localized temperature perturbations)
- Temperature zone classification for reward/penalty systems
- Gradient vector computation for navigation
"""

from enum import Enum

import numpy as np
from pydantic.dataclasses import dataclass


class TemperatureZone(Enum):
    """
    Temperature zones for C. elegans thermotaxis simulation.

    Zones are defined relative to the cultivation temperature (Tc).
    Real C. elegans exhibit strong preference for their cultivation
    temperature and avoid temperatures outside their comfort range.

    Zone thresholds (configurable):
    - LETHAL_COLD: < Tc - 15°C (e.g., < 5°C for Tc=20°C)
    - DANGER_COLD: Tc - 15°C to Tc - 10°C
    - DISCOMFORT_COLD: Tc - 10°C to Tc - 5°C
    - COMFORT: Tc - 5°C to Tc + 5°C (preferred range)
    - DISCOMFORT_HOT: Tc + 5°C to Tc + 10°C
    - DANGER_HOT: Tc + 10°C to Tc + 15°C
    - LETHAL_HOT: > Tc + 15°C (e.g., > 35°C for Tc=20°C)
    """

    LETHAL_COLD = "lethal_cold"
    DANGER_COLD = "danger_cold"
    DISCOMFORT_COLD = "discomfort_cold"
    COMFORT = "comfort"
    DISCOMFORT_HOT = "discomfort_hot"
    DANGER_HOT = "danger_hot"
    LETHAL_HOT = "lethal_hot"


@dataclass
class TemperatureZoneThresholds:
    """
    Configurable thresholds for temperature zone boundaries.

    All thresholds are relative to cultivation temperature (Tc).
    For example, with Tc=20°C and default thresholds:
    - comfort_low = 15°C (Tc - 5)
    - comfort_high = 25°C (Tc + 5)
    """

    comfort_delta: float = 5.0  # °C from Tc for comfort zone boundary
    discomfort_delta: float = 10.0  # °C from Tc for discomfort zone boundary
    danger_delta: float = 15.0  # °C from Tc for danger zone boundary


@dataclass
class TemperatureField:
    """
    Defines spatial temperature distribution for thermotaxis simulation.

    The temperature at any position is computed as:
    T(x, y) = base_temperature
              + gradient contribution (linear)
              + hot spot contributions (exponential falloff)
              + cold spot contributions (exponential falloff)

    Attributes
    ----------
    grid_size : int
        Size of the simulation grid.
    base_temperature : float
        Base temperature in °C, typically the cultivation temperature.
        Default is 20.0°C, a common C. elegans cultivation temperature.
    gradient_direction : float
        Direction of increasing temperature in radians.
        0 = temperature increases to the right (+x direction).
    gradient_strength : float
        Temperature change per cell in °C. Default 0.5°C/cell.
    hot_spots : list[tuple[int, int, float]]
        List of (x, y, intensity) tuples for localized hot spots.
        Intensity is the temperature increase at the center in °C.
    cold_spots : list[tuple[int, int, float]]
        List of (x, y, intensity) tuples for localized cold spots.
        Intensity is the temperature decrease at the center in °C (positive value).
    spot_decay_constant : float
        Decay constant for hot/cold spot exponential falloff.
        Controls how quickly spot effects diminish with distance.
    """

    grid_size: int
    base_temperature: float = 20.0
    gradient_direction: float = 0.0
    gradient_strength: float = 0.5
    hot_spots: list[tuple[int, int, float]] | None = None
    cold_spots: list[tuple[int, int, float]] | None = None
    spot_decay_constant: float = 5.0

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.hot_spots is None:
            self.hot_spots = []
        if self.cold_spots is None:
            self.cold_spots = []

    def get_temperature(self, position: tuple[int, int]) -> float:
        """
        Compute temperature at a given position.

        Parameters
        ----------
        position : tuple[int, int]
            (x, y) position to query.

        Returns
        -------
        float
            Temperature in °C at the given position.
        """
        x, y = position

        # Linear gradient contribution
        # Project position onto gradient direction vector
        gradient_component = x * np.cos(self.gradient_direction) + y * np.sin(
            self.gradient_direction,
        )
        temp = self.base_temperature + gradient_component * self.gradient_strength

        # Hot spot contributions (additive, exponential falloff)
        for hx, hy, intensity in self.hot_spots or []:
            distance = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
            temp += intensity * np.exp(-distance / self.spot_decay_constant)

        # Cold spot contributions (subtractive, exponential falloff)
        for cx, cy, intensity in self.cold_spots or []:
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            temp -= intensity * np.exp(-distance / self.spot_decay_constant)

        return float(temp)

    def get_gradient(self, position: tuple[int, int]) -> tuple[float, float]:
        """
        Compute temperature gradient vector at a given position.

        Uses central difference approximation:
        ∂T/∂x ≈ (T(x+1, y) - T(x-1, y)) / 2
        ∂T/∂y ≈ (T(x, y+1) - T(x, y-1)) / 2

        Parameters
        ----------
        position : tuple[int, int]
            (x, y) position to query gradient at.

        Returns
        -------
        tuple[float, float]
            (dx, dy) gradient vector pointing toward increasing temperature.
        """
        x, y = position

        # Central difference for x component
        # Clamp to grid boundaries
        x_plus = min(x + 1, self.grid_size - 1)
        x_minus = max(x - 1, 0)
        dx = (self.get_temperature((x_plus, y)) - self.get_temperature((x_minus, y))) / 2.0

        # Central difference for y component
        y_plus = min(y + 1, self.grid_size - 1)
        y_minus = max(y - 1, 0)
        dy = (self.get_temperature((x, y_plus)) - self.get_temperature((x, y_minus))) / 2.0

        return float(dx), float(dy)

    def get_gradient_polar(self, position: tuple[int, int]) -> tuple[float, float]:
        """
        Compute temperature gradient in polar coordinates.

        Parameters
        ----------
        position : tuple[int, int]
            (x, y) position to query gradient at.

        Returns
        -------
        tuple[float, float]
            (magnitude, direction) where:
            - magnitude: gradient strength in °C per cell
            - direction: angle in radians pointing toward increasing temperature
        """
        dx, dy = self.get_gradient(position)
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx) if magnitude > 0 else 0.0
        return float(magnitude), float(direction)

    def get_zone(  # noqa: PLR0911
        self,
        temperature: float,
        cultivation_temperature: float,
        thresholds: TemperatureZoneThresholds | None = None,
    ) -> TemperatureZone:
        """
        Classify a temperature into a zone relative to cultivation temperature.

        Parameters
        ----------
        temperature : float
            Current temperature in °C.
        cultivation_temperature : float
            The worm's cultivation/preferred temperature in °C.
        thresholds : TemperatureZoneThresholds | None
            Zone boundary thresholds. Uses defaults if None.

        Returns
        -------
        TemperatureZone
            The zone classification for this temperature.
        """
        if thresholds is None:
            thresholds = TemperatureZoneThresholds()

        deviation = temperature - cultivation_temperature

        # Cold side (negative deviation)
        if deviation < -thresholds.danger_delta:
            return TemperatureZone.LETHAL_COLD
        if deviation < -thresholds.discomfort_delta:
            return TemperatureZone.DANGER_COLD
        if deviation < -thresholds.comfort_delta:
            return TemperatureZone.DISCOMFORT_COLD

        # Hot side (positive deviation)
        if deviation > thresholds.danger_delta:
            return TemperatureZone.LETHAL_HOT
        if deviation > thresholds.discomfort_delta:
            return TemperatureZone.DANGER_HOT
        if deviation > thresholds.comfort_delta:
            return TemperatureZone.DISCOMFORT_HOT

        return TemperatureZone.COMFORT
