"""Food consumption handling for the quantum nematode agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantumnematode.agent.satiety import SatietyManager
    from quantumnematode.env import DynamicForagingEnvironment


@dataclass
class FoodConsumptionResult:
    """Result of checking and potentially consuming food.

    Attributes
    ----------
    food_consumed : bool
        Whether food was consumed at the current position.
    satiety_restored : float
        Amount of satiety restored (0.0 if no food consumed).
    reward : float
        Reward for consuming food (0.0 if no food consumed).
    distance_efficiency : float | None
        For dynamic environments, the ratio of optimal distance to actual distance traveled.
        None when no food was consumed.
    health_restored : float
        Amount of HP restored by food (0.0 if no food consumed or health system disabled).
    """

    food_consumed: bool
    satiety_restored: float
    reward: float
    distance_efficiency: float | None = None
    health_restored: float = 0.0


class FoodConsumptionHandler:
    """Handles food consumption logic and environment-specific behavior.

    The food consumption handler abstracts food interaction logic, including
    detecting food consumption, restoring satiety, and calculating distance
    efficiency for dynamic environments.

    Parameters
    ----------
    env : DynamicForagingEnvironment
        The environment containing food.
    satiety_manager : SatietyManager
        The satiety manager to restore satiety when food is consumed.
    satiety_gain_fraction : float, optional
        Fraction of max satiety to restore per food consumed, by default 0.2.

    Attributes
    ----------
    env : DynamicForagingEnvironment
        The environment.
    satiety_manager : SatietyManager
        The satiety manager.
    satiety_gain_fraction : float
        Fraction of max satiety restored per food.
    _initial_distance : int | None
        For dynamic environments, the initial distance to the nearest food.
    _steps_since_last_food : int
        Number of steps taken since last food consumption.
    """

    def __init__(
        self,
        env: DynamicForagingEnvironment,
        satiety_manager: SatietyManager,
        satiety_gain_fraction: float = 0.2,
    ) -> None:
        """Initialize the food consumption handler.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The environment containing food.
        satiety_manager : SatietyManager
            The satiety manager to restore satiety when food is consumed.
        satiety_gain_fraction : float, optional
            Fraction of max satiety to restore per food consumed, by default 0.2.
        """
        self.env = env
        self.satiety_manager = satiety_manager
        self.satiety_gain_fraction = satiety_gain_fraction

        # For dynamic environment distance tracking
        self._initial_distance: int | None = None
        self._steps_since_last_food = 0

        # Initialize distance tracking
        self._initial_distance = env.get_nearest_food_distance()

    def track_step(self) -> None:
        """Track a step for distance efficiency calculation."""
        self._steps_since_last_food += 1

    def check_and_consume_food(self) -> FoodConsumptionResult:
        """Check if food is at current position and consume if present.

        When both health system and satiety system are enabled, food restores BOTH.

        Returns
        -------
        FoodConsumptionResult
            Result indicating whether food was consumed, satiety restored,
            health restored, reward, and distance efficiency (for dynamic environments).
        """
        if not self.env.reached_goal():
            return FoodConsumptionResult(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
                health_restored=0.0,
            )

        # Food is present - consume it
        distance_efficiency = None
        health_restored = 0.0

        # Consume food and calculate distance efficiency
        consumed = self.env.consume_food()
        if not consumed:
            return FoodConsumptionResult(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
                health_restored=0.0,
            )

        # Calculate distance efficiency
        if self._initial_distance is not None and self._initial_distance > 0:
            distance_efficiency = (
                self._initial_distance / self._steps_since_last_food
                if self._steps_since_last_food > 0
                else 1.0
            )

        # Update tracking for next food
        self._initial_distance = self.env.get_nearest_food_distance()
        self._steps_since_last_food = 0

        # Apply health healing if health system is enabled
        health_restored = self.env.apply_food_healing()

        # Restore satiety
        satiety_gain = self.satiety_manager.max_satiety * self.satiety_gain_fraction
        self.satiety_manager.restore_satiety(satiety_gain)

        return FoodConsumptionResult(
            food_consumed=True,
            satiety_restored=satiety_gain,
            reward=0.0,  # Reward is calculated separately by reward system
            distance_efficiency=distance_efficiency,
            health_restored=health_restored,
        )

    def reset(self) -> None:
        """Reset food consumption tracking."""
        self._steps_since_last_food = 0
        self._initial_distance = self.env.get_nearest_food_distance()
