"""Food consumption handling for the quantum nematode agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quantumnematode.env import DynamicForagingEnvironment

if TYPE_CHECKING:
    from quantumnematode.env import BaseEnvironment
    from quantumnematode.agent.satiety import SatietyManager


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
        None for static environments or when no food was consumed.
    """

    food_consumed: bool
    satiety_restored: float
    reward: float
    distance_efficiency: float | None = None


class FoodConsumptionHandler:
    """Handles food consumption logic and environment-specific behavior.

    The food consumption handler abstracts food interaction logic, including
    detecting food consumption, restoring satiety, and calculating distance
    efficiency for dynamic environments.

    Parameters
    ----------
    env : BaseEnvironment
        The environment containing food.
    satiety_manager : SatietyManager
        The satiety manager to restore satiety when food is consumed.
    satiety_gain_fraction : float, optional
        Fraction of max satiety to restore per food consumed, by default 0.2.

    Attributes
    ----------
    env : BaseEnvironment
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
        env: BaseEnvironment,
        satiety_manager: SatietyManager,
        satiety_gain_fraction: float = 0.2,
    ) -> None:
        """Initialize the food consumption handler.

        Parameters
        ----------
        env : BaseEnvironment
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

        # Initialize distance tracking for dynamic environments
        if isinstance(env, DynamicForagingEnvironment):
            self._initial_distance = env.get_nearest_food_distance()

    def track_step(self) -> None:
        """Track a step for distance efficiency calculation."""
        self._steps_since_last_food += 1

    def check_and_consume_food(
        self,
        foods_collected: int = 0,
    ) -> FoodConsumptionResult:
        """Check if food is at current position and consume if present.

        Parameters
        ----------
        foods_collected : int, optional
            Number of foods collected so far in the episode, by default 0.

        Returns
        -------
        FoodConsumptionResult
            Result indicating whether food was consumed, satiety restored,
            reward, and distance efficiency (for dynamic environments).
        """
        if not self.env.reached_goal():
            return FoodConsumptionResult(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            )

        # Food is present - consume it
        distance_efficiency = None

        # For dynamic environments, calculate distance efficiency and update tracking
        if isinstance(self.env, DynamicForagingEnvironment):
            consumed = self.env.consume_food(foods_collected=foods_collected)
            if not consumed:
                return FoodConsumptionResult(
                    food_consumed=False,
                    satiety_restored=0.0,
                    reward=0.0,
                    distance_efficiency=None,
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

        # Restore satiety
        satiety_gain = self.satiety_manager.max_satiety * self.satiety_gain_fraction
        self.satiety_manager.restore_satiety(satiety_gain)

        return FoodConsumptionResult(
            food_consumed=True,
            satiety_restored=satiety_gain,
            reward=0.0,  # Reward is calculated separately by reward system
            distance_efficiency=distance_efficiency,
        )

    def reset(self) -> None:
        """Reset food consumption tracking."""
        self._steps_since_last_food = 0
        if isinstance(self.env, DynamicForagingEnvironment):
            self._initial_distance = self.env.get_nearest_food_distance()
