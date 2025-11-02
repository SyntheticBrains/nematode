"""Single step processing logic for the quantum nematode agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.agent import StepResult
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.arch import Brain
    from quantumnematode.env import BaseEnvironment
    from quantumnematode.food_handler import FoodConsumptionHandler
    from quantumnematode.reward_calculator import RewardCalculator
    from quantumnematode.satiety import SatietyManager


class StepProcessor:
    """Processes a single step of agent execution.

    The step processor is responsible for running the brain, executing actions,
    calculating rewards, and handling food consumption. It delegates responsibility
    to specialized components while orchestrating the step-level logic.

    Parameters
    ----------
    brain : Brain
        The brain instance to use for decision-making.
    env : BaseEnvironment
        The environment instance.
    reward_calculator : RewardCalculator
        Calculator for reward computation.
    food_handler : FoodConsumptionHandler
        Handler for food consumption logic.
    satiety_manager : SatietyManager
        Manager for satiety (hunger) mechanics.

    Attributes
    ----------
    brain : Brain
        The brain instance.
    env : BaseEnvironment
        The environment instance.
    reward_calculator : RewardCalculator
        The reward calculator.
    food_handler : FoodConsumptionHandler
        The food consumption handler.
    satiety_manager : SatietyManager
        The satiety manager.
    """

    def __init__(
        self,
        brain: Brain,
        env: BaseEnvironment,
        reward_calculator: RewardCalculator,
        food_handler: FoodConsumptionHandler,
        satiety_manager: SatietyManager,
    ) -> None:
        """Initialize the step processor.

        Parameters
        ----------
        brain : Brain
            The brain instance to use for decision-making.
        env : BaseEnvironment
            The environment instance.
        reward_calculator : RewardCalculator
            Calculator for reward computation.
        food_handler : FoodConsumptionHandler
            Handler for food consumption logic.
        satiety_manager : SatietyManager
            Manager for satiety (hunger) mechanics.
        """
        self.brain = brain
        self.env = env
        self.reward_calculator = reward_calculator
        self.food_handler = food_handler
        self.satiety_manager = satiety_manager

    def prepare_brain_params(
        self,
        gradient_strength: float,
        gradient_direction: float,
        previous_action: Action | None,
    ) -> BrainParams:
        """Prepare parameters for brain execution.

        Parameters
        ----------
        gradient_strength : float
            Strength of the gradient (distance to goal/food).
        gradient_direction : float
            Direction of the gradient (angle in radians).
        previous_action : Action | None
            The previous action taken, or None if first step.

        Returns
        -------
        BrainParams
            Brain parameters ready for execution.
        """
        # Create action data from previous action
        action_data = None
        if previous_action is not None:
            action_data = ActionData(
                state=str(previous_action.value),
                action=previous_action,
                probability=1.0,
            )

        # Get agent position and direction
        agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
        if len(agent_pos) != 2:  # noqa: PLR2004
            agent_pos = (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))

        return BrainParams(
            gradient_strength=gradient_strength,
            gradient_direction=gradient_direction,
            agent_position=agent_pos,
            agent_direction=self.env.current_direction,
            action=action_data,
        )

    def process_step(  # noqa: PLR0913 - step processor needs all params
        self,
        gradient_strength: float,
        gradient_direction: float,
        previous_action: Action | None,
        previous_reward: float,
        path: list[tuple[int, ...]],
        stuck_position_count: int = 0,
        *,
        top_only: bool = False,
        top_randomize: bool = False,
    ) -> StepResult:
        """Process a single step of agent execution.

        Parameters
        ----------
        gradient_strength : float
            Strength of the gradient (distance to goal/food).
        gradient_direction : float
            Direction of the gradient (angle in radians).
        previous_action : Action | None
            The previous action taken, or None if first step.
        previous_reward : float
            The reward from the previous step.
        path : list[tuple[int, ...]]
            The agent's path history.
        stuck_position_count : int, optional
            Number of consecutive steps in the same position, by default 0.
        top_only : bool, optional
            Whether to use only the top action, by default False.
        top_randomize : bool, optional
            Whether to randomize the top action selection, by default False.

        Returns
        -------
        StepResult
            Result of the step including action, reward, done status, and info.
        """
        # Prepare brain parameters
        params = self.prepare_brain_params(
            gradient_strength,
            gradient_direction,
            previous_action,
        )

        # Run the brain to get the next action
        from quantumnematode.brain.arch import QuantumBrain

        input_data = None
        if isinstance(self.brain, QuantumBrain):
            input_data = [float(gradient_strength)] * self.brain.num_qubits

        action_results = self.brain.run_brain(
            params=params,
            reward=previous_reward,
            input_data=input_data,
            top_only=top_only,
            top_randomize=top_randomize,
        )

        # Get the selected action
        top_action = action_results[0] if action_results else None
        if top_action is None:
            # Fallback to FORWARD if no action returned
            top_action = ActionData(
                state="FORWARD",
                action=Action.FORWARD,
                probability=1.0,
            )

        # Execute the action in the environment
        self.env.move_agent(top_action.action)

        # Track step for food handler (distance efficiency calculation)
        self.food_handler.track_step()

        # Calculate reward for this step
        reward = self.reward_calculator.calculate_reward(
            env=self.env,
            path=path,
            stuck_position_count=stuck_position_count,
        )

        # Check for food consumption
        food_result = self.food_handler.check_and_consume_food()
        if food_result.food_consumed:
            reward += food_result.reward
            logger.info(
                f"Food collected! Satiety restored by {food_result.satiety_restored:.1f} "
                f"to {self.satiety_manager.current_satiety:.1f}",
            )

        # Check if episode is done
        done = False
        info: dict[str, bool | float | None] = {
            "goal_reached": self.env.reached_goal(),
            "food_consumed": food_result.food_consumed,
            "distance_efficiency": food_result.distance_efficiency,
        }

        # Check for starvation
        if self.satiety_manager.is_starved():
            done = True
            info["starved"] = True
            logger.warning("Agent starved!")

        logger.info(f"Action={top_action.action.value}, Reward={reward:.3f}")

        return StepResult(
            action=top_action.action,
            reward=reward,
            done=done,
            info=info,
        )
