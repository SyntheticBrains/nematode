"""Reward calculation logic for the quantum nematode agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.agent import RewardConfig
    from quantumnematode.env import BaseEnvironment, DynamicForagingEnvironment, StaticEnvironment


class RewardCalculator:
    """Calculates rewards based on agent movement and environment state.

    The reward calculator handles reward computation for both single-goal
    environments (StaticEnvironment) and multi-food environments (DynamicForagingEnvironment).
    It considers distance to goal/food, exploration, anti-dithering, and various penalties.

    Parameters
    ----------
    config : RewardConfig
        Configuration for reward parameters (scaling factors, penalties, etc.).

    Attributes
    ----------
    config : RewardConfig
        The reward configuration.
    """

    def __init__(self, config: RewardConfig) -> None:
        """Initialize the reward calculator.

        Parameters
        ----------
        config : RewardConfig
            Configuration for reward parameters.
        """
        self.config = config

    def calculate_reward(
        self,
        env: BaseEnvironment,
        path: list[tuple[int, ...]],
        stuck_position_count: int = 0,
        current_step: int = 0,
        max_steps: int = 100,
    ) -> float:
        """Calculate reward based on the agent's movement toward the goal.

        Handles both StaticEnvironment (single goal) and DynamicForagingEnvironment
        (multiple foods).

        Parameters
        ----------
        env : BaseEnvironment
            The environment instance.
        path : list[tuple[int, ...]]
            The agent's path history.
        stuck_position_count : int, optional
            Number of consecutive steps in the same position, by default 0.
        current_step : int, optional
            Current step number for efficiency calculation, by default 0.
        max_steps : int, optional
            Maximum steps for efficiency calculation, by default 100.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        # Import here to avoid circular dependency
        from quantumnematode.env import DynamicForagingEnvironment, StaticEnvironment

        reward = 0.0
        distance_reward = 0.0
        anti_dither_penalty = 0.0
        exploration_bonus = 0.0
        goal_bonus = 0.0

        # Handle distance-based rewards differently for each environment type
        if isinstance(env, StaticEnvironment):
            reward += self._calculate_maze_distance_reward(env, path)
        elif isinstance(env, DynamicForagingEnvironment):
            distance_reward = self._calculate_foraging_distance_reward(env, path)
            exploration_bonus = self._calculate_exploration_bonus(env)
            reward += distance_reward + exploration_bonus

        # Anti-dithering: penalize if agent oscillates (returns to previous cell)
        if len(path) > 2 and env.agent_pos == path[-3]:  # noqa: PLR2004
            anti_dither_penalty = self.config.penalty_anti_dithering
            reward -= anti_dither_penalty
            logger.debug(
                f"[Penalty] Anti-dithering penalty applied: "
                f"{-anti_dither_penalty} (oscillation detected)",
            )

        # Step penalty (applies every step)
        reward -= self.config.penalty_step
        logger.debug(f"[Penalty] Step penalty applied: {-self.config.penalty_step}.")

        # Proximity penalty for being near predators (dynamic foraging only)
        if (
            isinstance(env, DynamicForagingEnvironment)
            and env.predators_enabled
            and env.is_agent_in_danger()
        ):
            proximity_penalty = self.config.penalty_predator_proximity
            reward -= proximity_penalty
            logger.debug(
                f"[Penalty] Predator proximity penalty applied: {-proximity_penalty}",
            )

        # Stuck position penalty: penalize agent for staying in same position
        if stuck_position_count > self.config.stuck_position_threshold:
            stuck_penalty = self.config.penalty_stuck_position * min(
                stuck_position_count - self.config.stuck_position_threshold,
                10,
            )
            reward -= stuck_penalty
            logger.debug(
                f"[Penalty] Stuck position penalty applied: {-stuck_penalty} "
                f"(count={stuck_position_count})",
            )

        # Bonus for reaching the goal, scaled by efficiency
        if env.reached_goal():
            efficiency = max(0.1, 1 - (current_step / max_steps))
            goal_bonus = self.config.reward_goal * efficiency
            reward += goal_bonus
            logger.debug(
                f"[Reward] Goal reached! Efficiency bonus applied: "
                f"{goal_bonus} (efficiency={efficiency}).",
            )

        return reward

    def _calculate_maze_distance_reward(
        self,
        env: StaticEnvironment,
        path: list[tuple[int, ...]],
    ) -> float:
        """Calculate distance-based reward for maze environments.

        Parameters
        ----------
        env : StaticEnvironment
            The maze environment.
        path : list[tuple[int, ...]]
            The agent's path history.

        Returns
        -------
        float
            Distance-based reward.
        """
        curr_pos = env.agent_pos
        curr_dist = abs(curr_pos[0] - env.goal[0]) + abs(curr_pos[1] - env.goal[1])

        if len(path) > 1:
            prev_pos = path[-2]
            prev_dist = abs(prev_pos[0] - env.goal[0]) + abs(prev_pos[1] - env.goal[1])
            distance_reward = self.config.reward_distance_scale * (prev_dist - curr_dist)
            logger.debug(
                f"[Reward] Scaled distance reward: {distance_reward} "
                f"(prev_dist={prev_dist}, curr_dist={curr_dist})",
            )
            return distance_reward

        return 0.0

    def _calculate_foraging_distance_reward(
        self,
        env: DynamicForagingEnvironment,
        path: list[tuple[int, ...]],
    ) -> float:
        """Calculate distance-based reward for foraging environments.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The foraging environment.
        path : list[tuple[int, ...]]
            The agent's path history.

        Returns
        -------
        float
            Distance-based reward.
        """
        curr_dist = env.get_nearest_food_distance()
        if curr_dist is None or len(path) <= 1:
            return 0.0

        # Calculate previous nearest food distance
        prev_pos = path[-2]
        prev_distances = [
            abs(prev_pos[0] - food[0]) + abs(prev_pos[1] - food[1]) for food in env.foods
        ]
        prev_dist = min(prev_distances) if prev_distances else None

        if prev_dist is not None:
            distance_reward = self.config.reward_distance_scale * (prev_dist - curr_dist)
            logger.debug(
                f"[Reward] Scaled distance reward (nearest food): {distance_reward} "
                f"(prev_dist={prev_dist}, curr_dist={curr_dist})",
            )
            return distance_reward

        return 0.0

    def _calculate_exploration_bonus(
        self,
        env: DynamicForagingEnvironment,
    ) -> float:
        """Calculate exploration bonus for visiting new cells.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The foraging environment.

        Returns
        -------
        float
            Exploration bonus.
        """
        curr_pos_tuple = (env.agent_pos[0], env.agent_pos[1])
        if curr_pos_tuple not in env.visited_cells:
            exploration_bonus = self.config.reward_exploration
            env.visited_cells.add(curr_pos_tuple)
            logger.debug(f"[Reward] Exploration bonus: {exploration_bonus}")
            return exploration_bonus

        return 0.0
