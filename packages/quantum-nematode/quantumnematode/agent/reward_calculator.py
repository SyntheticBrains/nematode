"""Reward calculation logic for the quantum nematode agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.agent import RewardConfig
    from quantumnematode.env import DynamicForagingEnvironment


class RewardCalculator:
    """Calculates rewards based on agent movement and environment state.

    The reward calculator handles reward computation for multi-food environments
    (DynamicForagingEnvironment). It considers distance to food, exploration,
    anti-dithering, and various penalties.

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

    def calculate_reward(  # noqa: PLR0913
        self,
        env: DynamicForagingEnvironment,
        path: list[tuple[int, ...]],
        stuck_position_count: int = 0,
        current_step: int = 0,
        max_steps: int = 100,
        agent_id: str = "default",
    ) -> float:
        """Calculate reward based on the agent's movement toward the goal.

        Handles DynamicForagingEnvironment (multiple foods).

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The environment instance.
        path : list[tuple[int, ...]]
            The agent's path history.
        stuck_position_count : int, optional
            Number of consecutive steps in the same position, by default 0.
        current_step : int, optional
            Current step number for efficiency calculation, by default 0.
        max_steps : int, optional
            Maximum steps for efficiency calculation, by default 100.
        agent_id : str, optional
            Agent identifier for multi-agent reward computation, by default "default".

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        reward = 0.0
        anti_dither_penalty = 0.0

        # Resolve agent position
        agent_pos = env.agents[agent_id].position

        # Handle distance-based rewards for foraging environment
        distance_reward = self._calculate_foraging_distance_reward(env, path, agent_id)
        exploration_bonus = self._calculate_exploration_bonus(env, agent_id)
        reward += distance_reward + exploration_bonus

        # Anti-dithering: penalize if agent oscillates (returns to previous cell)
        if len(path) > 2 and agent_pos == path[-3]:  # noqa: PLR2004
            anti_dither_penalty = self.config.penalty_anti_dithering
            reward -= anti_dither_penalty
            logger.debug(
                f"[Penalty] Anti-dithering penalty applied: "
                f"{-anti_dither_penalty} (oscillation detected)",
            )

        # Step penalty (applies every step)
        reward -= self.config.penalty_step
        logger.debug(f"[Penalty] Step penalty applied: {-self.config.penalty_step}.")

        # Distance-scaled predator evasion: reward for moving away, penalize for closer
        # Also applies a contact penalty when predator is on or adjacent to agent
        if env.predator.enabled and env.is_agent_in_danger_for(agent_id):
            curr_pred_dist = env.get_nearest_predator_distance_for(agent_id)
            if curr_pred_dist is not None and len(path) > 1:
                prev_pos = path[-2]
                prev_pred_distances = [
                    abs(prev_pos[0] - pred.position[0]) + abs(prev_pos[1] - pred.position[1])
                    for pred in env.predators
                ]
                prev_pred_dist = min(prev_pred_distances)
                # Positive when moving AWAY (curr > prev), negative when CLOSER
                evasion_reward = self.config.penalty_predator_proximity * (
                    curr_pred_dist - prev_pred_dist
                )
                # Contact penalty: when predator is on or adjacent (dist ≤ 1),
                # apply flat penalty so agent always has incentive to escape
                if curr_pred_dist <= 1:
                    evasion_reward -= self.config.penalty_predator_proximity
                reward += evasion_reward
                logger.debug(
                    f"[Reward] Predator evasion reward: {evasion_reward:.3f} "
                    f"(prev_dist={prev_pred_dist}, curr_dist={curr_pred_dist})",
                )
            else:
                # Fallback: flat penalty for first step or edge cases
                reward -= self.config.penalty_predator_proximity
                logger.debug(
                    f"[Penalty] Predator proximity penalty (flat fallback): "
                    f"{-self.config.penalty_predator_proximity}",
                )

        # Distance-scaled temperature avoidance: reward for moving toward cultivation temp
        reward += self._calculate_temperature_avoidance_reward(env, path, agent_id)

        # Boundary collision penalty (mechanosensation)
        # Penalizes when agent attempts to move into a wall, not just for being at edge
        if env.agents[agent_id].wall_collision_occurred:
            boundary_penalty = self.config.penalty_boundary_collision
            reward -= boundary_penalty
            logger.debug(
                f"[Penalty] Boundary collision penalty applied: {-boundary_penalty}",
            )

        # Stuck position penalty: penalize agent for staying in same position
        if (
            self.config.stuck_position_threshold > 0
            and stuck_position_count > self.config.stuck_position_threshold
        ):
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
        if env.reached_goal_for(agent_id):
            efficiency = max(0.1, 1 - (current_step / max_steps))
            goal_bonus = self.config.reward_goal * efficiency
            reward += goal_bonus
            logger.debug(
                f"[Reward] Goal reached! Efficiency bonus applied: "
                f"{goal_bonus} (efficiency={efficiency}).",
            )

        return reward

    def _calculate_foraging_distance_reward(
        self,
        env: DynamicForagingEnvironment,
        path: list[tuple[int, ...]],
        agent_id: str = "default",
    ) -> float:
        """Calculate distance-based reward for foraging environments.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The foraging environment.
        path : list[tuple[int, ...]]
            The agent's path history.
        agent_id : str, optional
            Agent identifier, by default "default".

        Returns
        -------
        float
            Distance-based reward.
        """
        curr_dist = env.get_nearest_food_distance_for(agent_id)
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
        agent_id: str = "default",
    ) -> float:
        """Calculate exploration bonus for visiting new cells.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The foraging environment.
        agent_id : str, optional
            Agent identifier, by default "default".

        Returns
        -------
        float
            Exploration bonus.
        """
        agent_state = env.agents[agent_id]
        curr_pos_tuple = (agent_state.position[0], agent_state.position[1])
        if curr_pos_tuple not in agent_state.visited_cells:
            exploration_bonus = self.config.reward_exploration
            agent_state.visited_cells.add(curr_pos_tuple)
            logger.debug(f"[Reward] Exploration bonus: {exploration_bonus}")
            return exploration_bonus

        return 0.0

    def _calculate_temperature_avoidance_reward(
        self,
        env: DynamicForagingEnvironment,
        path: list[tuple[int, ...]],
        agent_id: str = "default",
    ) -> float:
        """Calculate distance-scaled temperature avoidance reward.

        Rewards the agent for moving toward the cultivation temperature
        (reducing absolute deviation) and penalizes moving away from it.
        Only active outside the comfort zone.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            The environment instance.
        path : list[tuple[int, ...]]
            The agent's path history.
        agent_id : str, optional
            Agent identifier, by default "default".

        Returns
        -------
        float
            Temperature avoidance reward (positive for moving toward Tc).
        """
        from quantumnematode.env.temperature import TemperatureZone

        if (
            self.config.penalty_temperature_proximity <= 0
            or not env.thermotaxis.enabled
            or len(path) <= 1
        ):
            return 0.0

        zone = env.get_temperature_zone_for(agent_id)
        if zone is None or zone == TemperatureZone.COMFORT:
            return 0.0

        agent_pos = env.agents[agent_id].position
        curr_temp = env.get_temperature(agent_pos)
        prev_pos = (path[-2][0], path[-2][1])
        prev_temp = env.get_temperature(prev_pos)
        if curr_temp is None or prev_temp is None:
            return 0.0

        cultivation_temp = env.thermotaxis.cultivation_temperature
        curr_dev = abs(curr_temp - cultivation_temp)
        prev_dev = abs(prev_temp - cultivation_temp)
        # Negative deviation_delta = moving toward Tc (good)
        temp_reward = self.config.penalty_temperature_proximity * -(curr_dev - prev_dev)
        logger.debug(
            f"[Reward] Temperature avoidance reward: {temp_reward:.3f} "
            f"(prev_dev={prev_dev:.2f}, curr_dev={curr_dev:.2f}, "
            f"zone={zone.value})",
        )
        return temp_reward
