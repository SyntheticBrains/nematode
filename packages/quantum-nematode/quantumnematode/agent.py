"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os

from .brain._brain import Brain
from .env import MazeEnvironment
from .logging_config import logger

PENALTY_STAY = -0.1
PENALTY_STEP = -0.1
REWARD_GOAL = 50
REWARD_GOAL_PROXIMITY_1 = 20
REWARD_GOAL_PROXIMITY_2 = 10
REWARD_GOAL_PROXIMITY_3 = 5


class QuantumNematodeAgent:
    """
    Quantum nematode agent that navigates a grid environment using a quantum brain.

    Attributes
    ----------
    env : MazeEnvironment
        The grid environment for the agent.
    steps : int
        Number of steps taken by the agent.
    path : list[tuple]
        Path taken by the agent.
    body_length : int
        Maximum length of the agent's body.
    """

    def __init__(self, brain: Brain, maze_grid_size: int = 5) -> None:
        """
        Initialize the quantum nematode agent.

        Parameters
        ----------
        brain : Brain
            The quantum brain architecture used by the agent.
        maze_grid_size : int, optional
            Size of the grid environment, by default 5.
        """
        self.brain = brain
        self.env = MazeEnvironment(grid_size=maze_grid_size)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.body_length = min(maze_grid_size - 1, 6)  # Set the maximum body length
        self.success_count = 0
        self.total_steps = 0
        self.total_rewards = 0

    def run_episode(
        self,
        max_steps: int = 100,
        *,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
        """
        Run a single episode of the simulation.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps for the episode.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation, by default False.

        Returns
        -------
        list[tuple]
            The path taken by the agent during the episode.
        """
        self.env.current_direction = "up"  # Initialize the agent's direction

        for _ in range(max_steps):
            state = self.env.get_state()
            reward = self.calculate_reward()
            counts = self.brain.run_brain(*state, reward=reward)
            action = self.brain.interpret_counts(counts)

            self.env.move_agent(action)

            # Update the body length dynamically
            if len(self.env.body) < self.body_length:
                self.env.body.append(self.env.body[-1])

            # Calculate reward based on efficiency and collision avoidance
            self.brain.update_memory(reward)

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={action}, Reward={reward}")

            if action == "unknown":
                logger.warning("Invalid action received: staying in place.")

            if self.env.reached_goal():
                # Run the brain with the final state and reward
                reward = self.calculate_reward()
                counts = self.brain.run_brain(*state, reward=reward)

                # Calculate reward based on efficiency and collision avoidance
                self.brain.update_memory(reward)

                self.path.append(tuple(self.env.agent_pos))
                self.steps += 1

                logger.info(f"Step {self.steps}: Action={action}, Reward={reward}")

                self.total_rewards += reward
                logger.info("Reward: goal reached!")
                self.success_count += 1
                break

            self.total_steps += 1
            self.total_rewards += reward

            # Log action counts for debugging
            logger.debug(f"Action counts: {counts}")

            # Log distance to the goal
            distance_to_goal = abs(self.env.agent_pos[0] - self.env.goal[0]) + abs(
                self.env.agent_pos[1] - self.env.goal[1],
            )
            logger.debug(f"Distance to goal: {distance_to_goal}")

            # Log cumulative reward and average reward per step at the end of each run
            if self.steps > 0:
                average_reward = self.total_rewards / self.steps
                logger.info(
                    f"Cumulative reward: {self.total_rewards}, "
                    f"Average reward per step: {average_reward}",
                )

            if show_last_frame_only:
                if os.name == "nt":  # For Windows
                    os.system("cls")  # noqa: S605, S607
                else:  # For macOS and Linux
                    os.system("clear")  # noqa: S605, S607

            grid = self.env.render()
            for frame in grid:
                print(frame)  # noqa: T201
                logger.debug(frame)

        return self.path

    def calculate_reward(self) -> float:
        """
        Calculate reward based on the agent's current state.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        # Decaying penalty for staying in the same position
        if len(self.path) > 1 and self.path[-1] == self.path[-2]:
            penalty = PENALTY_STAY * self.steps  # Penalty increases with steps
            logger.warning(f"Penalty: staying in the same position. Penalty: {penalty}")
            return penalty

        # Reward for reaching or in proximity of the goal
        distance_to_goal = abs(self.env.agent_pos[0] - self.env.goal[0]) + abs(
            self.env.agent_pos[1] - self.env.goal[1],
        )

        # Adjust reward for proximity to the goal
        if distance_to_goal == 0:
            logger.info("Reward: goal reached!")
            return REWARD_GOAL
        if distance_to_goal == 1:
            logger.debug("Reward: close to the goal!")
            return REWARD_GOAL_PROXIMITY_1
        if distance_to_goal == 2:  # noqa: PLR2004
            logger.debug("Reward: two spaces away from the goal.")
            return REWARD_GOAL_PROXIMITY_2
        if distance_to_goal == 3:  # noqa: PLR2004
            logger.debug("Reward: three spaces away from the goal.")
            return REWARD_GOAL_PROXIMITY_3

        return PENALTY_STEP  # Small penalty for each step to encourage efficiency

    def reset_environment(self) -> None:
        """
        Reset the environment while retaining the agent's learned data.

        Returns
        -------
        None
        """
        self.env = MazeEnvironment(grid_size=self.env.grid_size)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        logger.info("Environment reset. Retaining learned data.")

    def calculate_metrics(self, total_runs: int) -> dict:
        """
        Calculate and return performance metrics.

        Returns
        -------
        dict
            A dictionary containing success rate, average steps, and average reward.
        """
        return {
            "success_rate": self.success_count / total_runs,
            "average_steps": self.total_steps / total_runs,
            "average_reward": self.total_rewards / total_runs,
        }
