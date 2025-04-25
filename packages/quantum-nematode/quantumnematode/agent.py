"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os

from .brain._brain import Brain
from .env import MazeEnvironment
from .logging_config import logger

PENALTY_STAY = -0.5
PENALTY_STEP = -0.1
REWARD_GOAL = 50


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
            logger.debug("--- New Step ---")
            gradient_strength, gradient_direction = self.env.get_state(self.path[-1])
            reward = self.calculate_reward(max_steps=max_steps)
            counts = self.brain.run_brain(gradient_strength, gradient_direction, reward=reward)
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
                reward = self.calculate_reward(max_steps=max_steps)
                counts = self.brain.run_brain(gradient_strength, gradient_direction, reward=reward)

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
            logger.debug(f"Sorted action counts: {sorted(counts.items(), key=lambda x: x[1], reverse=True)}")

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

    def calculate_reward(self, max_steps: int) -> float:
        """
        Calculate reward based on the agent's current state using gradient strength.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        reward = 0.0
        
        # Get the current gradient strength from the environment
        gradient_strength, _ = self.env.get_state(self.path[-1])

        # Calculate the change in gradient strength since the last step
        previous_gradient_strength = None
        if len(self.path) > 1:
            previous_gradient_strength, _ = self.env.get_state(self.path[-2])
            gradient_change = gradient_strength - previous_gradient_strength
        else:
            gradient_change = 0

        # Enhance reward signal for gradient improvement
        if previous_gradient_strength is not None:
            if gradient_change > 0:
                reward = gradient_change * 30  # Increased reward for improving gradient strength
            elif gradient_change < 0:
                reward -= gradient_change * 15  # Stronger penalty for weakening gradient strength
            else:
                reward = PENALTY_STEP * 2  # Small penalty for no change

        # Strengthen penalties for revisiting positions
        if self.path.count(tuple(self.env.agent_pos)) > 1:
            reward += PENALTY_STEP * 10  # Stronger penalty for revisiting positions

        # Strengthen penalties for collisions
        if len(self.path) > 1 and self.path[-1] == self.path[-2]:
            reward += PENALTY_STAY * 5  # Stronger penalty for staying in place due to collision

        # Reward efficient paths by scaling inversely with steps
        efficiency_factor = None
        if self.env.reached_goal():
            efficiency_factor = max(0.1, 1 - (self.steps / max_steps))  # Scale inversely with steps
            reward += REWARD_GOAL * 10 * efficiency_factor  # Further scale goal reward dynamically based on speed

        logger.debug(
            f"Revisit penalty applied: {PENALTY_STEP * 10 if self.path.count(tuple(self.env.agent_pos)) > 1 else 0}, "
            f"Collision penalty applied: {PENALTY_STAY * 5 if len(self.path) > 1 and self.path[-1] == self.path[-2] else 0}, "
            f"Efficiency factor: {efficiency_factor if self.env.reached_goal() else 'N/A'}, Reward: {reward}"
        )

        logger.debug(
            f"Gradient strength: {gradient_strength}, Gradient change: {gradient_change}, Reward: {reward}"
        )

        return reward

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
