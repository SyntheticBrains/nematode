"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os

from .brain import interpret_counts, run_brain
from .env import MazeEnvironment
from .logging_config import logger


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

    def __init__(self, maze_grid_size: int = 5) -> None:
        """
        Initialize the quantum nematode agent.

        Parameters
        ----------
        maze_grid_size : int, optional
            Size of the grid environment, by default 5.
        """
        self.env = MazeEnvironment(grid_size=maze_grid_size)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.body_length = min(maze_grid_size - 1, 6)  # Set the maximum body length

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
        max_steps : int, optional
            Maximum number of steps for the episode, by default 100.
        show_last_frame_only : bool, optional
            Whether to display only the last frame, by default False.

        Returns
        -------
        list[tuple]
            Path taken by the agent during the episode.
        """
        total_reward = 0
        while not self.env.reached_goal() and self.steps < max_steps:
            dx, dy = self.env.get_state()
            counts = run_brain(dx, dy, self.env.grid_size, reward=total_reward)
            action = interpret_counts(counts, self.env.agent_pos, self.env.grid_size)
            self.env.move_agent(action)

            # Calculate reward based on efficiency and collision avoidance
            reward = self.calculate_reward()
            total_reward += reward

            # Update the body length dynamically
            if len(self.env.body) < self.body_length:
                self.env.body.append(self.env.body[-1])

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={action}, Reward={reward}")

            if show_last_frame_only:
                os.system("clear")  # noqa: S605, S607

            grid = self.env.render()
            for frame in grid:
                print(frame)  # noqa: T201

        return self.path

    def calculate_reward(self) -> float:
        """
        Calculate reward based on the agent's current state.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        if self.env.reached_goal():
            return 10  # High reward for reaching the goal
        if tuple(self.env.agent_pos) in self.env.body:
            return -5  # Penalty for colliding with its own body
        return -0.1  # Small penalty for each step to encourage efficiency

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
