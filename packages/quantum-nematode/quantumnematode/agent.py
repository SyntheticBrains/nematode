import os

from quantumnematode.logging_config import logger

from .brain import interpret_counts, run_brain
from .env import MazeEnvironment


class QuantumNematodeAgent:
    def __init__(self, maze_grid_size: int = 5) -> None:
        self.env = MazeEnvironment(grid_size=maze_grid_size)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.body_length = min(maze_grid_size - 1, 6)  # Set the maximum body length

    def run_episode(
        self,
        max_steps: int = 100,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
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
                os.system("clear")

            grid = self.env.render()
            for frame in grid:
                print(frame)

        return self.path

    def calculate_reward(self) -> float:
        """Calculate reward based on the agent's current state."""
        if self.env.reached_goal():
            return 10  # High reward for reaching the goal
        if tuple(self.env.agent_pos) in self.env.body:
            return -5  # Penalty for colliding with its own body
        return -0.1  # Small penalty for each step to encourage efficiency
