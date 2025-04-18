import logging
import os

from .env import MazeEnvironment
from .brain import run_brain, interpret_counts
from quantumnematode.logging_config import logger


class QuantumNematodeAgent:
    def __init__(self, maze_grid_size=5):
        self.env = MazeEnvironment(grid_size=maze_grid_size)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.body_length = min(maze_grid_size - 1, 6)  # Set the maximum body length

    def run_episode(self, max_steps=100, show_last_frame_only=False):
        while not self.env.reached_goal() and self.steps < max_steps:
            dx, dy = self.env.get_state()
            counts = run_brain(dx, dy, self.env.grid_size)
            action = interpret_counts(counts, self.env.agent_pos, self.env.grid_size)
            self.env.move_agent(action)

            # Update the body length dynamically
            if len(self.env.body) < self.body_length:
                self.env.body.append(self.env.body[-1])

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={action}")

            if show_last_frame_only:
                os.system("clear")

            grid = self.env.render()
            for frame in grid:
                print(frame)

        return self.path
