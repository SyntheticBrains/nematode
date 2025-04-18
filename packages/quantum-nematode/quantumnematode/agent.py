import logging

from .env import GRID_SIZE, MazeEnvironment
from .brain import run_brain, interpret_counts
from quantumnematode.logging_config import logger


class QuantumNematodeAgent:
    def __init__(self):
        self.env = MazeEnvironment()
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]

    def run_episode(self, max_steps=30):
        while not self.env.reached_goal() and self.steps < max_steps:
            dx, dy = self.env.get_state()
            counts = run_brain(dx, dy)
            action = interpret_counts(counts, self.env.agent_pos, GRID_SIZE)
            self.env.move_agent(action)
            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            # Render the maze after each step
            logger.info(f"Step {self.steps}: Action={action}")
            grid = self.env.render()
            for frame in grid:
                print(frame)

        return self.path
