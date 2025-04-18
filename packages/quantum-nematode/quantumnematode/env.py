from quantumnematode.logging_config import logger


class MazeEnvironment:
    def __init__(self, grid_size=5, start_pos=(1, 1), food_pos=None):
        self.grid_size = grid_size
        self.agent_pos = list(start_pos)
        self.body = [tuple(start_pos)]  # Initialize the body with the head position
        self.goal = (grid_size - 1, grid_size - 1) if food_pos is None else food_pos

    def get_state(self):
        dx = self.goal[0] - self.agent_pos[0] + 1
        dy = self.goal[1] - self.agent_pos[1] + 1

        logger.debug(
            f"Agent position: {self.agent_pos}, Goal: {self.goal}, dx={dx}, dy={dy}"
        )
        return dx, dy

    def move_agent(self, action):
        logger.debug(f"Action received: {action}, Current position: {self.agent_pos}")

        # Update the body positions
        self.body = [tuple(self.agent_pos)] + self.body[:-1]

        if action == "up" and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == "down" and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == "right" and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == "left" and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        else:
            logger.warning(f"Invalid action: {action}, staying in place.")

        logger.debug(f"New position: {self.agent_pos}")

    def reached_goal(self):
        return tuple(self.agent_pos) == self.goal

    def render(self) -> list[str]:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal[1]][self.goal[0]] = "*"  # Mark the goal

        # Mark the body
        for segment in self.body:
            grid[segment[1]][segment[0]] = "O"

        grid[self.agent_pos[1]][self.agent_pos[0]] = "@"  # Mark the agent

        return [" ".join(row) for row in reversed(grid)] + [""]
