"""
Maze environment for the Quantum Nematode agent.

This environment simulates a simple maze where the agent navigates to reach a goal position.
The agent can move in four directions: up, down, left, or right.
The agent must avoid colliding with itself.
The environment provides methods to get the current state, move the agent,
    check if the goal is reached, and render the maze.
"""

from .logging_config import logger


class MazeEnvironment:
    """
    A simple maze environment for the Quantum Nematode agent.

    The agent navigates a grid to reach a goal position while avoiding its own body.
    The agent can move in four directions: up, down, left, or right.

    Attributes
    ----------
    grid_size : int
        Size of the maze grid.
    agent_pos : list[int]
        Current position of the agent in the grid.
    body : list[tuple[int, int]]
        Positions of the agent's body segments.
    goal : tuple[int, int]
        Position of the goal in the grid.
    """

    def __init__(
        self,
        grid_size: int = 5,
        start_pos: tuple[int, int] = (1, 1),
        food_pos: tuple[int, int] | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.agent_pos = list(start_pos)
        self.body = [tuple(start_pos)]  # Initialize the body with the head position
        self.goal = (grid_size - 1, grid_size - 1) if food_pos is None else food_pos
        self.current_direction = "up"  # Initialize the agent's direction

    def get_state(self) -> tuple[int, int]:
        """
        Get the current state of the agent in relation to the goal.

        Returns
        -------
        tuple
            A tuple containing the x and y distances to the goal.
        """
        dx = self.goal[0] - self.agent_pos[0] + 1
        dy = self.goal[1] - self.agent_pos[1] + 1

        logger.debug(
            f"Agent position: {self.agent_pos}, Body positions: {self.body}, "
            f"Goal: {self.goal}, dx={dx}, dy={dy}",
        )
        return dx, dy

    def move_agent(self, action: str) -> None:
        """
        Move the agent based on its perspective.

        Parameters
        ----------
        action : str
            The action to take. Can be "forward", "left", or "right".
        """
        logger.debug(f"Action received: {action}, Current position: {self.agent_pos}")

        if action == "stay":
            logger.info("Action is stay: staying in place.")
            return
        if action == "unknown":
            logger.warning("Action is either unknown: staying in place.")
            return

        # Define direction mappings
        direction_map = {
            "up": {"forward": "up", "left": "left", "right": "right"},
            "down": {"forward": "down", "left": "right", "right": "left"},
            "left": {"forward": "left", "left": "down", "right": "up"},
            "right": {"forward": "right", "left": "up", "right": "down"},
        }

        # Determine the new direction based on the current direction and action
        new_direction = direction_map[self.current_direction][action]
        self.current_direction = new_direction

        # Calculate the new position based on the new direction
        new_pos = list(self.agent_pos)
        if new_direction == "up" and self.agent_pos[1] < self.grid_size - 1:
            new_pos[1] += 1
        elif new_direction == "down" and self.agent_pos[1] > 0:
            new_pos[1] -= 1
        elif new_direction == "right" and self.agent_pos[0] < self.grid_size - 1:
            new_pos[0] += 1
        elif new_direction == "left" and self.agent_pos[0] > 0:
            new_pos[0] -= 1
        else:
            logger.warning(f"Invalid action: {action}, staying in place.")
            return

        # Check for collision with the body
        if tuple(new_pos) in self.body:
            logger.warning(f"Collision detected at {new_pos}, staying in place.")
            return

        # Update the body positions
        self.body = [tuple(self.agent_pos)] + self.body[:-1]

        # Update the agent's position
        self.agent_pos = new_pos

        logger.debug(f"New position: {self.agent_pos}, New direction: {self.current_direction}")

    def reached_goal(self) -> bool:
        """
        Check if the agent has reached the goal.

        Returns
        -------
        bool
            True if the agent's position matches the goal position, False otherwise.
        """
        return tuple(self.agent_pos) == self.goal

    def render(self) -> list[str]:
        """
        Render the current state of the maze.

        Returns
        -------
        list[str]
            A string representation of the maze grid.
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal[1]][self.goal[0]] = "*"  # Mark the goal

        # Mark the body
        for segment in self.body:
            grid[segment[1]][segment[0]] = "O"

        # Map the agent's direction to an arrow symbol
        direction_map = {
            "up": "^",
            "down": "v",
            "left": "<",
            "right": ">",
        }
        agent_symbol = direction_map.get(self.current_direction, "@")

        grid[self.agent_pos[1]][self.agent_pos[0]] = agent_symbol  # Mark the agent with an arrow

        return [" ".join(row) for row in reversed(grid)] + [""]
