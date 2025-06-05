"""
Maze environment for the Quantum Nematode agent.

This environment simulates a simple maze where the agent navigates to reach a goal position.
The agent can move in four directions: up, down, left, or right.
The agent must avoid colliding with itself.
The environment provides methods to get the current state, move the agent,
    check if the goal is reached, and render the maze.
"""

import secrets
from enum import Enum

import numpy as np  # pyright: ignore[reportMissingImports]

from .constants import MIN_GRID_SIZE
from .logging_config import logger

GRADIENT_SCALING_TANH_FACTOR = 1.0


class ScalingMethod(Enum):
    """Enum for scaling methods used in the maze environment."""

    EXPONENTIAL = "exponential"
    TANH = "tanh"


class MazeEnvironment:
    """
    A simple maze environment for the Quantum Nematode agent.

    The agent navigates a grid to reach a goal position while avoiding its own body.
    The agent can move in four directions: up, down, left, or right.

    Attributes
    ----------
    grid_size : int
        Size of the maze grid.
    agent_pos : tuple[int, int]
        Current position of the agent in the grid.
    body : list[tuple[int, int]]
        Positions of the agent's body segments.
    goal : tuple[int, int]
        Position of the goal in the grid.
    """

    def __init__(
        self,
        grid_size: int = 5,
        start_pos: tuple[int, int] | None = None,
        food_pos: tuple[int, int] | None = None,
        max_body_length: int = 6,
        theme: str = "ascii",
    ) -> None:
        if grid_size < MIN_GRID_SIZE:
            error_message = (
                f"Grid size must be at least {MIN_GRID_SIZE}. Provided grid size: {grid_size}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        self.grid_size = grid_size

        # Randomize agent and goal positions to all 4 corners
        corners = [
            (1, 1),
            (1, self.grid_size - 2),
            (self.grid_size - 2, 1),
            (self.grid_size - 2, self.grid_size - 2),
        ]

        corners_map = {
            "top_left": corners[0],
            "top_right": corners[1],
            "bottom_left": corners[2],
            "bottom_right": corners[3],
        }

        agent_chosen_corner = None
        if start_pos is None:
            agent_chosen_corner = secrets.choice(list(corners_map.keys()))
            start_pos = corners_map[agent_chosen_corner]

        if food_pos is None:
            if agent_chosen_corner is not None:
                if agent_chosen_corner == "top_left":
                    food_pos = corners_map["bottom_right"]
                elif agent_chosen_corner == "top_right":
                    food_pos = corners_map["bottom_left"]
                elif agent_chosen_corner == "bottom_left":
                    food_pos = corners_map["top_right"]
                elif agent_chosen_corner == "bottom_right":
                    food_pos = corners_map["top_left"]
                else:
                    food_pos = secrets.choice(corners)
            else:
                food_pos = secrets.choice(corners)

        self.agent_pos = start_pos
        self.goal = food_pos

        # Adjust body initialization
        self.body = (
            [tuple(self.agent_pos)] if max_body_length > 0 else []
        )  # Initialize the body with the head position
        self.current_direction = "up"  # Initialize the agent's direction
        self.theme = theme

    def get_state(
        self,
        position: tuple[int, ...],
        *,
        scaling_method: ScalingMethod = ScalingMethod.TANH,
        disable_log: bool = False,
    ) -> tuple[float, float]:
        """
        Get the current state of the agent in relation to the goal (chemical gradient).

        Returns
        -------
        tuple
            A tuple containing the gradient strength and direction.
        """
        # Simulate chemical gradient strength and direction
        dx = self.goal[0] - position[0]
        dy = self.goal[1] - position[1]

        # Refine gradient strength scaling to emphasize proximity to the goal
        max_distance = self.grid_size * 2  # Maximum possible Manhattan distance in the grid
        distance_to_goal = abs(dx) + abs(dy)
        gradient_strength = max(0.0, 1.0 - (distance_to_goal / max_distance))

        if scaling_method == ScalingMethod.EXPONENTIAL:
            gradient_strength = np.exp(
                -distance_to_goal / max_distance,
            )  # Apply exponential scaling
        elif scaling_method == ScalingMethod.TANH:
            gradient_strength = np.tanh(
                gradient_strength * GRADIENT_SCALING_TANH_FACTOR,
            )  # Apply non-linear scaling with tanh
        gradient_direction = np.arctan2(dy, dx) if dx != 0 or dy != 0 else 0.0

        if not disable_log:
            # Debugging: Log detailed information about gradient computation
            logger.debug(
                f"Gradient computation details: dx={dx}, dy={dy}, "
                f"gradient_strength={gradient_strength}, gradient_direction={gradient_direction}",
            )

            logger.debug(
                f"Agent position: {self.agent_pos}, Body positions: {self.body}, Goal: {self.goal}",
            )

        return gradient_strength, gradient_direction

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

        # Store the previous direction before attempting to move
        previous_direction = self.current_direction

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
            logger.warning(f"Collision against boundary with action: {action}, staying in place.")
            self.current_direction = previous_direction  # Revert to the previous direction
            return

        # Check for collision with the body
        if tuple(new_pos) in self.body:
            logger.warning(f"Collision detected at {new_pos}, staying in place.")
            self.current_direction = previous_direction  # Revert to the previous direction
            return

        # Update the body positions
        self.body = [tuple(self.agent_pos)] + self.body[:-1] if len(self.body) > 0 else []

        # Update the agent's position
        self.agent_pos = tuple(new_pos)

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
        """Render the current state of the maze using the selected theme."""
        # Theme symbol sets
        ascii_symbols = {
            "goal": "*",
            "body": "O",
            "up": "^",
            "down": "v",
            "left": "<",
            "right": ">",
            "empty": ".",
        }
        emoji_symbols = {
            "goal": "🦠",
            "body": "🔵",
            "up": "⬆️ ",
            "down": "⬇️ ",
            "left": "⬅️ ",
            "right": "➡️ ",
            "empty": "⬜️",
        }
        symbols = ascii_symbols if self.theme == "ascii" else emoji_symbols

        grid = [[symbols["empty"] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal[1]][self.goal[0]] = symbols["goal"]  # Mark the goal

        # Mark the body
        for segment in self.body:
            grid[segment[1]][segment[0]] = symbols["body"]

        agent_symbol = symbols.get(self.current_direction, "@")
        grid[self.agent_pos[1]][self.agent_pos[0]] = agent_symbol  # Mark the agent

        return [" ".join(row) for row in reversed(grid)] + [""]

    def copy(self) -> "MazeEnvironment":
        """
        Create a deep copy of the MazeEnvironment instance.

        Returns
        -------
        MazeEnvironment
            A new instance of MazeEnvironment with the same state.
        """
        new_env = MazeEnvironment(
            grid_size=self.grid_size,
            start_pos=(self.agent_pos[0], self.agent_pos[1])
            if self.agent_pos is not None
            else None,
            food_pos=self.goal,
            max_body_length=len(self.body),
        )
        new_env.body = self.body.copy()
        new_env.current_direction = self.current_direction
        return new_env
