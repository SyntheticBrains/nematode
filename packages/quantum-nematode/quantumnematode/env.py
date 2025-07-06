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

import numpy as np

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action
from quantumnematode.theme import THEME_SYMBOLS, Theme

from .constants import MIN_GRID_SIZE
from .logging_config import logger

GRADIENT_SCALING_TANH_FACTOR = 1.0


class Corner(Enum):
    """Enum for the corners of the maze grid."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class Direction(Enum):
    """Enum for the directions the agent can move."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


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

    def __init__(  # noqa: PLR0913
        self,
        grid_size: int = 5,
        start_pos: tuple[int, int] | None = None,
        food_pos: tuple[int, int] | None = None,
        max_body_length: int = 6,
        theme: Theme = Theme.ASCII,
        action_set: list[Action] = DEFAULT_ACTIONS,
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
            Corner.TOP_LEFT: corners[0],
            Corner.TOP_RIGHT: corners[1],
            Corner.BOTTOM_LEFT: corners[2],
            Corner.BOTTOM_RIGHT: corners[3],
        }

        agent_chosen_corner = None
        if start_pos is None:
            agent_chosen_corner = secrets.choice(list(Corner))
            start_pos = corners_map[agent_chosen_corner]

        if food_pos is None:
            if agent_chosen_corner is not None:
                if agent_chosen_corner == Corner.TOP_LEFT:
                    food_pos = corners_map[Corner.BOTTOM_RIGHT]
                elif agent_chosen_corner == Corner.TOP_RIGHT:
                    food_pos = corners_map[Corner.BOTTOM_LEFT]
                elif agent_chosen_corner == Corner.BOTTOM_LEFT:
                    food_pos = corners_map[Corner.TOP_RIGHT]
                elif agent_chosen_corner == Corner.BOTTOM_RIGHT:
                    food_pos = corners_map[Corner.TOP_LEFT]
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
        self.current_direction = Direction.UP  # Initialize the agent's direction
        self.theme = theme
        self.action_set = action_set

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

    def move_agent(self, action: Action) -> None:
        """
        Move the agent based on its perspective.

        Parameters
        ----------
        action : Action
            The action to take. Can be from the action set.
        """
        logger.debug(f"Action received: {action.value}, Current position: {self.agent_pos}")

        if self.action_set != DEFAULT_ACTIONS:
            error_message = (
                f"Action set {self.action_set} is not supported. "
                f"Only {DEFAULT_ACTIONS} are supported in this environment."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if action == Action.STAY:
            logger.info("Action is stay: staying in place.")
            return

        # Define direction mappings
        direction_map = {
            Direction.UP: {
                Action.FORWARD: Direction.UP,
                Action.LEFT: Direction.LEFT,
                Action.RIGHT: Direction.RIGHT,
            },
            Direction.DOWN: {
                Action.FORWARD: Direction.DOWN,
                Action.LEFT: Direction.RIGHT,
                Action.RIGHT: Direction.LEFT,
            },
            Direction.LEFT: {
                Action.FORWARD: Direction.LEFT,
                Action.LEFT: Direction.DOWN,
                Action.RIGHT: Direction.UP,
            },
            Direction.RIGHT: {
                Action.FORWARD: Direction.RIGHT,
                Action.LEFT: Direction.UP,
                Action.RIGHT: Direction.DOWN,
            },
        }

        # Store the previous direction before attempting to move
        previous_direction = self.current_direction

        # Determine the new direction based on the current direction and action
        new_direction = direction_map[self.current_direction][action]
        self.current_direction = new_direction

        # Calculate the new position based on the new direction
        new_pos = list(self.agent_pos)
        if new_direction == Direction.UP and self.agent_pos[1] < self.grid_size - 1:
            new_pos[1] += 1
        elif new_direction == Direction.DOWN and self.agent_pos[1] > 0:
            new_pos[1] -= 1
        elif new_direction == Direction.RIGHT and self.agent_pos[0] < self.grid_size - 1:
            new_pos[0] += 1
        elif new_direction == Direction.LEFT and self.agent_pos[0] > 0:
            new_pos[0] -= 1
        else:
            logger.warning(
                f"Collision against boundary with action: {action.value}, staying in place.",
            )
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

        logger.debug(
            f"New position: {self.agent_pos}, New direction: {self.current_direction.value}",
        )

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
        symbols = THEME_SYMBOLS[self.theme]

        grid = [[symbols.empty for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal[1]][self.goal[0]] = symbols.goal  # Mark the goal

        # Mark the body
        for segment in self.body:
            grid[segment[1]][segment[0]] = symbols.body

        agent_symbol = getattr(symbols, self.current_direction.value, "@")
        grid[self.agent_pos[1]][self.agent_pos[0]] = agent_symbol  # Mark the agent

        # For emoji theme, join with empty string to avoid extra spaces
        if self.theme == Theme.EMOJI:
            return ["".join(row) for row in reversed(grid)] + [""]
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
            theme=self.theme,
        )
        new_env.body = self.body.copy()
        new_env.current_direction = self.current_direction
        return new_env
