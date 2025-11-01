"""
Maze environment for the Nematode agent.

This environment simulates a simple maze where the agent navigates to reach a goal position.
The agent can move in four directions: up, down, left, or right.
The agent must avoid colliding with itself.
The environment provides methods to get the current state, move the agent,
    check if the goal is reached, and render the maze.
"""

import secrets
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text as RichText

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action
from quantumnematode.theme import THEME_SYMBOLS, DarkColorRichStyleConfig, Theme

from .logging_config import logger

# Validation
MIN_GRID_SIZE = 5
MAX_POISSON_ATTEMPTS = 100

# Constants for gradient scaling
GRADIENT_SCALING_TANH_FACTOR = 1.0


class Corner(Enum):
    """Corners of the maze grid."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class Direction(Enum):
    """Directions the agent can move."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


class ScalingMethod(Enum):
    """Scaling methods used in the maze environment."""

    EXPONENTIAL = "exponential"
    TANH = "tanh"


class BaseEnvironment(ABC):
    """
    Abstract base class for nematode environments.

    Provides common functionality for grid-based navigation environments.

    Attributes
    ----------
    grid_size : int
        Size of the maze grid.
    agent_pos : tuple[int, int]
        Current position of the agent in the grid.
    body : list[tuple[int, int]]
        Positions of the agent's body segments.
    current_direction : Direction
        Current facing direction of the agent.
    theme : Theme
        Visual theme for rendering.
    """

    def __init__(  # noqa: PLR0913
        self,
        grid_size: int,
        start_pos: tuple[int, int] | None,
        max_body_length: int,
        theme: Theme,
        action_set: list[Action],
        rich_style_config: DarkColorRichStyleConfig | None,
    ) -> None:
        """Initialize the base environment."""
        if grid_size < MIN_GRID_SIZE:
            error_message = (
                f"Grid size must be at least {MIN_GRID_SIZE}. Provided grid size: {grid_size}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        self.grid_size = grid_size
        self.agent_pos = start_pos if start_pos else (1, 1)
        self.body = [tuple(self.agent_pos)] if max_body_length > 0 else []
        self.current_direction = Direction.UP
        self.theme = theme
        self.action_set = action_set
        self.rich_style_config = rich_style_config or DarkColorRichStyleConfig()

    @abstractmethod
    def get_state(
        self,
        position: tuple[int, ...],
        *,
        scaling_method: ScalingMethod = ScalingMethod.TANH,
        disable_log: bool = False,
    ) -> tuple[float, float]:
        """
        Get the current state of the agent (gradient strength and direction).

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.
        scaling_method : ScalingMethod
            Method to scale gradient strength.
        disable_log : bool
            Whether to disable debug logging.

        Returns
        -------
        tuple[float, float]
            Gradient strength and direction.
        """

    @abstractmethod
    def reached_goal(self) -> bool:
        """
        Check if the agent has reached a goal.

        Returns
        -------
        bool
            True if goal is reached, False otherwise.
        """

    def _compute_single_gradient(
        self,
        position: tuple[int, ...],
        goal: tuple[int, int],
        scaling_method: ScalingMethod,
    ) -> tuple[float, float, int]:
        """
        Compute gradient from a single goal position.

        Parameters
        ----------
        position : tuple[int, ...]
            Current position.
        goal : tuple[int, int]
            Goal position.
        scaling_method : ScalingMethod
            Scaling method for gradient strength.

        Returns
        -------
        tuple[float, float, int]
            Gradient strength, direction, and Manhattan distance.
        """
        dx = goal[0] - position[0]
        dy = goal[1] - position[1]

        max_distance = self.grid_size * 2
        distance_to_goal = abs(dx) + abs(dy)
        gradient_strength = max(0.0, 1.0 - (distance_to_goal / max_distance))

        if scaling_method == ScalingMethod.EXPONENTIAL:
            gradient_strength = np.exp(-distance_to_goal / max_distance)
        elif scaling_method == ScalingMethod.TANH:
            gradient_strength = np.tanh(gradient_strength * GRADIENT_SCALING_TANH_FACTOR)

        gradient_direction = np.arctan2(dy, dx) if dx != 0 or dy != 0 else 0.0

        return gradient_strength, gradient_direction, distance_to_goal

    def move_agent(self, action: Action) -> None:
        """
        Move the agent based on its perspective.

        Parameters
        ----------
        action : Action
            The action to take.
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

        previous_direction = self.current_direction
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
            self.current_direction = previous_direction
            return

        # Check for collision with the body
        if tuple(new_pos) in self.body:
            logger.warning(f"Collision detected at {new_pos}, staying in place.")
            self.current_direction = previous_direction
            return

        # Update the body positions
        self.body = [tuple(self.agent_pos), *self.body[:-1]] if len(self.body) > 0 else []

        # Update the agent's position
        self.agent_pos = tuple(new_pos)

        logger.debug(
            f"New position: {self.agent_pos}, New direction: {self.current_direction.value}",
        )

    @abstractmethod
    def render(self) -> list[str]:
        """
        Render the current state of the environment.

        Returns
        -------
        list[str]
            List of strings representing the rendered environment.
        """

    def _render_grid(
        self,
        goals: list[tuple[int, int]],
        viewport: tuple[int, int, int, int] | None = None,
    ) -> list[list[str]]:
        """
        Render the grid with goals and agent.

        Parameters
        ----------
        goals : list[tuple[int, int]]
            List of goal positions.
        viewport : tuple[int, int, int, int] | None
            Viewport bounds (min_x, min_y, max_x, max_y) or None for full grid.

        Returns
        -------
        list[list[str]]
            2D grid of symbols.
        """
        symbols = THEME_SYMBOLS[self.theme]

        if viewport:
            min_x, min_y, max_x, max_y = viewport
            width = max_x - min_x
            height = max_y - min_y
            grid = [[symbols.empty for _ in range(width)] for _ in range(height)]

            # Mark goals (only if in viewport)
            for goal in goals:
                if min_x <= goal[0] < max_x and min_y <= goal[1] < max_y:
                    grid_y = goal[1] - min_y
                    grid_x = goal[0] - min_x
                    grid[grid_y][grid_x] = symbols.goal

            # Mark body (only if in viewport)
            for segment in self.body:
                if min_x <= segment[0] < max_x and min_y <= segment[1] < max_y:
                    grid_y = segment[1] - min_y
                    grid_x = segment[0] - min_x
                    grid[grid_y][grid_x] = symbols.body

            # Mark agent (only if in viewport)
            if min_x <= self.agent_pos[0] < max_x and min_y <= self.agent_pos[1] < max_y:
                agent_symbol = getattr(symbols, self.current_direction.value, "@")
                grid_y = self.agent_pos[1] - min_y
                grid_x = self.agent_pos[0] - min_x
                grid[grid_y][grid_x] = agent_symbol
        else:
            # Full grid rendering
            grid = [[symbols.empty for _ in range(self.grid_size)] for _ in range(self.grid_size)]

            for goal in goals:
                grid[goal[1]][goal[0]] = symbols.goal

            for segment in self.body:
                grid[segment[1]][segment[0]] = symbols.body

            agent_symbol = getattr(symbols, self.current_direction.value, "@")
            grid[self.agent_pos[1]][self.agent_pos[0]] = agent_symbol

        return grid

    def _render_grid_to_strings(self, grid: list[list[str]]) -> list[str]:
        """Convert grid to string representation based on theme."""
        if self.theme in (Theme.RICH, Theme.EMOJI_RICH):
            return self._render_rich(grid, theme=self.theme)

        if self.theme == Theme.EMOJI:
            return ["".join(row) for row in reversed(grid)] + [""]
        if self.theme == Theme.UNICODE:
            return [" ".join(row) for row in reversed(grid)] + [""]
        if self.theme == Theme.RICH:
            return ["".join(row) for row in reversed(grid)] + [""]
        return [" ".join(row) for row in reversed(grid)] + [""]

    def _render_rich(self, grid: list[list[str]], theme: Theme) -> list[str]:
        """Render the grid with Rich styling and colors as strings."""
        width_extra_pad = 1 if theme == Theme.EMOJI_RICH else 0
        grid_width = len(grid[0])
        table_width = (grid_width * (4 + width_extra_pad)) + 1

        console = Console(
            record=True,
            width=table_width,
            legacy_windows=False,
            force_terminal=True,
        )

        symbols = THEME_SYMBOLS[self.theme]

        table = Table(
            show_header=False,
            show_lines=True,
            box=box.SQUARE,
            padding=(0, 0),
            pad_edge=False,
            style=self.rich_style_config.grid_background,
        )

        for _ in range(grid_width):
            table.add_column(
                justify="center",
                width=3 + width_extra_pad,
                min_width=3 + width_extra_pad,
                max_width=3 + width_extra_pad,
                no_wrap=True,
            )

        for row in reversed(grid):
            styled_cells = []
            for cell in row:
                if cell == symbols.goal:
                    styled_cell = RichText(cell, style=self.rich_style_config.goal_style)
                elif cell == symbols.body:
                    styled_cell = RichText(cell, style=self.rich_style_config.body_style)
                elif cell in [symbols.up, symbols.down, symbols.left, symbols.right]:
                    styled_cell = RichText(cell, style=self.rich_style_config.agent_style)
                else:
                    styled_cell = RichText(
                        cell,
                        style=self.rich_style_config.empty_style,
                        justify="center",
                    )
                styled_cells.append(styled_cell)

            table.add_row(*styled_cells)

        with console.capture() as capture:
            console.print(table, crop=True)

        output_lines = capture.get().splitlines()
        cleaned_lines = [line.rstrip() for line in output_lines if line.strip()]

        return [*cleaned_lines, ""]

    @abstractmethod
    def copy(self) -> "BaseEnvironment":
        """
        Create a deep copy of the environment.

        Returns
        -------
        BaseEnvironment
            A new instance with the same state.
        """


class MazeEnvironment(BaseEnvironment):
    """
    A simple maze environment for the Nematode agent.

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
        rich_style_config: DarkColorRichStyleConfig | None = None,
    ) -> None:
        # Randomize agent and goal positions to all 4 corners
        corners = [
            (1, 1),
            (1, grid_size - 2),
            (grid_size - 2, 1),
            (grid_size - 2, grid_size - 2),
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

        super().__init__(
            grid_size=grid_size,
            start_pos=start_pos,
            max_body_length=max_body_length,
            theme=theme,
            action_set=action_set,
            rich_style_config=rich_style_config,
        )

        self.goal = food_pos

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
        gradient_strength, gradient_direction, _ = self._compute_single_gradient(
            position,
            self.goal,
            scaling_method,
        )

        if not disable_log:
            dx = self.goal[0] - position[0]
            dy = self.goal[1] - position[1]
            logger.debug(
                f"Gradient computation details: dx={dx}, dy={dy}, "
                f"gradient_strength={gradient_strength}, gradient_direction={gradient_direction}",
            )
            logger.debug(
                f"Agent position: {self.agent_pos}, Body positions: {self.body}, Goal: {self.goal}",
            )

        return gradient_strength, gradient_direction

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
        grid = self._render_grid([self.goal])
        return self._render_grid_to_strings(grid)

    def print_rich(self) -> None:
        """Print the grid using Rich console with colors (Rich theme only)."""
        console = Console()
        styled_grid = self.render()
        for row in styled_grid:
            if row:
                console.print(" ".join(str(cell) for cell in row))
            else:
                console.print()

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


class DynamicForagingEnvironment(BaseEnvironment):
    """
    Dynamic multi-food foraging environment for the Nematode agent.

    The agent navigates a large grid to find multiple food sources using gradient
    superposition. Food sources respawn when consumed, and the agent has a satiety
    level that decays over time.

    Attributes
    ----------
    grid_size : int
        Size of the maze grid.
    agent_pos : tuple[int, int]
        Current position of the agent.
    body : list[tuple[int, int]]
        Positions of the agent's body segments.
    foods : list[tuple[int, int]]
        Positions of active food sources.
    visited_cells : set[tuple[int, int]]
        Set of cells the agent has visited.
    satiety : float
        Current satiety (hunger) level of the agent.
    """

    def __init__(  # noqa: PLR0913
        self,
        grid_size: int = 50,
        start_pos: tuple[int, int] | None = None,
        num_initial_foods: int = 10,
        max_active_foods: int = 15,
        min_food_distance: int = 5,
        agent_exclusion_radius: int = 10,
        gradient_decay_constant: float = 10.0,
        gradient_strength: float = 1.0,
        viewport_size: tuple[int, int] = (11, 11),
        max_body_length: int = 6,
        theme: Theme = Theme.ASCII,
        action_set: list[Action] = DEFAULT_ACTIONS,
        rich_style_config: DarkColorRichStyleConfig | None = None,
    ) -> None:
        """Initialize the dynamic foraging environment."""
        if start_pos is None:
            start_pos = (grid_size // 2, grid_size // 2)

        super().__init__(
            grid_size=grid_size,
            start_pos=start_pos,
            max_body_length=max_body_length,
            theme=theme,
            action_set=action_set,
            rich_style_config=rich_style_config,
        )

        self.num_initial_foods = num_initial_foods
        self.max_active_foods = max_active_foods
        self.min_food_distance = min_food_distance
        self.agent_exclusion_radius = agent_exclusion_radius
        self.gradient_decay_constant = gradient_decay_constant
        self.gradient_strength_base = gradient_strength
        self.viewport_size = viewport_size

        # Initialize food sources using Poisson disk sampling
        self.foods: list[tuple[int, int]] = []
        self._initialize_foods()

        # Track visited cells for exploration bonus
        self.visited_cells: set[tuple[int, int]] = {(self.agent_pos[0], self.agent_pos[1])}

        # Satiety tracking (will be set by agent)
        self.satiety = 0.0

    def _initialize_foods(self) -> None:
        """Initialize food sources using Poisson disk sampling."""
        self.foods = []
        attempts = 0
        max_total_attempts = MAX_POISSON_ATTEMPTS * self.num_initial_foods

        while len(self.foods) < self.num_initial_foods and attempts < max_total_attempts:
            candidate = (
                secrets.randbelow(self.grid_size),
                secrets.randbelow(self.grid_size),
            )

            if self._is_valid_food_position(candidate):
                self.foods.append(candidate)

            attempts += 1

        if len(self.foods) < self.num_initial_foods:
            logger.warning(
                f"Could only place {len(self.foods)}/{self.num_initial_foods} initial foods "
                f"after {attempts} attempts.",
            )

    def _is_valid_food_position(self, pos: tuple[int, int]) -> bool:
        """
        Check if a position is valid for food placement.

        Uses Euclidean distance to ensure proper Poisson disk sampling.

        Parameters
        ----------
        pos : tuple[int, int]
            Position to check.

        Returns
        -------
        bool
            True if position is valid, False otherwise.
        """
        # Check Euclidean distance from agent
        agent_dist = np.sqrt(
            (pos[0] - self.agent_pos[0]) ** 2 + (pos[1] - self.agent_pos[1]) ** 2,
        )
        if agent_dist < self.agent_exclusion_radius:
            return False

        # Check Euclidean distance from other foods
        for food in self.foods:
            dist = np.sqrt((pos[0] - food[0]) ** 2 + (pos[1] - food[1]) ** 2)
            if dist < self.min_food_distance:
                return False

        return True

    def spawn_food(self) -> bool:
        """
        Spawn a new food source if under the maximum limit.

        Returns
        -------
        bool
            True if food was spawned, False otherwise.
        """
        if len(self.foods) >= self.max_active_foods:
            return False

        for _ in range(MAX_POISSON_ATTEMPTS):
            candidate = (
                secrets.randbelow(self.grid_size),
                secrets.randbelow(self.grid_size),
            )

            if self._is_valid_food_position(candidate):
                self.foods.append(candidate)
                logger.debug(f"Spawned new food at {candidate}")
                return True

        logger.warning("Failed to spawn new food after maximum attempts")
        return False

    def get_state(
        self,
        position: tuple[int, ...],
        *,
        scaling_method: ScalingMethod = ScalingMethod.EXPONENTIAL,  # noqa: ARG002
        disable_log: bool = False,
    ) -> tuple[float, float]:
        """
        Get the current state with gradient superposition from all food sources.

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.
        scaling_method : ScalingMethod
            Scaling method for gradient strength.
        disable_log : bool
            Whether to disable debug logging.

        Returns
        -------
        tuple[float, float]
            Total gradient strength and direction.
        """
        if not self.foods:
            return 0.0, 0.0

        # Compute gradient from each food source
        total_vector_x = 0.0
        total_vector_y = 0.0

        for food in self.foods:
            dx = food[0] - position[0]
            dy = food[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance == 0:
                continue

            # Exponential decay gradient
            strength = self.gradient_strength_base * np.exp(
                -distance / self.gradient_decay_constant,
            )

            # Compute unit direction vector
            direction = np.arctan2(dy, dx)
            vector_x = strength * np.cos(direction)
            vector_y = strength * np.sin(direction)

            total_vector_x += vector_x
            total_vector_y += vector_y

        # Compute magnitude and direction of superposed gradient
        gradient_magnitude = np.sqrt(total_vector_x**2 + total_vector_y**2)
        gradient_direction = np.arctan2(total_vector_y, total_vector_x)

        if not disable_log:
            logger.debug(
                f"Gradient superposition: magnitude={gradient_magnitude}, "
                f"direction={gradient_direction}, num_foods={len(self.foods)}",
            )

        return float(gradient_magnitude), float(gradient_direction)

    def reached_goal(self) -> bool:
        """
        Check if the agent has reached any food source.

        Returns
        -------
        bool
            True if agent is at a food position, False otherwise.
        """
        return tuple(self.agent_pos) in self.foods

    def consume_food(self) -> tuple[int, int] | None:
        """
        Consume food at the agent's current position.

        Returns
        -------
        tuple[int, int] | None
            Position of consumed food, or None if no food at current position.
        """
        agent_tuple = (self.agent_pos[0], self.agent_pos[1])
        if agent_tuple in self.foods:
            self.foods.remove(agent_tuple)
            logger.info(f"Food consumed at {agent_tuple}")

            # Spawn new food
            self.spawn_food()

            return agent_tuple

        return None

    def get_nearest_food_distance(self) -> int | None:
        """
        Get Manhattan distance to the nearest food source.

        Returns
        -------
        int | None
            Distance to nearest food, or None if no foods exist.
        """
        if not self.foods:
            return None

        distances = [
            abs(self.agent_pos[0] - food[0]) + abs(self.agent_pos[1] - food[1])
            for food in self.foods
        ]
        return min(distances)

    def _get_viewport_bounds(self) -> tuple[int, int, int, int]:
        """
        Calculate viewport bounds centered on the agent.

        Returns
        -------
        tuple[int, int, int, int]
            Viewport bounds (min_x, min_y, max_x, max_y).
        """
        half_width = self.viewport_size[0] // 2
        half_height = self.viewport_size[1] // 2

        min_x = max(0, self.agent_pos[0] - half_width)
        min_y = max(0, self.agent_pos[1] - half_height)
        max_x = min(self.grid_size, self.agent_pos[0] + half_width + 1)
        max_y = min(self.grid_size, self.agent_pos[1] + half_height + 1)

        return min_x, min_y, max_x, max_y

    def render(self) -> list[str]:
        """Render the viewport centered on the agent."""
        viewport = self._get_viewport_bounds()
        grid = self._render_grid(self.foods, viewport=viewport)
        return self._render_grid_to_strings(grid)

    def render_full(self) -> list[str]:
        """Render the entire environment (for logging/debugging)."""
        grid = self._render_grid(self.foods)
        return self._render_grid_to_strings(grid)

    def copy(self) -> "DynamicForagingEnvironment":
        """
        Create a deep copy of the DynamicForagingEnvironment.

        Returns
        -------
        DynamicForagingEnvironment
            A new instance with the same state.
        """
        new_env = DynamicForagingEnvironment(
            grid_size=self.grid_size,
            start_pos=(self.agent_pos[0], self.agent_pos[1]),
            num_initial_foods=self.num_initial_foods,
            max_active_foods=self.max_active_foods,
            min_food_distance=self.min_food_distance,
            agent_exclusion_radius=self.agent_exclusion_radius,
            gradient_decay_constant=self.gradient_decay_constant,
            gradient_strength=self.gradient_strength_base,
            viewport_size=self.viewport_size,
            max_body_length=len(self.body),
            theme=self.theme,
            action_set=self.action_set,
            rich_style_config=self.rich_style_config,
        )
        new_env.body = self.body.copy()
        new_env.current_direction = self.current_direction
        new_env.foods = self.foods.copy()
        new_env.visited_cells = self.visited_cells.copy()
        new_env.satiety = self.satiety
        return new_env
