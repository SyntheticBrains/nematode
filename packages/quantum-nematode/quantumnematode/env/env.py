"""
Foraging environments for the Nematode agent.

These environments simulate foraging tasks where the agent navigates to collect
food sources in a DynamicForagingEnvironment. The agent can move in four directions:
up, down, left, or right. The agent must avoid colliding with itself.
The environment provides methods to get the current state, move the agent,
check if the goal is reached, and render the environment.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np
from pydantic.dataclasses import dataclass
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text as RichText

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action
from quantumnematode.dtypes import GradientPolar, GridPosition, TemperatureSpot
from quantumnematode.env.temperature import (
    TemperatureField,
    TemperatureZone,
    TemperatureZoneThresholds,
)
from quantumnematode.env.theme import (
    THEME_SYMBOLS,
    DarkColorRichStyleConfig,
    Theme,
    ThemeSymbolSet,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng

# Validation
MIN_GRID_SIZE = 5
MAX_POISSON_ATTEMPTS = 100

# Constants for gradient scaling
GRADIENT_SCALING_TANH_FACTOR = 1.0

# Type aliases
type Viewport = tuple[int, int, int, int]
"""Viewport bounds as (min_x, min_y, max_x, max_y) in world coordinates."""


class Direction(Enum):
    """Directions the agent can move."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


class PredatorType(Enum):
    """
    Types of predator movement behaviors.

    RANDOM: Moves randomly in one of four directions each step (default).
    STATIONARY: Does not move, acts as a toxic zone with larger damage radius.
    PURSUIT: Moves toward the agent when within detection radius, otherwise random.
    """

    RANDOM = "random"
    STATIONARY = "stationary"
    PURSUIT = "pursuit"


# Health system defaults
DEFAULT_MAX_HP = 100.0
DEFAULT_PREDATOR_DAMAGE = 10.0
DEFAULT_FOOD_HEALING = 5.0


@dataclass
class ForagingParams:
    """
    Parameters for food/foraging configuration in the environment.

    TODO: Freeze this class once the following Pylance issue is resolved:
    https://github.com/microsoft/pylance-release/issues/7801

    Attributes
    ----------
    foods_on_grid : int
        Number of food items to maintain on the grid.
    target_foods_to_collect : int
        Number of foods needed to complete the task.
    min_food_distance : int
        Minimum distance between food items (Poisson disk sampling).
    agent_exclusion_radius : int
        Minimum distance from agent for initial food placement.
    gradient_decay_constant : float
        Decay constant for food gradient sensing.
    gradient_strength : float
        Strength of food gradient signal.
    safe_zone_food_bias : float
        Probability (0.0-1.0) that food spawns in safe temperature zones
        (COMFORT or DISCOMFORT). Set to 0.0 to disable (uniform spawning).
        Requires thermotaxis to be enabled; ignored otherwise.
    """

    foods_on_grid: int = 10
    target_foods_to_collect: int = 15
    min_food_distance: int = 5
    agent_exclusion_radius: int = 10
    gradient_decay_constant: float = 10.0
    gradient_strength: float = 1.0
    safe_zone_food_bias: float = 0.0


@dataclass
class PredatorParams:
    """
    Parameters for predator configuration in the environment.

    TODO: Freeze this class once the following Pylance issue is resolved:
    https://github.com/microsoft/pylance-release/issues/7801

    Attributes
    ----------
    enabled : bool
        Whether predators are active in the environment.
    count : int
        Number of predators to spawn.
    predator_type : PredatorType
        Movement behavior type (RANDOM, STATIONARY, or PURSUIT).
    speed : float
        Movement speed relative to agent.
    detection_radius : int
        Distance at which pursuit predators detect the agent.
    kill_radius : int
        Distance for instant death (when health system disabled).
    damage_radius : int
        Distance at which predators deal damage (when health system enabled).
        Stationary predators typically have larger damage_radius (toxic zones).
    gradient_decay_constant : float
        Controls how quickly predator gradient signal decays with distance.
    gradient_strength : float
        Multiplier for predator gradient signal strength.
    """

    enabled: bool = False
    count: int = 2
    predator_type: PredatorType = PredatorType.RANDOM
    speed: float = 1.0
    detection_radius: int = 8
    kill_radius: int = 0
    damage_radius: int = 0
    gradient_decay_constant: float = 12.0
    gradient_strength: float = 1.0


@dataclass
class HealthParams:
    """
    Parameters for health system configuration in the environment.

    TODO: Freeze this class once the following Pylance issue is resolved:
    https://github.com/microsoft/pylance-release/issues/7801
    """

    enabled: bool = False
    max_hp: float = DEFAULT_MAX_HP
    predator_damage: float = DEFAULT_PREDATOR_DAMAGE
    food_healing: float = DEFAULT_FOOD_HEALING


# Thermotaxis system defaults
DEFAULT_CULTIVATION_TEMPERATURE = 20.0
DEFAULT_TEMPERATURE_GRADIENT_STRENGTH = 0.5
DEFAULT_COMFORT_REWARD = 0.05
DEFAULT_DISCOMFORT_PENALTY = -0.1
DEFAULT_DANGER_PENALTY = -0.3
DEFAULT_DANGER_HP_DAMAGE = 2.0
DEFAULT_LETHAL_HP_DAMAGE = 10.0


@dataclass
class ThermotaxisParams:
    """
    Parameters for thermotaxis configuration in the environment.

    Thermotaxis simulates C. elegans temperature-guided navigation behavior.
    Worms navigate toward their cultivation temperature (Tc) using AFD
    thermosensory neurons.

    TODO: Freeze this class once the following Pylance issue is resolved:
    https://github.com/microsoft/pylance-release/issues/7801

    Attributes
    ----------
    enabled : bool
        Whether thermotaxis is active in the environment.
    cultivation_temperature : float
        The worm's preferred temperature (Tc) in °C.
    base_temperature : float
        Base temperature of the environment in °C.
    gradient_direction : float
        Direction of increasing temperature in radians (0 = rightward).
    gradient_strength : float
        Temperature change per cell in °C.
    hot_spots : list[TemperatureSpot] | None
        Localized hot spots as (x, y, intensity) tuples.
    cold_spots : list[TemperatureSpot] | None
        Localized cold spots as (x, y, intensity) tuples.
    spot_decay_constant : float
        Decay constant for hot/cold spot exponential falloff.
    comfort_reward : float
        Reward per step when in comfort zone.
    discomfort_penalty : float
        Penalty per step when in discomfort zone.
    danger_penalty : float
        Penalty per step when in danger zone.
    danger_hp_damage : float
        HP damage per step when in danger zone (requires health system).
    lethal_hp_damage : float
        HP damage per step when in lethal zone (requires health system).
    comfort_delta : float
        Temperature deviation from Tc for comfort zone boundary (°C).
    discomfort_delta : float
        Temperature deviation from Tc for discomfort zone boundary (°C).
    danger_delta : float
        Temperature deviation from Tc for danger zone boundary (°C).
    reward_discomfort_food : float
        Bonus reward for collecting food while in a discomfort zone.
        Encourages "brave foraging" - entering uncomfortable but safe zones for food.
    """

    enabled: bool = False
    cultivation_temperature: float = DEFAULT_CULTIVATION_TEMPERATURE
    base_temperature: float = DEFAULT_CULTIVATION_TEMPERATURE
    gradient_direction: float = 0.0
    gradient_strength: float = DEFAULT_TEMPERATURE_GRADIENT_STRENGTH
    hot_spots: list[TemperatureSpot] | None = None
    cold_spots: list[TemperatureSpot] | None = None
    spot_decay_constant: float = 5.0
    comfort_reward: float = DEFAULT_COMFORT_REWARD
    discomfort_penalty: float = DEFAULT_DISCOMFORT_PENALTY
    danger_penalty: float = DEFAULT_DANGER_PENALTY
    danger_hp_damage: float = DEFAULT_DANGER_HP_DAMAGE
    lethal_hp_damage: float = DEFAULT_LETHAL_HP_DAMAGE
    comfort_delta: float = 5.0
    discomfort_delta: float = 10.0
    danger_delta: float = 15.0
    reward_discomfort_food: float = 0.0


class Predator:
    """
    Represents a predator entity in the environment.

    Attributes
    ----------
    position : tuple[int, int]
        Current position of the predator.
    predator_type : PredatorType
        Movement behavior type (RANDOM, STATIONARY, or PURSUIT).
    speed : float
        Movement speed relative to agent (1.0 = same speed).
        Supports fractional speeds (< 1.0) and multi-step movement (> 1.0).
        Capped at 10 steps per update for safety.
    movement_accumulator : float
        Accumulator for fractional movement (handles both speed < 1.0 and > 1.0).
    detection_radius : int
        Distance at which pursuit predators detect and chase the agent.
        Only used for PURSUIT type predators.
    damage_radius : int
        Distance at which this predator deals damage (when health system enabled).
        Stationary predators typically have larger damage_radius (toxic zones).
    """

    def __init__(  # noqa: PLR0913
        self,
        position: tuple[int, int],
        predator_type: PredatorType = PredatorType.RANDOM,
        speed: float = 1.0,
        movement_accumulator: float = 0.0,
        detection_radius: int = 8,
        damage_radius: int = 0,
    ) -> None:
        """
        Initialize a predator.

        Parameters
        ----------
        position : tuple[int, int]
            Starting position of the predator.
        predator_type : PredatorType
            Movement behavior type (default RANDOM).
        speed : float
            Movement speed (default 1.0).
        movement_accumulator : float
            Initial movement accumulator (default 0.0).
        detection_radius : int
            Detection radius for pursuit predators (default 8).
        damage_radius : int
            Distance at which this predator deals damage (default 0).
        """
        self.position = position
        self.predator_type = predator_type
        self.speed = speed
        self.movement_accumulator = movement_accumulator
        self.detection_radius = detection_radius
        self.damage_radius = damage_radius

    def update_position(
        self,
        grid_size: int,
        rng: np.random.Generator,
        agent_pos: tuple[int, int] | None = None,
    ) -> None:
        """
        Update predator position based on its type.

        Parameters
        ----------
        grid_size : int
            Size of the grid (for boundary checking).
        rng : np.random.Generator
            Random number generator for reproducible movement.
        agent_pos : tuple[int, int] | None
            Current agent position. Required for PURSUIT type predators.
        """
        if self.predator_type == PredatorType.STATIONARY:
            return  # Stationary predators don't move

        if self.predator_type == PredatorType.PURSUIT and agent_pos is not None:
            self._update_pursuit(grid_size, rng, agent_pos)
        else:
            self._update_random(grid_size, rng)

    def _update_random(self, grid_size: int, rng: np.random.Generator) -> None:
        """
        Update predator position with random movement.

        Supports speeds > 1.0 for multiple steps per update.
        For safety, caps maximum steps per update at 10 to prevent
        pathological behavior with very high speeds.

        Parameters
        ----------
        grid_size : int
            Size of the grid (for boundary checking).
        rng : np.random.Generator
            Random number generator for reproducible movement.
        """
        # Handle fractional and multi-step movement
        self.movement_accumulator += self.speed
        if self.movement_accumulator < 1.0:
            return  # Don't move yet

        # Take multiple steps if speed > 1.0
        # Cap at 10 steps per update for safety
        max_steps_per_update = 10
        steps_taken = 0

        while self.movement_accumulator >= 1.0 and steps_taken < max_steps_per_update:
            self.movement_accumulator -= 1.0
            steps_taken += 1

            # Random movement in one of four directions
            up = 0
            down = 1
            left = 2
            right = 3

            direction_choice = rng.integers(4)
            x, y = self.position

            if direction_choice == up:
                y = max(0, y - 1)
            elif direction_choice == down:
                y = min(grid_size - 1, y + 1)
            elif direction_choice == left:
                x = max(0, x - 1)
            elif direction_choice == right:
                x = min(grid_size - 1, x + 1)

            self.position = (x, y)

    def _update_pursuit(
        self,
        grid_size: int,
        rng: np.random.Generator,
        agent_pos: tuple[int, int],
    ) -> None:
        """
        Update predator position with pursuit behavior.

        When within detection radius, moves toward the agent.
        When outside detection radius, moves randomly.

        Parameters
        ----------
        grid_size : int
            Size of the grid (for boundary checking).
        rng : np.random.Generator
            Random number generator for reproducible movement.
        agent_pos : tuple[int, int]
            Current agent position.
        """
        # Calculate Manhattan distance to agent
        px, py = self.position
        ax, ay = agent_pos
        distance = abs(px - ax) + abs(py - ay)

        # If outside detection radius, move randomly
        if distance > self.detection_radius:
            self._update_random(grid_size, rng)
            return

        # Inside detection radius: pursue the agent
        self.movement_accumulator += self.speed
        if self.movement_accumulator < 1.0:
            return  # Don't move yet

        max_steps_per_update = 10
        steps_taken = 0

        while self.movement_accumulator >= 1.0 and steps_taken < max_steps_per_update:
            self.movement_accumulator -= 1.0
            steps_taken += 1

            x, y = self.position

            # Calculate direction toward agent
            dx = ax - x
            dy = ay - y

            # Move in the direction with larger distance first (greedy pursuit)
            if abs(dx) >= abs(dy):
                # Move horizontally
                if dx > 0:
                    x = min(grid_size - 1, x + 1)
                elif dx < 0:
                    x = max(0, x - 1)
            # Move vertically
            elif dy > 0:
                y = min(grid_size - 1, y + 1)
            elif dy < 0:
                y = max(0, y - 1)

            self.position = (x, y)


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
    seed : int
        Random seed for reproducibility.
    rng : np.random.Generator
        Seeded random number generator for all random operations.
    """

    # Maps current direction + action to resulting direction
    # Used for relative movement (FORWARD/LEFT/RIGHT from agent's perspective)
    DIRECTION_MAP: ClassVar[dict[Direction, dict[Action, Direction]]] = {
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

    def __init__(  # noqa: PLR0913
        self,
        grid_size: int,
        start_pos: tuple[int, int] | None,
        max_body_length: int,
        theme: Theme,
        action_set: list[Action],
        rich_style_config: DarkColorRichStyleConfig | None,
        seed: int | None = None,
    ) -> None:
        """Initialize the base environment."""
        if grid_size < MIN_GRID_SIZE:
            error_message = (
                f"Grid size must be at least {MIN_GRID_SIZE}. Provided grid size: {grid_size}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Initialize seeding first as it may be used for random positions
        self.seed = ensure_seed(seed)
        self.rng = get_rng(self.seed)

        self.grid_size = grid_size
        self.agent_pos = start_pos or (1, 1)
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
        disable_log: bool = False,
    ) -> tuple[float, float]:
        """
        Get the current state of the agent (gradient strength and direction).

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.
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

    def _get_new_position_if_valid(
        self,
        direction: Direction,
    ) -> tuple[int, int] | None:
        """
        Calculate the new position if moving in the given direction is valid.

        Parameters
        ----------
        direction : Direction
            The direction to move.

        Returns
        -------
        tuple[int, int] | None
            The new position if the move is valid (within grid bounds),
            or None if the move would hit a wall.
        """
        x, y = self.agent_pos
        match direction:
            case Direction.UP:
                return (x, y + 1) if y < self.grid_size - 1 else None
            case Direction.DOWN:
                return (x, y - 1) if y > 0 else None
            case Direction.RIGHT:
                return (x + 1, y) if x < self.grid_size - 1 else None
            case Direction.LEFT:
                return (x - 1, y) if x > 0 else None
            case _:
                return None

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
            logger.debug("Action is stay: staying in place.")
            return

        previous_direction = self.current_direction
        new_direction = self.DIRECTION_MAP[self.current_direction][action]
        self.current_direction = new_direction

        # Calculate the new position based on the new direction
        new_pos = self._get_new_position_if_valid(new_direction)
        if new_pos is None:
            logger.debug(
                f"Collision against boundary with action: {action.value}, staying in place.",
            )
            self.current_direction = previous_direction
            return

        # Check for collision with the body
        if tuple(new_pos) in self.body:
            logger.debug(f"Collision detected at {new_pos}, staying in place.")
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
        viewport: Viewport | None = None,
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

    def _render_grid_to_strings(
        self,
        grid: list[list[str]],
        viewport: Viewport | None = None,
    ) -> list[str]:
        """Convert grid to string representation based on theme.

        Parameters
        ----------
        grid : list[list[str]]
            The grid of symbols to render.
        viewport : tuple[int, int, int, int] | None
            Viewport bounds (min_x, min_y, max_x, max_y) for coordinate mapping.
            Used by subclasses to determine zone backgrounds.
        """
        if self.theme in (Theme.RICH, Theme.EMOJI_RICH):
            return self._render_rich(grid, theme=self.theme, viewport=viewport)

        if self.theme == Theme.EMOJI:
            return ["".join(row) for row in reversed(grid)] + [""]
        if self.theme == Theme.UNICODE:
            return [" ".join(row) for row in reversed(grid)] + [""]
        if self.theme == Theme.RICH:
            return ["".join(row) for row in reversed(grid)] + [""]
        return [" ".join(row) for row in reversed(grid)] + [""]

    def _render_rich(
        self,
        grid: list[list[str]],
        theme: Theme,
        viewport: Viewport | None = None,  # noqa: ARG002 - used by subclass override
    ) -> list[str]:
        """Render the grid with Rich styling and colors as strings.

        Parameters
        ----------
        grid : list[list[str]]
            2D grid of symbols to render.
        theme : Theme
            The theme to use for rendering.
        viewport : Viewport | None
            Viewport bounds for coordinate mapping. Unused in base class but
            used by subclasses (e.g., DynamicForagingEnvironment) for zone rendering.
        """
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
        viewport_size: tuple[int, int] = (11, 11),
        max_body_length: int = 6,
        theme: Theme = Theme.ASCII,
        action_set: list[Action] = DEFAULT_ACTIONS,
        rich_style_config: DarkColorRichStyleConfig | None = None,
        seed: int | None = None,
        # Encapsulated parameter objects
        foraging: ForagingParams | None = None,
        predator: PredatorParams | None = None,
        health: HealthParams | None = None,
        thermotaxis: ThermotaxisParams | None = None,
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
            seed=seed,
        )

        # Store parameter objects (use defaults if None)
        self.foraging = foraging or ForagingParams()
        self.predator = predator or PredatorParams()
        self.health = health or HealthParams()
        self.thermotaxis = thermotaxis or ThermotaxisParams()

        self.viewport_size = viewport_size

        # Health state (runtime, not config)
        self.agent_hp: float = self.health.max_hp if self.health.enabled else 0.0

        # Temperature field (runtime, created from thermotaxis config)
        self.temperature_field: TemperatureField | None = None
        if self.thermotaxis.enabled:
            self.temperature_field = TemperatureField(
                grid_size=grid_size,
                base_temperature=self.thermotaxis.base_temperature,
                gradient_direction=self.thermotaxis.gradient_direction,
                gradient_strength=self.thermotaxis.gradient_strength,
                hot_spots=self.thermotaxis.hot_spots,
                cold_spots=self.thermotaxis.cold_spots,
                spot_decay_constant=self.thermotaxis.spot_decay_constant,
            )

        # Thermotaxis tracking (comfort score calculation)
        self.steps_in_comfort_zone: int = 0
        self.total_thermotaxis_steps: int = 0

        # Validate gradient parameters to prevent divide-by-zero in exp(-distance/decay)
        if self.foraging.gradient_decay_constant <= 0:
            msg = (
                f"gradient_decay_constant must be > 0, got {self.foraging.gradient_decay_constant}"
            )
            raise ValueError(msg)
        # Only validate predator gradient decay if predators are enabled
        if self.predator.enabled and self.predator.gradient_decay_constant <= 0:
            msg = (
                f"predator_gradient_decay must be > 0, got {self.predator.gradient_decay_constant}"
            )
            raise ValueError(msg)

        # Initialize food sources using Poisson disk sampling
        self.foods: list[tuple[int, int]] = []
        self._initialize_foods()

        # Initialize predators if enabled
        self.predators: list[Predator] = []
        if self.predator.enabled:
            self._initialize_predators()

        # Track visited cells for exploration bonus
        self.visited_cells: set[tuple[int, int]] = {(self.agent_pos[0], self.agent_pos[1])}

        # Track wall collision for boundary penalty (reset each step)
        self.wall_collision_occurred: bool = False

    def _initialize_foods(self) -> None:
        """
        Initialize food sources using Poisson disk sampling.

        Respects safe_zone_food_bias: with probability `bias`, food is only
        placed in safe temperature zones (COMFORT or DISCOMFORT). Otherwise,
        food can be placed anywhere valid.
        """
        self.foods = []
        attempts = 0
        max_total_attempts = MAX_POISSON_ATTEMPTS * self.foraging.foods_on_grid
        bias = self.foraging.safe_zone_food_bias

        while len(self.foods) < self.foraging.foods_on_grid and attempts < max_total_attempts:
            candidate = (
                int(self.rng.integers(self.grid_size)),
                int(self.rng.integers(self.grid_size)),
            )

            if self._is_valid_food_position(candidate):
                # Apply safe zone bias if thermotaxis is enabled
                if bias > 0 and self.thermotaxis.enabled:
                    require_safe_zone = self.rng.random() < bias
                    if require_safe_zone and not self._is_safe_temperature_zone(candidate):
                        attempts += 1
                        continue
                self.foods.append(candidate)

            attempts += 1

        if len(self.foods) < self.foraging.foods_on_grid:
            logger.warning(
                f"Could only place {len(self.foods)}/{self.foraging.foods_on_grid} initial foods "
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
        if agent_dist < self.foraging.agent_exclusion_radius:
            return False

        # Check Euclidean distance from other foods
        for food in self.foods:
            dist = np.sqrt((pos[0] - food[0]) ** 2 + (pos[1] - food[1]) ** 2)
            if dist < self.foraging.min_food_distance:
                return False

        return True

    def _initialize_predators(self) -> None:
        """Initialize predators at random positions outside damage radius of agent."""
        self.predators = []
        # Use the larger of detection_radius or damage_radius as minimum spawn distance
        # to ensure agent doesn't start in danger zone or immediately detected
        min_spawn_distance = max(
            self.predator.detection_radius,
            self.predator.damage_radius,
        )
        for _ in range(self.predator.count):
            # Spawn predators outside danger zone to avoid immediate damage
            candidate = (0, 0)  # Default position in case loop never runs
            for _ in range(MAX_POISSON_ATTEMPTS):
                candidate = (
                    int(self.rng.integers(self.grid_size)),
                    int(self.rng.integers(self.grid_size)),
                )
                # Calculate Euclidean distance to agent (consistent with Predator methods)
                distance_to_agent = np.sqrt(
                    (candidate[0] - self.agent_pos[0]) ** 2
                    + (candidate[1] - self.agent_pos[1]) ** 2,
                )
                # Ensure predator spawns outside both detection and damage radius
                if distance_to_agent > min_spawn_distance:
                    predator = Predator(
                        position=candidate,
                        predator_type=self.predator.predator_type,
                        speed=self.predator.speed,
                        detection_radius=self.predator.detection_radius,
                        damage_radius=self.predator.damage_radius,
                    )
                    self.predators.append(predator)
                    logger.debug(
                        f"Initialized {self.predator.predator_type.value} predator "
                        f"at {candidate} (euclidean distance to agent: {distance_to_agent:.2f})",
                    )
                    break
            else:
                # If we couldn't find a valid position after max attempts, log warning
                # but still spawn the predator (edge case for very small grids)
                logger.warning(
                    "Could not find safe spawn position for predator "
                    f"after {MAX_POISSON_ATTEMPTS} attempts. "
                    f"Spawning at {candidate} anyway.",
                )
                predator = Predator(
                    position=candidate,
                    predator_type=self.predator.predator_type,
                    speed=self.predator.speed,
                    detection_radius=self.predator.detection_radius,
                    damage_radius=self.predator.damage_radius,
                )
                self.predators.append(predator)

    def spawn_food(self) -> bool:
        """
        Spawn a new food source to maintain target count on grid.

        Always attempts to maintain foods_on_grid count on the grid.
        Food spawns immediately after collection to ensure constant supply.

        Respects safe_zone_food_bias: with probability `bias`, food is only
        placed in safe temperature zones (COMFORT or DISCOMFORT). Otherwise,
        food can be placed anywhere valid.

        Returns
        -------
        bool
            True if food was spawned, False otherwise.
        """
        # Check if we're already at target grid capacity
        if len(self.foods) >= self.foraging.foods_on_grid:
            return False

        bias = self.foraging.safe_zone_food_bias
        require_safe_zone = bias > 0 and self.thermotaxis.enabled and self.rng.random() < bias

        # Attempt to spawn food at valid location
        for _ in range(MAX_POISSON_ATTEMPTS):
            candidate = (
                int(self.rng.integers(self.grid_size)),
                int(self.rng.integers(self.grid_size)),
            )

            if self._is_valid_food_position(candidate):
                # Check safe zone requirement if applicable
                if require_safe_zone and not self._is_safe_temperature_zone(candidate):
                    continue
                self.foods.append(candidate)
                logger.debug(
                    f"Spawned food at {candidate} "
                    f"({len(self.foods)}/{self.foraging.foods_on_grid} on grid)",
                )
                return True

        logger.warning(f"Failed to spawn food after {MAX_POISSON_ATTEMPTS} attempts")
        return False

    def _compute_food_gradient_vector(
        self,
        position: tuple[int, ...],
    ) -> tuple[float, float]:
        """
        Compute the local food gradient vector (superposition of all food sources).

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.

        Returns
        -------
        tuple[float, float]
            Food gradient vector (x, y) components.
        """
        vector_x = 0.0
        vector_y = 0.0

        for food in self.foods:
            dx = food[0] - position[0]
            dy = food[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance == 0:
                continue

            # Exponential decay gradient (positive/attractive)
            strength = self.foraging.gradient_strength * np.exp(
                -distance / self.foraging.gradient_decay_constant,
            )

            # Compute direction vector
            direction = np.arctan2(dy, dx)
            vector_x += strength * np.cos(direction)
            vector_y += strength * np.sin(direction)

        return vector_x, vector_y

    def _compute_predator_gradient_vector(
        self,
        position: tuple[int, ...],
    ) -> tuple[float, float]:
        """
        Compute the local predator gradient vector (superposition of all predators).

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.

        Returns
        -------
        tuple[float, float]
            Predator gradient vector (x, y) components.
            Note: Values are negative (repulsive gradient).
        """
        vector_x = 0.0
        vector_y = 0.0

        if not self.predator.enabled:
            return vector_x, vector_y

        for pred in self.predators:
            dx = pred.position[0] - position[0]
            dy = pred.position[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance == 0:
                continue

            # Exponential decay gradient (negative/repulsive)
            strength = -self.predator.gradient_strength * np.exp(
                -distance / self.predator.gradient_decay_constant,
            )

            # Compute direction vector (pointing away from predator due to negative strength)
            direction = np.arctan2(dy, dx)
            vector_x += strength * np.cos(direction)
            vector_y += strength * np.sin(direction)

        return vector_x, vector_y

    def get_state(
        self,
        position: tuple[int, ...],
        *,
        disable_log: bool = False,
    ) -> tuple[float, float]:
        """
        Get the current state with gradient superposition from food and predators.

        Uses exponential decay for gradient calculations:
        strength = base_strength * exp(-distance / decay_constant)

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query gradient at.
        disable_log : bool
            Whether to disable debug logging.

        Returns
        -------
        tuple[float, float]
            Total gradient strength and direction (food attraction + predator repulsion).
        """
        # Compute gradient vectors using helper methods
        food_x, food_y = self._compute_food_gradient_vector(position)
        pred_x, pred_y = self._compute_predator_gradient_vector(position)

        # Superpose food (attractive) and predator (repulsive) gradients
        total_vector_x = food_x + pred_x
        total_vector_y = food_y + pred_y

        # Compute magnitude and direction of superposed gradient
        gradient_magnitude = np.sqrt(total_vector_x**2 + total_vector_y**2)
        gradient_direction = np.arctan2(total_vector_y, total_vector_x)

        if not disable_log:
            logger.debug(
                f"Gradient superposition: magnitude={gradient_magnitude}, "
                f"direction={gradient_direction}, num_foods={len(self.foods)}, "
                f"num_predators={len(self.predators) if self.predator.enabled else 0}",
            )

        return float(gradient_magnitude), float(gradient_direction)

    def get_separated_gradients(
        self,
        position: tuple[int, ...],
        *,
        disable_log: bool = False,
    ) -> dict[str, float]:
        """
        Get separated local gradient vectors for appetitive/aversive modules.

        Decomposes the superimposed gradient at the agent's current position into:
        - Appetitive component: local chemical gradient from food sources (attractive)
        - Aversive component: local chemical gradient from predators (repulsive)

        This provides egocentric sensory information that a nematode could actually sense
        through its chemoreceptors, without requiring global knowledge of object positions.

        Parameters
        ----------
        position : tuple[int, ...]
            Position to query local gradients at.
        disable_log : bool
            Whether to disable debug logging.

        Returns
        -------
        dict[str, float]
            Dictionary containing LOCAL sensory information:
            - food_gradient_strength: Magnitude of local food gradient vector
            - food_gradient_direction: Direction of local food gradient vector (radians)
            - predator_gradient_strength: Magnitude of local predator gradient vector
            - predator_gradient_direction: Direction of local predator gradient vector (radians)
        """
        # Compute gradient vectors using helper methods
        food_vector_x, food_vector_y = self._compute_food_gradient_vector(position)
        predator_vector_x, predator_vector_y = self._compute_predator_gradient_vector(position)

        # Convert vectors to magnitude + direction (what sensors detect)
        food_magnitude_raw = np.sqrt(food_vector_x**2 + food_vector_y**2)
        food_direction = np.arctan2(food_vector_y, food_vector_x) if food_magnitude_raw > 0 else 0.0

        predator_magnitude_raw = np.sqrt(predator_vector_x**2 + predator_vector_y**2)
        # Note: predator_vector is negative (repulsive), so we negate to get direction
        # TOWARD predators. This gives consistent semantics with food_direction:
        # - food_direction: points toward food (move this way to eat)
        # - predator_direction: points toward predator (move AWAY from this direction)
        predator_direction = (
            np.arctan2(-predator_vector_y, -predator_vector_x)
            if predator_magnitude_raw > 0
            else 0.0
        )

        # Normalize magnitudes to [0, 1] using tanh, matching combined gradient scaling.
        # This ensures food_chemotaxis and nociception modules receive properly scaled
        # inputs that work with the same feature extraction logic as chemotaxis.
        food_magnitude = float(np.tanh(food_magnitude_raw * GRADIENT_SCALING_TANH_FACTOR))
        predator_magnitude = float(np.tanh(predator_magnitude_raw * GRADIENT_SCALING_TANH_FACTOR))

        # TODO: Convert to dataclass
        result = {
            "food_gradient_strength": food_magnitude,
            "food_gradient_direction": float(food_direction),
            "predator_gradient_strength": predator_magnitude,
            "predator_gradient_direction": float(predator_direction),
        }

        if not disable_log:
            logger.debug(
                f"Local gradients: food_mag={food_magnitude:.3f}, "
                f"predator_mag={predator_magnitude:.3f}",
            )

        return result

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
        Consume food at the agent's current position and respawn immediately.

        Automatically respawns a new food to maintain constant foods_on_grid count.

        Returns
        -------
        tuple[int, int] | None
            Position of consumed food, or None if no food at current position.
        """
        agent_tuple = (self.agent_pos[0], self.agent_pos[1])
        if agent_tuple in self.foods:
            self.foods.remove(agent_tuple)
            logger.info(f"Food consumed at {agent_tuple}")

            # Immediately spawn new food to maintain constant supply
            self.spawn_food()

            return agent_tuple

        return None

    def has_collected_target_foods(self, foods_collected: int) -> bool:
        """
        Check if agent has collected enough foods to win.

        Parameters
        ----------
        foods_collected : int
            Number of foods collected so far in this episode.

        Returns
        -------
        bool
            True if agent has collected target_foods_to_collect, False otherwise.
        """
        return foods_collected >= self.foraging.target_foods_to_collect

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

    def update_predators(self) -> None:
        """Update all predator positions."""
        if not self.predator.enabled:
            return
        agent_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        for pred in self.predators:
            pred.update_position(self.grid_size, self.rng, agent_pos)

    def check_predator_collision(self) -> bool:
        """
        Check if agent collided with any predator.

        Returns
        -------
        bool
            True if collision detected (within kill_radius), False otherwise.
        """
        if not self.predator.enabled:
            return False

        agent_pos = self.agent_pos
        for pred in self.predators:
            # Manhattan distance for kill radius
            distance = abs(agent_pos[0] - pred.position[0]) + abs(
                agent_pos[1] - pred.position[1],
            )
            if distance <= self.predator.kill_radius:
                logger.debug(
                    f"Predator collision! Agent at {agent_pos}, Predator at {pred.position}",
                )
                return True
        return False

    def is_agent_in_danger(self) -> bool:
        """
        Check if agent is within detection radius of any predator.

        Returns
        -------
        bool
            True if within detection radius, False otherwise.
        """
        if not self.predator.enabled:
            return False

        agent_pos = self.agent_pos
        for pred in self.predators:
            # Manhattan distance for detection radius
            distance = abs(agent_pos[0] - pred.position[0]) + abs(
                agent_pos[1] - pred.position[1],
            )
            if distance <= self.predator.detection_radius:
                return True
        return False

    def is_agent_in_damage_radius(self) -> bool:
        """
        Check if agent is within damage radius of any predator.

        Uses per-predator damage_radius which may vary by predator type.
        Stationary predators typically have larger damage_radius (toxic zones).

        Returns
        -------
        bool
            True if within damage radius of any predator, False otherwise.
        """
        if not self.predator.enabled:
            return False

        agent_pos = self.agent_pos
        for pred in self.predators:
            # Manhattan distance for damage radius (per-predator)
            distance = abs(agent_pos[0] - pred.position[0]) + abs(
                agent_pos[1] - pred.position[1],
            )
            if distance <= pred.damage_radius:
                logger.debug(
                    f"Agent in damage radius of {pred.predator_type.value} predator "
                    f"at {pred.position} (distance: {distance}, radius: {pred.damage_radius})",
                )
                return True
        return False

    # --- Mechanosensation methods ---

    def move_agent(self, action: Action) -> None:
        """
        Move the agent based on its perspective.

        Overrides base class to track wall collisions for boundary penalty.
        Distinguishes wall collisions from body collisions by checking
        if the intended move would exceed grid bounds.

        Parameters
        ----------
        action : Action
            The action to take.
        """
        # Reset wall collision flag at start of each move
        self.wall_collision_occurred = False

        # Check if this action would result in a wall collision BEFORE calling parent
        # This distinguishes wall collisions from body collisions
        if action != Action.STAY:
            self.wall_collision_occurred = self._would_hit_wall(action)

        # Call parent move_agent
        super().move_agent(action)

    def _would_hit_wall(self, action: Action) -> bool:
        """
        Check if the given action would cause a wall collision.

        Parameters
        ----------
        action : Action
            The action to check.

        Returns
        -------
        bool
            True if the action would cause a wall collision, False otherwise.
        """
        if action == Action.STAY:
            return False

        intended_direction = self.DIRECTION_MAP[self.current_direction][action]
        return self._get_new_position_if_valid(intended_direction) is None

    def is_agent_at_boundary(self) -> bool:
        """
        Check if agent is touching grid boundary.

        Mechanosensation: Detects physical contact with environment edges,
        modeled after C. elegans gentle touch neurons (ALM, PLM, AVM).

        Returns
        -------
        bool
            True if agent is at x=0, x=grid_size-1, y=0, or y=grid_size-1.
        """
        x, y = self.agent_pos
        return x == 0 or x == self.grid_size - 1 or y == 0 or y == self.grid_size - 1

    def is_agent_in_predator_contact(self) -> bool:
        """
        Check if agent is in physical contact with a predator.

        Mechanosensation: Detects harsh touch from predator contact,
        modeled after C. elegans harsh touch response (ASH, ADL neurons).

        This uses kill_radius for non-health environments and damage_radius
        for health-enabled environments to determine physical contact.

        Returns
        -------
        bool
            True if agent is within contact radius of any predator.
        """
        if not self.predator.enabled:
            return False

        # Use damage_radius if health system enabled, otherwise kill_radius
        if self.health.enabled:
            return self.is_agent_in_damage_radius()
        return self.check_predator_collision()

    # --- Health methods ---

    def apply_predator_damage(self) -> float:
        """
        Apply damage from predator contact.

        Returns
        -------
        float
            Actual amount of damage applied (0 if health system disabled).
            May be less than configured damage if HP was already low.
        """
        if not self.health.enabled:
            return 0.0

        old_hp = self.agent_hp
        self.agent_hp = max(0.0, self.agent_hp - self.health.predator_damage)
        actual_damage = old_hp - self.agent_hp
        logger.debug(
            f"Predator damage applied: {actual_damage} HP. "
            f"Current HP: {self.agent_hp}/{self.health.max_hp}",
        )
        return actual_damage

    def apply_food_healing(self) -> float:
        """
        Apply healing from food consumption.

        Returns
        -------
        float
            Amount of HP restored (0 if health system disabled).
        """
        if not self.health.enabled:
            return 0.0

        old_hp = self.agent_hp
        self.agent_hp = min(self.health.max_hp, self.agent_hp + self.health.food_healing)
        actual_healing = self.agent_hp - old_hp
        logger.debug(
            f"Food healing applied: {actual_healing} HP. "
            f"Current HP: {self.agent_hp}/{self.health.max_hp}",
        )
        return actual_healing

    def is_health_depleted(self) -> bool:
        """
        Check if agent's HP has reached zero.

        Returns
        -------
        bool
            True if HP <= 0 and health system is enabled, False otherwise.
        """
        return self.health.enabled and self.agent_hp <= 0.0

    def reset_health(self) -> None:
        """Reset agent HP to maximum (called at episode start)."""
        if self.health.enabled:
            self.agent_hp = self.health.max_hp

    # -------------------------------------------------------------------------
    # Thermotaxis Methods
    # -------------------------------------------------------------------------

    def get_temperature(self, position: GridPosition | None = None) -> float | None:
        """
        Get temperature at a position (or agent position if not specified).

        Parameters
        ----------
        position : GridPosition | None
            Position to query. Uses agent position if None.

        Returns
        -------
        float | None
            Temperature in °C, or None if thermotaxis is disabled.
        """
        if not self.thermotaxis.enabled or self.temperature_field is None:
            return None
        pos: GridPosition = position or (self.agent_pos[0], self.agent_pos[1])
        return self.temperature_field.get_temperature(pos)

    def get_temperature_gradient(
        self,
        position: GridPosition | None = None,
    ) -> GradientPolar | None:
        """
        Get temperature gradient (magnitude, direction) at a position.

        Parameters
        ----------
        position : GridPosition | None
            Position to query. Uses agent position if None.

        Returns
        -------
        GradientPolar | None
            (magnitude, direction) or None if thermotaxis is disabled.
            Direction points toward increasing temperature.
        """
        if not self.thermotaxis.enabled or self.temperature_field is None:
            return None
        pos: GridPosition = position or (self.agent_pos[0], self.agent_pos[1])
        return self.temperature_field.get_gradient_polar(pos)

    def get_temperature_zone(
        self,
        position: GridPosition | None = None,
    ) -> TemperatureZone | None:
        """
        Get the temperature zone at a position.

        Parameters
        ----------
        position : GridPosition | None
            Position to query. Uses agent position if None.

        Returns
        -------
        TemperatureZone | None
            The temperature zone, or None if thermotaxis is disabled.
        """
        if not self.thermotaxis.enabled or self.temperature_field is None:
            return None

        temp = self.get_temperature(position)
        if temp is None:
            return None

        thresholds = TemperatureZoneThresholds(
            comfort_delta=self.thermotaxis.comfort_delta,
            discomfort_delta=self.thermotaxis.discomfort_delta,
            danger_delta=self.thermotaxis.danger_delta,
        )
        return self.temperature_field.get_zone(
            temp,
            self.thermotaxis.cultivation_temperature,
            thresholds,
        )

    def _is_safe_temperature_zone(self, position: tuple[int, int]) -> bool:
        """
        Check if a position is in a safe temperature zone (COMFORT or DISCOMFORT).

        Safe zones are defined as zones where the agent does not take HP damage.
        This includes COMFORT, DISCOMFORT_COLD, and DISCOMFORT_HOT zones.

        Parameters
        ----------
        position : tuple[int, int]
            Position to check.

        Returns
        -------
        bool
            True if position is in a safe zone, False if in danger/lethal zone.
            Returns True if thermotaxis is disabled (all positions are "safe").
        """
        if not self.thermotaxis.enabled or self.temperature_field is None:
            return True

        zone = self.get_temperature_zone(position)
        if zone is None:
            return True

        return zone in (
            TemperatureZone.COMFORT,
            TemperatureZone.DISCOMFORT_COLD,
            TemperatureZone.DISCOMFORT_HOT,
        )

    def apply_temperature_effects(self) -> tuple[float, float]:
        """
        Apply temperature zone effects (rewards/penalties and HP damage).

        Updates the comfort zone tracking counters and applies appropriate
        rewards, penalties, and HP damage based on the current temperature zone.

        Returns
        -------
        tuple[float, float]
            (reward_delta, hp_damage) applied this step.
            reward_delta is positive for comfort, negative for discomfort/danger.
            hp_damage is always non-negative (0 if no damage).
        """
        if not self.thermotaxis.enabled:
            return 0.0, 0.0

        zone = self.get_temperature_zone()
        if zone is None:
            return 0.0, 0.0

        # Track comfort zone time
        self.total_thermotaxis_steps += 1
        if zone == TemperatureZone.COMFORT:
            self.steps_in_comfort_zone += 1

        reward_delta = 0.0
        hp_damage = 0.0

        if zone == TemperatureZone.COMFORT:
            reward_delta = self.thermotaxis.comfort_reward
        elif zone in (TemperatureZone.DISCOMFORT_COLD, TemperatureZone.DISCOMFORT_HOT):
            reward_delta = self.thermotaxis.discomfort_penalty
        elif zone in (TemperatureZone.DANGER_COLD, TemperatureZone.DANGER_HOT):
            reward_delta = self.thermotaxis.danger_penalty
            hp_damage = self.thermotaxis.danger_hp_damage
        elif zone in (TemperatureZone.LETHAL_COLD, TemperatureZone.LETHAL_HOT):
            reward_delta = self.thermotaxis.danger_penalty  # Use danger penalty for lethal too
            hp_damage = self.thermotaxis.lethal_hp_damage

        # Apply HP damage if health system is enabled
        if hp_damage > 0 and self.health.enabled:
            self.agent_hp = max(0.0, self.agent_hp - hp_damage)
            logger.debug(
                f"Temperature damage: zone={zone.value}, damage={hp_damage}, hp={self.agent_hp}",
            )

        return reward_delta, hp_damage

    def get_temperature_comfort_score(self) -> float:
        """
        Get the fraction of time spent in the comfort zone.

        Returns
        -------
        float
            Ratio of steps in comfort zone to total steps (0.0 to 1.0).
            Returns 0.0 if no thermotaxis steps have occurred.
        """
        if self.total_thermotaxis_steps == 0:
            return 0.0
        return self.steps_in_comfort_zone / self.total_thermotaxis_steps

    def reset_thermotaxis(self) -> None:
        """Reset thermotaxis tracking counters (called at episode start)."""
        self.steps_in_comfort_zone = 0
        self.total_thermotaxis_steps = 0

    def get_viewport_bounds(self) -> tuple[int, int, int, int]:
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
        viewport = self.get_viewport_bounds()
        grid = self._render_grid(self.foods, viewport=viewport)

        # Add predators to grid if enabled
        if self.predator.enabled:
            self._render_predators(grid, viewport)

        return self._render_grid_to_strings(grid, viewport=viewport)

    def render_full(self, *, theme_override: Theme | None = None) -> list[str]:
        """Render the entire environment (for logging/debugging).

        Parameters
        ----------
        theme_override : Theme | None
            If provided, temporarily use this theme for rendering.
            Useful when the active theme (e.g. PIXEL) has no meaningful
            text representation.
        """
        original_theme = self.theme
        if theme_override is not None:
            self.theme = theme_override
        try:
            grid = self._render_grid(self.foods)

            # Add predators to grid if enabled
            if self.predator.enabled:
                self._render_predators(grid, viewport=None)

            return self._render_grid_to_strings(grid, viewport=None)
        finally:
            self.theme = original_theme

    def _get_predator_symbol(
        self,
        predator: Predator,
        symbols: "ThemeSymbolSet",
    ) -> str:
        """
        Get the appropriate symbol for a predator based on its type.

        Parameters
        ----------
        predator : Predator
            The predator to get the symbol for.
        symbols : ThemeSymbolSet
            The current theme's symbol set.

        Returns
        -------
        str
            The symbol to use for rendering this predator.
        """
        if predator.predator_type == PredatorType.STATIONARY:
            return symbols.predator_stationary
        if predator.predator_type == PredatorType.PURSUIT:
            return symbols.predator_pursuit
        return symbols.predator

    def _render_predators(
        self,
        grid: list[list[str]],
        viewport: Viewport | None = None,
    ) -> None:
        """
        Add predators to the rendered grid.

        Parameters
        ----------
        grid : list[list[str]]
            The grid to render predators onto.
        viewport : tuple[int, int, int, int] | None
            Viewport bounds (min_x, min_y, max_x, max_y) or None for full grid.
        """
        symbols = THEME_SYMBOLS[self.theme]

        for predator in self.predators:
            predator_symbol = self._get_predator_symbol(predator, symbols)
            if viewport:
                min_x, min_y, max_x, max_y = viewport
                # Only render predator if in viewport
                if min_x <= predator.position[0] < max_x and min_y <= predator.position[1] < max_y:
                    grid_y = predator.position[1] - min_y
                    grid_x = predator.position[0] - min_x
                    # Don't overwrite agent position
                    agent_y = self.agent_pos[1] - min_y
                    agent_x = self.agent_pos[0] - min_x
                    if grid_y != agent_y or grid_x != agent_x:
                        grid[grid_y][grid_x] = predator_symbol
            # Full grid rendering
            # Don't overwrite agent position
            elif predator.position != tuple(self.agent_pos):
                grid[predator.position[1]][predator.position[0]] = predator_symbol

    def _is_in_toxic_zone(self, world_x: int, world_y: int) -> bool:
        """Check if a world position is within a toxic zone (stationary predator damage radius).

        Parameters
        ----------
        world_x : int
            X coordinate in world space.
        world_y : int
            Y coordinate in world space.

        Returns
        -------
        bool
            True if position is within damage radius of any stationary predator.
        """
        if not self.predator.enabled:
            return False

        for pred in self.predators:
            if pred.predator_type != PredatorType.STATIONARY:
                continue
            if pred.damage_radius <= 0:
                continue
            # Manhattan distance check
            distance = abs(pred.position[0] - world_x) + abs(pred.position[1] - world_y)
            if distance <= pred.damage_radius:
                return True
        return False

    def _get_zone_background_style(self, world_x: int, world_y: int) -> str:
        """Get the background style for a cell based on zone type.

        Priority order (highest to lowest):
        1. Toxic zone (stationary predator damage radius) - purple
        2. Temperature zone (if thermotaxis enabled) - cold/hot gradient
        3. Default (no background override)

        Parameters
        ----------
        world_x : int
            X coordinate in world space.
        world_y : int
            Y coordinate in world space.

        Returns
        -------
        str
            Rich style string for background (e.g., "on blue") or empty string.
        """
        # Priority 1: Toxic zones
        if self._is_in_toxic_zone(world_x, world_y):
            return self.rich_style_config.zone_toxic_bg

        # Priority 2: Temperature zones
        if self.thermotaxis.enabled and self.temperature_field is not None:
            temp = self.temperature_field.get_temperature((world_x, world_y))
            thresholds = TemperatureZoneThresholds(
                comfort_delta=self.thermotaxis.comfort_delta,
                discomfort_delta=self.thermotaxis.discomfort_delta,
                danger_delta=self.thermotaxis.danger_delta,
            )
            zone = self.temperature_field.get_zone(
                temp,
                cultivation_temperature=self.thermotaxis.cultivation_temperature,
                thresholds=thresholds,
            )

            zone_bg_map = {
                TemperatureZone.LETHAL_COLD: self.rich_style_config.zone_lethal_cold_bg,
                TemperatureZone.DANGER_COLD: self.rich_style_config.zone_danger_cold_bg,
                TemperatureZone.DISCOMFORT_COLD: self.rich_style_config.zone_discomfort_cold_bg,
                TemperatureZone.COMFORT: self.rich_style_config.zone_comfort_bg,
                TemperatureZone.DISCOMFORT_HOT: self.rich_style_config.zone_discomfort_hot_bg,
                TemperatureZone.DANGER_HOT: self.rich_style_config.zone_danger_hot_bg,
                TemperatureZone.LETHAL_HOT: self.rich_style_config.zone_lethal_hot_bg,
            }
            return zone_bg_map.get(zone, "")

        return ""

    def _get_zone_symbol(self, world_x: int, world_y: int) -> str:
        """Get the zone symbol for a cell based on zone type.

        Used by EMOJI and COLORED_ASCII themes to replace empty cells with
        zone-specific symbols or apply zone background styling.

        Priority order (highest to lowest):
        1. Toxic zone (stationary predator damage radius)
        2. Temperature zone (if thermotaxis enabled)
        3. Default (empty string, use normal empty symbol)

        Parameters
        ----------
        world_x : int
            X coordinate in world space.
        world_y : int
            Y coordinate in world space.

        Returns
        -------
        str
            Zone symbol/background style or empty string for default.
        """
        symbols = THEME_SYMBOLS[self.theme]

        # Priority 1: Toxic zones
        if self._is_in_toxic_zone(world_x, world_y):
            return symbols.zone_toxic

        # Priority 2: Temperature zones
        if self.thermotaxis.enabled and self.temperature_field is not None:
            temp = self.temperature_field.get_temperature((world_x, world_y))
            thresholds = TemperatureZoneThresholds(
                comfort_delta=self.thermotaxis.comfort_delta,
                discomfort_delta=self.thermotaxis.discomfort_delta,
                danger_delta=self.thermotaxis.danger_delta,
            )
            zone = self.temperature_field.get_zone(
                temp,
                cultivation_temperature=self.thermotaxis.cultivation_temperature,
                thresholds=thresholds,
            )

            zone_symbol_map = {
                TemperatureZone.LETHAL_COLD: symbols.zone_lethal_cold,
                TemperatureZone.DANGER_COLD: symbols.zone_danger_cold,
                TemperatureZone.DISCOMFORT_COLD: symbols.zone_discomfort_cold,
                TemperatureZone.COMFORT: symbols.zone_comfort,
                TemperatureZone.DISCOMFORT_HOT: symbols.zone_discomfort_hot,
                TemperatureZone.DANGER_HOT: symbols.zone_danger_hot,
                TemperatureZone.LETHAL_HOT: symbols.zone_lethal_hot,
            }
            return zone_symbol_map.get(zone, "")

        return ""

    def _render_grid_to_strings(  # noqa: C901, PLR0912
        self,
        grid: list[list[str]],
        viewport: Viewport | None = None,
    ) -> list[str]:
        """Convert grid to string representation with zone visualization.

        Overrides base class to add zone backgrounds for EMOJI and COLORED_ASCII themes.

        Parameters
        ----------
        grid : list[list[str]]
            The grid of symbols to render.
        viewport : tuple[int, int, int, int] | None
            Viewport bounds (min_x, min_y, max_x, max_y) for coordinate mapping.
        """
        if self.theme in (Theme.RICH, Theme.EMOJI_RICH):
            return self._render_rich(grid, theme=self.theme, viewport=viewport)

        symbols = THEME_SYMBOLS[self.theme]
        grid_height = len(grid)

        # Determine coordinate offset from viewport
        offset_x = viewport[0] if viewport else 0
        offset_y = viewport[1] if viewport else 0

        if self.theme == Theme.EMOJI:
            rendered_rows = []
            for grid_row_idx, row in enumerate(reversed(grid)):
                world_y = offset_y + (grid_height - 1 - grid_row_idx)
                rendered_row = []
                for grid_col_idx, cell in enumerate(row):
                    world_x = offset_x + grid_col_idx
                    # For empty cells, use zone symbol if available
                    if cell == symbols.empty:
                        zone_symbol = self._get_zone_symbol(world_x, world_y)
                        if zone_symbol:
                            rendered_row.append(zone_symbol)
                        else:
                            rendered_row.append(cell)
                    else:
                        rendered_row.append(cell)
                rendered_rows.append("".join(rendered_row))
            return [*rendered_rows, ""]

        if self.theme == Theme.COLORED_ASCII:
            rendered_rows = []
            for grid_row_idx, row in enumerate(reversed(grid)):
                world_y = offset_y + (grid_height - 1 - grid_row_idx)
                rendered_row = []
                for grid_col_idx, cell in enumerate(row):
                    world_x = offset_x + grid_col_idx
                    # For empty cells, use zone-colored dot if available
                    if cell == symbols.empty:
                        zone_symbol = self._get_zone_symbol(world_x, world_y)
                        if zone_symbol:
                            rendered_row.append(zone_symbol)
                        else:
                            rendered_row.append(cell)
                    else:
                        rendered_row.append(cell)
                rendered_rows.append(" ".join(rendered_row))
            return [*rendered_rows, ""]

        # Default handling for other themes (ASCII, UNICODE)
        if self.theme == Theme.UNICODE:
            return [" ".join(row) for row in reversed(grid)] + [""]

        return [" ".join(row) for row in reversed(grid)] + [""]

    def _render_rich(
        self,
        grid: list[list[str]],
        theme: Theme,
        viewport: Viewport | None = None,
    ) -> list[str]:
        """Render the grid with Rich styling, zone backgrounds, and colors.

        Overrides base class to add temperature zone and toxic zone backgrounds.

        Parameters
        ----------
        grid : list[list[str]]
            2D grid of symbols to render.
        theme : Theme
            The theme to use for rendering.
        viewport : tuple[int, int, int, int] | None
            Viewport bounds (min_x, min_y, max_x, max_y) for coordinate mapping.
            If None, assumes full grid starting at (0, 0).

        Returns
        -------
        list[str]
            List of strings representing the rendered Rich table.
        """
        width_extra_pad = 1 if theme == Theme.EMOJI_RICH else 0
        grid_width = len(grid[0])
        grid_height = len(grid)
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

        # Determine coordinate offset from viewport
        offset_x = viewport[0] if viewport else 0
        offset_y = viewport[1] if viewport else 0

        # Grid is rendered reversed (top row is highest y)
        for grid_row_idx, row in enumerate(reversed(grid)):
            styled_cells = []
            # Convert grid row index to world y coordinate
            # reversed grid means grid_row_idx 0 = highest y in grid
            world_y = offset_y + (grid_height - 1 - grid_row_idx)

            for grid_col_idx, cell in enumerate(row):
                world_x = offset_x + grid_col_idx

                # Determine foreground style based on entity
                if cell == symbols.goal:
                    fg_style = self.rich_style_config.goal_style
                elif cell == symbols.body:
                    fg_style = self.rich_style_config.body_style
                elif cell in [symbols.up, symbols.down, symbols.left, symbols.right]:
                    fg_style = self.rich_style_config.agent_style
                elif cell == symbols.predator:
                    fg_style = self.rich_style_config.predator_style
                elif cell == symbols.predator_stationary:
                    fg_style = self.rich_style_config.predator_stationary_style
                elif cell == symbols.predator_pursuit:
                    fg_style = self.rich_style_config.predator_pursuit_style
                else:
                    fg_style = self.rich_style_config.empty_style

                # Get background style based on zone
                bg_style = self._get_zone_background_style(world_x, world_y)

                # Combine foreground and background styles
                combined_style = f"{fg_style} {bg_style}" if bg_style else fg_style

                styled_cell = RichText(cell, style=combined_style, justify="center")
                styled_cells.append(styled_cell)

            table.add_row(*styled_cells)

        with console.capture() as capture:
            console.print(table, crop=True)

        output_lines = capture.get().splitlines()
        cleaned_lines = [line.rstrip() for line in output_lines if line.strip()]

        return [*cleaned_lines, ""]

    def copy(self) -> "DynamicForagingEnvironment":
        """
        Create a deep copy of the DynamicForagingEnvironment.

        Returns
        -------
        DynamicForagingEnvironment
            A new instance with the same state.

        Notes
        -----
        Config objects (ForagingParams, PredatorParams, HealthParams, ThermotaxisParams)
        are shared between the original and copy, not deep-copied. This is intentional
        as these are treated as immutable configuration. Runtime state (foods,
        visited_cells, agent_hp, predator positions, thermotaxis counters) is properly
        copied.
        """
        new_env = DynamicForagingEnvironment(
            grid_size=self.grid_size,
            start_pos=(self.agent_pos[0], self.agent_pos[1]),
            viewport_size=self.viewport_size,
            max_body_length=len(self.body),
            theme=self.theme,
            action_set=self.action_set,
            rich_style_config=self.rich_style_config,
            seed=self.seed,
            foraging=self.foraging,
            predator=self.predator,
            health=self.health,
            thermotaxis=self.thermotaxis,
        )
        new_env.body = self.body.copy()
        new_env.current_direction = self.current_direction
        new_env.foods = self.foods.copy()
        new_env.visited_cells = self.visited_cells.copy()
        # Copy RNG state for reproducibility
        new_env.rng = get_rng(self.seed)
        # Copy health state
        new_env.agent_hp = self.agent_hp
        # Copy thermotaxis tracking state
        new_env.steps_in_comfort_zone = self.steps_in_comfort_zone
        new_env.total_thermotaxis_steps = self.total_thermotaxis_steps
        if self.predator.enabled:
            new_env.predators = [
                Predator(
                    position=p.position,
                    predator_type=p.predator_type,
                    speed=p.speed,
                    movement_accumulator=p.movement_accumulator,
                    detection_radius=p.detection_radius,
                    damage_radius=p.damage_radius,
                )
                for p in self.predators
            ]
        return new_env


type EnvironmentType = DynamicForagingEnvironment
