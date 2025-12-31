"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

from quantumnematode.agent.tracker import EpisodeTracker
from quantumnematode.brain.actions import ActionData  # noqa: TC001 - needed at runtime
from quantumnematode.brain.arch import Brain, BrainParams, QuantumBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.dtypes import FoodHistory, GridPosition  # noqa: TC001 - used at runtime
from quantumnematode.env import (
    BaseEnvironment,
    DynamicForagingEnvironment,
    EnvironmentType,
    StaticEnvironment,
)
from quantumnematode.env.theme import DEFAULT_THEME, DarkColorRichStyleConfig, Theme
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import PerformanceMetrics

if TYPE_CHECKING:
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.agent.runners import EpisodeResult

# Defaults
DEFAULT_AGENT_BODY_LENGTH = 2
DEFAULT_MAX_AGENT_BODY_LENGTH = 6
DEFAULT_MAX_STEPS = 100
DEFAULT_MAZE_GRID_SIZE = 5
DEFAULT_PENALTY_ANTI_DITHERING = 0.02
DEFAULT_PENALTY_STEP = 0.05
DEFAULT_PENALTY_STUCK_POSITION = 0.5
DEFAULT_PENALTY_STARVATION = 10.0
DEFAULT_PENALTY_PREDATOR_DEATH = 10.0
DEFAULT_PENALTY_PREDATOR_PROXIMITY = 0.1
DEFAULT_PENALTY_HEALTH_DAMAGE = 0.5  # Penalty when taking damage (per hit)
DEFAULT_REWARD_HEALTH_GAIN = 0.1  # Reward when healing (per healing event)
DEFAULT_REWARD_DISTANCE_SCALE = 0.3
DEFAULT_REWARD_GOAL = 0.2
DEFAULT_REWARD_EXPLORATION = 0.05
DEFAULT_MANYWORLDS_MODE_MAX_COLUMNS = 4
DEFAULT_MANYWORLDS_MODE_MAX_SUPERPOSITIONS = 16
DEFAULT_MANYWORLDS_MODE_RENDER_SLEEP_SECONDS = 1.0
DEFAULT_MANYWORLDS_MODE_TOP_N_ACTIONS = 2
DEFAULT_MANYWORLDS_MODE_TOP_N_RANDOMIZE = True
DEFAULT_STUCK_POSITION_THRESHOLD = 2
DEFAULT_SATIETY_INITIAL = 200.0
DEFAULT_SATIETY_DECAY_RATE = 1.0
DEFAULT_SATIETY_GAIN_PER_FOOD = 0.2


class SatietyConfig(BaseModel):
    """Configuration for the satiety (hunger) system."""

    initial_satiety: float = DEFAULT_SATIETY_INITIAL
    satiety_decay_rate: float = DEFAULT_SATIETY_DECAY_RATE
    satiety_gain_per_food: float = DEFAULT_SATIETY_GAIN_PER_FOOD  # Fraction of max


class RewardConfig(BaseModel):
    """Configuration for the reward function."""

    penalty_anti_dithering: float = (
        DEFAULT_PENALTY_ANTI_DITHERING  # Penalty for oscillating (revisiting previous cell)
    )
    penalty_step: float = DEFAULT_PENALTY_STEP
    penalty_stuck_position: float = (
        DEFAULT_PENALTY_STUCK_POSITION  # Penalty for staying in same position, disabled if 0
    )
    stuck_position_threshold: int = (
        DEFAULT_STUCK_POSITION_THRESHOLD  # Steps before stuck penalty applies
    )
    reward_distance_scale: float = (
        DEFAULT_REWARD_DISTANCE_SCALE  # Scale the distance reward for smoother learning
    )
    reward_goal: float = DEFAULT_REWARD_GOAL
    reward_exploration: float = DEFAULT_REWARD_EXPLORATION  # Bonus for visiting new cells
    penalty_starvation: float = DEFAULT_PENALTY_STARVATION  # Penalty when satiety reaches 0
    penalty_predator_death: float = (
        DEFAULT_PENALTY_PREDATOR_DEATH  # Penalty when caught by predator
    )
    penalty_predator_proximity: float = (
        DEFAULT_PENALTY_PREDATOR_PROXIMITY  # Penalty per step within predator detection radius
    )
    # Health system rewards (only applied when health system is enabled)
    penalty_health_damage: float = (
        DEFAULT_PENALTY_HEALTH_DAMAGE  # Penalty when taking damage from predators
    )
    reward_health_gain: float = (
        DEFAULT_REWARD_HEALTH_GAIN  # Reward when healing from food consumption
    )


class ManyworldsModeConfig(BaseModel):
    """Configuration for the many-worlds mode."""

    max_superpositions: int = DEFAULT_MANYWORLDS_MODE_MAX_SUPERPOSITIONS
    max_columns: int = DEFAULT_MANYWORLDS_MODE_MAX_COLUMNS
    render_sleep_seconds: float = DEFAULT_MANYWORLDS_MODE_RENDER_SLEEP_SECONDS
    top_n_actions: int = DEFAULT_MANYWORLDS_MODE_TOP_N_ACTIONS
    top_n_randomize: bool = DEFAULT_MANYWORLDS_MODE_TOP_N_RANDOMIZE


class QuantumNematodeAgent:
    """
    Nematode agent that navigates a grid environment using a quantum brain.

    Attributes
    ----------
    env : EnvironmentType
        The grid environment for the agent.
    steps : int
        Number of steps taken by the agent.
    path : list[tuple]
        Path taken by the agent.
    body_length : int
        Maximum length of the agent's body.

    Notes
    -----
    Satiety is managed internally by the SatietyManager component.
    Access via `agent.current_satiety`.
    """

    def __init__(  # noqa: PLR0913
        self,
        brain: Brain,
        env: EnvironmentType | None = None,
        maze_grid_size: int = DEFAULT_MAZE_GRID_SIZE,
        max_body_length: int = DEFAULT_MAX_AGENT_BODY_LENGTH,
        theme: Theme = DEFAULT_THEME,
        rich_style_config: DarkColorRichStyleConfig | None = None,
        satiety_config: SatietyConfig | None = None,
        *,
        use_separated_gradients: bool = False,
    ) -> None:
        """
        Initialize the nematode agent.

        Parameters
        ----------
        brain : Brain
            The brain architecture used by the agent.
        env : EnvironmentType | None
            The environment to use. If None, creates a default StaticEnvironment.
        maze_grid_size : int, optional
            Size of the grid environment, by default 5 (only used if env is None).
        max_body_length : int, optional
            Maximum body length.
        theme : Theme, optional
            Visual theme for rendering.
        rich_style_config : DarkColorRichStyleConfig | None, optional
            Rich styling configuration.
        satiety_config : SatietyConfig | None, optional
            Satiety system configuration.
        use_separated_gradients : bool, optional
            Whether to use separated food/predator gradients for appetitive/aversive modules.
            Only valid for dynamic environments. Default is False (unified gradients).
        """
        self.brain = brain
        self.satiety_config = satiety_config or SatietyConfig()
        self.use_separated_gradients = use_separated_gradients

        if env is None:
            self.env = StaticEnvironment(
                grid_size=maze_grid_size,
                max_body_length=max_body_length,
                theme=theme,
                rich_style_config=rich_style_config,
            )
        else:
            self.env = env

        self.path: list[GridPosition] = [(self.env.agent_pos[0], self.env.agent_pos[1])]
        # Track food positions at each step for chemotaxis validation
        self.food_history: FoodHistory = []
        if isinstance(self.env, DynamicForagingEnvironment):
            self.food_history = [list(self.env.foods)]
        self.max_body_length = min(
            self.env.grid_size - 1,
            max_body_length,
        )

        # For dynamic environments, track initial distance for metrics
        self.initial_distance_to_food: int | None = None

        # Component instantiation
        # Import at runtime to avoid circular dependencies
        from quantumnematode.agent.food_handler import FoodConsumptionHandler
        from quantumnematode.agent.metrics import MetricsTracker
        from quantumnematode.agent.reward_calculator import RewardCalculator
        from quantumnematode.agent.runners import ManyworldsEpisodeRunner, StandardEpisodeRunner
        from quantumnematode.agent.satiety import SatietyManager

        self._episode_tracker = EpisodeTracker()
        self._satiety_manager = SatietyManager(self.satiety_config)
        self._metrics_tracker = MetricsTracker()
        self._reward_calculator = RewardCalculator(RewardConfig())  # Default config
        self._food_handler = FoodConsumptionHandler(
            env=self.env,
            satiety_manager=self._satiety_manager,
            satiety_gain_fraction=self.satiety_config.satiety_gain_per_food,
        )
        self._standard_runner = StandardEpisodeRunner()
        self._manyworlds_runner = ManyworldsEpisodeRunner()

    @property
    def current_satiety(self) -> float:
        """Get current satiety level from the satiety manager.

        Returns
        -------
        float
            Current satiety level.
        """
        return self._satiety_manager.current_satiety

    @property
    def max_satiety(self) -> float:
        """Get maximum satiety level from the satiety manager.

        Returns
        -------
        float
            Maximum satiety level.
        """
        return self._satiety_manager.max_satiety

    def run_episode(
        self,
        reward_config: RewardConfig,
        max_steps: int = DEFAULT_MAX_STEPS,
        render_text: str | None = None,
        *,
        show_last_frame_only: bool = False,
    ) -> EpisodeResult:
        """Run a single episode using StandardEpisodeRunner.

        Parameters
        ----------
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        render_text : str | None, optional
            Text to display during rendering.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation, by default False.

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        return self._standard_runner.run(
            agent=self,
            reward_config=reward_config,
            max_steps=max_steps,
            render_text=render_text,
            show_last_frame_only=show_last_frame_only,
        )

    def run_manyworlds_mode(
        self,
        config: ManyworldsModeConfig,
        reward_config: RewardConfig,
        max_steps: int = DEFAULT_MAX_STEPS,
        *,
        show_last_frame_only: bool = False,
    ) -> EpisodeResult:
        """Run the agent in many-worlds mode using ManyworldsEpisodeRunner.

        Runs the agent in "many-worlds mode", inspired by the many-worlds interpretation in
        quantum mechanics, where all possible outcomes of a decision are explored in parallel.
        In this mode, the agent simulates multiple parallel universes by branching at each step
        according to the top N actions, visualizing how different choices lead to divergent paths
        and outcomes.

        At each step, the agent considers the top N actions (as set in the configuration) and
        creates new superpositions (parallel environments) for each action, up to a maximum number
        of superpositions. This allows users to observe how the agent's trajectory diverges based
        on different decisions, providing insight into the agent's decision-making process and the
        landscape of possible futures.

        Parameters
        ----------
        config : ManyworldsModeConfig
            Configuration for many-worlds mode, including rendering and branching options.
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int, optional
            Maximum number of steps for the episode (default: DEFAULT_MAX_STEPS).
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation.

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        return self._manyworlds_runner.run(
            agent=self,
            reward_config=reward_config,
            max_steps=max_steps,
            config=config,
            show_last_frame_only=show_last_frame_only,
        )

    def _get_agent_position_tuple(self) -> tuple[float, float]:
        """Get agent position as a 2-element float tuple.

        Returns
        -------
        tuple[float, float]
            Agent position (x, y) as floats.
        """
        agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
        if len(agent_pos) != 2:  # noqa: PLR2004
            return (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))
        return agent_pos  # type: ignore[return-value]

    def _prepare_input_data(self, gradient_strength: float) -> list[float] | None:
        """Prepare input data for quantum brain data re-uploading.

        For quantum brains, returns a list of gradient_strength repeated for each qubit.
        For classical brains, returns None.

        Parameters
        ----------
        gradient_strength : float
            The gradient strength value to use for data re-uploading.

        Returns
        -------
        list[float] | None
            List of floats for quantum brains, None for classical brains.
        """
        if isinstance(self.brain, QuantumBrain):
            return [float(gradient_strength)] * self.brain.num_qubits
        return None

    def _create_brain_params(
        self,
        gradient_strength: float,
        gradient_direction: float,
        action: ActionData | None = None,
    ) -> BrainParams:
        """Create BrainParams for brain execution.

        Parameters
        ----------
        gradient_strength : float
            Strength of the combined gradient (food + predator).
        gradient_direction : float
            Direction of the combined gradient (angle in radians).
        action : ActionData | None, optional
            Previous action taken, by default None.

        Returns
        -------
        BrainParams
            Brain parameters ready for execution.
        """
        # Get separated gradients for appetitive/aversive modules if configured
        separated_grads = {}
        if self.use_separated_gradients:
            if isinstance(self.env, DynamicForagingEnvironment):
                separated_grads = self.env.get_separated_gradients(
                    self.env.agent_pos,
                    disable_log=True,
                )
            else:
                error_message = (
                    "Separated gradients requested but "
                    "environment is not DynamicForagingEnvironment."
                )
                logger.error(error_message)
                raise ValueError(error_message)

        return BrainParams(
            # Combined gradients
            gradient_strength=gradient_strength,
            gradient_direction=gradient_direction,
            # Separated LOCAL gradients (egocentric sensing)
            food_gradient_strength=separated_grads.get("food_gradient_strength"),
            food_gradient_direction=separated_grads.get("food_gradient_direction"),
            predator_gradient_strength=separated_grads.get("predator_gradient_strength"),
            predator_gradient_direction=separated_grads.get("predator_gradient_direction"),
            # Internal state (hunger)
            satiety=self.current_satiety,
            # Agent proprioception
            agent_position=self._get_agent_position_tuple(),
            agent_direction=self.env.current_direction,
            action=action,
        )

    def _render_step(
        self,
        max_steps: int,
        render_text: str | None = None,
        *,
        show_last_frame_only: bool = False,
    ) -> None:
        """Render the current step with environment state and status.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps for the episode.
        render_text : str | None, optional
            Additional text to display, by default None.
        show_last_frame_only : bool, optional
            Whether to clear screen before rendering, by default False.
        """
        # Clear screen if showing last frame only
        if show_last_frame_only:
            if os.name == "nt":  # For Windows
                os.system("cls")  # noqa: S605, S607
            else:  # For macOS and Linux
                os.system("clear")  # noqa: S605, S607

        # Render environment grid
        grid = self.env.render()
        for frame in grid:
            print(frame)  # noqa: T201
            logger.debug(frame)

        # Display custom render text
        if render_text:
            print(render_text)  # noqa: T201

        # Display environment-specific status
        print("Run:\n----")  # noqa: T201
        print(f"Step:\t\t{self._episode_tracker.steps}/{max_steps}")  # noqa: T201
        match self.env:
            case StaticEnvironment():
                pass
            case DynamicForagingEnvironment():
                print(  # noqa: T201
                    f"Eaten:\t\t{self._episode_tracker.foods_collected}/{self.env.target_foods_to_collect}",
                )
                print(f"Satiety:\t{self.current_satiety:.1f}/{self.max_satiety}")  # noqa: T201
                # Display danger status if predators are enabled
                if self.env.predators_enabled:
                    danger_status = "IN DANGER" if self.env.is_agent_in_danger() else "SAFE"
                    print(f"Status:\t\t{danger_status}")  # noqa: T201

    def calculate_reward(
        self,
        config: RewardConfig,
        env: BaseEnvironment,
        path: list[tuple[int, ...]],
        max_steps: int,
        stuck_position_count: int = 0,
    ) -> float:
        """
        Calculate reward based on the agent's movement toward the goal.

        Handles both StaticEnvironment (single goal) and DynamicForagingEnvironment (multiple foods)

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        # Delegate to RewardCalculator component
        self._reward_calculator.config = config
        return self._reward_calculator.calculate_reward(
            env=env,
            path=path,
            stuck_position_count=stuck_position_count,
            current_step=self._episode_tracker.steps,
            max_steps=max_steps,
        )

    def reset_environment(self) -> None:
        """
        Reset the environment while retaining the agent's learned data.

        Returns
        -------
        None
        """
        if isinstance(self.env, DynamicForagingEnvironment):
            self.env = DynamicForagingEnvironment(
                grid_size=self.env.grid_size,
                foods_on_grid=self.env.foods_on_grid,
                target_foods_to_collect=self.env.target_foods_to_collect,
                min_food_distance=self.env.min_food_distance,
                agent_exclusion_radius=self.env.agent_exclusion_radius,
                gradient_decay_constant=self.env.gradient_decay_constant,
                gradient_strength=self.env.gradient_strength_base,
                viewport_size=self.env.viewport_size,
                max_body_length=self.max_body_length,
                theme=self.env.theme,
                rich_style_config=self.env.rich_style_config,
                # Predator parameters (preserve from original env)
                predators_enabled=self.env.predators_enabled,
                num_predators=self.env.num_predators,
                predator_speed=self.env.predator_speed,
                predator_detection_radius=self.env.predator_detection_radius,
                predator_kill_radius=self.env.predator_kill_radius,
                predator_gradient_decay=self.env.predator_gradient_decay,
                predator_gradient_strength=self.env.predator_gradient_strength,
                # Health (preserve from original env)
                health_enabled=self.env.health_enabled,
                max_hp=self.env.max_hp,
                predator_damage=self.env.predator_damage,
                food_healing=self.env.food_healing,
                # Reproducibility: preserve seed from original environment
                seed=self.env.seed,
            )
        else:
            self.env = StaticEnvironment(
                grid_size=self.env.grid_size,
                max_body_length=self.max_body_length,
                theme=self.env.theme,
                rich_style_config=self.env.rich_style_config,
                # Reproducibility: preserve seed from original environment
                seed=self.env.seed,
            )
        self.path = [(self.env.agent_pos[0], self.env.agent_pos[1])]
        # Track food positions at each step for chemotaxis validation
        self.food_history = []
        if isinstance(self.env, DynamicForagingEnvironment):
            self.food_history = [list(self.env.foods)]

        # Update component references to new environment instance
        self._food_handler.env = self.env

        # Reset satiety manager to initial satiety
        self._satiety_manager.reset()

        # Reset food handler tracking for new environment
        if isinstance(self.env, DynamicForagingEnvironment):
            self._food_handler.reset()

        # Reset episode tracker
        self._episode_tracker.reset()

        logger.info("Environment reset. Retaining learned data.")

    def reset_brain(self) -> None:
        """
        Reset the agent's brain state.

        Reset only brain data we do not want to persist between runs.
        This includes historical data saved in the brain.

        Returns
        -------
        None
        """
        # Reset the brain's history
        self.brain.history_data = BrainHistoryData()
        logger.info("Agent brain reset.")

    def calculate_metrics(self, total_runs: int) -> PerformanceMetrics:
        """
        Calculate and return performance metrics.

        Parameters
        ----------
        total_runs : int
            Total number of runs.

        Returns
        -------
        PerformanceMetrics
            An object containing success rate, average steps, average reward, and dynamic metrics.
        """
        # Determine if predators are enabled for proper metrics calculation
        predators_enabled = (
            isinstance(self.env, DynamicForagingEnvironment) and self.env.predators_enabled
        )

        metrics = self._metrics_tracker.calculate_metrics(
            total_runs=total_runs,
            predators_enabled=predators_enabled,
        )

        # For non-dynamic environments, set foraging metrics to None (agent's original behavior)
        if not isinstance(self.env, DynamicForagingEnvironment):
            # Replace the foraging efficiency with agent's original calculation
            return PerformanceMetrics(
                success_rate=metrics.success_rate,
                average_steps=metrics.average_steps,
                average_reward=metrics.average_reward,
                foraging_efficiency=None,
                average_distance_efficiency=None,
                average_foods_collected=None,
                total_successes=metrics.total_successes,
                total_starved=metrics.total_starved,
                total_predator_encounters=metrics.total_predator_encounters,
                total_predator_deaths=metrics.total_predator_deaths,
                total_successful_evasions=metrics.total_successful_evasions,
                total_max_steps=metrics.total_max_steps,
                total_interrupted=metrics.total_interrupted,
                average_predator_encounters=metrics.average_predator_encounters,
                average_successful_evasions=metrics.average_successful_evasions,
            )

        # For dynamic environments, convert foraging_efficiency from foods/run to foods/step
        foraging_efficiency_per_step = None
        if self._metrics_tracker.total_steps > 0:
            foraging_efficiency_per_step = (
                self._metrics_tracker.foods_collected / self._metrics_tracker.total_steps
            )

        return PerformanceMetrics(
            success_rate=metrics.success_rate,
            average_steps=metrics.average_steps,
            average_reward=metrics.average_reward,
            foraging_efficiency=foraging_efficiency_per_step,
            average_distance_efficiency=metrics.average_distance_efficiency,
            average_foods_collected=metrics.average_foods_collected,
            total_successes=metrics.total_successes,
            total_starved=metrics.total_starved,
            total_predator_encounters=metrics.total_predator_encounters,
            total_predator_deaths=metrics.total_predator_deaths,
            total_successful_evasions=metrics.total_successful_evasions,
            total_max_steps=metrics.total_max_steps,
            total_interrupted=metrics.total_interrupted,
            average_predator_encounters=metrics.average_predator_encounters,
            average_successful_evasions=metrics.average_successful_evasions,
        )
