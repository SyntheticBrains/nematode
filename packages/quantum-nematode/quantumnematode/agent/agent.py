"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from quantumnematode.agent.stam import STAMBuffer
from quantumnematode.agent.tracker import EpisodeTracker
from quantumnematode.brain.actions import ActionData  # noqa: TC001 - needed at runtime
from quantumnematode.brain.arch import Brain, BrainParams, QuantumBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.dtypes import FoodHistory, GridPosition  # noqa: TC001 - used at runtime
from quantumnematode.env import (
    DEFAULT_AGENT_ID,
    DynamicForagingEnvironment,
)
from quantumnematode.env.theme import DEFAULT_THEME, DarkColorRichStyleConfig, Theme
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import PerformanceMetrics

if TYPE_CHECKING:
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.agent.runners import EpisodeResult
    from quantumnematode.env.pygame_renderer import PygameRenderer
    from quantumnematode.utils.config_loader import SensingConfig

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
DEFAULT_PENALTY_BOUNDARY_COLLISION = 0.02  # Penalty for wall collision (mechanosensation)
DEFAULT_PENALTY_TEMPERATURE_PROXIMITY = 0.3  # Scale factor for temp deviation reward
DEFAULT_REWARD_DISTANCE_SCALE = 0.3
DEFAULT_REWARD_GOAL = 0.2
DEFAULT_REWARD_EXPLORATION = 0.05
DEFAULT_MANYWORLDS_MODE_MAX_COLUMNS = 4
DEFAULT_MANYWORLDS_MODE_MAX_SUPERPOSITIONS = 16
DEFAULT_MANYWORLDS_MODE_RENDER_SLEEP_SECONDS = 1.0
DEFAULT_MANYWORLDS_MODE_TOP_N_ACTIONS = 2
DEFAULT_MANYWORLDS_MODE_TOP_N_RANDOMIZE = True
DEFAULT_STUCK_POSITION_THRESHOLD = 2

# Action-to-index mapping for STAM action entropy computation
_ACTION_TO_IDX: dict[str, int] = {
    "forward": 0,
    "left": 1,
    "right": 2,
    "stay": 3,
    "forward-left": 4,
    "forward-right": 5,
}
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
    # Penalty for health depletion (predator damage or temperature)
    penalty_predator_death: float = DEFAULT_PENALTY_PREDATOR_DEATH
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
    # Mechanosensation penalties
    penalty_boundary_collision: float = (
        DEFAULT_PENALTY_BOUNDARY_COLLISION  # Penalty per step touching grid boundary
    )
    # Temperature avoidance (distance-scaled, mirrors predator evasion structure)
    penalty_temperature_proximity: float = (
        DEFAULT_PENALTY_TEMPERATURE_PROXIMITY  # Scale factor for temp deviation-based reward
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
        env: DynamicForagingEnvironment | None = None,
        maze_grid_size: int = DEFAULT_MAZE_GRID_SIZE,
        max_body_length: int = DEFAULT_MAX_AGENT_BODY_LENGTH,
        theme: Theme = DEFAULT_THEME,
        rich_style_config: DarkColorRichStyleConfig | None = None,
        satiety_config: SatietyConfig | None = None,
        sensing_config: SensingConfig | None = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> None:
        """
        Initialize the nematode agent.

        Parameters
        ----------
        brain : Brain
            The brain architecture used by the agent.
        env : DynamicForagingEnvironment | None
            The environment to use. If None, creates a default DynamicForagingEnvironment.
        maze_grid_size : int, optional
            Size of the grid environment, by default 50 (only used if env is None).
        max_body_length : int, optional
            Maximum body length.
        theme : Theme, optional
            Visual theme for rendering.
        rich_style_config : DarkColorRichStyleConfig | None, optional
            Rich styling configuration.
        satiety_config : SatietyConfig | None, optional
            Satiety system configuration.
        sensing_config : SensingConfig | None, optional
            Temporal sensing modes and STAM configuration.
        agent_id : str, optional
            Unique identifier for multi-agent mode. Defaults to "default".
        """
        from quantumnematode.utils.config_loader import SensingConfig

        self.agent_id = agent_id
        self.brain = brain
        self.satiety_config = satiety_config or SatietyConfig()
        self.sensing_config: SensingConfig = sensing_config or SensingConfig()

        # Initialize STAM buffer when enabled
        self._stam: STAMBuffer | None = None
        if self.sensing_config.stam_enabled:
            self._stam = STAMBuffer(
                buffer_size=self.sensing_config.stam_buffer_size,
                decay_rate=self.sensing_config.stam_decay_rate,
            )
        self._previous_position: tuple[int, ...] | None = None

        if env is None:
            self.env = DynamicForagingEnvironment(
                grid_size=maze_grid_size,
                max_body_length=max_body_length,
                theme=theme,
                rich_style_config=rich_style_config,
            )
        else:
            self.env = env

        self.path: list[GridPosition] = [(self.env.agent_pos[0], self.env.agent_pos[1])]
        # Track food positions at each step for chemotaxis validation
        self.food_history: FoodHistory = [list(self.env.foods)]
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

    def _compute_temporal_data(  # noqa: C901
        self,
        sensing: SensingConfig,
        temperature: float | None,
        separated_grads: dict[str, Any],
        action: ActionData | None,
    ) -> dict[str, Any]:
        """Compute temporal sensing data (scalar concentrations, STAM, derivatives).

        Parameters
        ----------
        sensing : SensingConfig
            Sensing configuration.
        temperature : float | None
            Current temperature at agent's position (None if thermotaxis disabled).
        separated_grads : dict
            Separated gradient dict (modified in-place to suppress oracle fields).
        action : ActionData | None
            Previous action taken.

        Returns
        -------
        dict
            Temporal sensing fields for BrainParams.
        """
        from quantumnematode.utils.config_loader import SensingMode

        result: dict[str, Any] = {}

        any_non_oracle = (
            sensing.chemotaxis_mode != SensingMode.ORACLE
            or sensing.nociception_mode != SensingMode.ORACLE
            or sensing.thermotaxis_mode != SensingMode.ORACLE
            or sensing.aerotaxis_mode != SensingMode.ORACLE
        )

        if not (any_non_oracle or sensing.stam_enabled):
            return result

        # (a) Get scalar concentrations from environment
        food_conc_val = self.env.get_food_concentration()
        pred_conc_val = self.env.get_predator_concentration()
        temp_val = temperature if temperature is not None else 0.0

        if sensing.chemotaxis_mode != SensingMode.ORACLE:
            result["food_concentration"] = food_conc_val
            separated_grads.pop("food_gradient_strength", None)
            separated_grads.pop("food_gradient_direction", None)

        if sensing.nociception_mode != SensingMode.ORACLE:
            result["predator_concentration"] = pred_conc_val
            separated_grads.pop("predator_gradient_strength", None)
            separated_grads.pop("predator_gradient_direction", None)

        # Oxygen scalar concentration (raw percentage, not tanh-normalized)
        o2_val = 0.0
        if self.env.aerotaxis.enabled:
            o2_raw = self.env.get_oxygen_concentration()
            o2_val = o2_raw if o2_raw is not None else 0.0
            if sensing.aerotaxis_mode != SensingMode.ORACLE:
                separated_grads.pop("oxygen_gradient_strength", None)
                separated_grads.pop("oxygen_gradient_direction", None)

        # (b) Compute position delta from previous position
        current_pos = tuple(self.env.agent_pos)
        pos_delta = (0.0, 0.0)
        if self._previous_position is not None:
            pos_delta = (
                float(current_pos[0] - self._previous_position[0]),
                float(current_pos[1] - self._previous_position[1]),
            )
        self._previous_position = current_pos

        # (c) Record to STAM
        if self._stam is not None:
            action_idx = 0
            if action is not None:
                action_str = str(action.action) if action.action is not None else "stay"
                action_idx = _ACTION_TO_IDX.get(action_str, 0)
            self._stam.record(
                np.array([food_conc_val, temp_val, pred_conc_val, o2_val]),
                pos_delta,
                action_idx,
            )

            # (d) Get temporal derivatives from STAM
            if sensing.chemotaxis_mode == SensingMode.DERIVATIVE:
                result["food_dconcentration_dt"] = self._stam.compute_temporal_derivative(0)
            if sensing.thermotaxis_mode == SensingMode.DERIVATIVE:
                result["temperature_ddt"] = self._stam.compute_temporal_derivative(1)
            if sensing.nociception_mode == SensingMode.DERIVATIVE:
                result["predator_dconcentration_dt"] = self._stam.compute_temporal_derivative(2)
            if sensing.aerotaxis_mode == SensingMode.DERIVATIVE:
                result["oxygen_dconcentration_dt"] = self._stam.compute_temporal_derivative(3)

            result["stam_state"] = tuple(self._stam.get_memory_state().tolist())

        return result

    def _create_brain_params(
        self,
        action: ActionData | None = None,
        nearby_agents_count: int | None = None,
    ) -> BrainParams:
        """Create BrainParams for brain execution.

        Step-0-safe ordering for temporal sensing:
        (a) get scalar concentrations from env
        (b) compute position delta from previous position
        (c) record to STAM
        (d) get temporal derivatives from STAM
        (e) build BrainParams with all fields

        Parameters
        ----------
        action : ActionData | None, optional
            Previous action taken, by default None.

        Returns
        -------
        BrainParams
            Brain parameters ready for execution.
        """
        sensing = self.sensing_config

        # Get separated food/predator gradients for sensory modules
        separated_grads = self.env.get_separated_gradients(
            self.env.agent_pos,
            disable_log=True,
        )

        # Mechanosensation: detect physical contact with boundaries and predators
        boundary_contact = self.env.is_agent_at_boundary()
        predator_contact = self.env.is_agent_in_predator_contact()

        # Health state
        health = self.env.agent_hp
        max_health = self.env.health.max_hp

        # Thermotaxis: temperature sensing
        temperature = None
        temperature_gradient_strength = None
        temperature_gradient_direction = None
        cultivation_temperature = None

        if self.env.thermotaxis.enabled:
            temperature = self.env.get_temperature()
            from quantumnematode.utils.config_loader import SensingMode

            if sensing.thermotaxis_mode == SensingMode.ORACLE:
                temp_gradient = self.env.get_temperature_gradient()
                if temp_gradient is not None:
                    temperature_gradient_strength = temp_gradient[0]
                    temperature_gradient_direction = temp_gradient[1]
            cultivation_temperature = self.env.thermotaxis.cultivation_temperature

        # Aerotaxis: oxygen sensing
        oxygen_concentration = None
        oxygen_gradient_strength = None
        oxygen_gradient_direction = None

        if self.env.aerotaxis.enabled:
            oxygen_concentration = self.env.get_oxygen_concentration()
            from quantumnematode.utils.config_loader import SensingMode

            if sensing.aerotaxis_mode == SensingMode.ORACLE:
                o2_gradient = self.env.get_oxygen_gradient()
                if o2_gradient is not None:
                    oxygen_gradient_strength = o2_gradient[0]
                    oxygen_gradient_direction = o2_gradient[1]

        # --- Temporal sensing: scalar concentrations ---
        temporal = self._compute_temporal_data(sensing, temperature, separated_grads, action)

        # (e) Build BrainParams with all fields
        return BrainParams(
            # Separated LOCAL gradients (egocentric sensing, oracle)
            food_gradient_strength=separated_grads.get("food_gradient_strength"),
            food_gradient_direction=separated_grads.get("food_gradient_direction"),
            predator_gradient_strength=separated_grads.get("predator_gradient_strength"),
            predator_gradient_direction=separated_grads.get("predator_gradient_direction"),
            # Internal state (hunger)
            satiety=self.current_satiety,
            # Health state
            health=health,
            max_health=max_health,
            # Mechanosensation (physical contact)
            boundary_contact=boundary_contact,
            predator_contact=predator_contact,
            # Thermotaxis (temperature sensing)
            temperature=temperature,
            temperature_gradient_strength=temperature_gradient_strength,
            temperature_gradient_direction=temperature_gradient_direction,
            cultivation_temperature=cultivation_temperature,
            # Aerotaxis (oxygen sensing)
            oxygen_concentration=oxygen_concentration,
            oxygen_gradient_strength=oxygen_gradient_strength,
            oxygen_gradient_direction=oxygen_gradient_direction,
            oxygen_dconcentration_dt=temporal.get("oxygen_dconcentration_dt"),
            oxygen_comfort_midpoint=(
                (self.env.aerotaxis.comfort_lower + self.env.aerotaxis.comfort_upper) / 2.0
                if self.env.aerotaxis.enabled
                else 8.5
            ),
            # Normalization = max(midpoint, half_width) so that the larger
            # possible deviation (hypoxia toward 0% or hyperoxia toward 21%)
            # maps to [-1, 1] after clipping.  For default 5-12% comfort:
            # midpoint=8.5, half_width=3.5 → norm=8.5 (hypoxia side dominates).
            oxygen_comfort_normalization=(
                max(
                    (self.env.aerotaxis.comfort_lower + self.env.aerotaxis.comfort_upper) / 2.0,
                    self.env.aerotaxis.comfort_upper
                    - (self.env.aerotaxis.comfort_lower + self.env.aerotaxis.comfort_upper) / 2.0,
                )
                if self.env.aerotaxis.enabled
                else 12.5
            ),
            # Temporal sensing (Phase 3)
            food_concentration=temporal.get("food_concentration"),
            predator_concentration=temporal.get("predator_concentration"),
            food_dconcentration_dt=temporal.get("food_dconcentration_dt"),
            predator_dconcentration_dt=temporal.get("predator_dconcentration_dt"),
            temperature_ddt=temporal.get("temperature_ddt"),
            stam_state=temporal.get("stam_state"),
            derivative_scale=sensing.derivative_scale,
            # Agent proprioception
            agent_position=self._get_agent_position_tuple(),
            agent_direction=self.env.current_direction,
            action=action,
            # Social sensing (Phase 4 multi-agent)
            nearby_agents_count=nearby_agents_count,
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
        if self.env.theme == Theme.HEADLESS:
            return

        # Pygame rendering for PIXEL theme
        if self.env.theme == Theme.PIXEL:
            self._render_step_pygame(max_steps, render_text=render_text)
            return

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
        print(  # noqa: T201
            f"Eaten:\t\t{self._episode_tracker.foods_collected}/{self.env.foraging.target_foods_to_collect}",
        )
        print(  # noqa: T201
            f"Health:\t\t{self.env.agent_hp:.1f}/{self.env.health.max_hp}",
        )
        print(f"Satiety:\t{self.current_satiety:.1f}/{self.max_satiety}")  # noqa: T201
        # Display danger status if predators are enabled
        if self.env.predator.enabled:
            danger_status = "IN DANGER" if self.env.is_agent_in_danger() else "SAFE"
            print(f"Status:\t\t{danger_status}")  # noqa: T201
        # Display temperature if thermotaxis is enabled
        if self.env.thermotaxis.enabled:
            temperature = self.env.get_temperature()
            zone = self.env.get_temperature_zone()
            if temperature is not None:
                zone_name = zone.value.upper().replace("_", " ") if zone else "UNKNOWN"
                print(f"Temp:\t\t{temperature:.2f}°C ({zone_name})")  # noqa: T201

    def _get_pygame_renderer(self) -> PygameRenderer:
        """Lazily initialize and return the Pygame renderer."""
        if not hasattr(self, "_pygame_renderer") or self._pygame_renderer is None:
            try:
                from quantumnematode.env.pygame_renderer import PygameRenderer

                self._pygame_renderer = PygameRenderer(
                    viewport_size=self.env.viewport_size,
                )
            except Exception as exc:  # pragma: no cover
                msg = (
                    "PIXEL theme requires pygame with an available video backend. "
                    "Use --theme headless (no rendering) or --theme ascii (text)."
                )
                raise RuntimeError(msg) from exc
        return self._pygame_renderer

    @property
    def pygame_renderer_closed(self) -> bool:
        """Whether the Pygame renderer window has been closed by the user."""
        if hasattr(self, "_pygame_renderer") and self._pygame_renderer is not None:
            return self._pygame_renderer.closed
        return False

    def _render_step_pygame(
        self,
        max_steps: int,
        render_text: str | None = None,
    ) -> None:
        """Render the current step using the Pygame renderer."""
        renderer = self._get_pygame_renderer()
        if renderer.closed:
            return

        temperature: float | None = None
        zone_name: str | None = None
        if self.env.thermotaxis.enabled:
            temperature = self.env.get_temperature()
            zone = self.env.get_temperature_zone()
            if zone is not None:
                zone_name = zone.value.upper().replace("_", " ")

        oxygen: float | None = None
        oxygen_zone_name: str | None = None
        if self.env.aerotaxis.enabled:
            oxygen = self.env.get_oxygen_concentration()
            o2_zone = self.env.get_oxygen_zone()
            if o2_zone is not None:
                oxygen_zone_name = o2_zone.value.upper().replace("_", " ")

        renderer.render_frame(
            env=self.env,
            step=self._episode_tracker.steps,
            max_steps=max_steps,
            foods_collected=self._episode_tracker.foods_collected,
            target_foods=self.env.foraging.target_foods_to_collect,
            health=self.env.agent_hp,
            max_health=self.env.health.max_hp,
            satiety=self.current_satiety,
            max_satiety=self.max_satiety,
            in_danger=self.env.is_agent_in_danger() if self.env.predator.enabled else False,
            temperature=temperature,
            zone_name=zone_name,
            oxygen=oxygen,
            oxygen_zone_name=oxygen_zone_name,
            session_text=render_text,
        )

    def calculate_reward(
        self,
        config: RewardConfig,
        env: DynamicForagingEnvironment,
        path: list[tuple[int, ...]],
        max_steps: int,
        stuck_position_count: int = 0,
    ) -> float:
        """
        Calculate reward based on the agent's movement toward the goal.

        Handles DynamicForagingEnvironment (multiple foods)

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
        self.env = DynamicForagingEnvironment(
            grid_size=self.env.grid_size,
            viewport_size=self.env.viewport_size,
            max_body_length=self.max_body_length,
            theme=self.env.theme,
            rich_style_config=self.env.rich_style_config,
            # Preserve params from original env
            foraging=self.env.foraging,
            predator=self.env.predator,
            health=self.env.health,
            thermotaxis=self.env.thermotaxis,
            aerotaxis=self.env.aerotaxis,
            # Reproducibility: preserve seed from original environment
            seed=self.env.seed,
        )
        self.path = [(self.env.agent_pos[0], self.env.agent_pos[1])]
        # Track food positions at each step for chemotaxis validation
        self.food_history = [list(self.env.foods)]

        # Update component references to new environment instance
        self._food_handler.env = self.env

        # Reset satiety manager to initial satiety
        self._satiety_manager.reset()

        # Reset food handler tracking for new environment
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
        predators_enabled = self.env.predator.enabled

        metrics = self._metrics_tracker.calculate_metrics(
            total_runs=total_runs,
            predators_enabled=predators_enabled,
        )

        # Convert foraging_efficiency from foods/run to foods/step
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
            total_successful_evasions=metrics.total_successful_evasions,
            total_max_steps=metrics.total_max_steps,
            total_interrupted=metrics.total_interrupted,
            average_predator_encounters=metrics.average_predator_encounters,
            average_successful_evasions=metrics.average_successful_evasions,
        )
