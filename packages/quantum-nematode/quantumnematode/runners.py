"""Episode execution strategies for the quantum nematode agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.agent import EpisodeResult
from quantumnematode.env import DynamicForagingEnvironment

if TYPE_CHECKING:
    from quantumnematode.agent import QuantumNematodeAgent, RewardConfig
    from quantumnematode.metrics import MetricsTracker
    from quantumnematode.rendering import EpisodeRenderer
    from quantumnematode.step_processor import StepProcessor


class StandardEpisodeRunner:
    """Runs a standard episode using step-by-step execution.

    This runner orchestrates the main episode loop, delegating step execution
    to the StepProcessor and handling episode-level concerns like rendering
    and metrics tracking.

    Parameters
    ----------
    step_processor : StepProcessor
        Processor for individual step execution.
    metrics_tracker : MetricsTracker
        Tracker for performance metrics.
    renderer : EpisodeRenderer
        Renderer for episode visualization.

    Attributes
    ----------
    step_processor : StepProcessor
        The step processor instance.
    metrics_tracker : MetricsTracker
        The metrics tracker instance.
    renderer : EpisodeRenderer
        The renderer instance.
    """

    def __init__(
        self,
        step_processor: StepProcessor,
        metrics_tracker: MetricsTracker,
        renderer: EpisodeRenderer,
    ) -> None:
        """Initialize the standard episode runner.

        Parameters
        ----------
        step_processor : StepProcessor
            Processor for individual step execution.
        metrics_tracker : MetricsTracker
            Tracker for performance metrics.
        renderer : EpisodeRenderer
            Renderer for episode visualization.
        """
        self.step_processor = step_processor
        self.metrics_tracker = metrics_tracker
        self.renderer = renderer

    def run(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,  # noqa: ARG002 - will be used in future refinements
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401 - flexible kwargs for episode options
    ) -> EpisodeResult:
        """Run a standard episode.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance (used for accessing path, steps, env).
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional keyword arguments:
            - render_text : str | None - Text to display during rendering
            - show_last_frame_only : bool - Whether to show only the last frame

        Returns
        -------
        EpisodeResult
            Complete result of running the episode.
        """
        render_text: str | None = kwargs.get("render_text")  # type: ignore[assignment]
        show_last_frame_only: bool = kwargs.get("show_last_frame_only", False)  # type: ignore[assignment]

        # Initialize episode state
        path = [tuple(agent.env.agent_pos)]
        steps = 0
        total_reward = 0.0
        previous_action = None
        previous_reward = 0.0
        stuck_position_count = 0
        previous_position = None
        success = False

        # Track stuck position
        for step in range(max_steps):
            # Get current state
            gradient_strength, gradient_direction = agent.env.get_state(path[-1])

            # Track if agent stays in same position
            current_position = tuple(agent.env.agent_pos)
            if current_position == previous_position:
                stuck_position_count += 1
            else:
                stuck_position_count = 0
            previous_position = current_position

            # Process step
            step_result = self.step_processor.process_step(
                gradient_strength=gradient_strength,
                gradient_direction=gradient_direction,
                previous_action=previous_action,
                previous_reward=previous_reward,
                path=path,
                stuck_position_count=stuck_position_count,
                top_only=True,
                top_randomize=True,
            )

            # Update state
            path.append(tuple(agent.env.agent_pos))
            steps += 1
            total_reward += step_result.reward
            previous_action = step_result.action
            previous_reward = step_result.reward

            # Render if needed
            self.renderer.render_if_needed(
                env=agent.env,
                step=step,
                max_steps=max_steps,
                show_last_frame_only=show_last_frame_only,
                text=render_text,
            )

            # Track food consumption
            if step_result.info.get("food_consumed"):
                self.metrics_tracker.track_food_collection(
                    distance_efficiency=step_result.info.get("distance_efficiency"),
                )

            # Check termination conditions
            if step_result.done:
                # Starvation or other terminal condition
                break

            # Check for goal reached (single-goal environments)
            if step_result.info.get("goal_reached") and not isinstance(
                agent.env,
                DynamicForagingEnvironment,
            ):
                success = True
                break

        # Track episode completion
        self.metrics_tracker.track_episode_completion(
            success=success,
            steps=steps,
            total_reward=total_reward,
        )

        # Calculate final metrics
        metrics = self.metrics_tracker.calculate_metrics(total_runs=1)

        # Convert path to correct type (EpisodeResult expects 2D tuples)
        path_2d: list[tuple[int, int]] = [(int(p[0]), int(p[1])) for p in path]

        return EpisodeResult(
            path=path_2d,
            success=success,
            total_reward=total_reward,
            steps_taken=steps,
            metrics=metrics.model_dump(),
        )
