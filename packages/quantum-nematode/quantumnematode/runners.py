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


class ManyworldsEpisodeRunner:
    """Runs an episode with many-worlds branching.

    This runner implements branching trajectories inspired by the many-worlds
    interpretation of quantum mechanics. At each step, multiple action branches
    are explored in parallel, creating a tree of possible futures.

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
        """Initialize the manyworlds episode runner.

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

    def run(  # noqa: C901, PLR0915 - complex branching logic inherent to many-worlds
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,  # noqa: ARG002 - will be used in future refinements
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401 - flexible kwargs for episode options
    ) -> EpisodeResult:
        """Run an episode with many-worlds branching.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance (used for accessing brain, env).
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional keyword arguments:
            - manyworlds_config : ManyworldsModeConfig - Configuration for branching
            - render_text : str | None - Text to display during rendering
            - show_last_frame_only : bool - Whether to show only the last frame

        Returns
        -------
        EpisodeResult
            Complete result of running the episode (path of best trajectory).
        """
        from quantumnematode.agent import ManyworldsModeConfig

        # Get configuration
        manyworlds_config: ManyworldsModeConfig = kwargs.get(
            "manyworlds_config",
            ManyworldsModeConfig(),
        )  # type: ignore[assignment]

        # Initialize superpositions (brain_copy, env_copy, path, total_reward)
        initial_path = [tuple(agent.env.agent_pos)]
        superpositions: list[tuple[Any, Any, list[tuple[int, ...]], float]] = [
            (agent.brain.copy(), agent.env.copy(), initial_path, 0.0),
        ]

        best_reward = float("-inf")
        best_path = initial_path

        for _ in range(max_steps):
            new_superpositions: list[tuple[Any, Any, list[tuple[int, ...]], float]] = []

            # Process each existing superposition
            for brain_copy, env_copy, path_copy, accumulated_reward in superpositions:
                # Skip if already reached goal
                if env_copy.reached_goal():
                    new_superpositions.append(
                        (brain_copy, env_copy, path_copy, accumulated_reward),
                    )
                    continue

                # Get state and run brain
                gradient_strength, gradient_direction = env_copy.get_state(path_copy[-1])

                # Get action probabilities from brain
                agent_pos_tuple = tuple(float(x) for x in env_copy.agent_pos[:2])
                agent_pos: tuple[float, float] = (
                    agent_pos_tuple[0] if len(agent_pos_tuple) > 0 else 0.0,
                    agent_pos_tuple[1] if len(agent_pos_tuple) > 1 else 0.0,
                )
                from quantumnematode.brain.arch import BrainParams

                params = BrainParams(
                    gradient_strength=gradient_strength,
                    gradient_direction=gradient_direction,
                    agent_position=agent_pos,
                    agent_direction=env_copy.current_direction,
                )

                actions = brain_copy.run_brain(
                    params=params,
                    reward=0.0,
                    input_data=None,
                    top_only=False,
                    top_randomize=True,
                )

                # Select top N actions
                import numpy as np

                if manyworlds_config.top_n_randomize:
                    # Sample actions by probability
                    probs = np.array([a.probability for a in actions], dtype=float)
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        norm_probs = probs / probs_sum
                    else:
                        norm_probs = np.ones_like(probs) / len(probs)

                    actions_arr = np.array(actions)
                    top_actions_data = np.random.default_rng().choice(
                        actions_arr,
                        p=norm_probs,
                        size=min(manyworlds_config.top_n_actions, len(actions)),
                        replace=True,
                    )
                    top_actions = [a.action for a in top_actions_data if a.action is not None]
                else:
                    # Take deterministic top N
                    top_actions = [a.action for a in actions if a.action is not None][
                        : manyworlds_config.top_n_actions
                    ]

                # Create branches for each top action
                for i, action in enumerate(top_actions):
                    # Create copies for this branch
                    new_brain = brain_copy.copy()
                    new_env = env_copy.copy()
                    new_path = path_copy.copy()

                    # Execute action
                    new_env.move_agent(action)
                    new_path.append(tuple(new_env.agent_pos))

                    # Calculate reward for this step
                    step_result = self.step_processor.process_step(
                        gradient_strength=gradient_strength,
                        gradient_direction=gradient_direction,
                        previous_action=action,
                        previous_reward=0.0,
                        path=new_path,
                        stuck_position_count=0,  # Many-worlds doesn't track stuck
                        top_only=True,
                        top_randomize=True,
                    )

                    new_accumulated_reward = accumulated_reward + step_result.reward
                    new_brain.update_memory(step_result.reward)

                    # Track best path
                    if new_accumulated_reward > best_reward:
                        best_reward = new_accumulated_reward
                        best_path = new_path

                    # Add to new superpositions if under limit
                    if len(new_superpositions) < manyworlds_config.max_superpositions:
                        new_superpositions.append(
                            (new_brain, new_env, new_path, new_accumulated_reward),
                        )

                    # Only keep first action branch for primary superposition
                    if i == 0:
                        break

            superpositions = new_superpositions

            # Check if all reached goal
            if all(env.reached_goal() for _, env, _, _ in superpositions):
                break

        # Use best trajectory for result
        success = any(env.reached_goal() for _, env, _, _ in superpositions)
        steps = len(best_path) - 1  # Subtract initial position

        # Track episode completion
        self.metrics_tracker.track_episode_completion(
            success=success,
            steps=steps,
            total_reward=best_reward,
        )

        # Calculate final metrics
        metrics = self.metrics_tracker.calculate_metrics(total_runs=1)

        # Convert path to correct type
        path_2d: list[tuple[int, int]] = [(int(p[0]), int(p[1])) for p in best_path]

        return EpisodeResult(
            path=path_2d,
            success=success,
            total_reward=best_reward,
            steps_taken=steps,
            metrics=metrics.model_dump(),
        )
