"""Episode execution strategies for the quantum nematode agent."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from quantumnematode.brain.arch import ClassicalBrain
from quantumnematode.env import Direction, DynamicForagingEnvironment, StaticEnvironment
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import TerminationReason

if TYPE_CHECKING:
    from quantumnematode.agent import QuantumNematodeAgent, RewardConfig


@dataclass
class EpisodeData:
    """Data collected during a single simulation episode.

    Attributes
    ----------
    steps : int
        The number of steps taken in the episode.
    rewards : float
        The total reward accumulated during the episode.
    foods_collected : int
        The number of food items collected during the episode.
    distance_efficiencies : list[float]
        The distance efficiencies recorded during the episode.
    satiety_history : list[float]
        The satiety levels at each step (for dynamic foraging environments).
    predator_encounters : int
        Number of times agent entered predator detection radius.
    successful_evasions : int
        Number of times agent exited predator detection radius without dying.
    in_danger : bool
        Whether agent is currently within predator detection radius.
    """

    steps: int
    rewards: float
    foods_collected: int
    distance_efficiencies: list[float]
    satiety_history: list[float]
    predator_encounters: int = 0
    successful_evasions: int = 0
    in_danger: bool = False


@dataclass
class EpisodeResult:
    """Result of processing a single simulation episode.

    Attributes
    ----------
    agent_path : list[tuple]
        The path taken by the agent in the episode.
    termination_reason : TerminationReason
        The reason for episode termination, if applicable.
    """

    agent_path: list[tuple]
    termination_reason: TerminationReason


class EpisodeRunner(Protocol):
    """Protocol for episode execution strategies.

    Episode runners encapsulate different modes of executing simulation episodes
    (e.g., standard single-trajectory, many-worlds branching). They access agent
    components and helper methods to execute the episode logic.
    """

    def run(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: dict[str, Any],
    ) -> EpisodeResult:
        """Execute an episode using this runner's strategy.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance containing brain, environment, and components.
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional runner-specific parameters.

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        ...


class StandardEpisodeRunner(EpisodeRunner):
    """Runs a standard episode using step-by-step execution.

    This runner orchestrates the main episode loop by directly accessing agent
    components and helper methods. The runner delegates to the agent's internal
    components (FoodConsumptionHandler, SatietyManager) and helper methods for
    episode execution.

    Notes
    -----
    The implementation directly accesses agent private members (_food_handler,
    _satiety_manager, helper methods) as part of the episode execution architecture.
    This provides a clean separation between episode orchestration (runner) and
    agent state management.
    """

    def __init__(self) -> None:
        """Initialize the standard episode runner."""

    def run(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        """Run a standard episode.

        Complete implementation that matches agent.run_episode() logic exactly.

        Note: This runner intentionally accesses private agent members (_food_handler,
        _satiety_manager, _prepare_input_data, _create_brain_params, _render_step)
        as part of the episode execution architecture.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance.
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
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        render_text: str | None = kwargs.get("render_text")
        show_last_frame_only: bool = kwargs.get("show_last_frame_only", False)

        # Initialize the agent's direction
        agent.env.current_direction = Direction.UP

        # Initialize distance tracking for dynamic environments
        if isinstance(agent.env, DynamicForagingEnvironment):
            # Reset food handler tracking for new episode
            agent._food_handler.reset()

        reward = 0.0
        top_action = None
        stuck_position_count = 0
        previous_position = None

        for _ in range(max_steps):
            logger.debug("--- New Step ---")
            gradient_strength, gradient_direction = agent.env.get_state(agent.path[-1])

            if logger.isEnabledFor(logging.DEBUG):
                print()  # noqa: T201
                print(f"Gradient strength: {gradient_strength}")  # noqa: T201
                print(f"Gradient direction: {gradient_direction}")  # noqa: T201

            # Track if agent stays in same position
            current_position = tuple(agent.env.agent_pos)
            if current_position == previous_position:
                stuck_position_count += 1
            else:
                stuck_position_count = 0
            previous_position = current_position

            # Calculate reward
            reward = agent.calculate_reward(
                reward_config,
                agent.env,
                agent.path,
                max_steps=max_steps,
                stuck_position_count=stuck_position_count,
            )
            agent._episode_tracker.track_reward(reward)

            if logger.isEnabledFor(logging.DEBUG):
                print(f"Reward: {reward}")  # noqa: T201

            # Prepare input_data and brain parameters
            input_data = agent._prepare_input_data(gradient_strength)
            params = agent._create_brain_params(
                gradient_strength,
                gradient_direction,
                action=top_action,
            )
            action = agent.brain.run_brain(
                params=params,
                reward=reward,
                input_data=input_data,
                top_only=True,
                top_randomize=True,
            )

            # Only one action is supported
            if len(action) != 1:
                error_msg = f"Invalid action length: {len(action)}. Expected 1."
                logger.error(error_msg)
                raise ValueError(error_msg)

            top_action = action[0]

            agent.env.move_agent(top_action.action)

            # Track step (will add satiety later if dynamic environment)
            agent._episode_tracker.track_step()

            # Food collection (must happen immediately after agent moves)
            if isinstance(agent.env, DynamicForagingEnvironment) and agent.env.reached_goal():
                # Multi-food environment: delegate to food handler
                food_result = agent._food_handler.check_and_consume_food(
                    foods_collected=agent._episode_tracker.foods_collected,
                )
                if food_result.food_consumed:
                    agent._episode_tracker.track_food_collection(
                        distance_efficiency=food_result.distance_efficiency,
                    )

                    # Log food collection with distance efficiency
                    dist_eff_msg = ""
                    if food_result.distance_efficiency is not None:
                        dist_eff = food_result.distance_efficiency
                        dist_eff_msg = f" (Distance efficiency: {dist_eff:.2f})"

                    logger.info(
                        f"Food #{agent._episode_tracker.foods_collected} collected! "
                        f"Satiety restored by {food_result.satiety_restored:.1f} to "
                        f"{agent.current_satiety:.1f}/{agent.max_satiety}{dist_eff_msg}",
                    )

            # Update predators and check for collision (dynamic environment only)
            if isinstance(agent.env, DynamicForagingEnvironment):
                # Update predator encounter tracking
                was_in_danger = agent._episode_tracker.in_danger
                is_in_danger = agent.env.is_agent_in_danger()

                if not was_in_danger and is_in_danger:
                    # Entered danger zone - increment encounters
                    agent._episode_tracker.predator_encounters += 1
                elif was_in_danger and not is_in_danger:
                    # Exited danger zone without dying - successful evasion
                    agent._episode_tracker.successful_evasions += 1

                agent._episode_tracker.in_danger = is_in_danger

                # Check for predator collision BEFORE predators move
                if agent.env.check_predator_collision():
                    logger.warning("Agent caught by predator!")
                    # Apply death penalty to both brain reward and episode tracker
                    penalty = -reward_config.penalty_predator_death
                    reward += penalty
                    agent._episode_tracker.track_reward(penalty)
                    agent.brain.update_memory(reward)
                    agent.brain.post_process_episode()
                    agent._metrics_tracker.track_episode_completion(
                        success=False,
                        steps=agent._episode_tracker.steps,
                        reward=agent._episode_tracker.rewards,
                        foods_collected=agent._episode_tracker.foods_collected,
                        distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                        predator_encounters=agent._episode_tracker.predator_encounters,
                        successful_evasions=agent._episode_tracker.successful_evasions,
                        termination_reason=TerminationReason.PREDATOR,
                    )
                    return EpisodeResult(
                        agent_path=agent.path,
                        termination_reason=TerminationReason.PREDATOR,
                    )

                # Move predators after agent moves
                agent.env.update_predators()

                # Check for predator collision AFTER predators move
                # (predator may step onto agent's position)
                if agent.env.check_predator_collision():
                    logger.warning("Agent caught by predator (after predator movement)!")
                    # Apply death penalty to both brain reward and episode tracker
                    penalty = -reward_config.penalty_predator_death
                    reward += penalty
                    agent._episode_tracker.track_reward(penalty)
                    agent.brain.update_memory(reward)
                    agent.brain.post_process_episode()
                    agent._metrics_tracker.track_episode_completion(
                        success=False,
                        steps=agent._episode_tracker.steps,
                        reward=agent._episode_tracker.rewards,
                        foods_collected=agent._episode_tracker.foods_collected,
                        distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                        predator_encounters=agent._episode_tracker.predator_encounters,
                        successful_evasions=agent._episode_tracker.successful_evasions,
                        termination_reason=TerminationReason.PREDATOR,
                    )
                    return EpisodeResult(
                        agent_path=agent.path,
                        termination_reason=TerminationReason.PREDATOR,
                    )

                # Satiety decay (after predator movement)
                # Delegate to satiety manager
                agent._satiety_manager.decay_satiety()

                # Track satiety after decay
                agent._episode_tracker.track_satiety(agent.current_satiety)

                logger.debug(
                    f"Satiety: {agent.current_satiety:.1f}/{agent.max_satiety}",
                )

                # Check for starvation (after satiety decay)
                if agent._satiety_manager.is_starved():
                    logger.warning("Agent starved!")
                    # Apply starvation penalty to both brain reward and episode tracker
                    penalty = -reward_config.penalty_starvation
                    reward += penalty
                    agent._episode_tracker.track_reward(penalty)
                    agent.brain.update_memory(reward)
                    agent.brain.post_process_episode()
                    agent._metrics_tracker.track_episode_completion(
                        success=False,
                        steps=agent._episode_tracker.steps,
                        reward=agent._episode_tracker.rewards,
                        foods_collected=agent._episode_tracker.foods_collected,
                        distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                        predator_encounters=agent._episode_tracker.predator_encounters,
                        successful_evasions=agent._episode_tracker.successful_evasions,
                        termination_reason=TerminationReason.STARVED,
                    )
                    return EpisodeResult(
                        agent_path=agent.path,
                        termination_reason=TerminationReason.STARVED,
                    )

            # Classical brain learning step
            if isinstance(agent.brain, ClassicalBrain):
                episode_done = bool(
                    agent._episode_tracker.steps >= max_steps or agent.env.reached_goal(),
                )
                agent.brain.learn(
                    params=params,
                    reward=reward,
                    episode_done=episode_done,
                )

            # Update the body length dynamically
            if agent.max_body_length > 0 and len(agent.env.body) < agent.max_body_length:
                agent.env.body.append(agent.env.body[-1])

            agent.brain.update_memory(reward)

            agent.path.append(tuple(agent.env.agent_pos))

            # Track step for food distance efficiency calculation
            if isinstance(agent.env, DynamicForagingEnvironment):
                agent._food_handler.track_step()

            logger.info(
                f"Step {agent._episode_tracker.steps}: "
                f"Action={top_action.action.value}, Reward={reward}",
            )

            # Check for goal reached (static maze only - dynamic food already handled)
            if agent.env.reached_goal() and not isinstance(agent.env, DynamicForagingEnvironment):
                # Single goal environment: episode ends when goal is reached
                # Run the brain with the final state and reward
                reward = agent.calculate_reward(
                    reward_config,
                    agent.env,
                    agent.path,
                    max_steps=max_steps,
                    stuck_position_count=stuck_position_count,
                )
                agent._episode_tracker.track_reward(reward)

                # Prepare input_data and brain parameters for final goal state
                input_data = agent._prepare_input_data(gradient_strength)
                params = agent._create_brain_params(
                    gradient_strength,
                    gradient_direction,
                    action=top_action,
                )
                _ = agent.brain.run_brain(
                    params=params,
                    reward=reward,
                    input_data=None,
                    top_only=True,
                    top_randomize=True,
                )

                agent.brain.update_memory(reward)
                agent.brain.post_process_episode()

                agent.path.append(tuple(agent.env.agent_pos))
                agent._episode_tracker.track_step(reward=reward)

                logger.info(
                    f"Step {agent._episode_tracker.steps}: "
                    f"Action={top_action.action.value}, Reward={reward}",
                )

                logger.info("Reward: goal reached!")
                agent._metrics_tracker.track_episode_completion(
                    success=True,
                    steps=agent._episode_tracker.steps,
                    reward=agent._episode_tracker.rewards,
                    foods_collected=agent._episode_tracker.foods_collected,
                    distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                    predator_encounters=agent._episode_tracker.predator_encounters,
                    successful_evasions=agent._episode_tracker.successful_evasions,
                    termination_reason=TerminationReason.GOAL_REACHED,
                )
                return EpisodeResult(
                    agent_path=agent.path,
                    termination_reason=TerminationReason.GOAL_REACHED,
                )

            # Log distance to the goal (only for StaticEnvironment)
            if isinstance(agent.env, StaticEnvironment) and agent.env.goal is not None:
                distance_to_goal = abs(agent.env.agent_pos[0] - agent.env.goal[0]) + abs(
                    agent.env.agent_pos[1] - agent.env.goal[1],
                )
                logger.debug(f"Distance to goal: {distance_to_goal}")

            # Log cumulative reward and average reward per step
            if agent._episode_tracker.steps > 0:
                average_reward = agent._episode_tracker.rewards / agent._episode_tracker.steps
                logger.info(
                    f"Cumulative reward: {agent._episode_tracker.rewards}, "
                    f"Average reward per step: {average_reward}",
                )

            # Render current step
            agent._render_step(max_steps, render_text, show_last_frame_only=show_last_frame_only)

            # Handle max steps reached
            if agent._episode_tracker.steps >= max_steps:
                logger.warning("Max steps reached.")
                agent.brain.post_process_episode()
                agent._metrics_tracker.track_episode_completion(
                    success=False,
                    steps=agent._episode_tracker.steps,
                    reward=agent._episode_tracker.rewards,
                    foods_collected=agent._episode_tracker.foods_collected,
                    distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                    predator_encounters=agent._episode_tracker.predator_encounters,
                    successful_evasions=agent._episode_tracker.successful_evasions,
                    termination_reason=TerminationReason.MAX_STEPS,
                )
                return EpisodeResult(
                    agent_path=agent.path,
                    termination_reason=TerminationReason.MAX_STEPS,
                )

            # Handle all food collected (for dynamic environments)
            if (
                isinstance(agent.env, DynamicForagingEnvironment)
                and agent._episode_tracker.foods_collected >= agent.env.max_active_foods
            ):
                logger.info("All food collected.")
                agent.brain.post_process_episode()
                agent._metrics_tracker.track_episode_completion(
                    success=True,
                    steps=agent._episode_tracker.steps,
                    reward=agent._episode_tracker.rewards,
                    foods_collected=agent._episode_tracker.foods_collected,
                    distance_efficiencies=agent._episode_tracker.distance_efficiencies,
                    predator_encounters=agent._episode_tracker.predator_encounters,
                    successful_evasions=agent._episode_tracker.successful_evasions,
                    termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                )
                return EpisodeResult(
                    agent_path=agent.path,
                    termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                )

        # Episode ended normally (loop completed without specific termination)
        agent._metrics_tracker.track_episode_completion(
            success=False,
            steps=agent._episode_tracker.steps,
            reward=agent._episode_tracker.rewards,
            foods_collected=agent._episode_tracker.foods_collected,
            distance_efficiencies=agent._episode_tracker.distance_efficiencies,
            predator_encounters=agent._episode_tracker.predator_encounters,
            successful_evasions=agent._episode_tracker.successful_evasions,
            termination_reason=TerminationReason.MAX_STEPS,
        )
        return EpisodeResult(
            agent_path=agent.path,
            termination_reason=TerminationReason.MAX_STEPS,
        )


class ManyworldsEpisodeRunner(EpisodeRunner):
    """Runs an episode with many-worlds branching.

    This runner implements branching trajectories inspired by the many-worlds
    interpretation of quantum mechanics. At each step, multiple action branches
    are explored in parallel, creating a tree of possible futures.

    Notes
    -----
    The implementation directly accesses agent helper methods (_create_brain_params)
    and uses inline rendering for visualizing parallel universes. This specialized
    execution mode is fundamentally different from standard single-trajectory episodes.
    """

    def __init__(self) -> None:
        """Initialize the manyworlds episode runner."""

    def run(  # noqa: C901, PLR0912, PLR0915
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        """Run an episode with many-worlds branching.

        Complete implementation that matches agent.run_manyworlds_mode() logic exactly.

        Note: This runner intentionally accesses private agent member (_create_brain_params)
        as part of the episode execution architecture.

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
            - config : ManyworldsModeConfig - Configuration for branching
            - show_last_frame_only : bool - Whether to show only the last frame

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        from quantumnematode.agent import ManyworldsModeConfig

        # Get configuration
        config: ManyworldsModeConfig = kwargs.get("config", ManyworldsModeConfig())  # type: ignore[assignment]
        show_last_frame_only: bool = kwargs.get("show_last_frame_only", False)

        # Initialize many-worlds mode
        agent.env.current_direction = Direction.UP

        if show_last_frame_only:
            if os.name == "nt":  # For Windows
                os.system("cls")  # noqa: S605, S607
            else:  # For macOS and Linux
                os.system("clear")  # noqa: S605, S607

        # Render the initial grid
        grid = agent.env.render()
        for frame in grid:
            print(frame)  # noqa: T201
            logger.debug(frame)
        print("#1")  # noqa: T201

        time.sleep(config.render_sleep_seconds)  # Wait before the next render

        logger.info(
            "Many-worlds mode enabled. "
            f"Visualizing top {config.top_n_actions} decisions at each step.",
        )
        superpositions = [(agent.brain.copy(), agent.env.copy(), agent.path.copy())]

        reward = 0.0
        for _ in range(max_steps):
            total_superpositions = len(superpositions)
            i = 0
            for brain_copy, env_copy, path_copy in superpositions:
                gradient_strength, gradient_direction = env_copy.get_state(path_copy[-1])
                reward = agent.calculate_reward(
                    reward_config,
                    env_copy,
                    path_copy,
                    max_steps=max_steps,
                    stuck_position_count=0,  # Many-worlds mode doesn't track stuck positions
                )
                agent._episode_tracker.track_reward(reward)

                params = agent._create_brain_params(
                    gradient_strength,
                    gradient_direction,
                )
                actions = brain_copy.run_brain(
                    params=params,
                    reward=reward,
                    input_data=None,
                    top_only=False,
                    top_randomize=True,
                )

                if config.top_n_randomize:
                    rng = np.random.default_rng()
                    probs = np.array([a.probability for a in actions], dtype=float)
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        norm_probs = probs / probs_sum
                    else:
                        norm_probs = np.ones_like(probs) / len(probs)
                    actions_arr = np.array(actions)
                    top_actions_and_probs = rng.choice(
                        actions_arr,
                        p=norm_probs,
                        size=config.top_n_actions,
                        replace=True,
                    )
                    top_actions = [a.action for a in top_actions_and_probs if a.action is not None]
                else:
                    top_actions = [a.action for a in actions if a.action is not None][
                        : config.top_n_actions
                    ]

                # Update the body length dynamically
                if agent.max_body_length > 0 and len(env_copy.body) < agent.max_body_length:
                    env_copy.body.append(env_copy.body[-1])

                if len(superpositions) < config.max_superpositions and top_actions:
                    new_env = env_copy.copy()
                    new_path = path_copy.copy()
                    new_brain = agent.brain.copy()
                    runner_up_action = top_actions[1] if len(top_actions) > 1 else top_actions[0]
                    if runner_up_action is not None:
                        new_env.move_agent(runner_up_action)
                        agent._episode_tracker.track_step()
                        new_brain.update_memory(reward)
                        new_path.append(new_env.agent_pos)
                        superpositions.append((new_brain, new_env, new_path))

                if env_copy.reached_goal():
                    continue

                if top_actions:
                    env_copy.move_agent(top_actions[0])
                    agent._episode_tracker.track_step()
                    brain_copy.update_memory(reward)
                    path_copy.append(env_copy.agent_pos)

                i += 1
                if i >= total_superpositions:
                    break

            if show_last_frame_only:
                if os.name == "nt":  # For Windows
                    os.system("cls")  # noqa: S605, S607
                else:  # For macOS and Linux
                    os.system("clear")  # noqa: S605, S607

            # Render all grids for superpositions at each step
            row = []
            labels = []
            label_padding_first = " " * 8
            label_padding_all = " " * agent.env.grid_size
            for i, (_, env_copy, _) in enumerate(superpositions):
                grid = env_copy.render()
                label = (
                    f"#{i + 1} <= #{i // 2 + 1}{label_padding_all}"
                    if i > 0
                    else f"#{i + 1}{label_padding_first}{label_padding_all}"
                )
                row.append(grid)
                labels.append(label)

                # Print the row when reaching MAX_COLUMNS or the last grid
                if (i + 1) % config.max_columns == 0 or i == len(superpositions) - 1:
                    for line_set in zip(*row, strict=False):
                        # Render side by side
                        print("\t".join(line_set))  # noqa: T201
                    # Add labels below the grids
                    print("\t".join(labels))  # noqa: T201
                    # Add spacing between rows
                    print("\n")  # noqa: T201
                    row = []  # Reset the row buffer
                    labels = []  # Reset the labels buffer

            if len(superpositions) < config.max_superpositions:
                time.sleep(config.render_sleep_seconds)  # Wait before the next render

            # Stop if all superpositions have reached their goal
            if all(env_copy.reached_goal() for _, env_copy, _ in superpositions):
                msg = "All superpositions have reached their goal."
                logger.info(msg)
                print(msg)  # noqa: T201
                # Return the path from the first superposition (primary path)
                return EpisodeResult(
                    agent_path=superpositions[0][2],
                    termination_reason=TerminationReason.GOAL_REACHED,
                )
        msg = "Many-worlds mode completed as maximum number of steps reached."
        logger.info(msg)
        print(msg)  # noqa: T201
        # Return the path from the first superposition (primary path)
        return EpisodeResult(
            agent_path=superpositions[0][2],
            termination_reason=TerminationReason.MAX_STEPS,
        )
