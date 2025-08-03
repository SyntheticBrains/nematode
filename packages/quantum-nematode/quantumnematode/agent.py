"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os
import sys
import time

import numpy as np
from pydantic import BaseModel

from quantumnematode.brain.arch import ClassicalBrain, QuantumBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.report.dtypes import PerformanceMetrics
from quantumnematode.theme import DEFAULT_THEME, Theme

from .brain.arch import Brain, BrainParams
from .env import Direction, MazeEnvironment
from .logging_config import logger

# Defaults
DEFAULT_AGENT_BODY_LENGTH = 2
DEFAULT_MAX_AGENT_BODY_LENGTH = 6
DEFAULT_MAX_STEPS = 100
DEFAULT_MAZE_GRID_SIZE = 5
DEFAULT_PENALTY_ANTI_DITHERING = 0.02
DEFAULT_PENALTY_STEP = 0.005
DEFAULT_REWARD_DISTANCE_SCALE = 0.3
DEFAULT_REWARD_GOAL = 0.2
DEFAULT_SUPERPOSITION_MODE_MAX_COLUMNS = 4
DEFAULT_SUPERPOSITION_MODE_MAX_SUPERPOSITIONS = 16
DEFAULT_SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS = 1.0
DEFAULT_SUPERPOSITION_MODE_TOP_N_ACTIONS = 2
DEFAULT_SUPERPOSITION_MODE_TOP_N_RANDOMIZE = True


class RewardConfig(BaseModel):
    """Configuration for the reward function."""

    penalty_anti_dithering: float = (
        DEFAULT_PENALTY_ANTI_DITHERING  # Penalty for oscillating (revisiting previous cell)
    )
    penalty_step: float = DEFAULT_PENALTY_STEP
    reward_distance_scale: float = (
        DEFAULT_REWARD_DISTANCE_SCALE  # Scale the distance reward for smoother learning
    )
    reward_goal: float = DEFAULT_REWARD_GOAL


class SuperpositionModeConfig(BaseModel):
    """Configuration for the superposition mode."""

    max_superpositions: int = DEFAULT_SUPERPOSITION_MODE_MAX_SUPERPOSITIONS
    max_columns: int = DEFAULT_SUPERPOSITION_MODE_MAX_COLUMNS
    render_sleep_seconds: float = DEFAULT_SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS
    top_n_actions: int = DEFAULT_SUPERPOSITION_MODE_TOP_N_ACTIONS
    top_n_randomize: bool = DEFAULT_SUPERPOSITION_MODE_TOP_N_RANDOMIZE


class QuantumNematodeAgent:
    """
    Quantum nematode agent that navigates a grid environment using a quantum brain.

    Attributes
    ----------
    env : MazeEnvironment
        The grid environment for the agent.
    steps : int
        Number of steps taken by the agent.
    path : list[tuple]
        Path taken by the agent.
    body_length : int
        Maximum length of the agent's body.
    """

    def __init__(
        self,
        brain: Brain,
        maze_grid_size: int = DEFAULT_MAZE_GRID_SIZE,
        max_body_length: int = DEFAULT_MAX_AGENT_BODY_LENGTH,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        """
        Initialize the quantum nematode agent.

        Parameters
        ----------
        brain : Brain
            The quantum brain architecture used by the agent.
        maze_grid_size : int, optional
            Size of the grid environment, by default 5.
        """
        self.brain = brain
        self.env = MazeEnvironment(
            grid_size=maze_grid_size,
            max_body_length=max_body_length,
            theme=theme,
        )
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.max_body_length = min(
            maze_grid_size - 1,
            max_body_length,
        )  # Set the maximum body length
        self.success_count = 0
        self.total_steps = 0
        self.total_rewards = 0

    def run_episode(  # noqa: C901, PLR0912, PLR0915
        self,
        reward_config: RewardConfig,
        max_steps: int = DEFAULT_MAX_STEPS,
        render_text: str | None = None,
        *,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
        """
        Run a single episode of the simulation.

        Parameters
        ----------
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation, by default False.

        Returns
        -------
        list[tuple]
            The path taken by the agent during the episode.
        """
        self.env.current_direction = Direction.UP  # Initialize the agent's direction

        reward = 0.0
        top_action = None
        for _ in range(max_steps):
            logger.debug("--- New Step ---")
            gradient_strength, gradient_direction = self.env.get_state(self.path[-1])

            if logger.isEnabledFor(logging.DEBUG):
                print()  # noqa: T201
                print(f"Gradient strength: {gradient_strength}")  # noqa: T201
                print(f"Gradient direction: {gradient_direction}")  # noqa: T201

            # Calculate reward based on efficiency and collision avoidance
            reward = self.calculate_reward(
                reward_config,
                self.env,
                self.path,
                max_steps=max_steps,
            )

            if logger.isEnabledFor(logging.DEBUG):
                print(f"Reward: {reward}")  # noqa: T201

            # Prepare input_data for data re-uploading (one float per qubit)
            input_data = None
            if isinstance(self.brain, QuantumBrain):
                input_data = [float(gradient_strength)] * self.brain.num_qubits

            # Fix agent_position type for BrainParams (must be exactly 2 floats)
            agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
            if len(agent_pos) != 2:  # noqa: PLR2004
                agent_pos = (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))

            params = BrainParams(
                gradient_strength=gradient_strength,
                gradient_direction=gradient_direction,
                agent_position=agent_pos,
                agent_direction=self.env.current_direction,
                action=top_action,
            )
            action = self.brain.run_brain(
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

            self.env.move_agent(top_action.action)

            # Learning step
            if isinstance(self.brain, ClassicalBrain):
                action_idx = self.brain.action_set.index(top_action.action)
                self.brain.learn(
                    params=params,
                    action_idx=action_idx,
                    reward=reward,
                    episode_rewards=None,
                )

            # Update the body length dynamically
            if self.max_body_length > 0 and len(self.env.body) < self.max_body_length:
                self.env.body.append(self.env.body[-1])

            self.brain.update_memory(reward)

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={top_action.action.value}, Reward={reward}")

            if self.env.reached_goal():
                # Run the brain with the final state and reward
                reward = self.calculate_reward(
                    reward_config,
                    self.env,
                    self.path,
                    max_steps=max_steps,
                )

                # Prepare input_data for data re-uploading (one float per qubit)
                input_data = None
                if isinstance(self.brain, QuantumBrain):
                    input_data = [float(gradient_strength)] * self.brain.num_qubits

                agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
                if len(agent_pos) != 2:  # noqa: PLR2004
                    agent_pos = (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))

                params = BrainParams(
                    gradient_strength=gradient_strength,
                    gradient_direction=gradient_direction,
                    agent_position=agent_pos,
                    agent_direction=self.env.current_direction,
                    action=top_action,
                )
                _ = self.brain.run_brain(
                    params=params,
                    reward=reward,
                    input_data=None,
                    top_only=True,
                    top_randomize=True,
                )

                # Calculate reward based on efficiency and collision avoidance
                self.brain.update_memory(reward)

                self.brain.satiety = 1.0  # Set satiety to maximum

                self.path.append(tuple(self.env.agent_pos))
                self.steps += 1

                logger.info(f"Step {self.steps}: Action={top_action.action.value}, Reward={reward}")

                self.total_rewards += reward
                logger.info("Reward: goal reached!")
                self.success_count += 1
                break

            self.total_steps += 1
            self.total_rewards += reward

            # Log distance to the goal
            if self.env.goal is not None:
                distance_to_goal = self.calculate_goal_distance()
                logger.debug(f"Distance to goal: {distance_to_goal}")

            # Log cumulative reward and average reward per step at the end of each run
            if self.steps > 0:
                average_reward = self.total_rewards / self.steps
                logger.info(
                    f"Cumulative reward: {self.total_rewards}, "
                    f"Average reward per step: {average_reward}",
                )

            if show_last_frame_only:
                if os.name == "nt":  # For Windows
                    os.system("cls")  # noqa: S605, S607
                else:  # For macOS and Linux
                    os.system("clear")  # noqa: S605, S607

            grid = self.env.render()
            for frame in grid:
                print(frame)  # noqa: T201
                logger.debug(frame)

            if render_text:
                print(render_text)  # noqa: T201

            print(f"Step:\t\t{self.steps}/{max_steps}")  # noqa: T201
            print(f"Wins:\t\t{self.success_count}")  # noqa: T201

        return self.path

    def run_superposition_mode(  # noqa: C901, PLR0912, PLR0915
        self,
        config: SuperpositionModeConfig,
        reward_config: RewardConfig,
        max_steps: int = DEFAULT_MAX_STEPS,
        *,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
        """
        Run the agent in superposition mode.

        Parameters
        ----------
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        max_superpositions : int
            Maximum number of superpositions to maintain.
        max_columns : int
            Maximum number of columns to render side by side.
        render_sleep_seconds : float
            Seconds to wait before rendering the next frame.
        top_n_actions : int
            Number of top actions to consider at each step.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation.
        top_n_randomize : bool, optional
            Whether to randomize the top N actions.

        Returns
        -------
        list[tuple]
            The path taken by the agent during the episode.
        """
        # Initialize superposition mode
        self.env.current_direction = Direction.UP

        if show_last_frame_only:
            if os.name == "nt":  # For Windows
                os.system("cls")  # noqa: S605, S607
            else:  # For macOS and Linux
                os.system("clear")  # noqa: S605, S607

        # Render the initial grid
        grid = self.env.render()
        for frame in grid:
            print(frame)  # noqa: T201
            logger.debug(frame)
        print("#1")  # noqa: T201

        time.sleep(config.render_sleep_seconds)  # Wait before the next render

        logger.info(
            "Superposition mode enabled. "
            f"Visualizing top {config.top_n_actions} decisions at each step.",
        )
        superpositions = [(self.brain.copy(), self.env.copy(), self.path.copy())]

        reward = 0.0
        for _ in range(max_steps):
            total_superpositions = len(superpositions)
            i = 0
            for brain_copy, env_copy, path_copy in superpositions:
                gradient_strength, gradient_direction = env_copy.get_state(path_copy[-1])
                reward = self.calculate_reward(
                    reward_config,
                    env_copy,
                    path_copy,
                    max_steps=max_steps,
                )

                agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
                if len(agent_pos) != 2:  # noqa: PLR2004
                    agent_pos = (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))

                params = BrainParams(
                    gradient_strength=gradient_strength,
                    gradient_direction=gradient_direction,
                    agent_position=agent_pos,
                    agent_direction=self.env.current_direction,
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
                if self.max_body_length > 0 and len(env_copy.body) < self.max_body_length:
                    env_copy.body.append(env_copy.body[-1])

                if len(superpositions) < config.max_superpositions and top_actions:
                    new_env = env_copy.copy()
                    new_path = path_copy.copy()
                    new_brain = self.brain.copy()
                    runner_up_action = top_actions[1] if len(top_actions) > 1 else top_actions[0]
                    if runner_up_action is not None:
                        new_env.move_agent(runner_up_action)
                        new_brain.update_memory(reward)
                        new_path.append(new_env.agent_pos)
                        superpositions.append((new_brain, new_env, new_path))

                if env_copy.reached_goal():
                    continue

                if top_actions:
                    env_copy.move_agent(top_actions[0])
                    brain_copy.update_memory(reward)
                    path_copy.append(env_copy.agent_pos)

                i += 1
                if i >= total_superpositions:
                    break

            self.steps += 1

            if show_last_frame_only:
                if os.name == "nt":  # For Windows
                    os.system("cls")  # noqa: S605, S607
                else:  # For macOS and Linux
                    os.system("clear")  # noqa: S605, S607

            # Render all grids for superpositions at each step
            row = []
            labels = []
            for i, (_, env_copy, _) in enumerate(superpositions):
                grid = env_copy.render()
                label = f"#{i + 1} <= #{i // 2 + 1}" if i > 0 else f"#{i + 1}      "
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
                sys.exit(0)  # Exit the program
        msg = "Superposition mode completed as maximum number of steps reached."
        logger.info(msg)
        print(msg)  # noqa: T201
        sys.exit(0)  # Exit the program

    def calculate_goal_distance(self) -> int:
        """
        Calculate the Manhattan distance to the goal.

        Returns
        -------
        int
            The Manhattan distance to the goal.
        """
        return abs(self.env.agent_pos[0] - self.env.goal[0]) + abs(
            self.env.agent_pos[1] - self.env.goal[1],
        )

    def calculate_reward(
        self,
        config: RewardConfig,
        env: MazeEnvironment,
        path: list[tuple[int, ...]],
        max_steps: int,
    ) -> float:
        """
        Calculate reward based on the agent's movement toward the goal.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        reward = 0.0
        distance_reward = 0.0
        goal_bonus = 0.0
        anti_dither_penalty = 0.0
        prev_dist = None
        curr_dist = None

        # Only compute distance-based reward if goal exists
        if env.goal is not None:
            curr_pos = env.agent_pos
            curr_dist = abs(curr_pos[0] - env.goal[0]) + abs(curr_pos[1] - env.goal[1])
            if len(path) > 1:
                prev_pos = path[-2]
                prev_dist = abs(prev_pos[0] - env.goal[0]) + abs(prev_pos[1] - env.goal[1])
                # Magnitude-based reward: positive if agent gets closer, negative if further
                distance_reward = config.reward_distance_scale * (prev_dist - curr_dist)
                reward += distance_reward
                logger.debug(
                    f"[Reward] Scaled distance reward: {distance_reward} "
                    f"(prev_dist={prev_dist}, curr_dist={curr_dist})",
                )
                # Anti-dithering: penalize if agent oscillates (returns to previous cell)
                if len(path) > 2 and curr_pos == path[-3]:  # noqa: PLR2004
                    anti_dither_penalty = config.penalty_anti_dithering
                    reward -= anti_dither_penalty
                    logger.debug(
                        f"[Penalty] Anti-dithering penalty applied: "
                        f"{-anti_dither_penalty} (oscillation detected)",
                    )
            else:
                logger.debug("[Reward] First step, no previous position for distance reward.")

        # Step penalty (applies every step)
        reward -= config.penalty_step
        logger.debug(f"[Penalty] Step penalty applied: {-config.penalty_step}.")

        # Bonus for reaching the goal, scaled by efficiency (fewer steps = higher bonus)
        if env.reached_goal():
            efficiency = max(0.1, 1 - (self.steps / max_steps))
            goal_bonus = config.reward_goal * efficiency
            reward += goal_bonus
            logger.debug(
                f"[Reward] Goal reached! Efficiency bonus applied: "
                f"{goal_bonus} (efficiency={efficiency}).",
            )

        logger.debug(
            f"Reward breakdown: distance_reward={distance_reward}, "
            f"step_penalty={-config.penalty_step}, anti_dither_penalty={-anti_dither_penalty}, "
            f"goal_bonus={goal_bonus}, total_reward={reward}",
        )
        return reward

    def reset_environment(self) -> None:
        """
        Reset the environment while retaining the agent's learned data.

        Returns
        -------
        None
        """
        self.env = MazeEnvironment(
            grid_size=self.env.grid_size,
            max_body_length=self.max_body_length,
            theme=self.env.theme,
        )
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
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

        Returns
        -------
        PerformanceMetrics
            An object containing success rate, average steps, and average reward.
        """
        return PerformanceMetrics(
            success_rate=self.success_count / total_runs,
            average_steps=self.total_steps / total_runs,
            average_reward=self.total_rewards / total_runs,
        )
