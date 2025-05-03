"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os
import sys
import time

from quantumnematode.constants import (
    SUPERPOSITION_MODE_MAX_COLUMNS,
    SUPERPOSITION_MODE_MAX_SUPERPOSITIONS,
    SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS,
    SUPERPOSITION_MODE_TOP_N_ACTIONS,
)

from .brain._brain import Brain
from .env import MazeEnvironment
from .logging_config import logger

PENALTY_STAY = -0.5
PENALTY_STEP = -0.1
REWARD_GOAL = 50


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

    def __init__(self, brain: Brain, maze_grid_size: int = 5, max_body_length: int = 6) -> None:
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
        self.env = MazeEnvironment(grid_size=maze_grid_size, max_body_length=max_body_length)
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        self.max_body_length = min(
            maze_grid_size - 1,
            max_body_length,
        )  # Set the maximum body length
        self.success_count = 0
        self.total_steps = 0
        self.total_rewards = 0

    def run_episode(  # noqa: C901
        self,
        max_steps: int = 100,
        *,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
        """
        Run a single episode of the simulation.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps for the episode.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation, by default False.

        Returns
        -------
        list[tuple]
            The path taken by the agent during the episode.
        """
        self.env.current_direction = "up"  # Initialize the agent's direction

        for _ in range(max_steps):
            logger.debug("--- New Step ---")
            gradient_strength, gradient_direction = self.env.get_state(self.path[-1])
            reward = self.calculate_reward(max_steps=max_steps)
            counts = self.brain.run_brain(gradient_strength, gradient_direction, reward=reward)
            action = self.brain.interpret_counts(counts)

            if not isinstance(action, str):
                error_msg = f"Invalid action type: {type(action)}. Expected str."
                logger.error(error_msg)
                raise TypeError(error_msg)

            self.env.move_agent(action)

            # Update the body length dynamically
            if self.max_body_length > 0 and len(self.env.body) < self.max_body_length:
                self.env.body.append(self.env.body[-1])

            # Calculate reward based on efficiency and collision avoidance
            self.brain.update_memory(reward)

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={action}, Reward={reward}")

            if action == "unknown":
                logger.warning("Invalid action received: staying in place.")

            if self.env.reached_goal():
                # Run the brain with the final state and reward
                reward = self.calculate_reward(max_steps=max_steps)
                counts = self.brain.run_brain(gradient_strength, gradient_direction, reward=reward)

                # Calculate reward based on efficiency and collision avoidance
                self.brain.update_memory(reward)

                self.brain.satiety = 1.0  # Set satiety to maximum

                self.path.append(tuple(self.env.agent_pos))
                self.steps += 1

                logger.info(f"Step {self.steps}: Action={action}, Reward={reward}")

                self.total_rewards += reward
                logger.info("Reward: goal reached!")
                self.success_count += 1
                break

            self.total_steps += 1
            self.total_rewards += reward

            # Log action counts for debugging
            logger.debug(
                f"Sorted action counts: {sorted(counts.items(), key=lambda x: x[1], reverse=True)}",
            )

            # Log distance to the goal
            if self.env.goal is not None:
                distance_to_goal = abs(self.env.agent_pos[0] - self.env.goal[0]) + abs(
                    self.env.agent_pos[1] - self.env.goal[1],
                )
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

        return self.path

    def run_superposition_mode(  # noqa: C901, PLR0912, PLR0915
        self,
        max_steps: int = 100,
        *,
        show_last_frame_only: bool = False,
    ) -> list[tuple]:
        """
        Run the agent in superposition mode.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps for the episode.
        show_last_frame_only : bool, optional
            Whether to show only the last frame of the simulation, by default False.

        Returns
        -------
        list[tuple]
            The path taken by the agent during the episode.
        """
        # Initialize superposition mode
        self.env.current_direction = "up"

        logger.info(
            "Superposition mode enabled. Visualizing top "
            "{SUPERPOSITION_MODE_TOP_N_ACTIONS} decisions at each step.",
        )
        superpositions = [(self.brain.copy(), self.env.copy(), self.path.copy())]

        new_superpositions = []
        for _ in range(max_steps):
            for brain_copy, env_copy, path_copy in superpositions:
                gradient_strength, gradient_direction = env_copy.get_state(path_copy[-1])
                reward = self.calculate_reward(max_steps=max_steps)
                counts = brain_copy.run_brain(gradient_strength, gradient_direction, reward=reward)
                actions = brain_copy.interpret_counts(counts, best_only=False)
                top_actions = list(dict.fromkeys([key for key, _ in actions]))[
                    :SUPERPOSITION_MODE_TOP_N_ACTIONS
                ]

                # Update the body length dynamically
                if self.max_body_length > 0 and len(env_copy.body) < self.max_body_length:
                    env_copy.body.append(env_copy.body[-1])

                for action in top_actions:
                    if len(new_superpositions) < SUPERPOSITION_MODE_MAX_SUPERPOSITIONS:
                        new_env = env_copy.copy()
                        new_path = path_copy.copy()
                        new_brain = self.brain.copy()
                        new_env.move_agent(action)
                        new_path.append(new_env.agent_pos)
                        new_superpositions.append((new_brain, new_env, new_path))
                    else:
                        # Stop splitting and let existing superpositions finish
                        if env_copy.reached_goal():
                            continue
                        env_copy.move_agent(action)
                        path_copy.append(env_copy.agent_pos)

            superpositions = new_superpositions

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
                label = f"#{i + 1} <= #{i // 2 + 1}"
                row.append(grid)
                labels.append(label)

                # Print the row when reaching MAX_COLUMNS or the last grid
                if (i + 1) % SUPERPOSITION_MODE_MAX_COLUMNS == 0 or i == len(superpositions) - 1:
                    for line_set in zip(*row, strict=False):
                        # Render side by side
                        print("\t".join(line_set))  # noqa: T201
                    # Add labels below the grids
                    print("\t".join(labels))  # noqa: T201
                    # Add spacing between rows
                    print("\n")  # noqa: T201
                    row = []  # Reset the row buffer
                    labels = []  # Reset the labels buffer

            time.sleep(SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS)  # Wait before the next render

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

    def calculate_reward(self, max_steps: int) -> float:
        """
        Calculate reward based on the agent's current state using gradient strength.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        reward = 0.0

        # Get the current gradient strength from the environment
        gradient_strength, _ = self.env.get_state(self.path[-1])

        # Calculate the change in gradient strength since the last step
        previous_gradient_strength = None
        if len(self.path) > 1:
            previous_gradient_strength, _ = self.env.get_state(self.path[-2])
            gradient_change = gradient_strength - previous_gradient_strength
        else:
            gradient_change = 0

        # Enhance reward signal for gradient improvement and vice versa
        if previous_gradient_strength is not None:
            if gradient_change > 0:
                reward_amount = gradient_strength
                reward += reward_amount
                logger.debug(f"[Reward] Gradient improvement reward applied: {reward_amount}.")
            elif gradient_change < 0:
                penalty_amount = -(gradient_strength / 2)
                reward += penalty_amount
                logger.debug(f"[Penalty] Gradient weakening penalty applied: {penalty_amount}.")

        # Strengthen penalties for no movements
        if len(self.path) > 1 and self.path[-1] == self.path[-2]:
            penalty_amount = PENALTY_STAY * 3
            reward += penalty_amount
            logger.debug(f"[Penalty] No movement penalty applied: {penalty_amount}.")
        # Strengthen penalties for revisiting positions
        elif self.path.count(tuple(self.env.agent_pos)) > 1:
            penalty_amount = PENALTY_STEP * 3
            reward += penalty_amount
            logger.debug(f"[Penalty] Revisit penalty applied: {penalty_amount}.")

        # Reward efficient paths by scaling inversely with steps
        efficiency_factor = None
        if self.env.reached_goal():
            efficiency_factor = max(0.1, 1 - (self.steps / max_steps))  # Scale inversely with steps
            reward_amount = REWARD_GOAL * efficiency_factor * 2
            reward += reward_amount  # Further scale goal reward dynamically based on speed
            logger.debug(f"[Reward] Goal reached, efficiency factor applied: {reward_amount}.")

        logger.debug(
            f"Gradient strength: {gradient_strength}, "
            f"Gradient change: {gradient_change}, Reward: {reward}",
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
        )
        self.steps = 0
        self.path = [tuple(self.env.agent_pos)]
        logger.info("Environment reset. Retaining learned data.")

    def calculate_metrics(self, total_runs: int) -> dict:
        """
        Calculate and return performance metrics.

        Returns
        -------
        dict
            A dictionary containing success rate, average steps, and average reward.
        """
        return {
            "success_rate": self.success_count / total_runs,
            "average_steps": self.total_steps / total_runs,
            "average_reward": self.total_rewards / total_runs,
        }
