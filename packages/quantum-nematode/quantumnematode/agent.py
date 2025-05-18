"""The quantum nematode agent that navigates a grid environment using a quantum brain."""

import os
import sys
import time

import numpy as np  # pyright: ignore[reportMissingImports]

from quantumnematode.constants import (
    SUPERPOSITION_MODE_MAX_COLUMNS,
    SUPERPOSITION_MODE_MAX_SUPERPOSITIONS,
    SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS,
    SUPERPOSITION_MODE_TOP_N_ACTIONS,
    SUPERPOSITION_MODE_TOP_N_RANDOMIZE,
)
from quantumnematode.models import ActionData

from .brain._brain import Brain, BrainParams
from .env import MazeEnvironment
from .logging_config import logger

PENALTY_STAY = 0
REWARD_GOAL = 0.1
REWARD_GOAL_PROXIMITY_FACTOR = 2


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

    def run_episode(  # noqa: C901, PLR0912, PLR0915
        self,
        max_steps: int = 100,
        render_text: str | None = None,
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

            print()  # noqa: T201
            print(f"Gradient strength: {gradient_strength}")  # noqa: T201
            print(f"Gradient direction: {gradient_direction}")  # noqa: T201

            # Calculate reward based on efficiency and collision avoidance
            reward = self.calculate_reward(self.env, self.path, max_steps=max_steps)

            print(f"Reward: {reward}")  # noqa: T201

            # Prepare input_data for data re-uploading (one float per qubit)
            input_data = None
            if hasattr(self.brain, "num_qubits"):
                input_data = [float(gradient_strength)] * int(getattr(self.brain, "num_qubits", 1))

            # Fix agent_position type for BrainParams (must be exactly 2 floats)
            agent_pos = tuple(float(x) for x in self.env.agent_pos[:2])
            if len(agent_pos) != 2:  # noqa: PLR2004
                agent_pos = (float(self.env.agent_pos[0]), float(self.env.agent_pos[1]))

            params = BrainParams(
                gradient_strength=gradient_strength,
                gradient_direction=gradient_direction,
                agent_position=agent_pos,
                agent_direction=self.env.current_direction,
            )
            # Only pass input_data if supported by run_brain (DynamicBrain)
            if (
                hasattr(self.brain, "run_brain")
                and "input_data" in self.brain.run_brain.__code__.co_varnames
            ):
                counts = self.brain.run_brain(
                    params=params,
                    reward=reward,
                    input_data=input_data,
                )
            else:
                counts = self.brain.run_brain(
                    params=params,
                    reward=reward,
                )

            action = self.brain.interpret_counts(counts, top_only=True, top_randomize=True)

            if not isinstance(action, ActionData):
                error_msg = f"Invalid action type: {type(action)}. Expected ActionData."
                logger.error(error_msg)
                raise TypeError(error_msg)

            self.env.move_agent(action.action)

            # Update the body length dynamically
            if self.max_body_length > 0 and len(self.env.body) < self.max_body_length:
                self.env.body.append(self.env.body[-1])

            self.brain.update_memory(reward)

            self.path.append(tuple(self.env.agent_pos))
            self.steps += 1

            logger.info(f"Step {self.steps}: Action={action.action}, Reward={reward}")

            if self.env.reached_goal():
                # Run the brain with the final state and reward
                reward = self.calculate_reward(self.env, self.path, max_steps=max_steps)

                # Prepare input_data for data re-uploading (one float per qubit)
                input_data = None
                if hasattr(self.brain, "num_qubits"):
                    input_data = [float(gradient_strength)] * int(
                        getattr(self.brain, "num_qubits", 1),
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
                counts = self.brain.run_brain(
                    params=params,
                    reward=reward,
                )

                # Calculate reward based on efficiency and collision avoidance
                self.brain.update_memory(reward)

                self.brain.satiety = 1.0  # Set satiety to maximum

                # Reset the brain's history (currently only for DynamicBrain)
                if hasattr(self.brain, "history_params"):
                    self.brain.history_params = []  # type: ignore[assignment]
                if hasattr(self.brain, "history_gradients"):
                    self.brain.history_gradients = []  # type: ignore[assignment]

                self.path.append(tuple(self.env.agent_pos))
                self.steps += 1

                logger.info(f"Step {self.steps}: Action={action.action}, Reward={reward}")

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

            print(f"Step:\t{self.steps}/{max_steps}")  # noqa: T201
            print(f"Wins:\t{self.success_count}")  # noqa: T201

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

        time.sleep(SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS)  # Wait before the next render

        logger.info(
            "Superposition mode enabled. Visualizing top "
            "{SUPERPOSITION_MODE_TOP_N_ACTIONS} decisions at each step.",
        )
        superpositions = [(self.brain.copy(), self.env.copy(), self.path.copy())]

        for _ in range(max_steps):
            total_superpositions = len(superpositions)
            i = 0
            for brain_copy, env_copy, path_copy in superpositions:
                gradient_strength, gradient_direction = env_copy.get_state(path_copy[-1])
                reward = self.calculate_reward(env_copy, path_copy, max_steps=max_steps)
                params = BrainParams(
                    gradient_strength=gradient_strength,
                    gradient_direction=gradient_direction,
                    agent_position=self.env.agent_pos,
                    agent_direction=self.env.current_direction,
                )
                counts = brain_copy.run_brain(
                    params=params,
                    reward=reward,
                )
                actions = brain_copy.interpret_counts(counts, top_only=False)

                def get_action_and_prob(a: ActionData) -> tuple:
                    if hasattr(a, "action") and hasattr(a, "probability"):
                        return a.action, a.probability
                    if isinstance(a, tuple) and len(a) >= 2:  # noqa: PLR2004
                        return a[0], a[1]
                    return None, None

                if SUPERPOSITION_MODE_TOP_N_RANDOMIZE:
                    rng = np.random.default_rng()
                    filtered_actions = [
                        a
                        for a in actions
                        if isinstance(get_action_and_prob(a)[1], float)
                        and get_action_and_prob(a)[0] is not None
                    ]
                    if filtered_actions:
                        probs = [get_action_and_prob(a)[1] for a in filtered_actions]
                        # Remove None values from probs and corresponding actions
                        filtered_pairs = [
                            (a, p)
                            for a, p in zip(filtered_actions, probs, strict=False)
                            if p is not None
                        ]
                        if filtered_pairs:
                            filtered_actions, probs = zip(*filtered_pairs, strict=False)
                            probs = np.array(probs, dtype=float)
                            probs_sum = probs.sum()
                            if probs_sum > 0:
                                norm_probs = probs / probs_sum
                            else:
                                norm_probs = np.ones_like(probs) / len(probs)
                            filtered_actions = np.array(filtered_actions)
                            top_actions_and_probs = rng.choice(
                                filtered_actions,
                                p=norm_probs,
                                size=SUPERPOSITION_MODE_TOP_N_ACTIONS,
                                replace=True,
                            )
                            top_actions = [
                                get_action_and_prob(a)[0]
                                for a in top_actions_and_probs
                                if get_action_and_prob(a)[0] is not None
                            ]
                        else:
                            top_actions = []
                    else:
                        top_actions = []
                else:
                    top_actions = [
                        get_action_and_prob(a)[0]
                        for a in actions
                        if get_action_and_prob(a)[0] is not None
                    ][:SUPERPOSITION_MODE_TOP_N_ACTIONS]

                # Defensive: filter out None from top_actions
                top_actions = [a for a in top_actions if a is not None]

                # Update the body length dynamically
                if self.max_body_length > 0 and len(env_copy.body) < self.max_body_length:
                    env_copy.body.append(env_copy.body[-1])

                if len(superpositions) < SUPERPOSITION_MODE_MAX_SUPERPOSITIONS and top_actions:
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

            if len(superpositions) < SUPERPOSITION_MODE_MAX_SUPERPOSITIONS:
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
        env: MazeEnvironment,
        path: list[tuple[int, ...]],
        max_steps: int,
    ) -> float:
        """
        Calculate reward based on the agent's current state using gradient strength.

        Returns
        -------
        float
            Reward value based on the agent's performance.
        """
        reward = 0.0

        # Get the current gradient strength from the environment
        gradient_strength, _ = env.get_state(path[-1])

        # Calculate the change in gradient strength since the last step
        previous_gradient_strength = None
        if len(path) > 1:
            previous_gradient_strength, _ = env.get_state(path[-2])
            gradient_change = gradient_strength - previous_gradient_strength
        else:
            gradient_change = 0

        # Enhance reward signal for gradient improvement and vice versa
        if previous_gradient_strength is not None:
            if gradient_change > 0:
                reward_amount = gradient_strength / REWARD_GOAL_PROXIMITY_FACTOR
                reward += reward_amount
                logger.debug(f"[Reward] Gradient improvement reward applied: {reward_amount}.")
            elif gradient_change < 0:
                penalty_amount = -(gradient_strength / REWARD_GOAL_PROXIMITY_FACTOR)
                reward += penalty_amount
                logger.debug(f"[Penalty] Gradient weakening penalty applied: {penalty_amount}.")

        # Strengthen penalties for no movements
        if PENALTY_STAY != 0 and len(path) > 1 and path[-1] == path[-2]:
            penalty_amount = PENALTY_STAY
            reward += penalty_amount
            logger.debug(f"[Penalty] No movement penalty applied: {penalty_amount}.")

        # Reward efficient paths by scaling inversely with steps
        efficiency_factor = None
        if env.reached_goal():
            efficiency_factor = max(0.1, 1 - (self.steps / max_steps))  # Scale inversely with steps
            reward_amount = REWARD_GOAL * efficiency_factor
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
