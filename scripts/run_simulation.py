"""Run the Quantum Nematode simulation."""

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from quantumnematode.agent import (  # pyright: ignore[reportMissingImports]
    QuantumNematodeAgent,
)
from quantumnematode.constants import MIN_GRID_SIZE  # pyright: ignore[reportMissingImports]
from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)
from quantumnematode.report import summary  # pyright: ignore[reportMissingImports]

DEFAULT_QUBITS = 2

# Feature flags
# NOTE: Pausing is not implemented properly yet.
TOGGLE_PAUSE = os.getenv("TOGGLE_PAUSE", "False").lower() == "false"


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the Quantum Nematode simulation."""
    parser = argparse.ArgumentParser(description="Run the Quantum Nematode simulation.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps for the simulation (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use 'NONE' to disable logging.",
    )
    parser.add_argument(
        "--maze-grid-size",
        type=int,
        default=5,
        help="Size of the maze grid (minimum: 5, default: 5)",
    )
    parser.add_argument(
        "--show-last-frame-only",
        action="store_true",
        help="Only display the last frame in the CLI output.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of simulation runs to perform (default: 1)",
    )
    parser.add_argument(
        "--brain",
        type=str,
        choices=["simple", "complex", "reduced", "memory", "dynamic"],
        default="simple",
        help="Choose the quantum brain architecture to use (default: simple)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use for AerSimulator",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="Number of shots for the AerSimulator",
    )
    parser.add_argument(
        "--body-length",
        type=int,
        default=0,
        help="Length of the agent's body, excluding head (default: 0)",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=DEFAULT_QUBITS,
        help="Number of qubits to use in the quantum brain (default: 2). "
        "Only supported by DynamicBrain.",
    )

    args = parser.parse_args()

    if args.maze_grid_size < MIN_GRID_SIZE:
        error_message = (
            f"Grid size must be at least {MIN_GRID_SIZE}. "
            f"Provided grid size: {args.maze_grid_size}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Configure logging level
    if args.log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(args.log_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(args.log_level)

    # Set up the timestamp for saving results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger.info(f"Session ID: {timestamp}")

    # Pass the device and shots arguments to the brain classes
    device = args.device.upper()
    shots = args.shots

    logger.info("Simulation parameters:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # Select the brain architecture
    if args.brain == "simple":
        from quantumnematode.brain.simple import (  # pyright: ignore[reportMissingImports]
            SimpleBrain,
        )

        brain = SimpleBrain(device=device, shots=shots)
    elif args.brain == "complex":
        from quantumnematode.brain.complex import (  # pyright: ignore[reportMissingImports]
            ComplexBrain,
        )

        if device != "CPU":
            logger.warning(
                "ComplexBrain is not optimized for GPU. Using CPU instead.",
            )
        brain = ComplexBrain(device=device, shots=shots)
    elif args.brain == "reduced":
        from quantumnematode.brain.reduced import (  # pyright: ignore[reportMissingImports]
            ReducedBrain,
        )

        brain = ReducedBrain(device=device, shots=shots)
    elif args.brain == "memory":
        from quantumnematode.brain.memory import (  # pyright: ignore[reportMissingImports]
            MemoryBrain,
        )

        brain = MemoryBrain(device=device, shots=shots)
    elif args.brain == "dynamic":
        from quantumnematode.brain.dynamic import (  # pyright: ignore[reportMissingImports]
            DynamicBrain,
        )

        brain = DynamicBrain(device=device, shots=shots, num_qubits=args.qubits)
    else:
        error_message = f"Unknown brain architecture: {args.brain}"
        raise ValueError(error_message)

    if args.brain != "dynamic" and args.qubits != DEFAULT_QUBITS:
        error_message = (
            f"The 'qubits' parameter is only supported by the DynamicBrain architecture. "
            f"Provided brain: {args.brain}, qubits: {args.qubits}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Update the agent to use the selected brain architecture
    agent = QuantumNematodeAgent(
        maze_grid_size=args.maze_grid_size,
        brain=brain,
        max_body_length=args.body_length,
    )

    # Initialize tracking variables for plotting
    tracking_data = {
        "run": [],
        "input_parameters": [],
        "computed_gradients": [],
        "learning_rate": [],
        "updated_parameters": [],
        "exploration_factor": [],
        "temperature": [],
    }

    all_results = []

    total_runs_done = 0

    try:
        for run in range(total_runs_done, args.runs):
            logger.info(f"Starting run {run + 1} of {args.runs}")
            path = agent.run_episode(
                max_steps=args.max_steps,
                show_last_frame_only=args.show_last_frame_only,
            )

            steps = len(path)
            total_reward = sum(
                agent.env.get_state(pos, disable_log=True)[0] for pos in path
            )  # Calculate total reward for the run
            all_results.append(
                (run + 1, steps, path, total_reward, agent.total_rewards),
            )  # Include total reward in results

            logger.info(f"Run {run + 1}/{args.runs} completed in {steps} steps.")

            if run < args.runs - 1:
                agent.reset_environment()

            total_runs_done += 1

            # Track data for plotting, only supported for dynamic brain
            if args.brain == "dynamic":
                tracking_data["run"].append(run + 1)
                tracking_data["input_parameters"].append(agent.brain.latest_input_parameters)
                tracking_data["computed_gradients"].append(agent.brain.latest_gradients)
                tracking_data["learning_rate"].append(agent.brain.latest_learning_rate)
                tracking_data["updated_parameters"].append(agent.brain.latest_updated_parameters)
                tracking_data["exploration_factor"].append(agent.brain.latest_exploration_factor)
                tracking_data["temperature"].append(agent.brain.latest_temperature)

    except KeyboardInterrupt:
        if TOGGLE_PAUSE == "true":
            resume_from = manage_simulation_pause(
                args,
                timestamp,
                agent,
                all_results,
                total_runs_done,
                tracking_data,
            )
            if resume_from == -1:
                return
            total_runs_done = resume_from
            main()  # Restart the main function to re-enter the loop
        else:
            logger.info("KeyboardInterrupt detected. Exiting the simulation.")
            return

    # Calculate and log performance metrics
    metrics = agent.calculate_metrics(total_runs=total_runs_done)
    logger.info("\nPerformance Metrics:")
    logger.info(f"Success Rate: {metrics['success_rate']:.2f}")
    logger.info(f"Average Steps: {metrics['average_steps']:.2f}")
    logger.info(f"Average Reward: {metrics['average_reward']:.2f}")

    # Final summary of all runs.
    summary(total_runs_done, args.max_steps, all_results)

    # Generate plots after the simulation
    plot_results(all_results, metrics, timestamp, args.max_steps)

    # Generate additional plots for tracking data
    if args.brain == "dynamic":
        plot_tracking_data(tracking_data, timestamp, args.brain, args.qubits)

    return


def manage_simulation_pause(  # noqa: PLR0913
    args: argparse.Namespace,
    timestamp: str,
    agent: QuantumNematodeAgent,
    all_results: list[tuple[int, int, list[tuple[int, int]], float, float]],
    total_runs_done: int,
    tracking_data: dict[str, list],
) -> int:
    """
    Handle simulation pause triggered by a KeyboardInterrupt.

    This function provides options to the user to either continue the simulation,
    output partial results and plots, print circuit details, or exit the session.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        timestamp (str): Timestamp for the current session.
        agent (QuantumNematodeAgent): The simulation agent.
        all_results (list[tuple[int, int, list[tuple[int, int]], float, float]]):
            List of results for each run, including run number, steps, path,
            total reward, and cumulative rewards.
        total_runs_done (int): Total number of runs completed so far.
        tracking_data (dict[str, list]): Data tracked during the simulation for plotting.

    Returns
    -------
        int: The run number to resume from, or -1 to exit.
    """
    while True:
        prompt_intro_message = (
            "KeyboardInterrupt detected. The simulation is paused. "
            "You can choose to continue or output the results up to this point."
        )
        logger.warning(prompt_intro_message)
        print(prompt_intro_message)  # noqa: T201
        print("1. Continue")  # noqa: T201
        print("2. Output the summary, plots, and tracking until this point in time.")  # noqa: T201
        print("3. Print the circuit's details.")  # noqa: T201
        print("4. Exit")  # noqa: T201

        try:
            choice = int(input("Enter your choice (1-4): "))
        except ValueError:
            logger.error("Invalid input. Please enter a number between 1 and 4.")
            continue
        except KeyboardInterrupt:
            continue

        if choice == 1:
            logger.info("Resuming the session.")
            return total_runs_done
        if choice == 2:  # noqa: PLR2004
            logger.info("Generating partial results and plots.")
            metrics = agent.calculate_metrics(total_runs=total_runs_done)
            logger.info("\nPerformance Metrics:")
            logger.info(f"Success Rate: {metrics['success_rate']:.2f}")
            logger.info(f"Average Steps: {metrics['average_steps']:.2f}")
            logger.info(f"Average Reward: {metrics['average_reward']:.2f}")

            # Generate partial summary
            summary(total_runs_done, args.max_steps, all_results)

            # Generate plots with current timestamp
            file_prefix = f"{total_runs_done}_"
            plot_results(all_results, metrics, timestamp, args.max_steps, file_prefix=file_prefix)

            if args.brain == "dynamic":
                plot_tracking_data(
                    tracking_data,
                    timestamp,
                    args.brain,
                    args.qubits,
                    file_prefix=file_prefix,
                )
        elif choice == 3:  # noqa: PLR2004
            logger.info("Printing circuit details.")
            circuit = agent.brain.inspect_circuit()
            logger.info(f"Circuit details:\n{circuit}")
            print(circuit)  # noqa: T201
        elif choice == 4:  # noqa: PLR2004
            logger.info("Exiting the session.")
            return -1
        else:
            logger.error("Invalid choice. Please enter a number between 1 and 4.")


def plot_results(
    all_results: list[tuple[int, int, list[tuple[int, int]], float, float]],
    metrics: dict[str, float],
    timestamp: str,
    max_steps: int,
    file_prefix: str = "",
) -> None:
    """Generate and save plots for the simulation results."""
    runs: list[int] = [result[0] for result in all_results]
    steps: list[int] = [result[1] for result in all_results]

    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot: Steps per Run
    plt.figure(figsize=(10, 6))
    plt.plot(runs, steps, marker="o", label="Steps per Run")
    plt.axhline(y=metrics["average_steps"], color="r", linestyle="--", label="Average Steps")
    plt.title("Steps per Run")
    plt.xlabel("Run")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}steps_per_run.png")
    plt.close()

    # Plot: Cumulative Reward per Run
    cumulative_rewards: list[float] = [
        result[3] for result in all_results
    ]  # Assuming rewards are stored as result[3]
    plt.figure(figsize=(10, 6))
    plt.plot(runs, cumulative_rewards, marker="o", label="Cumulative Reward per Run")
    plt.title("Cumulative Reward per Run")
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_reward_per_run.png")
    plt.close()

    # Plot: Last Cumulative Rewards Over Runs
    last_cumulative_rewards: list[float] = [
        result[4] for result in all_results
    ]  # Assuming rewards are stored as result[4]
    plt.figure(figsize=(10, 6))
    plt.plot(runs, last_cumulative_rewards, marker="o", label="Last Cumulative Rewards")
    plt.title("Last Cumulative Reward Over Time")
    plt.xlabel("Run")
    plt.ylabel("Last Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_last_reward_over_time.png")
    plt.close()

    # Plot: Success Rate Over Time
    success_rates: list[float] = [
        sum(1 for r in all_results[:i] if r[1] < max_steps) / i
        for i in range(1, len(all_results) + 1)
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(runs, success_rates, marker="o", label="Success Rate Over Time")
    plt.title("Success Rate Over Time")
    plt.xlabel("Run")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}success_rate_over_time.png")
    plt.close()


def plot_tracking_data(
    tracking_data: dict[str, list],
    timestamp: str,
    brain: str,
    qubits: int,
    file_prefix: str = "",
) -> None:
    """Generate and save plots for tracking data."""
    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_postfix: str = f" [{brain} {qubits}Q]" if brain == "dynamic" else f" [{brain}]"

    # Plot each tracked variable
    for key, values in tracking_data.items():
        if key == "run":
            continue

        logger.debug(f"Tracking data for {key}: {str(values).replace('θ', 'theta_')}")

        if isinstance(values, list) and all(isinstance(v, dict) for v in values):
            # Flatten dictionaries into lists of values for plotting
            title = key.replace("_", " ").title()
            label = next(list(param_dict.keys()) for param_dict in values)
            values = [list(param_dict.values()) for param_dict in values]  # noqa: PLW2901
        elif key == "computed_gradients":
            title = "Computed Gradients"
            label = [str(n + 1) for n in range(len(values[0]))]
        else:
            label = key.replace("_", " ").title()
            title = label

        title += title_postfix

        plt.figure(figsize=(10, 6))
        plt.plot(tracking_data["run"], values, marker="o", label=label)
        plt.title(title)
        plt.xlabel("Run")
        plt.ylabel(str(label))
        plt.legend()
        plt.grid()
        plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
        plt.close()


if __name__ == "__main__":
    main()
