"""Run the Quantum Nematode simulation."""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from quantumnematode.agent import (  # pyright: ignore[reportMissingImports]
    QuantumNematodeAgent,
)
from quantumnematode.brain.arch import Brain  # pyright: ignore[reportMissingImports]
from quantumnematode.constants import (  # pyright: ignore[reportMissingImports]
    DEFAULT_AGENT_BODY_LENGTH,
    DEFAULT_BRAIN,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAZE_GRID_SIZE,
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    MIN_GRID_SIZE,
)
from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)
from quantumnematode.models import SimulationResult  # pyright: ignore[reportMissingImports]
from quantumnematode.optimizer.gradient_methods import (  # pyright: ignore[reportMissingImports]
    GradientCalculationMethod,
)
from quantumnematode.optimizer.learning_rate import (  # pyright: ignore[reportMissingImports]
    AdamLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)
from quantumnematode.report.plots import (  # pyright: ignore[reportMissingImports]
    plot_cumulative_reward_per_run,
    plot_efficiency_score_over_time,
    plot_last_cumulative_rewards,
    plot_steps_per_run,
    plot_success_rate_over_time,
    plot_tracking_data_per_run,
    plot_tracking_data_per_session,
)
from quantumnematode.report.summary import summary  # pyright: ignore[reportMissingImports]
from quantumnematode.utils.config_loader import (  # pyright: ignore[reportMissingImports]
    configure_gradient_method,
    configure_learning_rate,
    load_simulation_config,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Quantum Nematode simulation.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use 'NONE' to disable logging.",
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use for AerSimulator",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    # Update the argument parser to include a flag for superposition mode
    parser.add_argument(
        "--superposition",
        action="store_true",
        help="Enable superposition mode to visualize top 2 decisions at each step. "
        "Only one run is allowed in this mode.",
    )
    parser.add_argument(
        "--track-per-run",
        action="store_true",
        help="If set, output tracked DynamicBrain data as separate plots per run in subfolders.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="ascii",
        choices=["ascii", "emoji"],
        help="Maze rendering theme: 'ascii' (default) or 'emoji' for emoji-based rendering.",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the Quantum Nematode simulation."""
    args = parse_arguments()

    config_file = args.config
    superposition_mode = args.superposition
    max_steps = DEFAULT_MAX_STEPS
    maze_grid_size = DEFAULT_MAZE_GRID_SIZE
    runs = args.runs
    brain_type = DEFAULT_BRAIN
    shots = DEFAULT_SHOTS
    body_length = DEFAULT_AGENT_BODY_LENGTH
    qubits = DEFAULT_QUBITS
    device = args.device.upper()
    show_last_frame_only = args.show_last_frame_only
    log_level = args.log_level.upper()
    learning_rate = DynamicLearningRate()
    gradient_method = GradientCalculationMethod.RAW
    track_per_run = args.track_per_run

    if config_file:
        config = load_simulation_config(config_file)

        max_steps = config.get("max_steps", max_steps)
        maze_grid_size = config.get("maze_grid_size", maze_grid_size)
        brain_type = config.get("brain", brain_type)
        shots = config.get("shots", shots)
        body_length = config.get("body_length", body_length)
        qubits = config.get("qubits", qubits)

        # Load learning rate method and parameters if specified
        learning_rate = configure_learning_rate(config)

        # Load gradient method if specified
        gradient_method = configure_gradient_method(gradient_method, config)

    validate_simulation_parameters(maze_grid_size, brain_type, qubits)

    # Configure logging level
    if log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(log_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)

    # Set up the timestamp for saving results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger.info(f"Session ID: {timestamp}")

    logger.info("Simulation parameters:")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Runs: {runs}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Device: {device}")
    logger.info(f"Grid size: {maze_grid_size}")
    logger.info(f"Brain type: {brain_type}")
    logger.info(f"Body length: {body_length}")
    logger.info(f"Qubits: {qubits}")
    logger.info(f"Shots: {shots}")

    # Select the brain architecture
    brain = setup_brain_model(brain_type, shots, qubits, device, learning_rate, gradient_method)

    # Update the agent to use the selected brain architecture
    agent = QuantumNematodeAgent(
        maze_grid_size=maze_grid_size,
        brain=brain,
        max_body_length=body_length,
        theme=args.theme,
    )

    # Initialize tracking variables for plotting
    tracking_data = {
        "run": [],
        "input_parameters": [],
        "computed_gradients": [],
        "learning_rate": [],
        "updated_parameters": [],
        "temperature": [],
    }

    all_results: list[SimulationResult] = []

    total_runs_done = 0

    if superposition_mode:
        try:
            agent.run_superposition_mode(
                max_steps=max_steps,
                show_last_frame_only=show_last_frame_only,
            )
        except KeyboardInterrupt:
            message = "KeyboardInterrupt detected. Exiting the simulation."
            logger.info(message)
            print(message)  # noqa: T201
            return

    try:
        for run in range(total_runs_done, runs):
            logger.info(f"Starting run {run + 1} of {runs}")

            # Calculate the initial distance to the goal
            initial_distance = agent.calculate_goal_distance()

            render_text = f"Run:\t{run + 1}/{runs}"
            path = agent.run_episode(
                max_steps=max_steps,
                render_text=render_text,
                show_last_frame_only=show_last_frame_only,
            )

            steps = len(path)
            total_reward = sum(
                agent.env.get_state(pos, disable_log=True)[0] for pos in path
            )  # Calculate total reward for the run

            # Calculate efficiency score for the run
            steps_taken = len(path)
            efficiency_score = initial_distance - steps_taken

            logger.info(f"Efficiency Score for run {run + 1}: {efficiency_score}")

            result = SimulationResult(
                run=run + 1,
                steps=steps,
                path=path,
                total_reward=total_reward,
                last_total_reward=agent.total_rewards,
                efficiency_score=efficiency_score,
            )
            all_results.append(result)

            logger.info(f"Run {run + 1}/{runs} completed in {steps} steps.")

            total_runs_done += 1

            # Track data for plotting, only supported for dynamic and modular brains
            if brain_type in ("dynamic", "modular"):
                tracking_data["run"].append(run + 1)
                tracking_data["input_parameters"].append(agent.brain.latest_input_parameters)
                tracking_data["computed_gradients"].append(agent.brain.latest_gradients)
                tracking_data["learning_rate"].append(agent.brain.latest_learning_rate)
                tracking_data["updated_parameters"].append(agent.brain.latest_updated_parameters)
                tracking_data["temperature"].append(agent.brain.latest_temperature)

            if track_per_run and brain_type in ("dynamic", "modular"):
                plot_tracking_data_per_run(timestamp, agent, run)

            if run < runs - 1:
                agent.reset_environment()
                agent.reset_brain()

    except KeyboardInterrupt:
        manage_simulation_halt(
            max_steps,
            brain_type,
            qubits,
            timestamp,
            agent,
            all_results,
            total_runs_done,
            tracking_data,
        )

    # Calculate and log performance metrics
    metrics = agent.calculate_metrics(total_runs=total_runs_done)
    logger.info("\nPerformance Metrics:")
    logger.info(f"Success Rate: {metrics['success_rate']:.2f}")
    logger.info(f"Average Steps: {metrics['average_steps']:.2f}")
    logger.info(f"Average Reward: {metrics['average_reward']:.2f}")

    print()  # noqa: T201
    print(f"Session ID: {timestamp}")  # noqa: T201

    # Final summary of all runs.
    summary(total_runs_done, max_steps, all_results)

    # Generate plots after the simulation
    plot_results(all_results, metrics, timestamp, max_steps)

    # Generate additional plots for tracking data
    if brain_type in ("dynamic", "modular"):
        plot_tracking_data_per_session(tracking_data, timestamp, brain_type, qubits)

    return


def validate_simulation_parameters(maze_grid_size: int, brain_type: str, qubits: int) -> None:
    """
    Validate the simulation parameters to ensure they meet the required constraints.

    Args:
        maze_grid_size (int): The size of the maze grid.
        brain_type (str): The type of brain architecture being used.
        qubits (int): The number of qubits specified for the simulation.

    Raises
    ------
        ValueError: If the maze grid size is smaller than the minimum allowed size.
        ValueError: If the 'qubits' parameter is used with a brain type
            other than 'dynamic' or 'modular'.
    """
    if maze_grid_size < MIN_GRID_SIZE:
        error_message = (
            f"Grid size must be at least {MIN_GRID_SIZE}. Provided grid size: {maze_grid_size}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if brain_type not in ("dynamic", "modular") and qubits != DEFAULT_QUBITS:
        error_message = (
            f"The 'qubits' parameter is only supported by the DynamicBrain "
            "and ModularBrain architectures. "
            f"Provided brain: {brain_type}, qubits: {qubits}."
        )
        logger.error(error_message)
        raise ValueError(error_message)


def setup_brain_model(  # noqa: PLR0913
    brain_type: str,
    shots: int,
    qubits: int,
    device: str,
    learning_rate: DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,
) -> Brain:
    """
    Set up the brain model based on the specified brain type.

    Args:
        brain_type (str): The type of brain architecture to use. Options include
            "simple", "complex", "reduced", "memory", "modular" and "dynamic".
        shots (int): The number of shots for quantum circuit execution.
        qubits (int): The number of qubits to use (only applicable for "dynamic" brain).
        device (str): The device to use for simulation ("CPU" or "GPU").
        learning_rate (DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate):
            The learning rate configuration for the "dynamic" brain.

    Returns
    -------
        Brain: An instance of the selected brain model.

    Raises
    ------
        ValueError: If an unknown brain type is provided.
    """
    if brain_type == "simple":
        from quantumnematode.brain.arch.simple import (  # pyright: ignore[reportMissingImports]
            SimpleBrain,
        )

        brain = SimpleBrain(device=device, shots=shots)
    elif brain_type == "complex":
        from quantumnematode.brain.arch.complex import (  # pyright: ignore[reportMissingImports]
            ComplexBrain,
        )

        if device != "CPU":
            logger.warning(
                "ComplexBrain is not optimized for GPU. Using CPU instead.",
            )
        brain = ComplexBrain(device=device, shots=shots)
    elif brain_type == "reduced":
        from quantumnematode.brain.arch.reduced import (  # pyright: ignore[reportMissingImports]
            ReducedBrain,
        )

        brain = ReducedBrain(device=device, shots=shots)
    elif brain_type == "memory":
        from quantumnematode.brain.arch.memory import (  # pyright: ignore[reportMissingImports]
            MemoryBrain,
        )

        brain = MemoryBrain(device=device, shots=shots)
    elif brain_type == "dynamic":
        from quantumnematode.brain.arch.dynamic import (  # pyright: ignore[reportMissingImports]
            DynamicBrain,
        )

        brain = DynamicBrain(
            device=device,
            shots=shots,
            num_qubits=qubits,
            learning_rate=learning_rate,
            gradient_method=gradient_method,
        )
    elif brain_type == "modular":
        from quantumnematode.brain.arch.modular import (  # pyright: ignore[reportMissingImports]
            ModularBrain,
        )

        modules = None
        brain = ModularBrain(
            num_qubits=qubits,
            modules=modules,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
        )
    elif brain_type == "mlp":
        from quantumnematode.brain.arch.mlp import (  # pyright: ignore[reportMissingImports]
            MLPBrain,
        )

        brain = MLPBrain(
            input_dim=2,
            num_actions=4,
            lr_scheduler=True,
            device=device.lower(),
        )
    else:
        error_message = f"Unknown brain architecture: {brain_type}"
        raise ValueError(error_message)
    return brain


def manage_simulation_halt(  # noqa: PLR0913
    max_steps: int,
    brain_type: str,
    qubits: int,
    timestamp: str,
    agent: QuantumNematodeAgent,
    all_results: list[SimulationResult],
    total_runs_done: int,
    tracking_data: dict[str, list],
) -> None:
    """
    Handle simulation halt triggered by a KeyboardInterrupt.

    This function provides options to the user to either exit the simulation,
    output partial results and plots, or print circuit details.

    Args:
        max_steps (int): Maximum number of steps for the simulation.
        brain_type (str): Type of brain architecture used in the simulation.
        qubits (int): Number of qubits used in the simulation.
        args (argparse.Namespace): Parsed command-line arguments.
        timestamp (str): Timestamp for the current session.
        agent (QuantumNematodeAgent): The simulation agent.
        all_results (list[tuple[int, int, list[tuple[int, int]], float, float]]):
            List of results for each run, including run number, steps, path,
            total reward, and cumulative rewards.
        total_runs_done (int): Total number of runs completed so far.
        tracking_data (dict[str, list]): Data tracked during the simulation for plotting.
    """
    while True:
        prompt_intro_message = (
            "KeyboardInterrupt detected. The simulation has halted. "
            "You can choose to exit or output the results up to this point."
        )
        logger.warning(prompt_intro_message)
        print(prompt_intro_message)  # noqa: T201
        print("0. Exit")  # noqa: T201
        print("1. Output the summary, plots, and tracking until this point in time.")  # noqa: T201
        print("2. Print the circuit's details.")  # noqa: T201

        try:
            choice = int(input("Enter your choice (0-2): "))
        except ValueError:
            logger.error("Invalid input. Please enter a number between 0 and 2.")
            continue
        except KeyboardInterrupt:
            continue

        if choice == 0:
            logger.info("Exiting the session.")
            sys.exit(0)
        elif choice == 1:
            logger.info("Generating partial results and plots.")
            metrics = agent.calculate_metrics(total_runs=total_runs_done)
            logger.info("\nPerformance Metrics:")
            logger.info(f"Success Rate: {metrics['success_rate']:.2f}")
            logger.info(f"Average Steps: {metrics['average_steps']:.2f}")
            logger.info(f"Average Reward: {metrics['average_reward']:.2f}")

            print()  # noqa: T201
            print(f"Session ID: {timestamp}")  # noqa: T201

            # Generate partial summary
            summary(total_runs_done, max_steps, all_results)

            # Generate plots with current timestamp
            file_prefix = f"{total_runs_done}_"
            plot_results(all_results, metrics, timestamp, max_steps, file_prefix=file_prefix)

            if brain_type in ("dynamic", "modular"):
                plot_tracking_data_per_session(
                    tracking_data,
                    timestamp,
                    brain_type,
                    qubits,
                    file_prefix=file_prefix,
                )
        elif choice == 2:  # noqa: PLR2004
            logger.info("Printing circuit details.")
            circuit = agent.brain.inspect_circuit()
            logger.info(f"Circuit details:\n{circuit}")
            print(circuit)  # noqa: T201
        else:
            logger.error("Invalid choice. Please enter a number between 1 and 4.")


def plot_results(
    all_results: list[SimulationResult],
    metrics: dict[str, float],
    timestamp: str,
    max_steps: int,
    file_prefix: str = "",
) -> None:
    """Generate and save plots for the simulation results."""
    runs: list[int] = [result.run for result in all_results]
    steps: list[int] = [result.steps for result in all_results]

    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot: Steps per Run
    plot_steps_per_run(metrics, file_prefix, runs, steps, plot_dir)

    # Plot: Cumulative Reward per Run
    cumulative_rewards: list[float] = [
        result.total_reward for result in all_results
    ]  # Assuming rewards are stored as result[3]
    plot_cumulative_reward_per_run(file_prefix, runs, plot_dir, cumulative_rewards)

    # Plot: Last Cumulative Rewards Over Runs
    last_cumulative_rewards: list[float] = [
        result.last_total_reward for result in all_results
    ]  # Assuming rewards are stored as result[4]
    plot_last_cumulative_rewards(file_prefix, runs, plot_dir, last_cumulative_rewards)

    # Plot: Success Rate Over Time
    success_rates: list[float] = [
        sum(1 for r in all_results[:i] if r.steps < max_steps) / i
        for i in range(1, len(all_results) + 1)
    ]
    plot_success_rate_over_time(file_prefix, runs, plot_dir, success_rates)

    # Plot: Efficiency Score Over Time
    efficiency_scores: list[float] = [
        result.efficiency_score for result in all_results
    ]  # Assuming efficiency scores are stored as result[5]
    plot_efficiency_score_over_time(file_prefix, runs, plot_dir, efficiency_scores)


if __name__ == "__main__":
    main()
