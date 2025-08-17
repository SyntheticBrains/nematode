"""Run the Quantum Nematode simulation."""

import argparse
import logging
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quantumnematode.agent import (
    DEFAULT_AGENT_BODY_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAZE_GRID_SIZE,
    QuantumNematodeAgent,
)
from quantumnematode.brain.arch import (
    Brain,
    MLPBrainConfig,
    ModularBrainConfig,
    QMLPBrainConfig,
    QuantumBrain,
)
from quantumnematode.brain.arch.dtypes import (
    DEFAULT_BRAIN_TYPE,
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    BrainType,
    DeviceType,
)
from quantumnematode.env import MIN_GRID_SIZE
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.optimizers.gradient_methods import (
    GradientCalculationMethod,
)
from quantumnematode.optimizers.learning_rate import (
    AdamLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)
from quantumnematode.report.csv_export import (
    export_performance_metrics_to_csv,
    export_run_data_to_csv,
    export_simulation_results_to_csv,
    export_tracking_data_to_csv,
)
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult, TrackingData
from quantumnematode.report.plots import (
    plot_cumulative_reward_per_run,
    plot_efficiency_score_over_time,
    plot_last_cumulative_rewards,
    plot_running_average_steps,
    plot_steps_per_run,
    plot_success_rate_over_time,
    plot_tracking_data_by_latest_run,
    plot_tracking_data_by_session,
)
from quantumnematode.report.summary import summary
from quantumnematode.theme import Theme
from quantumnematode.utils.config_loader import (
    BrainContainerConfig,
    ManyworldsModeConfig,
    RewardConfig,
    configure_brain,
    configure_gradient_method,
    configure_learning_rate,
    configure_manyworlds_mode,
    configure_reward,
    load_simulation_config,
)

DEFAULT_DEVICE = DeviceType.CPU
DEFAULT_RUNS = 1

if TYPE_CHECKING:
    from qiskit_serverless.core.function import RunnableQiskitFunction


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
        default=DEFAULT_RUNS,
        help=f"Number of simulation runs to perform (default: {DEFAULT_RUNS}).",
    )
    device_choices = [dt.value for dt in DeviceType]
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE.value,
        choices=device_choices,
        help="Device to use for quantum execution "
        f"({', '.join(device_choices)}; default: '{DEFAULT_DEVICE.value}').",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    # Update the argument parser to include a flag for many-worlds mode
    parser.add_argument(
        "--manyworlds",
        action="store_true",
        help="Enable many-worlds mode to visualize top 2 decisions at each step. "
        "Only one run is allowed in this mode.",
    )
    parser.add_argument(
        "--track-per-run",
        action="store_true",
        help="If set, output tracked brain data as separate plots per run in subfolders.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=Theme.ASCII.value,
        choices=[
            Theme.ASCII.value,
            Theme.EMOJI.value,
            Theme.UNICODE.value,
            Theme.COLORED_ASCII.value,
            Theme.RICH.value,
            Theme.EMOJI_RICH.value,
        ],
        help="Maze rendering theme: 'ascii' (default), "
        "'emoji', 'unicode', 'colored_ascii', 'rich', or 'emoji_rich'.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Q-CTRL's Fire Opal error suppression techniques on QPUs.",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the Quantum Nematode simulation."""
    args = parse_arguments()

    config_file = args.config
    manyworlds_mode = args.manyworlds
    max_steps = DEFAULT_MAX_STEPS
    maze_grid_size = DEFAULT_MAZE_GRID_SIZE
    runs = args.runs
    brain_type: BrainType = DEFAULT_BRAIN_TYPE
    shots = DEFAULT_SHOTS
    body_length = DEFAULT_AGENT_BODY_LENGTH
    qubits = DEFAULT_QUBITS
    device = DeviceType(args.device.lower())
    show_last_frame_only = args.show_last_frame_only
    log_level = args.log_level.upper()
    learning_rate = DynamicLearningRate()
    gradient_method = GradientCalculationMethod.RAW
    reward_config = RewardConfig()
    manyworlds_mode_config = ManyworldsModeConfig()
    track_per_run = args.track_per_run
    theme = Theme(args.theme)
    optimize_quantum_performance = args.optimize

    match brain_type:
        case BrainType.MODULAR:
            brain_config = ModularBrainConfig()
        case BrainType.MLP:
            brain_config = MLPBrainConfig()
        case BrainType.QMLP:
            brain_config = QMLPBrainConfig()

    # Authenticate and setup Q-CTRL if needed
    perf_mgmt = None
    if device == DeviceType.QPU:
        from quantumnematode.auth.ibm_quantum import IBMQuantumAuthenticator

        ibmq_authenticator = IBMQuantumAuthenticator()

        if optimize_quantum_performance:
            catalog = ibmq_authenticator.get_functions_catalog()
            perf_mgmt = catalog.load("q-ctrl/performance-management")
            logger.info("Q-CTRL Fire Opal Performance Management loaded successfully")
        else:
            ibmq_authenticator.authenticate_runtime_service()

    if config_file:
        config = load_simulation_config(config_file)

        brain_config = configure_brain(config)
        brain_type = (
            BrainType(config.brain.name)
            if config.brain is not None and isinstance(config.brain, BrainContainerConfig)
            else brain_type
        )
        max_steps = config.max_steps if config.max_steps is not None else max_steps
        maze_grid_size = (
            config.maze_grid_size if config.maze_grid_size is not None else maze_grid_size
        )
        shots = config.shots if config.shots is not None else shots
        body_length = config.body_length if config.body_length is not None else body_length
        qubits = config.qubits if config.qubits is not None else qubits

        # Load learning rate method and parameters if specified
        learning_rate = configure_learning_rate(config)

        # Load gradient method if specified
        gradient_method = configure_gradient_method(gradient_method, config)

        # Load reward configuration if specified
        reward_config = configure_reward(config)

        # Load many-worlds mode configuration if specified
        manyworlds_mode_config = configure_manyworlds_mode(config)

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
    logger.info(f"Brain type: {brain_type.value}")
    logger.info(f"Body length: {body_length}")
    logger.info(f"Qubits: {qubits}")
    logger.info(f"Shots: {shots}")

    # Select the brain architecture
    brain = setup_brain_model(
        brain_type=brain_type,
        brain_config=brain_config,
        shots=shots,
        qubits=qubits,
        device=device,
        learning_rate=learning_rate,
        gradient_method=gradient_method,
        perf_mgmt=perf_mgmt,
    )

    # Update the agent to use the selected brain architecture
    agent = QuantumNematodeAgent(
        maze_grid_size=maze_grid_size,
        brain=brain,
        max_body_length=body_length,
        theme=theme,
    )

    # Set the plot and data directories
    plot_dir = Path.cwd() / "exports" / timestamp / "session" / "plots"
    data_dir = Path.cwd() / "exports" / timestamp / "session" / "data"

    # Initialize tracking variables for plotting
    tracking_data = TrackingData()

    all_results: list[SimulationResult] = []

    total_runs_done = 0

    if manyworlds_mode:
        try:
            agent.run_manyworlds_mode(
                config=manyworlds_mode_config,
                reward_config=reward_config,
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
            run_num = run + 1
            logger.info(f"Starting run {run_num} of {runs}")

            # Calculate the initial distance to the goal
            initial_distance = agent.calculate_goal_distance()

            render_text = f"Run:\t\t{run_num}/{runs}\n"

            if len(all_results) > 1:
                # Running average steps per run
                total_steps_all_runs = (
                    sum([result.steps for result in all_results]) / total_runs_done + 1
                )
                render_text += f"Steps(Avg):\t{total_steps_all_runs:.2f}/{total_runs_done + 1}\n"

            path = agent.run_episode(
                reward_config=reward_config,
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

            logger.info(f"Efficiency Score for run {run_num}: {efficiency_score}")

            result = SimulationResult(
                run=run_num,
                steps=steps,
                path=path,
                total_reward=total_reward,
                last_total_reward=agent.total_rewards,
                efficiency_score=efficiency_score,
            )
            all_results.append(result)

            logger.info(f"Run {run_num}/{runs} completed in {steps} steps.")

            total_runs_done += 1

            tracking_data.data[run_num] = deepcopy(agent.brain.history_data)

            if track_per_run:
                plot_tracking_data_by_latest_run(
                    tracking_data=tracking_data,
                    timestamp=timestamp,
                    run=run_num,
                )
                export_run_data_to_csv(
                    tracking_data=tracking_data,
                    run=run_num,
                    timestamp=timestamp,
                )

            if run_num < runs:
                agent.reset_environment()
                agent.reset_brain()

    except KeyboardInterrupt:
        manage_simulation_halt(
            max_steps=max_steps,
            brain_type=brain_type,
            qubits=qubits,
            timestamp=timestamp,
            agent=agent,
            all_results=all_results,
            total_runs_done=total_runs_done,
            tracking_data=tracking_data,
            plot_dir=plot_dir,
        )

    # Calculate and log performance metrics
    metrics = agent.calculate_metrics(total_runs=total_runs_done)
    logger.info("\nPerformance Metrics:")
    logger.info(f"Success Rate: {metrics.success_rate:.2f}")
    logger.info(f"Average Steps: {metrics.average_steps:.2f}")
    logger.info(f"Average Reward: {metrics.average_reward:.2f}")

    print()  # noqa: T201
    print(f"Session ID: {timestamp}")  # noqa: T201

    # Final summary of all runs.
    summary(total_runs_done, max_steps, all_results)

    # Generate plots after the simulation
    plot_results(all_results=all_results, metrics=metrics, max_steps=max_steps, plot_dir=plot_dir)

    # Export data to CSV files
    export_simulation_results_to_csv(all_results=all_results, data_dir=data_dir)
    export_performance_metrics_to_csv(metrics=metrics, data_dir=data_dir)

    # Generate additional plots for tracking data
    plot_tracking_data_by_session(
        tracking_data=tracking_data,
        brain_type=brain_type,
        plot_dir=plot_dir,
        qubits=qubits,
    )

    # Export tracking data to CSV files
    export_tracking_data_to_csv(
        tracking_data=tracking_data,
        brain_type=brain_type,
        data_dir=data_dir,
        qubits=qubits,
    )

    return


def validate_simulation_parameters(maze_grid_size: int, brain_type: BrainType, qubits: int) -> None:
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
            other than 'modular'.
    """
    if maze_grid_size < MIN_GRID_SIZE:
        error_message = (
            f"Grid size must be at least {MIN_GRID_SIZE}. Provided grid size: {maze_grid_size}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Validate qubits parameter for classical brain types
    if brain_type in (BrainType.MLP, BrainType.QMLP) and qubits != DEFAULT_QUBITS:
        error_message = (
            f"The 'qubits' parameter is only supported by "
            "quantum brain architectures. "
            f"Provided brain: {brain_type.value}, qubits: {qubits}."
        )
        logger.error(error_message)
        raise ValueError(error_message)


def setup_brain_model(  # noqa: PLR0913
    brain_type: BrainType,
    brain_config: ModularBrainConfig | MLPBrainConfig | QMLPBrainConfig,
    shots: int,
    qubits: int,  # noqa: ARG001
    device: DeviceType,
    learning_rate: DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,  # noqa: ARG001
    perf_mgmt: "RunnableQiskitFunction | None" = None,
) -> Brain:
    """
    Set up the brain model based on the specified brain type.

    Args:
        brain_type (str): The type of brain architecture to use.
        brain_config (BrainConfig): Configuration for the brain architecture.
        shots (int): The number of shots for quantum circuit execution.
        qubits (int): The number of qubits to use (only applicable for "modular" brain).
        device (str): The device to use for simulation ("CPU" or "GPU").
        learning_rate (DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate):
            The learning rate configuration for the brain.
        gradient_method: The gradient calculation method.
        perf_mgmt: Q-CTRL performance management function instance.

    Returns
    -------
        Brain: An instance of the selected brain model.

    Raises
    ------
        ValueError: If an unknown brain type is provided.
    """
    if brain_type == BrainType.MODULAR:
        if not isinstance(brain_config, ModularBrainConfig):
            error_message = (
                "The 'modular' brain architecture requires a ModularBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if not isinstance(learning_rate, DynamicLearningRate):
            error_message = (
                "The 'modular' brain architecture requires a DynamicLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        from quantumnematode.brain.arch.modular import ModularBrain

        brain = ModularBrain(
            config=brain_config,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
            perf_mgmt=perf_mgmt,
        )
    elif brain_type == BrainType.MLP:
        from quantumnematode.brain.arch.mlp import MLPBrain

        if not isinstance(brain_config, MLPBrainConfig):
            error_message = (
                "The 'mlp' brain architecture requires an MLPBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        brain = MLPBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            lr_scheduler=True,
            device=device,
        )
    elif brain_type == BrainType.QMLP:
        from quantumnematode.brain.arch.qmlp import QMLPBrain

        if not isinstance(brain_config, QMLPBrainConfig):
            error_message = (
                "The 'qmlp' brain architecture requires a QMLPBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        brain = QMLPBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
        )
    else:
        error_message = f"Unknown brain architecture: {brain_type}"
        raise ValueError(error_message)
    return brain


def manage_simulation_halt(  # noqa: PLR0913
    max_steps: int,
    brain_type: BrainType,
    qubits: int,
    timestamp: str,
    agent: QuantumNematodeAgent,
    all_results: list[SimulationResult],
    total_runs_done: int,
    tracking_data: TrackingData,
    plot_dir: Path,
) -> None:
    """
    Handle simulation halt triggered by a KeyboardInterrupt.

    This function provides options to the user to either exit the simulation,
    output partial results and plots, or print circuit details.

    Args:
        max_steps (int): Maximum number of steps for the simulation.
        brain_type (str): Type of brain architecture used in the simulation.
        qubits (int): Number of qubits used in the simulation.
        timestamp (str): Timestamp for the current session.
        agent (QuantumNematodeAgent): The simulation agent.
        all_results (list[SimulationResult]):
            List of results for each run.
        total_runs_done (int): Total number of runs completed so far.
        tracking_data TrackingData: Data tracked during the simulation for plotting.
        plot_dir (Path): Directory where plots will be saved.
    """
    data_dir = Path.cwd() / "exports" / timestamp / "session" / "data"
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
            logger.info(f"Success Rate: {metrics.success_rate:.2f}")
            logger.info(f"Average Steps: {metrics.average_steps:.2f}")
            logger.info(f"Average Reward: {metrics.average_reward:.2f}")

            print()  # noqa: T201
            print(f"Session ID: {timestamp}")  # noqa: T201

            # Generate partial summary
            summary(total_runs_done, max_steps, all_results)

            # Generate plots with current timestamp
            file_prefix = f"{total_runs_done}_"
            plot_results(
                all_results=all_results,
                metrics=metrics,
                max_steps=max_steps,
                file_prefix=file_prefix,
                plot_dir=plot_dir,
            )
            plot_tracking_data_by_session(
                tracking_data=tracking_data,
                plot_dir=plot_dir,
                brain_type=brain_type,
                qubits=qubits,
                file_prefix=file_prefix,
            )

            # Export partial results to CSV
            export_simulation_results_to_csv(
                all_results=all_results,
                data_dir=data_dir,
                file_prefix=file_prefix,
            )
            export_performance_metrics_to_csv(
                metrics=metrics,
                data_dir=data_dir,
                file_prefix=file_prefix,
            )
            export_tracking_data_to_csv(
                tracking_data=tracking_data,
                brain_type=brain_type,
                data_dir=data_dir,
                qubits=qubits,
                file_prefix=file_prefix,
            )
        elif choice == 2:  # noqa: PLR2004
            logger.info("Printing circuit details.")
            if isinstance(agent.brain, QuantumBrain):
                circuit = agent.brain.inspect_circuit()
                logger.info(f"Circuit details:\n{circuit}")
                print(circuit)  # noqa: T201
            else:
                logger.error(
                    "Circuit details are only available for QuantumBrain architectures.",
                )
                print("Circuit details are only available for QuantumBrain architectures.")  # noqa: T201
        else:
            logger.error("Invalid choice. Please enter a number between 1 and 4.")


def plot_results(
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,
    max_steps: int,
    plot_dir: Path,
    file_prefix: str = "",
) -> None:
    """Generate and save plots for the simulation results."""
    runs: list[int] = [result.run for result in all_results]
    steps: list[int] = [result.steps for result in all_results]

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot: Steps per Run
    plot_steps_per_run(metrics, file_prefix, runs, steps, plot_dir)

    # Plot: Running Average Steps per Run
    plot_running_average_steps(file_prefix, runs, steps, plot_dir)

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
