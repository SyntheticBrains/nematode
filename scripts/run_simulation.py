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
    QuantumNematodeAgent,
    SatietyConfig,
)
from quantumnematode.benchmark import save_benchmark
from quantumnematode.brain.arch import (
    Brain,
    MLPBrainConfig,
    ModularBrainConfig,
    QMLPBrainConfig,
    QModularBrainConfig,
    QuantumBrain,
    SpikingBrainConfig,
)
from quantumnematode.brain.arch.dtypes import (
    DEFAULT_BRAIN_TYPE,
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    QUANTUM_BRAIN_TYPES,
    BrainType,
    DeviceType,
)
from quantumnematode.env import MIN_GRID_SIZE, DynamicForagingEnvironment, StaticEnvironment
from quantumnematode.env.theme import Theme
from quantumnematode.experiment import capture_experiment_metadata, save_experiment
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
    export_distance_efficiencies_to_csv,
    export_foraging_results_to_csv,
    export_foraging_session_metrics_to_csv,
    export_performance_metrics_to_csv,
    export_run_data_to_csv,
    export_simulation_results_to_csv,
    export_tracking_data_to_csv,
)
from quantumnematode.report.dtypes import (
    EpisodeTrackingData,
    PerformanceMetrics,
    SimulationResult,
    TerminationReason,
    TrackingData,
)
from quantumnematode.report.plots import (
    plot_all_distance_efficiencies_distribution,
    plot_cumulative_reward_per_run,
    plot_distance_efficiency_trend,
    plot_efficiency_score_over_time,
    plot_foods_collected_per_run,
    plot_foods_vs_reward_correlation,
    plot_foods_vs_steps_correlation,
    plot_foraging_efficiency_per_run,
    plot_last_cumulative_rewards,
    plot_running_average_steps,
    plot_satiety_at_episode_end,
    plot_steps_per_run,
    plot_success_rate_over_time,
    plot_termination_reasons_breakdown,
    plot_tracking_data_by_latest_run,
    plot_tracking_data_by_session,
)
from quantumnematode.report.summary import summary
from quantumnematode.utils.config_loader import (
    BrainContainerConfig,
    DynamicEnvironmentConfig,
    EnvironmentConfig,
    ManyworldsModeConfig,
    ParameterInitializerConfig,
    RewardConfig,
    StaticEnvironmentConfig,
    configure_brain,
    configure_environment,
    configure_gradient_method,
    configure_learning_rate,
    configure_manyworlds_mode,
    configure_parameter_initializer,
    configure_reward,
    configure_satiety,
    create_parameter_initializer_instance,
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
    parser.add_argument(
        "--track-experiment",
        action="store_true",
        help="Save experiment metadata for reproducibility and comparison.",
    )
    parser.add_argument(
        "--save-benchmark",
        action="store_true",
        help="Save experiment as a benchmark submission (implies --track-experiment).",
    )
    parser.add_argument(
        "--benchmark-notes",
        type=str,
        help="Optional notes about optimization approach (requires --save-benchmark).",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the Quantum Nematode simulation."""
    args = parse_arguments()

    config_file = args.config
    manyworlds_mode = args.manyworlds
    max_steps = DEFAULT_MAX_STEPS
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
    parameter_initializer_config = ParameterInitializerConfig()
    reward_config = RewardConfig()
    satiety_config = SatietyConfig()
    environment_config = EnvironmentConfig()
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
        case BrainType.QMODULAR:
            brain_config = QModularBrainConfig()
        case BrainType.SPIKING:
            brain_config = SpikingBrainConfig()

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
        shots = config.shots if config.shots is not None else shots
        body_length = config.body_length if config.body_length is not None else body_length
        qubits = config.qubits if config.qubits is not None else qubits

        # Load learning rate method and parameters if specified
        learning_rate = configure_learning_rate(config)

        # Load gradient method if specified
        gradient_method = configure_gradient_method(gradient_method, config)

        # Load parameter initializer configuration if specified
        parameter_initializer_config = configure_parameter_initializer(config)

        # Load reward configuration if specified
        reward_config = configure_reward(config)

        # Load satiety configuration if specified
        satiety_config = configure_satiety(config)

        # Load environment configuration if specified
        environment_config = configure_environment(config)

        # Load many-worlds mode configuration if specified
        manyworlds_mode_config = configure_manyworlds_mode(config)

    # Get grid size from environment config for validation and logging
    if environment_config.type == "dynamic":
        grid_size = (environment_config.dynamic or DynamicEnvironmentConfig()).grid_size
    else:
        grid_size = (environment_config.static or StaticEnvironmentConfig()).grid_size

    validate_simulation_parameters(grid_size, brain_type, qubits)

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
    logger.info(f"Grid size: {grid_size}")
    logger.info(f"Environment type: {environment_config.type}")
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
        parameter_initializer_config=parameter_initializer_config,
        perf_mgmt=perf_mgmt,
    )

    # Create the environment based on configuration
    env = None
    if environment_config.type == "dynamic":
        logger.info("Using dynamic foraging environment")
        dynamic_config = environment_config.dynamic or DynamicEnvironmentConfig()
        env = DynamicForagingEnvironment(
            grid_size=dynamic_config.grid_size,
            num_initial_foods=dynamic_config.num_initial_foods,
            max_active_foods=dynamic_config.max_active_foods,
            min_food_distance=dynamic_config.min_food_distance,
            agent_exclusion_radius=dynamic_config.agent_exclusion_radius,
            gradient_decay_constant=dynamic_config.gradient_decay_constant,
            gradient_strength=dynamic_config.gradient_strength,
            viewport_size=dynamic_config.viewport_size,
            max_body_length=body_length,
            theme=theme,
        )
        logger.info(
            f"Dynamic environment: {dynamic_config.grid_size}x{dynamic_config.grid_size} grid, "
            f"{dynamic_config.num_initial_foods} initial foods",
        )
    else:
        logger.info("Using static maze environment")
        static_config = environment_config.static or StaticEnvironmentConfig()
        env = StaticEnvironment(
            grid_size=static_config.grid_size,
            max_body_length=body_length,
            theme=theme,
        )
        logger.info(
            f"Static maze environment: {static_config.grid_size}x{static_config.grid_size} grid",
        )

    # Update the agent to use the selected brain architecture and environment
    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        max_body_length=body_length,
        theme=theme,
        satiety_config=satiety_config,
    )

    # Set the plot and data directories
    plot_dir = Path.cwd() / "exports" / timestamp / "session" / "plots"
    data_dir = Path.cwd() / "exports" / timestamp / "session" / "data"

    # Initialize tracking variables for plotting
    tracking_data = TrackingData()

    all_results: list[SimulationResult] = []

    total_runs_done = 0

    # Session-level termination tracking
    total_successes = 0
    total_starved = 0
    total_max_steps = 0
    total_interrupted = 0

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
            print(message)
            return

    try:
        for run in range(total_runs_done, runs):
            run_num = run + 1
            logger.info(f"Starting run {run_num} of {runs}")

            # Log full initial environment state
            logger.info(f"Initial agent position: {tuple(agent.env.agent_pos)}")
            if isinstance(agent.env, DynamicForagingEnvironment):
                logger.info(f"Initial food positions: {agent.env.foods}")
                logger.info(f"Active foods: {len(agent.env.foods)}/{agent.env.max_active_foods}")
                logger.info(f"Initial satiety: {agent.current_satiety}/{agent.max_satiety}")

                # Log full environment render
                logger.info("Initial environment state (full render):")
                full_render = agent.env.render_full()
                for line in full_render:
                    logger.info(line)
            elif isinstance(agent.env, StaticEnvironment):
                logger.info(f"Goal position: {agent.env.goal}")

                # Log full environment render
                logger.info("Initial environment state (full render):")
                full_render = agent.env.render()
                for line in full_render:
                    logger.info(line)

            # Calculate the initial distance to the goal
            if isinstance(agent.env, DynamicForagingEnvironment):
                dist = agent.env.get_nearest_food_distance()
                initial_distance = dist if dist is not None else 0
            elif isinstance(agent.env, StaticEnvironment):
                initial_distance = abs(agent.env.agent_pos[0] - agent.env.goal[0]) + abs(
                    agent.env.agent_pos[1] - agent.env.goal[1],
                )
            else:
                initial_distance = 0

            render_text = "Session:\n--------\n"
            render_text += f"Run:\t\t{run_num}/{runs}\n"

            wins = 0
            match agent.env:
                case DynamicForagingEnvironment():
                    # NOTE: Total food calculation won't be accurate if we introduce different max
                    # active foods per run
                    wins = sum(result.success for result in all_results)
                    render_text += f"Wins:\t\t{wins}/{runs}\n"
                    total_foods_collected = sum(
                        result.foods_collected
                        for result in all_results
                        if result.foods_collected is not None
                    )
                    total_foods_available = sum(
                        result.foods_available
                        for result in all_results
                        if result.foods_available is not None
                    )
                    render_text += f"Eaten:\t\t{total_foods_collected}/{total_foods_available}\n"
                case StaticEnvironment():
                    wins = agent._metrics_tracker.success_count  # noqa: SLF001
                    render_text += f"Wins:\t\t{wins}/{runs}\n"

            if len(all_results) > 1:
                # Running average steps per run
                total_steps_all_runs = (
                    sum([result.steps for result in all_results]) / total_runs_done + 1
                )
                render_text += f"Steps(Avg):\t{total_steps_all_runs:.2f}/{total_runs_done + 1}\n"

            step_result = agent.run_episode(
                reward_config=reward_config,
                max_steps=max_steps,
                render_text=render_text,
                show_last_frame_only=show_last_frame_only,
            )

            steps_taken = agent._episode_tracker.steps  # noqa: SLF001
            total_reward = agent._episode_tracker.rewards  # noqa: SLF001

            # Determine success and track termination types
            success = step_result.termination_reason in (
                TerminationReason.GOAL_REACHED,
                TerminationReason.COMPLETED_ALL_FOOD,
            )

            # Update session-level counters
            if success:
                total_successes += 1
            if step_result.termination_reason == TerminationReason.STARVED:
                total_starved += 1
            elif step_result.termination_reason == TerminationReason.MAX_STEPS:
                total_max_steps += 1

            # Get environment specific data
            efficiency_score = None
            foods_collected_this_run = None
            foods_available_this_run = None
            satiety_remaining_this_run = None
            average_distance_efficiency = None
            satiety_history_this_run = None
            match agent.env:
                case StaticEnvironment():
                    # Calculate efficiency score for the run
                    efficiency_score = initial_distance - steps_taken
                case DynamicForagingEnvironment():
                    foods_collected_this_run = agent._episode_tracker.foods_collected  # noqa: SLF001
                    foods_available_this_run = agent.env.max_active_foods
                    satiety_remaining_this_run = agent.current_satiety
                    distance_efficiencies = agent._episode_tracker.distance_efficiencies  # noqa: SLF001
                    average_distance_efficiency = (
                        sum(distance_efficiencies) / len(distance_efficiencies)
                        if distance_efficiencies
                        else 0.0
                    )
                    satiety_history_this_run = agent._episode_tracker.satiety_history.copy()  # noqa: SLF001
                    # Efficiency score not defined for dynamic environment
                case _:
                    pass

            result = SimulationResult(
                run=run_num,
                steps=steps_taken,
                path=step_result.agent_path,
                total_reward=total_reward,
                last_total_reward=agent._episode_tracker.rewards,  # noqa: SLF001
                efficiency_score=efficiency_score,
                termination_reason=step_result.termination_reason,
                success=success,
                foods_collected=foods_collected_this_run,
                foods_available=foods_available_this_run,
                satiety_remaining=satiety_remaining_this_run,
                average_distance_efficiency=average_distance_efficiency,
                satiety_history=satiety_history_this_run,
            )
            all_results.append(result)

            # Log run outcome clearly
            outcome_msg = f"Run {run_num}/{runs} completed in {steps_taken} steps - "
            if success:
                if step_result.termination_reason == TerminationReason.GOAL_REACHED:
                    outcome_msg += "SUCCESS: Goal reached"
                elif step_result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD:
                    outcome_msg += f"SUCCESS: All food collected ({foods_collected_this_run} foods)"
            elif step_result.termination_reason == TerminationReason.STARVED:
                outcome_msg += "FAILED: Agent starved"
            elif step_result.termination_reason == TerminationReason.MAX_STEPS:
                outcome_msg += "FAILED: Max steps reached"

            logger.info(outcome_msg)

            total_runs_done += 1

            tracking_data.brain_data[run_num] = deepcopy(agent.brain.history_data)

            # Store episode tracking data for foraging environments
            if isinstance(agent.env, DynamicForagingEnvironment):
                tracking_data.episode_data[run_num] = EpisodeTrackingData(
                    satiety_history=satiety_history_this_run.copy()
                    if satiety_history_this_run
                    else [],
                    foods_collected=foods_collected_this_run or 0,
                    distance_efficiencies=agent._episode_tracker.distance_efficiencies.copy(),  # noqa: SLF001
                )

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
        total_interrupted = runs - total_runs_done
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

    # Update metrics with session-level termination counts
    metrics.total_successes = total_successes
    metrics.total_starved = total_starved
    metrics.total_max_steps = total_max_steps
    metrics.total_interrupted = total_interrupted

    # Final summary of all runs.
    summary(
        metrics=metrics,
        session_id=timestamp,
        num_runs=total_runs_done,
        max_steps=max_steps,
        all_results=all_results,
        env_type=agent.env,
    )

    # Generate plots after the simulation
    plot_results(all_results=all_results, metrics=metrics, max_steps=max_steps, plot_dir=plot_dir)

    # Export data to CSV files
    export_simulation_results_to_csv(all_results=all_results, data_dir=data_dir)
    export_performance_metrics_to_csv(metrics=metrics, data_dir=data_dir)

    # Export foraging-specific data (if dynamic environment)
    foraging_results = [r for r in all_results if r.foods_collected is not None]
    if foraging_results:
        export_foraging_results_to_csv(all_results=all_results, data_dir=data_dir)
        export_foraging_session_metrics_to_csv(
            all_results=all_results,
            metrics=metrics,
            data_dir=data_dir,
        )
        export_distance_efficiencies_to_csv(
            tracking_data=tracking_data,
            data_dir=data_dir,
        )

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

    # Experiment tracking (opt-in)
    track_experiment = args.track_experiment or args.save_benchmark
    if track_experiment:
        try:
            # Capture experiment metadata
            config_path = Path(config_file) if config_file else Path("config.yml")
            exports_rel_path = f"exports/{timestamp}"

            experiment_metadata = capture_experiment_metadata(
                config_path=config_path,
                env=agent.env,
                brain=agent.brain,
                brain_type=brain_type.value,
                config={
                    "brain": {
                        "config": brain_config.__dict__
                        if hasattr(brain_config, "__dict__")
                        else {},
                        "qubits": qubits if brain_type in QUANTUM_BRAIN_TYPES else None,
                        "shots": shots if brain_type in QUANTUM_BRAIN_TYPES else None,
                        "learning_rate": learning_rate.__dict__
                        if hasattr(learning_rate, "__dict__")
                        else {},
                    },
                    "satiety": {
                        "initial": satiety_config.initial_satiety,
                        "decay_rate": satiety_config.satiety_decay_rate,
                    },
                },
                all_results=all_results,
                metrics=metrics,
                device_type=device,
                qpu_backend=None,  # TODO: Implement extracting QPU backend
                exports_path=exports_rel_path,
            )

            if device == DeviceType.QPU:
                print(
                    "Warning: You will need to manually add the QPU backend "
                    "information to the experiment metadata. Example: `ibm_strasbourg`.",
                )

            # Save experiment
            experiment_path = save_experiment(experiment_metadata)
            print(f"\n✓ Experiment metadata saved: {experiment_path}")
            print(f"  Experiment ID: {experiment_metadata.experiment_id}")
            print(
                f"  Query with: uv run scripts/experiment_query.py show {experiment_metadata.experiment_id}",
            )

            if args.save_benchmark:
                # Interactive benchmark submission
                print("\n" + "=" * 80)
                print("Benchmark Submission")
                print("=" * 80)

                contributor = input("\nContributor name (required): ").strip()
                if not contributor:
                    logger.error("Contributor name is required for benchmark submission")
                else:
                    github_username = input(
                        "GitHub username (optional, press Enter to skip): ",
                    ).strip()
                    github_username = github_username if github_username else None

                    notes = args.benchmark_notes
                    if not notes:
                        notes = input(
                            "Optimization notes (optional, press Enter to skip): ",
                        ).strip()
                        notes = notes if notes else None

                    # Save benchmark
                    benchmark_path = save_benchmark(
                        metadata=experiment_metadata,
                        contributor=contributor,
                        github_username=github_username,
                        notes=notes,
                    )
                    print(f"\n✓ Benchmark saved: {benchmark_path}")

        except Exception as e:
            logger.error(f"Failed to save experiment metadata: {e}")

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
            other than quantum based architectures.
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


def setup_brain_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    brain_type: BrainType,
    brain_config: ModularBrainConfig
    | MLPBrainConfig
    | QMLPBrainConfig
    | QModularBrainConfig
    | SpikingBrainConfig,
    shots: int,
    qubits: int,  # noqa: ARG001
    device: DeviceType,
    learning_rate: DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,  # noqa: ARG001
    parameter_initializer_config: ParameterInitializerConfig,
    perf_mgmt: "RunnableQiskitFunction | None" = None,
) -> Brain:
    """
    Set up the brain model based on the specified brain type.

    Args:
        brain_type (str): The type of brain architecture to use.
        brain_config (BrainConfig): Configuration for the brain architecture.
        shots (int): The number of shots for quantum circuit execution.
        qubits (int): The number of qubits to use (only applicable for quantum brain architectures).
        device (str): The device to use for simulation ("CPU" or "GPU").
        learning_rate (DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate):
            The learning rate configuration for the brain.
        gradient_method: The gradient calculation method.
        parameter_initializer_config: Configuration for parameter initialization.
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

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = ModularBrain(
            config=brain_config,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
            parameter_initializer=parameter_initializer,
            perf_mgmt=perf_mgmt,
        )
    elif brain_type == BrainType.QMODULAR:
        if not isinstance(brain_config, QModularBrainConfig):
            error_message = (
                "The 'qmodular' brain architecture requires a QModularBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if not isinstance(learning_rate, DynamicLearningRate):
            error_message = (
                "The 'qmodular' brain architecture requires a DynamicLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        from quantumnematode.brain.arch.qmodular import QModularBrain

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = QModularBrain(
            config=brain_config,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
            parameter_initializer=parameter_initializer,
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

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            lr_scheduler=True,
            device=device,
            parameter_initializer=parameter_initializer,
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

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = QMLPBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type == BrainType.SPIKING:
        from quantumnematode.brain.arch.spiking import SpikingBrain

        if not isinstance(brain_config, SpikingBrainConfig):
            error_message = (
                "The 'spiking' brain architecture requires a SpikingBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = SpikingBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    else:
        error_message = f"Unknown brain type: {brain_type}"
        logger.error(error_message)
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
        print(prompt_intro_message)
        print("0. Exit")
        print("1. Output the summary, plots, and tracking until this point in time.")
        print("2. Print the circuit's details.")

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

            # Generate partial summary
            summary(
                metrics=metrics,
                session_id=timestamp,
                num_runs=total_runs_done,
                max_steps=max_steps,
                all_results=all_results,
                env_type=agent.env,
            )

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
        elif choice == 2:
            logger.info("Printing circuit details.")
            if isinstance(agent.brain, QuantumBrain):
                circuit = agent.brain.inspect_circuit()
                logger.info(f"Circuit details:\n{circuit}")
                print(circuit)
            else:
                logger.error(
                    "Circuit details are only available for QuantumBrain architectures.",
                )
                print("Circuit details are only available for QuantumBrain architectures.")
        else:
            logger.error("Invalid choice. Please enter a number between 1 and 4.")


def plot_results(  # noqa: C901
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
    cumulative_rewards: list[float] = [result.total_reward for result in all_results]
    plot_cumulative_reward_per_run(file_prefix, runs, plot_dir, cumulative_rewards)

    # Plot: Last Cumulative Rewards Over Runs
    last_cumulative_rewards: list[float] = [result.last_total_reward for result in all_results]
    plot_last_cumulative_rewards(file_prefix, runs, plot_dir, last_cumulative_rewards)

    # Plot: Success Rate Over Time
    success_rates: list[float] = [
        sum(1 for r in all_results[:i] if r.steps < max_steps) / i
        for i in range(1, len(all_results) + 1)
    ]
    plot_success_rate_over_time(file_prefix, runs, plot_dir, success_rates)

    # Plot: Efficiency Score Over Time
    efficiency_scores: list[float] = [
        result.efficiency_score for result in all_results if result.efficiency_score is not None
    ]
    if len(efficiency_scores) > 0:
        plot_efficiency_score_over_time(file_prefix, runs, plot_dir, efficiency_scores)

    # Dynamic Foraging Environment Specific Plots
    foraging_results = [r for r in all_results if r.foods_collected is not None]
    if foraging_results:
        # Extract foraging-specific data
        foods_collected_list = [
            r.foods_collected for r in foraging_results if r.foods_collected is not None
        ]
        foods_available = (
            foraging_results[0].foods_available
            if foraging_results[0].foods_available is not None
            else 0
        )
        satiety_remaining_list = [
            r.satiety_remaining for r in foraging_results if r.satiety_remaining is not None
        ]
        avg_distance_effs = [
            r.average_distance_efficiency
            for r in foraging_results
            if r.average_distance_efficiency is not None
        ]
        foraging_runs = [r.run for r in foraging_results]
        foraging_steps = [r.steps for r in foraging_results]
        foraging_rewards = [r.total_reward for r in foraging_results]

        # Plot: Foods Collected Per Run
        if foods_collected_list:
            plot_foods_collected_per_run(
                file_prefix,
                foraging_runs,
                plot_dir,
                foods_collected_list,
                foods_available,
            )

        # Plot: Distance Efficiency Trend
        if avg_distance_effs:
            plot_distance_efficiency_trend(
                file_prefix,
                foraging_runs,
                plot_dir,
                avg_distance_effs,
            )

        # Plot: Foraging Efficiency (foods per step)
        if foods_collected_list:
            plot_foraging_efficiency_per_run(
                file_prefix,
                foraging_runs,
                plot_dir,
                foods_collected_list,
                foraging_steps,
            )

        # Plot: Satiety at Episode End
        if satiety_remaining_list:
            max_satiety = max(satiety_remaining_list) if satiety_remaining_list else 100.0
            # Get actual max satiety from first result's history if available
            if foraging_results[0].satiety_history:
                max_satiety = max(foraging_results[0].satiety_history)
            plot_satiety_at_episode_end(
                file_prefix,
                foraging_runs,
                plot_dir,
                satiety_remaining_list,
                max_satiety,
            )

        # Plot: All Distance Efficiencies Distribution
        if avg_distance_effs:
            plot_all_distance_efficiencies_distribution(
                file_prefix,
                plot_dir,
                avg_distance_effs,
            )

        # Plot: Termination Reasons Breakdown
        termination_counts: dict[str, int] = {}
        for result in foraging_results:
            reason = result.termination_reason.value
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        if termination_counts:
            plot_termination_reasons_breakdown(
                file_prefix,
                plot_dir,
                termination_counts,
            )

        # Plot: Foods vs Reward Correlation
        if foods_collected_list and foraging_rewards:
            plot_foods_vs_reward_correlation(
                file_prefix,
                plot_dir,
                foods_collected_list,
                foraging_rewards,
            )

        # Plot: Foods vs Steps Correlation
        if foods_collected_list and foraging_steps:
            plot_foods_vs_steps_correlation(
                file_prefix,
                plot_dir,
                foods_collected_list,
                foraging_steps,
            )


if __name__ == "__main__":
    main()
