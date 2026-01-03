# pragma: no cover

"""Run the Quantum Nematode simulation."""

import argparse
import logging
import shutil
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
from quantumnematode.brain.arch import (
    Brain,
    MLPBrainConfig,
    ModularBrainConfig,
    PPOBrainConfig,
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
    ConstantLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)
from quantumnematode.report.csv_export import (
    export_convergence_metrics_to_csv,
    export_distance_efficiencies_to_csv,
    export_foraging_results_to_csv,
    export_foraging_session_metrics_to_csv,
    export_performance_metrics_to_csv,
    export_predator_results_to_csv,
    export_predator_session_metrics_to_csv,
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
    plot_evasion_success_rate_over_time,
    plot_foods_collected_per_run,
    plot_foods_vs_reward_correlation,
    plot_foods_vs_steps_correlation,
    plot_foraging_efficiency_per_run,
    plot_health_at_episode_end,
    plot_health_progression_single_run,
    plot_last_cumulative_rewards,
    plot_predator_encounters_over_time,
    plot_running_average_steps,
    plot_satiety_at_episode_end,
    plot_satiety_progression_single_run,
    plot_steps_per_run,
    plot_success_rate_over_time,
    plot_survival_vs_food_collection,
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
from quantumnematode.utils.seeding import derive_run_seed, ensure_seed, get_rng, set_global_seed

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
        "--validate-chemotaxis",
        action="store_true",
        help="Display chemotaxis validation against C. elegans literature data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not provided, a random seed is auto-generated.",
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

    # Initialize seed for reproducibility (auto-generate if not provided)
    simulation_seed = ensure_seed(args.seed)
    logger.info(f"Using simulation seed: {simulation_seed}")
    shots = DEFAULT_SHOTS
    body_length = DEFAULT_AGENT_BODY_LENGTH
    qubits = DEFAULT_QUBITS
    device = DeviceType(args.device.lower())
    show_last_frame_only = args.show_last_frame_only
    log_level = args.log_level.upper()
    learning_rate = DynamicLearningRate()
    gradient_method = GradientCalculationMethod.RAW
    gradient_max_norm = None
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
            brain_config = ModularBrainConfig(seed=simulation_seed)
        case BrainType.MLP:
            brain_config = MLPBrainConfig(seed=simulation_seed)
        case BrainType.PPO:
            brain_config = PPOBrainConfig(seed=simulation_seed)
        case BrainType.QMLP:
            brain_config = QMLPBrainConfig(seed=simulation_seed)
        case BrainType.QMODULAR:
            brain_config = QModularBrainConfig(seed=simulation_seed)
        case BrainType.SPIKING:
            brain_config = SpikingBrainConfig(seed=simulation_seed)

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

        # Handle seed precedence: CLI > config file > auto-generated
        if args.seed is not None:
            # CLI seed takes highest precedence (already set in simulation_seed)
            pass
        elif config.seed is not None:
            # Use seed from config file root level
            simulation_seed = config.seed
            logger.info(f"Using seed from config file: {simulation_seed}")

        brain_config = configure_brain(config)
        # Always update brain config with the resolved simulation seed
        brain_config = brain_config.model_copy(update={"seed": simulation_seed})

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

        # Load gradient method and max_norm if specified
        gradient_method, gradient_max_norm = configure_gradient_method(gradient_method, config)

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
    logger.info(f"Seed: {simulation_seed}")

    # Select the brain architecture
    brain = setup_brain_model(
        brain_type=brain_type,
        brain_config=brain_config,
        shots=shots,
        qubits=qubits,
        device=device,
        learning_rate=learning_rate,
        gradient_method=gradient_method,
        gradient_max_norm=gradient_max_norm,
        parameter_initializer_config=parameter_initializer_config,
        perf_mgmt=perf_mgmt,
    )

    # Create the environment based on configuration
    env = None
    if environment_config.type == "dynamic":
        logger.info("Using dynamic foraging environment")
        dynamic_config = environment_config.dynamic or DynamicEnvironmentConfig()

        # Get foraging and predator configs (with automatic migration)
        foraging_config = dynamic_config.get_foraging_config()
        predator_config = dynamic_config.get_predator_config()
        health_config = dynamic_config.get_health_config()
        thermotaxis_config = dynamic_config.get_thermotaxis_config()

        env = DynamicForagingEnvironment(
            grid_size=dynamic_config.grid_size,
            viewport_size=dynamic_config.viewport_size,
            max_body_length=body_length,
            theme=theme,
            seed=simulation_seed,
            foraging=foraging_config.to_params(),
            predator=predator_config.to_params(),
            health=health_config.to_params(),
            thermotaxis=thermotaxis_config.to_params(),
        )
        predator_info = ""
        if predator_config.enabled:
            predator_info = (
                f", {predator_config.count} predators "
                f"(detection_radius={predator_config.detection_radius}, "
                f"kill_radius={predator_config.kill_radius}, "
                f"damage_radius={predator_config.damage_radius})"
            )
        health_info = ""
        if health_config.enabled:
            health_info = (
                f", health (max_hp={health_config.max_hp}, "
                f"predator_damage={health_config.predator_damage}, "
                f"food_healing={health_config.food_healing})"
            )
        thermotaxis_info = ""
        if thermotaxis_config.enabled:
            thermotaxis_info = (
                f", thermotaxis (Tc={thermotaxis_config.cultivation_temperature}°C, "
                f"gradient={thermotaxis_config.gradient_strength}°C/cell)"
            )
        logger.info(
            f"Dynamic environment: {dynamic_config.grid_size}x{dynamic_config.grid_size} grid, "
            f"{foraging_config.foods_on_grid} foods on grid, "
            f"target {foraging_config.target_foods_to_collect} to collect"
            f"{predator_info}{health_info}{thermotaxis_info}",
        )
    else:
        logger.info("Using static maze environment")
        static_config = environment_config.static or StaticEnvironmentConfig()
        env = StaticEnvironment(
            grid_size=static_config.grid_size,
            max_body_length=body_length,
            theme=theme,
            seed=simulation_seed,
        )
        logger.info(
            f"Static maze environment: {static_config.grid_size}x{static_config.grid_size} grid",
        )

    # Update the agent to use the selected brain architecture and environment
    # Check env capability directly to avoid config/env drift
    # Separated gradients can be enabled via environment config OR brain config (for spiking)
    use_separated_gradients = False
    if isinstance(env, DynamicForagingEnvironment):
        dynamic_config = environment_config.dynamic or DynamicEnvironmentConfig()
        use_separated_gradients = dynamic_config.use_separated_gradients

        # For spiking brain, also check brain config for use_separated_gradients
        if isinstance(brain_config, SpikingBrainConfig) and brain_config.use_separated_gradients:
            use_separated_gradients = True

    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        max_body_length=body_length,
        theme=theme,
        satiety_config=satiety_config,
        use_separated_gradients=use_separated_gradients,
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
    total_health_depleted = 0
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

            # Reset global RNG with per-run derived seed for reproducibility
            # This ensures each run starts with a deterministic state
            run_seed = derive_run_seed(simulation_seed, run)
            set_global_seed(run_seed)
            logger.debug(f"Run {run_num} using derived seed: {run_seed}")

            logger.info(f"Starting run {run_num} of {runs}")

            # Log full initial environment state
            logger.info(f"Initial agent position: {tuple(agent.env.agent_pos)}")
            if isinstance(agent.env, DynamicForagingEnvironment):
                logger.info(f"Initial food positions: {agent.env.foods}")
                logger.info(
                    f"Foods on grid: {len(agent.env.foods)}/{agent.env.foraging.foods_on_grid}",
                )
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
            elif step_result.termination_reason == TerminationReason.HEALTH_DEPLETED:
                total_health_depleted += 1
            elif step_result.termination_reason == TerminationReason.MAX_STEPS:
                total_max_steps += 1

            # Get environment specific data
            efficiency_score = None
            foods_collected_this_run = None
            foods_available_this_run = None
            satiety_remaining_this_run = None
            average_distance_efficiency = None
            satiety_history_this_run = None
            predator_encounters_this_run = None
            successful_evasions_this_run = None
            died_to_predator_this_run = None
            died_to_health_depletion_this_run = None
            health_history_this_run = None
            temperature_history_this_run = None
            match agent.env:
                case StaticEnvironment():
                    # Calculate efficiency score for the run
                    efficiency_score = initial_distance - steps_taken
                case DynamicForagingEnvironment():
                    foods_collected_this_run = agent._episode_tracker.foods_collected  # noqa: SLF001
                    foods_available_this_run = agent.env.foraging.target_foods_to_collect
                    satiety_remaining_this_run = agent.current_satiety
                    distance_efficiencies = agent._episode_tracker.distance_efficiencies  # noqa: SLF001
                    average_distance_efficiency = (
                        sum(distance_efficiencies) / len(distance_efficiencies)
                        if distance_efficiencies
                        else 0.0
                    )
                    satiety_history_this_run = agent._episode_tracker.satiety_history.copy()  # noqa: SLF001
                    predator_encounters_this_run = agent._episode_tracker.predator_encounters  # noqa: SLF001
                    successful_evasions_this_run = agent._episode_tracker.successful_evasions  # noqa: SLF001
                    died_to_predator_this_run = (
                        step_result.termination_reason == TerminationReason.PREDATOR
                    )
                    died_to_health_depletion_this_run = (
                        step_result.termination_reason == TerminationReason.HEALTH_DEPLETED
                    )
                    # Copy health history if health system is enabled
                    if agent.env.health.enabled:
                        health_history_this_run = agent._episode_tracker.health_history.copy()  # noqa: SLF001
                    # Copy temperature history if thermotaxis is enabled
                    if agent.env.thermotaxis.enabled:
                        temperature_history_this_run = (
                            agent._episode_tracker.temperature_history.copy()  # noqa: SLF001
                        )
                    # Efficiency score not defined for dynamic environment
                case _:
                    pass

            # Calculate multi-objective scores
            survival_score_this_run = None
            temperature_comfort_score_this_run = None

            if isinstance(agent.env, DynamicForagingEnvironment):
                # Survival score: final_hp / max_hp
                if agent.env.health.enabled and health_history_this_run:
                    final_hp = health_history_this_run[-1]
                    max_hp = agent.env.health.max_hp
                    survival_score_this_run = final_hp / max_hp if max_hp > 0 else 0.0

                # Temperature comfort score: fraction of time in comfort zone
                if agent.env.thermotaxis.enabled and temperature_history_this_run:
                    comfort_delta = agent.env.thermotaxis.comfort_delta
                    cultivation_temp = agent.env.thermotaxis.cultivation_temperature
                    comfort_min = cultivation_temp - comfort_delta
                    comfort_max = cultivation_temp + comfort_delta
                    steps_in_comfort = sum(
                        1
                        for temp in temperature_history_this_run
                        if comfort_min <= temp <= comfort_max
                    )
                    temperature_comfort_score_this_run = (
                        steps_in_comfort / len(temperature_history_this_run)
                        if temperature_history_this_run
                        else 0.0
                    )

            result = SimulationResult(
                run=run_num,
                seed=run_seed,
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
                health_history=health_history_this_run,
                temperature_history=temperature_history_this_run,
                predator_encounters=predator_encounters_this_run,
                successful_evasions=successful_evasions_this_run,
                died_to_predator=died_to_predator_this_run,
                died_to_health_depletion=died_to_health_depletion_this_run,
                food_history=step_result.food_history,
                survival_score=survival_score_this_run,
                temperature_comfort_score=temperature_comfort_score_this_run,
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
            elif step_result.termination_reason == TerminationReason.PREDATOR:
                outcome_msg += "FAILED: Killed by predator"
            elif step_result.termination_reason == TerminationReason.HEALTH_DEPLETED:
                outcome_msg += "FAILED: Health depleted"

            logger.info(outcome_msg)

            total_runs_done += 1

            tracking_data.brain_data[run_num] = deepcopy(agent.brain.history_data)

            # Store episode tracking data for foraging environments
            if isinstance(agent.env, DynamicForagingEnvironment):
                tracking_data.episode_data[run_num] = EpisodeTrackingData(
                    satiety_history=satiety_history_this_run.copy()
                    if satiety_history_this_run
                    else [],
                    health_history=health_history_this_run.copy()
                    if health_history_this_run
                    else [],
                    temperature_history=temperature_history_this_run.copy()
                    if temperature_history_this_run
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
                # Derive seed for next run and update environment before reset
                # This ensures each run has unique but reproducible food/predator placements
                next_run_seed = derive_run_seed(
                    simulation_seed,
                    run_num,
                )  # run_num is 1-indexed, so this gives seed for next run
                agent.env.seed = next_run_seed
                agent.env.rng = get_rng(next_run_seed)
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
    metrics.total_health_depleted = total_health_depleted
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
    plot_results(all_results=all_results, metrics=metrics, plot_dir=plot_dir)

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

    # Export predator-specific data (only if predator mechanics were actually in play)
    predator_results = [
        r
        for r in all_results
        if r.predator_encounters is not None
        and (r.predator_encounters > 0 or r.successful_evasions or r.died_to_predator)
    ]
    if predator_results:
        export_predator_results_to_csv(all_results=all_results, data_dir=data_dir)
        export_predator_session_metrics_to_csv(
            all_results=all_results,
            metrics=metrics,
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
    track_experiment = args.track_experiment
    if track_experiment:
        try:
            # Capture experiment metadata
            config_path = Path(config_file) if config_file else Path("config.yml")
            exports_rel_path = f"exports/{timestamp}"

            learning_rate_metadata = None
            if isinstance(learning_rate, DynamicLearningRate):
                learning_rate_metadata = {
                    "method": "dynamic",
                    "parameters": {
                        "initial_learning_rate": learning_rate.initial_learning_rate,
                        "decay_rate": learning_rate.decay_rate,
                        "decay_type": learning_rate.decay_type,
                        "decay_factor": learning_rate.decay_factor,
                        "step_size": learning_rate.step_size,
                        "max_steps": learning_rate.max_steps,
                        "power": learning_rate.power,
                        "min_lr": learning_rate.min_lr,
                    },
                }
            elif isinstance(learning_rate, ConstantLearningRate):
                learning_rate_metadata = {
                    "method": "constant",
                    "parameters": {
                        "initial_learning_rate": learning_rate.learning_rate,
                        "learning_rate": learning_rate.learning_rate,
                    },
                }
            else:
                logger.warning(
                    "Learning rate metadata capture not implemented for "
                    f"{type(learning_rate).__name__}",
                )
                # Provide empty dict to avoid None.get() errors
                learning_rate_metadata = {"method": "unknown", "parameters": {}}

            experiment_metadata = capture_experiment_metadata(
                config_path=config_path,
                env=agent.env,
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
                    "gradient": {
                        "method": gradient_method.value,
                        "max_norm": gradient_max_norm,
                    },
                    "learning_rate": learning_rate_metadata,
                    "satiety": {
                        "initial": satiety_config.initial_satiety,
                        "decay_rate": satiety_config.satiety_decay_rate,
                    },
                    "parameter_initializer": {
                        "type": parameter_initializer_config.type,
                        "manual_parameter_values": parameter_initializer_config.manual_parameter_values,
                    },
                    "reward": {
                        "reward_goal": reward_config.reward_goal,
                        "reward_distance_scale": reward_config.reward_distance_scale,
                        "reward_exploration": reward_config.reward_exploration,
                        "penalty_step": reward_config.penalty_step,
                        "penalty_anti_dithering": reward_config.penalty_anti_dithering,
                        "penalty_stuck_position": reward_config.penalty_stuck_position,
                        "stuck_position_threshold": reward_config.stuck_position_threshold,
                        "penalty_starvation": reward_config.penalty_starvation,
                        "penalty_predator_death": reward_config.penalty_predator_death,
                        "penalty_predator_proximity": reward_config.penalty_predator_proximity,
                    },
                },
                all_results=all_results,
                metrics=metrics,
                device_type=device,
                qpu_backend=None,  # TODO: Implement extracting QPU backend
                exports_path=exports_rel_path,
                session_id=timestamp,
            )

            # Export convergence metrics to CSV (includes composite score and all convergence analysis)
            export_convergence_metrics_to_csv(
                experiment_metadata=experiment_metadata,
                data_dir=data_dir,
            )

            if device == DeviceType.QPU:
                print(
                    "Warning: You will need to manually add the QPU backend "
                    "information to the experiment metadata. Example: `ibm_strasbourg`.",
                )

            # Save experiment to experiments/<id>/ folder structure
            # This creates a self-contained experiment folder for potential benchmark submission
            experiment_dir = Path.cwd() / "experiments" / experiment_metadata.experiment_id

            # Save experiment metadata JSON
            experiment_path = save_experiment(
                experiment_metadata,
                base_dir=experiment_dir,
            )

            # Copy config file to experiment folder for reproducibility
            if config_file:
                config_source = Path(config_file)
                if config_source.exists():
                    config_dest = experiment_dir / config_source.name
                    shutil.copy2(config_source, config_dest)
                    logger.info(f"Config file copied to: {config_dest}")

            print(f"\n✓ Experiment saved: {experiment_dir}")
            print(f"  Experiment ID: {experiment_metadata.experiment_id}")
            print(f"  Metadata: {experiment_path}")
            if config_file:
                print(f"  Config: {experiment_dir / Path(config_file).name}")
            print(
                f"  Query with: uv run scripts/experiment_query.py show {experiment_metadata.experiment_id}",
            )
            print(
                f"\n  To submit as benchmark: uv run scripts/benchmark_submit.py --experiments {experiment_dir}",
            )

            # Display chemotaxis validation if requested
            if (
                args.validate_chemotaxis
                and experiment_metadata.results.post_convergence_chemotaxis_index is not None
            ):
                results = experiment_metadata.results
                print("\n" + "=" * 60)
                print("Chemotaxis Validation (C. elegans Literature Comparison)")
                print("=" * 60)
                # Show post-convergence metrics (trained behavior, used for validation)
                print("  Post-Convergence (Trained Behavior):")
                print(
                    f"    Chemotaxis Index (CI):  {results.post_convergence_chemotaxis_index:.3f}",
                )
                print(
                    f"    Time in Attractant:     {results.post_convergence_time_in_attractant:.1%}",
                )
                print(
                    f"    Approach Frequency:     {results.post_convergence_approach_frequency:.1%}",
                )
                print(
                    f"    Path Efficiency:        {results.post_convergence_path_efficiency:.3f}",
                )
                print(
                    f"  Validation Level:         {results.chemotaxis_validation_level}",
                )
                # Show all-run metrics for comparison (smaller font/indented)
                if results.avg_chemotaxis_index is not None:
                    print(f"  (All-run CI:              {results.avg_chemotaxis_index:.3f})")
                print("-" * 60)
                # Use dynamic literature source from benchmark
                if results.literature_source:
                    print(f"  Literature Reference: {results.literature_source}")
                if results.biological_ci_range:
                    ci_min, ci_max = results.biological_ci_range
                    print(f"  Biological CI Range:  {ci_min:.2f} - {ci_max:.2f}")
                if results.biological_ci_typical:
                    print(f"  Typical Wild-Type CI: {results.biological_ci_typical:.2f}")
                # Show match status
                if results.matches_biology is True:
                    print("  Status: MATCHES biological range")
                elif results.matches_biology is False:
                    level = results.chemotaxis_validation_level
                    if level in ("minimum", "target"):
                        print("  Status: ~ Approaching biological range")
                    else:
                        print("  Status: Below biological range")
            elif args.validate_chemotaxis:
                print(
                    "\n[Chemotaxis validation not available - requires dynamic foraging environment]",
                )

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
    | PPOBrainConfig
    | QMLPBrainConfig
    | QModularBrainConfig
    | SpikingBrainConfig,
    shots: int,
    qubits: int,  # noqa: ARG001
    device: DeviceType,
    learning_rate: ConstantLearningRate
    | DynamicLearningRate
    | AdamLearningRate
    | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,
    gradient_max_norm: float | None,
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
        learning_rate (ConstantLearningRate | DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate):
            The learning rate configuration for the brain. Note: modular/qmodular brains
            only support ConstantLearningRate and DynamicLearningRate.
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

        if not isinstance(learning_rate, (DynamicLearningRate, ConstantLearningRate)):
            error_message = (
                "The 'modular' brain architecture requires a DynamicLearningRate or ConstantLearningRate. "
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
            gradient_method=gradient_method,
            gradient_max_norm=gradient_max_norm,
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
    elif brain_type == BrainType.PPO:
        from quantumnematode.brain.arch.ppo import PPOBrain

        if not isinstance(brain_config, PPOBrainConfig):
            error_message = (
                "The 'ppo' brain architecture requires a PPOBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Create parameter initializer instance from config
        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = PPOBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
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

        # Determine input dimension based on separated gradients config
        # 2 features for combined gradient, 4 features for separated food/predator gradients
        input_dim = 4 if brain_config.use_separated_gradients else 2

        brain = SpikingBrain(
            config=brain_config,
            input_dim=input_dim,
            num_actions=4,
            device=device,
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


def plot_results(  # noqa: C901, PLR0912, PLR0915
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,
    plot_dir: Path,
    file_prefix: str = "",
) -> None:
    """
    Generate and save plots for the simulation results.

    Args:
        all_results (list[SimulationResult]):
            List of results for each run.
        metrics (PerformanceMetrics): Performance metrics calculated from the simulation.
        plot_dir (Path): Directory where plots will be saved.
        file_prefix (str, optional): Prefix to add to plot filenames. Defaults to "".

    """
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
        sum(1 for r in all_results[:i] if r.success) / i for i in range(1, len(all_results) + 1)
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

        # Plot: Health at Episode End (for runs with health system enabled)
        health_results = [r for r in foraging_results if r.health_history]
        if health_results:
            health_remaining_list = [
                r.health_history[-1] for r in health_results if r.health_history
            ]
            health_runs = [r.run for r in health_results]
            # Get max health from first result's history
            max_health = (
                max(health_results[0].health_history) if health_results[0].health_history else 100.0
            )
            if health_remaining_list:
                plot_health_at_episode_end(
                    file_prefix,
                    health_runs,
                    plot_dir,
                    health_remaining_list,
                    max_health,
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

    # Predator Evasion Environment Specific Plots (only if predator mechanics were in play)
    predator_results = [
        r
        for r in all_results
        if r.predator_encounters is not None
        and (r.predator_encounters > 0 or r.successful_evasions or r.died_to_predator)
    ]
    if predator_results:
        # Extract predator-specific data for encounters plot
        predator_encounters_list = [
            r.predator_encounters for r in predator_results if r.predator_encounters is not None
        ]
        predator_runs = [r.run for r in predator_results]

        # Plot: Predator Encounters Over Time
        if predator_encounters_list:
            plot_predator_encounters_over_time(
                file_prefix,
                predator_runs,
                plot_dir,
                predator_encounters_list,
            )

        # Plot: Evasion Success Rate Over Time
        # Build aligned lists in single loop to ensure index correspondence
        predator_runs_evasion: list[int] = []
        predator_encounters_evasion: list[int] = []
        successful_evasions_evasion: list[int] = []

        for r in predator_results:
            if (
                r.predator_encounters is not None
                and r.predator_encounters > 0
                and r.successful_evasions is not None
            ):
                predator_runs_evasion.append(r.run)
                predator_encounters_evasion.append(r.predator_encounters)
                successful_evasions_evasion.append(r.successful_evasions)

        if predator_encounters_evasion and successful_evasions_evasion:
            plot_evasion_success_rate_over_time(
                file_prefix,
                predator_runs_evasion,
                plot_dir,
                predator_encounters_evasion,
                successful_evasions_evasion,
            )

        # Plot: Survival vs Food Collection (for predator environments that also track food)
        # Build paired lists in single loop to ensure correspondence
        predator_foraging_results = [
            r
            for r in predator_results
            if r.foods_collected is not None and r.died_to_predator is not None
        ]
        if predator_foraging_results:
            foods_in_predator = []
            deaths_in_predator = []
            for r in predator_foraging_results:
                foods_in_predator.append(r.foods_collected)
                deaths_in_predator.append(r.died_to_predator)

            if foods_in_predator and deaths_in_predator:
                plot_survival_vs_food_collection(
                    file_prefix,
                    plot_dir,
                    foods_in_predator,
                    deaths_in_predator,
                )

        # Plot: Satiety Progression for Single Run (if requested via --track-per-run)
        # This is typically generated in the per-run tracking section, but we can add
        # a sample run here for predator environments
        # Get the last run's satiety history as an example
        last_run = predator_results[-1]
        if last_run.satiety_history:
            max_satiety_pred = max(last_run.satiety_history)
            plot_satiety_progression_single_run(
                file_prefix,
                plot_dir,
                last_run.run,
                last_run.satiety_history,
                max_satiety_pred,
            )

        # Plot: Health Progression for Single Run (for health-enabled environments)
        if last_run.health_history:
            max_health_pred = max(last_run.health_history)
            plot_health_progression_single_run(
                file_prefix,
                plot_dir,
                last_run.run,
                last_run.health_history,
                max_health_pred,
            )


if __name__ == "__main__":
    main()
