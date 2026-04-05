# pragma: no cover

"""Run the Quantum Nematode simulation."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantumnematode.validation.chemotaxis import ChemotaxisMetrics

from quantumnematode.agent import (
    DEFAULT_AGENT_BODY_LENGTH,
    DEFAULT_MAX_STEPS,
    QuantumNematodeAgent,
    SatietyConfig,
)
from quantumnematode.brain.arch import (
    MLPDQNBrainConfig,
    MLPPPOBrainConfig,
    MLPReinforceBrainConfig,
    QQLearningBrainConfig,
    QVarCircuitBrainConfig,
    SpikingReinforceBrainConfig,
)
from quantumnematode.brain.arch.dtypes import (
    DEFAULT_BRAIN_TYPE,
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    QUANTUM_BRAIN_TYPES,
    BrainType,
    DeviceType,
)
from quantumnematode.brain.weights import WeightPersistence, load_weights, save_weights
from quantumnematode.env import MIN_GRID_SIZE
from quantumnematode.env.theme import DEFAULT_THEME, Theme
from quantumnematode.experiment import capture_experiment_metadata, save_experiment
from quantumnematode.logging_config import (
    configure_file_logging,
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
    IncrementalDetailedTrackingWriter,
    create_path_csv_writer,
    create_simulation_results_csv_writer,
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
    write_path_data_row,
    write_simulation_result_row,
)
from quantumnematode.report.dtypes import (
    BrainDataSnapshot,
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
from quantumnematode.utils.brain_factory import setup_brain_model
from quantumnematode.utils.config_loader import (
    AgentConfig,
    BrainConfigType,
    BrainContainerConfig,
    EnvironmentConfig,
    ManyworldsModeConfig,
    MultiAgentConfig,
    ParameterInitializerConfig,
    RewardConfig,
    SensingConfig,
    SimulationConfig,
    configure_brain,
    configure_environment,
    configure_gradient_method,
    configure_learning_rate,
    configure_manyworlds_mode,
    configure_parameter_initializer,
    configure_reward,
    configure_satiety,
    create_env_from_config,
    load_simulation_config,
    validate_sensing_config,
)
from quantumnematode.utils.interrupt_handler import manage_simulation_halt
from quantumnematode.utils.seeding import derive_run_seed, ensure_seed, get_rng, set_global_seed
from quantumnematode.utils.session import generate_session_id

DEFAULT_DEVICE = DeviceType.CPU
DEFAULT_RUNS = 1


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
        default=DEFAULT_THEME.value,
        choices=[
            Theme.PIXEL.value,
            Theme.ASCII.value,
            Theme.EMOJI.value,
            Theme.UNICODE.value,
            Theme.COLORED_ASCII.value,
            Theme.RICH.value,
            Theme.EMOJI_RICH.value,
            Theme.HEADLESS.value,
        ],
        help="Maze rendering theme: 'pixel' (default), "
        "'ascii', 'emoji', 'unicode', 'colored_ascii', 'rich', 'emoji_rich', "
        "or 'headless' (no rendering — fastest for batch training).",
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
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to saved weights to load before training.",
    )
    parser.add_argument(
        "--save-weights",
        type=str,
        default=None,
        help="Path to save weights after training completes.",
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

    # Generate unique session ID and configure file logging early
    # so all logger.info() calls below are captured in the log file
    session_id = generate_session_id()
    configure_file_logging(session_id)

    # Configure logging level
    log_level = args.log_level.upper()
    if log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(log_level)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)

    # Initialize seed for reproducibility (auto-generate if not provided)
    simulation_seed = ensure_seed(args.seed)
    logger.info(f"Using simulation seed: {simulation_seed}")
    shots = DEFAULT_SHOTS
    body_length = DEFAULT_AGENT_BODY_LENGTH
    qubits = DEFAULT_QUBITS
    device = DeviceType(args.device.lower())
    show_last_frame_only = args.show_last_frame_only
    learning_rate = DynamicLearningRate()
    gradient_method = GradientCalculationMethod.RAW
    gradient_max_norm = None
    parameter_initializer_config = ParameterInitializerConfig()
    reward_config = RewardConfig()
    satiety_config = SatietyConfig()
    environment_config = EnvironmentConfig()
    sensing_config = SensingConfig()
    manyworlds_mode_config = ManyworldsModeConfig()
    track_per_run = args.track_per_run
    theme = Theme(args.theme)
    optimize_quantum_performance = args.optimize

    match brain_type:
        case BrainType.QVARCIRCUIT:
            brain_config = QVarCircuitBrainConfig(seed=simulation_seed)
        case BrainType.MLP_REINFORCE:
            brain_config = MLPReinforceBrainConfig(seed=simulation_seed)
        case BrainType.MLP_PPO:
            brain_config = MLPPPOBrainConfig(seed=simulation_seed)
        case BrainType.MLP_DQN:
            brain_config = MLPDQNBrainConfig(seed=simulation_seed)
        case BrainType.QQLEARNING:
            brain_config = QQLearningBrainConfig(seed=simulation_seed)
        case BrainType.SPIKING_REINFORCE:
            brain_config = SpikingReinforceBrainConfig(seed=simulation_seed)

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

    config: SimulationConfig | None = None
    if config_file:
        config = load_simulation_config(config_file)
        logger.info(f"Initializing simulation config: {config}")

        # Handle seed precedence: CLI > config file > auto-generated
        if args.seed is not None:
            # CLI seed takes highest precedence (already set in simulation_seed)
            pass
        elif config.seed is not None:
            # Use seed from config file root level
            simulation_seed = config.seed
            logger.info(f"Using seed from config file: {simulation_seed}")

        # For heterogeneous multi-agent (agents list, no top-level brain),
        # skip single-agent brain configuration — agents configure their own brains.
        _is_heterogeneous_multi_agent = (
            config.multi_agent is not None
            and config.multi_agent.enabled
            and config.multi_agent.agents is not None
        )

        if not _is_heterogeneous_multi_agent:
            brain_config = configure_brain(config)
            # Always update brain config with the resolved simulation seed
            brain_config = brain_config.model_copy(update={"seed": simulation_seed})

        if config.brain is not None and isinstance(config.brain, BrainContainerConfig):
            brain_type = BrainType(config.brain.name)

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

        # Load and validate sensing configuration
        sensing_config = validate_sensing_config(environment_config.get_sensing_config())

        # Apply sensing mode translation to brain's sensory modules
        sensory_modules_attr = (
            getattr(brain_config, "sensory_modules", None)
            if not _is_heterogeneous_multi_agent
            else None
        )
        if sensory_modules_attr is not None:
            from quantumnematode.brain.modules import ModuleName
            from quantumnematode.utils.config_loader import apply_sensing_mode

            original_modules = [m.value for m in sensory_modules_attr]
            translated = apply_sensing_mode(original_modules, sensing_config)
            translated_modules = [ModuleName(m) for m in translated]
            if translated_modules != list(sensory_modules_attr):
                brain_config = brain_config.model_copy(
                    update={"sensory_modules": translated_modules},
                )
                logger.info(
                    f"Sensing mode translation: {original_modules} → "
                    f"{[m.value for m in translated_modules]}",
                )

        # Load many-worlds mode configuration if specified
        manyworlds_mode_config = configure_manyworlds_mode(config)

    # ── Multi-agent branch ────────────────────────────────────────────
    # If multi-agent is enabled in config, delegate to the multi-agent
    # simulation runner and return. The rest of main() is single-agent.
    multi_agent_config = config.multi_agent if config_file and config is not None else None
    if multi_agent_config is not None and multi_agent_config.enabled and config is not None:
        # Reject single-agent-only CLI flags in multi-agent mode
        unsupported_flags = []
        if args.load_weights or args.save_weights:
            unsupported_flags.append("--load-weights/--save-weights")
        if args.manyworlds:
            unsupported_flags.append("--manyworlds")
        if args.track_per_run:
            unsupported_flags.append("--track-per-run")
        if args.track_experiment:
            unsupported_flags.append("--track-experiment")
        if args.validate_chemotaxis:
            unsupported_flags.append("--validate-chemotaxis")
        if unsupported_flags:
            msg = (
                f"Cannot use {', '.join(unsupported_flags)} with multi-agent mode. "
                "Use per-agent config options in multi_agent.agents instead."
            )
            raise ValueError(msg)
        _run_multi_agent(
            config=config,
            multi_agent_config=multi_agent_config,
            environment_config=environment_config,
            reward_config=reward_config,
            satiety_config=satiety_config,
            sensing_config=sensing_config,
            simulation_seed=simulation_seed,
            max_steps=max_steps,
            shots=shots,
            body_length=body_length,
            qubits=qubits,
            device=device,
            runs=runs,
            session_id=session_id,
            learning_rate=learning_rate,
            gradient_method=gradient_method,
            gradient_max_norm=gradient_max_norm,
            parameter_initializer_config=parameter_initializer_config,
            theme=theme,
            perf_mgmt=perf_mgmt,
        )
        return

    # Get grid size from environment config for validation and logging
    grid_size = environment_config.grid_size

    validate_simulation_parameters(grid_size, brain_type, qubits)

    logger.info(f"Session ID: {session_id}")

    logger.info("Simulation parameters:")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Runs: {runs}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Device: {device}")
    logger.info(f"Grid size: {grid_size}")
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

    # Pass session ID to brain for weight export alignment
    set_session_id = getattr(brain, "set_session_id", None)
    if callable(set_session_id):
        set_session_id(session_id)

    # Weight persistence: resolve load path (CLI overrides config)
    load_weights_path = args.load_weights or getattr(brain_config, "weights_path", None)
    save_weights_path = args.save_weights

    # Validate: if weight persistence requested, brain must implement it
    if (load_weights_path or save_weights_path) and not isinstance(
        brain,
        WeightPersistence,
    ):
        source = (
            "--load-weights/--save-weights"
            if (args.load_weights or args.save_weights)
            else "config.weights_path"
        )
        msg = (
            f"Brain {type(brain).__name__} does not implement "
            f"WeightPersistence. Cannot use {source} for weight persistence."
        )
        raise TypeError(msg)

    if load_weights_path:
        load_weights(brain, Path(load_weights_path))

    # Create the environment
    logger.info("Using dynamic foraging environment")

    foraging_config = environment_config.get_foraging_config()
    predator_config = environment_config.get_predator_config()
    health_config = environment_config.get_health_config()
    thermotaxis_config = environment_config.get_thermotaxis_config()

    env = create_env_from_config(
        environment_config,
        seed=simulation_seed,
        max_body_length=body_length,
        theme=theme,
    )
    predator_info = ""
    if predator_config.enabled:
        predator_info = (
            f", {predator_config.count} predators "
            f"(detection_radius={predator_config.detection_radius}, "
            f"damage_radius={predator_config.damage_radius})"
        )
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
        f"Dynamic environment: {environment_config.grid_size}x{environment_config.grid_size} grid, "
        f"{foraging_config.foods_on_grid} foods on grid, "
        f"target {foraging_config.target_foods_to_collect} to collect"
        f"{predator_info}{health_info}{thermotaxis_info}",
    )

    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        max_body_length=body_length,
        theme=theme,
        satiety_config=satiety_config,
        sensing_config=sensing_config,
    )

    # Set the plot and data directories
    plot_dir = Path.cwd() / "exports" / session_id / "session" / "plots"
    data_dir = Path.cwd() / "exports" / session_id / "session" / "data"

    # Initialize tracking variables for plotting
    tracking_data = TrackingData()

    all_results: list[SimulationResult] = []
    running_wins = 0
    running_foods_collected = 0
    running_foods_available = 0
    running_total_steps = 0

    total_runs_done = 0

    # Session-level termination tracking
    total_successes = 0
    total_starved = 0
    total_health_depleted = 0
    total_max_steps = 0
    total_interrupted = 0

    # Incremental CSV writers — write heavy per-step data each episode, then flush from memory
    data_dir.mkdir(parents=True, exist_ok=True)
    path_csv_file, path_csv_writer = create_path_csv_writer(data_dir / "paths.csv")
    sim_results_csv_file, sim_results_csv_writer = create_simulation_results_csv_writer(
        data_dir / "simulation_results.csv",
    )
    detailed_tracking_writer = IncrementalDetailedTrackingWriter(data_dir)

    # Pre-computed chemotaxis metrics (populated per-episode when --track-experiment is set)
    track_experiment = args.track_experiment
    chemotaxis_metrics_per_run: list[tuple[int, ChemotaxisMetrics]] = []

    # Keep last run's per-step histories for session-level single-run plots
    # (overwritten each iteration; only the final run's data survives to plot_results)
    last_run_satiety_history: list[float] = []
    last_run_health_history: list[float] = []

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
        finally:
            path_csv_file.close()
            sim_results_csv_file.close()
            detailed_tracking_writer.close()
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
            logger.info(f"Initial food positions: {agent.env.foods}")
            logger.info(
                f"Foods on grid: {len(agent.env.foods)}/{agent.env.foraging.foods_on_grid}",
            )
            logger.info(f"Initial satiety: {agent.current_satiety}/{agent.max_satiety}")

            # Log full environment render (use EMOJI for PIXEL since it has no text output)
            logger.info("Initial environment state (full render):")
            log_theme = Theme.EMOJI if theme == Theme.PIXEL else None
            full_render = agent.env.render_full(theme_override=log_theme)
            for line in full_render:
                logger.info(line)

            render_text = "Session:\n--------\n"
            render_text += f"Run:\t\t{run_num}/{runs}\n"
            render_text += f"Wins:\t\t{running_wins}/{runs}\n"
            render_text += f"Eaten:\t\t{running_foods_collected}/{running_foods_available}\n"

            if total_runs_done > 1:
                avg_steps = running_total_steps / total_runs_done
                render_text += f"Steps(Avg):\t{avg_steps:.2f}/{max_steps}\n"

            step_result = agent.run_episode(
                reward_config=reward_config,
                max_steps=max_steps,
                render_text=render_text,
                show_last_frame_only=show_last_frame_only,
            )

            steps_taken = agent._episode_tracker.steps
            total_reward = agent._episode_tracker.rewards

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
            elif step_result.termination_reason == TerminationReason.INTERRUPTED:
                total_interrupted += 1

            # Get environment specific data
            foods_collected_this_run = agent._episode_tracker.foods_collected
            foods_available_this_run = agent.env.foraging.target_foods_to_collect
            satiety_remaining_this_run = agent.current_satiety
            distance_efficiencies = agent._episode_tracker.distance_efficiencies
            average_distance_efficiency = (
                sum(distance_efficiencies) / len(distance_efficiencies)
                if distance_efficiencies
                else 0.0
            )
            satiety_history_this_run = agent._episode_tracker.satiety_history.copy()
            predator_encounters_this_run = agent._episode_tracker.predator_encounters
            successful_evasions_this_run = agent._episode_tracker.successful_evasions
            died_to_health_depletion_this_run = (
                step_result.termination_reason == TerminationReason.HEALTH_DEPLETED
            )
            health_history_this_run = agent._episode_tracker.health_history.copy()
            temperature_history_this_run = None
            # Copy temperature history if thermotaxis is enabled
            if agent.env.thermotaxis.enabled:
                temperature_history_this_run = agent._episode_tracker.temperature_history.copy()

            oxygen_history_this_run = None
            # Copy oxygen history if aerotaxis is enabled
            if agent.env.aerotaxis.enabled:
                oxygen_history_this_run = agent._episode_tracker.oxygen_history.copy()

            # Calculate multi-objective scores
            survival_score_this_run = None
            temperature_comfort_score_this_run = None
            oxygen_comfort_score_this_run = None

            # Survival score: final_hp / max_hp
            if health_history_this_run:
                final_hp = health_history_this_run[-1]
                max_hp = agent.env.health.max_hp
                survival_score_this_run = final_hp / max_hp if max_hp > 0 else 0.0

            # Temperature comfort score: fraction of time in comfort zone
            if agent.env.thermotaxis.enabled and temperature_history_this_run:
                temperature_comfort_score_this_run = agent.env.get_temperature_comfort_score()

            # Oxygen comfort score: fraction of time in O2 comfort zone
            if agent.env.aerotaxis.enabled and oxygen_history_this_run:
                oxygen_comfort_score_this_run = agent.env.get_oxygen_comfort_score()

            result = SimulationResult(
                run=run_num,
                seed=run_seed,
                steps=steps_taken,
                path=step_result.agent_path,
                total_reward=total_reward,
                last_total_reward=agent._episode_tracker.rewards,
                termination_reason=step_result.termination_reason,
                success=success,
                foods_collected=foods_collected_this_run,
                foods_available=foods_available_this_run,
                satiety_remaining=satiety_remaining_this_run,
                average_distance_efficiency=average_distance_efficiency,
                satiety_history=satiety_history_this_run,
                health_history=health_history_this_run,
                temperature_history=temperature_history_this_run,
                oxygen_history=oxygen_history_this_run,
                predator_encounters=predator_encounters_this_run,
                successful_evasions=successful_evasions_this_run,
                died_to_health_depletion=died_to_health_depletion_this_run,
                food_history=step_result.food_history,
                survival_score=survival_score_this_run,
                temperature_comfort_score=temperature_comfort_score_this_run,
                oxygen_comfort_score=oxygen_comfort_score_this_run,
            )
            all_results.append(result)
            running_wins += int(result.success)
            running_foods_collected += result.foods_collected or 0
            running_foods_available += result.foods_available or 0
            running_total_steps += result.steps

            # --- Incremental exports (before data flush) ---
            # Write simulation results and path data to CSV incrementally
            write_simulation_result_row(sim_results_csv_writer, result, sim_results_csv_file)
            write_path_data_row(path_csv_writer, result)

            # Write detailed brain tracking data incrementally
            detailed_tracking_writer.write_run(run_num, agent.brain.history_data)

            # Compute chemotaxis metrics per-episode (before path/food_history flush)
            if track_experiment and result.food_history:
                from quantumnematode.validation.chemotaxis import (
                    calculate_chemotaxis_metrics_stepwise,
                )

                positions = [(float(x), float(y)) for x, y in result.path]
                food_history_float = [
                    [(float(x), float(y)) for x, y in step_foods]
                    for step_foods in result.food_history
                ]
                if food_history_float:
                    cm = calculate_chemotaxis_metrics_stepwise(
                        positions=positions,
                        food_history=food_history_float,
                        attractant_zone_radius=5.0,
                    )
                    chemotaxis_metrics_per_run.append((run_num, cm))

            # If Pygame window was closed, stop all remaining runs
            if agent.pygame_renderer_closed:
                logger.info(
                    f"Pygame window closed - stopping simulation after run {run_num}/{runs}.",
                )
                total_runs_done = run_num
                break

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
            elif step_result.termination_reason == TerminationReason.HEALTH_DEPLETED:
                outcome_msg += "FAILED: Health depleted"

            logger.info(outcome_msg)

            total_runs_done += 1

            if track_per_run:
                # Full deepcopy + episode data only needed for per-run plots/exports
                tracking_data.brain_data[run_num] = deepcopy(agent.brain.history_data)
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
                    oxygen_history=oxygen_history_this_run.copy()
                    if oxygen_history_this_run
                    else [],
                    foods_collected=foods_collected_this_run or 0,
                    distance_efficiencies=agent._episode_tracker.distance_efficiencies.copy(),
                )
                plot_tracking_data_by_latest_run(
                    tracking_data=tracking_data,
                    session_id=session_id,
                    run=run_num,
                )
                export_run_data_to_csv(
                    tracking_data=tracking_data,
                    run=run_num,
                    session_id=session_id,
                )

            # --- Extract scalar snapshots and flush heavy data ---
            result.path_length = len(result.path)
            if result.satiety_history:
                result.max_satiety = max(result.satiety_history)
                last_run_satiety_history = result.satiety_history.copy()
            else:
                last_run_satiety_history = []
            if result.health_history:
                result.final_health = result.health_history[-1]
                result.max_health = max(result.health_history)
                last_run_health_history = result.health_history.copy()
            else:
                last_run_health_history = []

            # Replace full brain history with last-value snapshot
            snapshot_last_values: dict[str, object] = {}
            for attr_name in vars(agent.brain.history_data):
                attr_val = getattr(agent.brain.history_data, attr_name)
                if isinstance(attr_val, list) and attr_val:
                    snapshot_last_values[attr_name] = attr_val[-1]
            tracking_data.brain_data[run_num] = BrainDataSnapshot(
                last_values=snapshot_last_values,
            )

            # Flush heavy per-step data from result
            result.path = []
            result.food_history = None
            result.satiety_history = None
            result.health_history = None
            result.temperature_history = None
            result.oxygen_history = None

            # Replace episode tracking data with lightweight version
            tracking_data.episode_data[run_num] = EpisodeTrackingData(
                foods_collected=foods_collected_this_run or 0,
                distance_efficiencies=agent._episode_tracker.distance_efficiencies.copy(),
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
            session_id=session_id,
            agent=agent,
            all_results=all_results,
            total_runs_done=total_runs_done,
            tracking_data=tracking_data,
            plot_dir=plot_dir,
            plot_results_fn=plot_results,
        )
    finally:
        # Ensure incremental writers are always closed (safe to call twice)
        if not path_csv_file.closed:
            path_csv_file.close()
        if not sim_results_csv_file.closed:
            sim_results_csv_file.close()
        detailed_tracking_writer.close()

        # Auto-save final weights (covers both normal completion and KeyboardInterrupt).
        # Each save is isolated so I/O failures don't replace the original
        # exception or prevent the other save from running.
        weights_dir = Path.cwd() / "exports" / session_id / "weights"
        try:
            save_weights(brain, weights_dir / "final.pt")
        except Exception:
            logger.exception("Failed to auto-save weights to %s", weights_dir / "final.pt")

        # Explicit --save-weights path (in addition to auto-save)
        if save_weights_path:
            try:
                save_weights(brain, Path(save_weights_path))
            except Exception:
                logger.exception(
                    "Failed to save weights to explicit path %s",
                    save_weights_path,
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
        session_id=session_id,
        num_runs=total_runs_done,
        max_steps=max_steps,
        all_results=all_results,
        env_type=agent.env,
    )

    # Generate plots and exports after the simulation (skip if no results)
    if all_results:
        plot_results(
            all_results=all_results,
            metrics=metrics,
            plot_dir=plot_dir,
            last_run_satiety_history=last_run_satiety_history,
            last_run_health_history=last_run_health_history,
        )
        export_simulation_results_to_csv(
            all_results=all_results,
            data_dir=data_dir,
            skip_main_results=True,
            skip_path_data=True,
        )
        export_performance_metrics_to_csv(metrics=metrics, data_dir=data_dir)
    else:
        logger.warning("No completed runs - skipping plots and data export.")

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
        and (r.predator_encounters > 0 or r.successful_evasions or r.died_to_health_depletion)
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

    # Export tracking data to CSV files (detailed data already written incrementally)
    export_tracking_data_to_csv(
        tracking_data=tracking_data,
        brain_type=brain_type,
        data_dir=data_dir,
        qubits=qubits,
        skip_detailed=True,
    )

    # Experiment tracking (opt-in)
    if track_experiment:
        try:
            # Capture experiment metadata
            config_path = Path(config_file) if config_file else Path("config.yml")
            exports_rel_path = f"exports/{session_id}"

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
                session_id=session_id,
                precomputed_chemotaxis=chemotaxis_metrics_per_run,
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
    if brain_type in (BrainType.MLP_REINFORCE, BrainType.MLP_DQN) and qubits != DEFAULT_QUBITS:
        error_message = (
            f"The 'qubits' parameter is only supported by "
            "quantum brain architectures. "
            f"Provided brain: {brain_type.value}, qubits: {qubits}."
        )
        logger.error(error_message)
        raise ValueError(error_message)


def plot_results(  # noqa: C901, PLR0912, PLR0913, PLR0915
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,
    plot_dir: Path,
    file_prefix: str = "",
    last_run_satiety_history: list[float] | None = None,
    last_run_health_history: list[float] | None = None,
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
            # Get actual max satiety from snapshot or history
            first_result = foraging_results[0]
            if first_result.max_satiety is not None:
                max_satiety = first_result.max_satiety
            elif first_result.satiety_history:
                max_satiety = max(first_result.satiety_history)
            plot_satiety_at_episode_end(
                file_prefix,
                foraging_runs,
                plot_dir,
                satiety_remaining_list,
                max_satiety,
            )

        # Plot: Health at Episode End (for runs with health system enabled)
        health_results = [
            r for r in foraging_results if r.final_health is not None or r.health_history
        ]
        if health_results:
            health_remaining_list = [
                r.final_health
                if r.final_health is not None
                else (r.health_history[-1] if r.health_history else 0.0)
                for r in health_results
                if r.final_health is not None or r.health_history
            ]
            health_runs = [r.run for r in health_results]
            # Get max health from snapshot or history
            first_health = health_results[0]
            if first_health.max_health is not None:
                max_health = first_health.max_health
            elif first_health.health_history:
                max_health = max(first_health.health_history)
            else:
                max_health = 100.0
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
        and (r.predator_encounters > 0 or r.successful_evasions or r.died_to_health_depletion)
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
            if r.foods_collected is not None and r.died_to_health_depletion is not None
        ]
        if predator_foraging_results:
            foods_in_predator = []
            deaths_in_predator = []
            for r in predator_foraging_results:
                foods_in_predator.append(r.foods_collected)
                deaths_in_predator.append(r.died_to_health_depletion)

            if foods_in_predator and deaths_in_predator:
                plot_survival_vs_food_collection(
                    file_prefix,
                    plot_dir,
                    foods_in_predator,
                    deaths_in_predator,
                )

        # Plot: Satiety Progression for Single Run (sample for predator environments)
        last_run = predator_results[-1]
        satiety_hist = last_run.satiety_history or last_run_satiety_history
        health_hist = last_run.health_history or last_run_health_history
        if satiety_hist:
            max_satiety_pred = max(satiety_hist)
            plot_satiety_progression_single_run(
                file_prefix,
                plot_dir,
                last_run.run,
                satiety_hist,
                max_satiety_pred,
            )

        # Plot: Health Progression for Single Run (for health-enabled environments)
        if health_hist:
            max_health_pred = max(health_hist)
            plot_health_progression_single_run(
                file_prefix,
                plot_dir,
                last_run.run,
                health_hist,
                max_health_pred,
            )


def _run_multi_agent(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    config: SimulationConfig,
    multi_agent_config: MultiAgentConfig,
    environment_config: EnvironmentConfig,
    reward_config: RewardConfig,
    satiety_config: SatietyConfig,
    sensing_config: SensingConfig,
    simulation_seed: int,
    max_steps: int,
    shots: int,
    body_length: int,
    qubits: int,
    device: DeviceType,
    runs: int,
    session_id: str,
    learning_rate: ConstantLearningRate
    | DynamicLearningRate
    | AdamLearningRate
    | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,
    gradient_max_norm: float | None,
    parameter_initializer_config: ParameterInitializerConfig,
    theme: Theme,
    perf_mgmt: Any | None = None,  # noqa: ANN401 - RunnableQiskitFunction conditional import
) -> None:
    """Run a multi-agent simulation session.

    Called from main() when multi_agent.enabled=True. Creates multiple agents
    with independent brains in a shared environment and runs episodes via
    MultiAgentSimulation.
    """
    from quantumnematode.agent.multi_agent import (
        FoodCompetitionPolicy,
        MultiAgentSimulation,
        validate_multi_agent_grid,
    )

    # Determine agent configs
    if multi_agent_config.agents:
        agent_configs = multi_agent_config.agents
    else:
        # Homogeneous: create N agents with top-level brain config
        if config.brain is None:
            msg = "multi_agent.count requires a top-level 'brain' config."
            raise ValueError(msg)
        if multi_agent_config.count is None:
            msg = "multi_agent must specify 'count' or 'agents'."
            raise ValueError(msg)
        agent_configs = [
            AgentConfig(
                id=f"agent_{i}",
                brain=config.brain,
            )
            for i in range(multi_agent_config.count)
        ]

    num_agents = len(agent_configs)
    grid_size = environment_config.grid_size

    # Validate grid size
    validate_multi_agent_grid(grid_size, num_agents)

    logger.info(f"Multi-agent mode: {num_agents} agents, grid {grid_size}x{grid_size}")
    logger.info(f"Food competition: {multi_agent_config.food_competition}")
    logger.info(f"Termination policy: {multi_agent_config.termination_policy}")

    # Create initial shared environment (seeded for run 0)
    initial_run_seed = derive_run_seed(simulation_seed, 0)
    env = create_env_from_config(
        environment_config,
        seed=initial_run_seed,
        max_body_length=body_length,
        theme=theme,
    )

    # Create agents with independent brains
    agents: list[QuantumNematodeAgent] = []

    for ac in agent_configs:
        # Derive per-agent seed (stable across runs, unlike Python's hash())
        agent_seed = int.from_bytes(
            hashlib.blake2b(f"{simulation_seed}:{ac.id}".encode(), digest_size=4).digest(),
            "little",
        )

        # Configure brain for this agent
        agent_brain_config = _configure_brain_for_agent(ac.brain, agent_seed, sensing_config)

        agent_brain = setup_brain_model(
            brain_type=BrainType(ac.brain.name),
            brain_config=agent_brain_config,
            shots=shots,
            qubits=qubits,
            device=device,
            learning_rate=learning_rate,
            gradient_method=gradient_method,
            gradient_max_norm=gradient_max_norm,
            parameter_initializer_config=parameter_initializer_config,
            perf_mgmt=perf_mgmt,
        )

        # Load pre-trained weights if specified
        if ac.weights_path:
            if not isinstance(agent_brain, WeightPersistence):
                msg = (
                    f"Agent '{ac.id}' brain {type(agent_brain).__name__} does not implement "
                    f"WeightPersistence. Cannot load weights from {ac.weights_path}."
                )
                raise TypeError(msg)
            load_weights(agent_brain, Path(ac.weights_path))

        # Add agent to environment with min_distance separation
        env.add_agent(
            agent_id=ac.id,
            position=None,
            max_body_length=body_length,
            min_distance=multi_agent_config.min_agent_distance,
        )

        agent = QuantumNematodeAgent(
            brain=agent_brain,
            env=env,
            max_body_length=body_length,
            theme=theme,
            satiety_config=satiety_config,
            sensing_config=sensing_config,
            agent_id=ac.id,
        )
        agents.append(agent)

    # Create orchestrator
    food_policy = FoodCompetitionPolicy(multi_agent_config.food_competition)
    sim = MultiAgentSimulation(
        env=env,
        agents=agents,
        food_policy=food_policy,
        social_detection_radius=multi_agent_config.social_detection_radius,
        termination_policy=multi_agent_config.termination_policy,
    )

    # Set up export directories
    data_dir = Path.cwd() / "exports" / session_id / "session" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Multi-Agent Simulation: {num_agents} agents")
    print(f"Session: {session_id}")
    print(f"{'=' * 60}\n")

    # Run episodes
    for run in range(runs):
        run_seed = derive_run_seed(simulation_seed, run)
        set_global_seed(run_seed)

        # Create fresh env for this episode (run 0 uses initial env above)
        if run > 0:
            env = create_env_from_config(
                environment_config,
                seed=run_seed,
                max_body_length=body_length,
                theme=theme,
            )
            for ac in agent_configs:
                env.add_agent(
                    agent_id=ac.id,
                    position=None,
                    max_body_length=body_length,
                    min_distance=multi_agent_config.min_agent_distance,
                )
            for agent in agents:
                agent.env = env
                pos = env.agents[agent.agent_id].position
                agent.path = [(pos[0], pos[1])]
                agent.food_history = [list(env.foods)]
                agent._food_handler.env = env
                agent._satiety_manager.reset()
                agent._food_handler.reset()
                agent._episode_tracker.reset()
                agent.reset_brain()
            sim.env = env

        result = sim.run_episode(reward_config, max_steps)

        # Log per-episode results
        print(
            f"Run {run + 1}/{runs}: "
            f"food={result.total_food_collected}, "
            f"competition={result.food_competition_events}, "
            f"alive={result.agents_alive_at_end}/{num_agents}, "
            f"success={result.mean_agent_success:.0%}, "
            f"gini={result.food_gini_coefficient:.2f}",
        )

        for aid, agent_result in result.agent_results.items():
            print(
                f"  {aid}: {agent_result.termination_reason.value}, "
                f"food={result.per_agent_food.get(aid, 0)}, "
                f"steps={len(agent_result.agent_path)}",
            )

    # Save weights per agent
    weights_dir = data_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        if isinstance(agent.brain, WeightPersistence):
            weight_file = (
                "final.pt" if agent.agent_id == "default" else f"final_{agent.agent_id}.pt"
            )
            try:
                save_weights(agent.brain, weights_dir / weight_file)
            except Exception:
                logger.exception("Failed to save weights for %s", agent.agent_id)

    print(f"\nSession complete. Results in exports/{session_id}/")


def _configure_brain_for_agent(
    brain_container: BrainContainerConfig,
    seed: int,
    sensing_config: SensingConfig,
) -> BrainConfigType:
    """Configure a brain from a BrainContainerConfig with sensing mode translation.

    Parameters
    ----------
    brain_container : BrainContainerConfig
        Brain container config from YAML.
    seed : int
        Seed for the brain.
    sensing_config : SensingConfig
        Sensing configuration for module translation.

    Returns
    -------
    BrainConfigType
        Configured brain config ready for setup_brain_model().
    """
    from quantumnematode.utils.config_loader import configure_brain_from_container

    brain_config = configure_brain_from_container(brain_container)
    brain_config = brain_config.model_copy(update={"seed": seed})

    # Apply sensing mode translation
    sensory_modules_attr = getattr(brain_config, "sensory_modules", None)
    if sensory_modules_attr is not None:
        from quantumnematode.brain.modules import ModuleName
        from quantumnematode.utils.config_loader import apply_sensing_mode

        original_modules = [m.value for m in sensory_modules_attr]
        translated = apply_sensing_mode(original_modules, sensing_config)
        translated_modules = [ModuleName(m) for m in translated]
        if translated_modules != list(sensory_modules_attr):
            brain_config = brain_config.model_copy(
                update={"sensory_modules": translated_modules},
            )

    return brain_config


if __name__ == "__main__":
    main()
