# pragma: no cover

"""Experiment metadata capture and tracking integration."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from quantumnematode.benchmark.convergence import analyze_convergence
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env import DynamicForagingEnvironment
from quantumnematode.env.env import StaticEnvironment
from quantumnematode.experiment.git_utils import capture_git_context, get_relative_config_path
from quantumnematode.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    GradientMetadata,
    LearningRateMetadata,
    ParameterInitializer,
    ResultsMetadata,
    RewardMetadata,
    SystemMetadata,
)
from quantumnematode.experiment.system_utils import capture_system_info
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult, TerminationReason
from quantumnematode.validation.chemotaxis import (
    ChemotaxisMetrics,
    calculate_chemotaxis_metrics_stepwise,
    get_validation_level,
)
from quantumnematode.validation.datasets import ChemotaxisValidationBenchmark


def compute_config_hash(config_path: Path) -> str:
    """Compute SHA256 hash of configuration file.

    Parameters
    ----------
    config_path : Path
        Path to configuration file.

    Returns
    -------
    str
        SHA256 hash of file content.
    """
    with config_path.open("rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()


def extract_environment_metadata(
    env: DynamicForagingEnvironment | StaticEnvironment,
    satiety_config: dict,
) -> EnvironmentMetadata:
    """Extract environment metadata from environment instance.

    Parameters
    ----------
    env : DynamicForagingEnvironment | StaticEnvironment
        Environment instance.
    satiety_config : dict
        Satiety configuration dictionary.

    Returns
    -------
    EnvironmentMetadata
        Environment metadata.
    """
    if isinstance(env, DynamicForagingEnvironment):
        return EnvironmentMetadata(
            type="dynamic",
            grid_size=env.grid_size,
            num_foods=env.foods_on_grid,
            target_foods_to_collect=env.target_foods_to_collect,
            initial_satiety=satiety_config.get("initial"),
            satiety_decay_rate=satiety_config.get("decay_rate"),
            viewport_size=list(env.viewport_size) if hasattr(env, "viewport_size") else None,
            predators_enabled=env.predators_enabled,
            num_predators=env.num_predators if env.predators_enabled else None,
            predator_speed=env.predator_speed if env.predators_enabled else None,
            predator_detection_radius=env.predator_detection_radius
            if env.predators_enabled
            else None,
            predator_kill_radius=env.predator_kill_radius if env.predators_enabled else None,
            predator_gradient_decay=env.predator_gradient_decay if env.predators_enabled else None,
            predator_gradient_strength=env.predator_gradient_strength
            if env.predators_enabled
            else None,
        )
    # Static environment
    return EnvironmentMetadata(
        type="static",
        grid_size=env.grid_size,
    )


def extract_brain_metadata(
    brain_type: str,
    config: dict,
    parameter_initializer_config: dict | None = None,
) -> BrainMetadata:
    """Extract brain metadata from brain type and configuration.

    Parameters
    ----------
    brain_type : str
        Brain type string.
    config : dict
        Brain configuration dictionary.
    parameter_initializer_config : dict
        Parameter initializer configuration dictionary.

    Returns
    -------
    BrainMetadata
        Brain metadata.
    """
    brain_config = config.get("config", {})

    # Extract common parameters
    qubits = config.get("qubits")
    shots = config.get("shots")

    # Learning rate: prioritize brain-specific LR over global LR config
    # Some brains (spiking, mlp) have their own internal optimizer with LR in brain config
    brain_learning_rate = brain_config.get("learning_rate")
    learning_rate_config = config.get("learning_rate")
    learning_rate = None
    if brain_learning_rate is not None:
        # Use brain-specific learning rate (for spiking, mlp, etc.)
        learning_rate = brain_learning_rate
    elif isinstance(learning_rate_config, dict):
        # Fall back to global learning rate config (for modular, quantum brains)
        learning_rate = learning_rate_config.get("initial_learning_rate", None)

    # Extract architecture-specific parameters
    num_layers = brain_config.get("num_layers")
    hidden_dim = brain_config.get("hidden_dim")
    num_hidden_layers = brain_config.get("num_hidden_layers")
    modules = brain_config.get("modules")

    # Extract parameter initializer information
    parameter_initializer = None
    if parameter_initializer_config is not None and isinstance(parameter_initializer_config, dict):
        initializer_type = parameter_initializer_config.get("type")
        if initializer_type is not None:
            # If manual initialization, capture the parameter values
            manual_values = None
            if initializer_type == "manual":
                manual_values = parameter_initializer_config.get("manual_parameter_values")

            parameter_initializer = ParameterInitializer(
                type=initializer_type,
                manual_parameter_values=manual_values,
            )

    return BrainMetadata(
        type=brain_type,
        qubits=qubits,
        shots=shots,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
        modules=modules,
        parameter_initializer=parameter_initializer,
    )


def extract_reward_metadata(config: dict) -> RewardMetadata:
    """Extract reward configuration metadata.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    RewardMetadata
        Reward configuration metadata.
    """
    reward_config = config.get("reward", {})
    return RewardMetadata(
        reward_goal=reward_config.get("reward_goal"),
        reward_distance_scale=reward_config.get("reward_distance_scale"),
        reward_exploration=reward_config.get("reward_exploration"),
        penalty_step=reward_config.get("penalty_step"),
        penalty_anti_dithering=reward_config.get("penalty_anti_dithering"),
        penalty_stuck_position=reward_config.get("penalty_stuck_position"),
        stuck_position_threshold=reward_config.get("stuck_position_threshold"),
        penalty_starvation=reward_config.get("penalty_starvation"),
        penalty_predator_death=reward_config.get("penalty_predator_death"),
        penalty_predator_proximity=reward_config.get("penalty_predator_proximity"),
    )


def extract_learning_rate_metadata(config: dict) -> LearningRateMetadata:
    """Extract learning rate configuration metadata.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    LearningRateMetadata
        Learning rate configuration metadata.
    """
    lr_config = config.get("learning_rate", {})

    # Get method
    method = lr_config.get("method")

    # Get parameters
    params = lr_config.get("parameters", {})
    initial_lr = params.get("initial_learning_rate")
    decay_type = params.get("decay_type")
    decay_rate = params.get("decay_rate")
    decay_factor = params.get("decay_factor")
    step_size = params.get("step_size")
    max_steps = params.get("max_steps")
    power = params.get("power")
    min_lr = params.get("min_lr")

    return LearningRateMetadata(
        method=method,
        initial_learning_rate=initial_lr,
        decay_type=decay_type,
        decay_rate=decay_rate,
        decay_factor=decay_factor,
        step_size=step_size,
        min_lr=min_lr,
        max_steps=max_steps,
        power=power,
    )


def extract_gradient_metadata(config: dict) -> GradientMetadata:
    """Extract gradient calculation method metadata.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    GradientMetadata
        Gradient calculation method metadata.
    """
    gradient_config = config.get("gradient", {})
    method = gradient_config.get("method")
    max_norm = gradient_config.get("max_norm")
    return GradientMetadata(method=method, max_norm=max_norm)


def aggregate_results_metadata(all_results: list[SimulationResult]) -> ResultsMetadata:  # noqa: PLR0915, C901
    """Aggregate simulation results into metadata.

    Parameters
    ----------
    all_results : list[SimulationResult]
        List of simulation results.

    Returns
    -------
    ResultsMetadata
        Aggregated results metadata.
    """
    total_runs = len(all_results)
    if total_runs == 0:
        return ResultsMetadata(
            total_runs=0,
            success_rate=0.0,
            avg_steps=0.0,
            avg_reward=0.0,
            converged=False,
        )

    # Count successes
    successes = sum(1 for r in all_results if r.success)
    success_rate = successes / total_runs

    # Calculate averages
    avg_steps = sum(r.steps for r in all_results) / total_runs
    avg_reward = sum(r.total_reward for r in all_results) / total_runs

    # Foraging-specific metrics
    foods_collected = [r.foods_collected for r in all_results if r.foods_collected is not None]
    avg_foods_collected = sum(foods_collected) / len(foods_collected) if foods_collected else None

    distance_effs = [
        r.average_distance_efficiency
        for r in all_results
        if r.average_distance_efficiency is not None
    ]
    avg_distance_efficiency = sum(distance_effs) / len(distance_effs) if distance_effs else None

    # Count termination reasons
    completed_all_food = sum(
        1 for r in all_results if r.termination_reason == TerminationReason.COMPLETED_ALL_FOOD
    )
    starved = sum(1 for r in all_results if r.termination_reason == TerminationReason.STARVED)
    max_steps_reached = sum(
        1 for r in all_results if r.termination_reason == TerminationReason.MAX_STEPS
    )
    goal_reached = sum(
        1 for r in all_results if r.termination_reason == TerminationReason.GOAL_REACHED
    )

    # Predator-specific metrics
    predator_deaths = sum(
        1 for r in all_results if r.termination_reason == TerminationReason.PREDATOR
    )
    predator_encounters_list = [
        r.predator_encounters for r in all_results if r.predator_encounters is not None
    ]
    avg_predator_encounters = (
        sum(predator_encounters_list) / len(predator_encounters_list)
        if predator_encounters_list
        else None
    )
    successful_evasions_list = [
        r.successful_evasions for r in all_results if r.successful_evasions is not None
    ]
    avg_successful_evasions = (
        sum(successful_evasions_list) / len(successful_evasions_list)
        if successful_evasions_list
        else None
    )

    # CONVERGENCE ANALYSIS
    convergence_metrics = analyze_convergence(all_results, total_runs)

    # CHEMOTAXIS VALIDATION (for dynamic foraging environments)
    avg_chemotaxis_index = None
    avg_time_in_attractant = None
    avg_approach_frequency = None
    avg_path_efficiency = None
    post_convergence_chemotaxis_index = None
    post_convergence_time_in_attractant = None
    post_convergence_approach_frequency = None
    post_convergence_path_efficiency = None
    chemotaxis_validation_level = None
    biological_ci_range = None
    biological_ci_typical = None
    matches_biology = None
    literature_source = None

    # Helper function to calculate chemotaxis metrics for a result
    def _calculate_result_metrics(result: SimulationResult) -> ChemotaxisMetrics | None:
        if not result.food_history:
            return None
        positions = [(float(x), float(y)) for x, y in result.path]
        food_history_float = [
            [(float(x), float(y)) for x, y in step_foods] for step_foods in result.food_history
        ]
        if not food_history_float:
            return None
        return calculate_chemotaxis_metrics_stepwise(
            positions=positions,
            food_history=food_history_float,
            attractant_zone_radius=5.0,
        )

    # Calculate chemotaxis metrics for all runs with food history
    runs_with_food_history = [r for r in all_results if r.food_history]
    if runs_with_food_history:
        all_metrics: list[ChemotaxisMetrics] = []
        for result in runs_with_food_history:
            metrics = _calculate_result_metrics(result)
            if metrics:
                all_metrics.append(metrics)

        if all_metrics:
            # Calculate all-run averages
            avg_chemotaxis_index = sum(m.chemotaxis_index for m in all_metrics) / len(all_metrics)
            avg_time_in_attractant = sum(m.time_in_attractant for m in all_metrics) / len(
                all_metrics,
            )
            avg_approach_frequency = sum(m.approach_frequency for m in all_metrics) / len(
                all_metrics,
            )
            avg_path_efficiency = sum(m.path_efficiency for m in all_metrics) / len(all_metrics)

        # Calculate post-convergence metrics (used for biological validation)
        convergence_run = convergence_metrics.convergence_run
        if convergence_run is not None:
            # Filter to post-convergence runs with food history
            post_conv_runs = [r for r in all_results if r.run > convergence_run and r.food_history]
        else:
            # If not converged, use all runs with food history
            post_conv_runs = runs_with_food_history

        post_conv_metrics: list[ChemotaxisMetrics] = []
        for result in post_conv_runs:
            metrics = _calculate_result_metrics(result)
            if metrics:
                post_conv_metrics.append(metrics)

        if post_conv_metrics:
            # Calculate post-convergence averages
            post_convergence_chemotaxis_index = sum(
                m.chemotaxis_index for m in post_conv_metrics
            ) / len(post_conv_metrics)
            post_convergence_time_in_attractant = sum(
                m.time_in_attractant for m in post_conv_metrics
            ) / len(post_conv_metrics)
            post_convergence_approach_frequency = sum(
                m.approach_frequency for m in post_conv_metrics
            ) / len(post_conv_metrics)
            post_convergence_path_efficiency = sum(
                m.path_efficiency for m in post_conv_metrics
            ) / len(post_conv_metrics)

            # Validation level based on post-convergence CI (trained behavior)
            chemotaxis_validation_level = get_validation_level(
                post_convergence_chemotaxis_index,
            ).value

            # Use benchmark to compare against biological literature
            benchmark = ChemotaxisValidationBenchmark()
            validation_stats = benchmark.validate_multiple_runs(post_conv_metrics)
            if validation_stats["num_runs"] > 0:
                # Get source information from a single validation
                sample_result = benchmark.validate_agent(post_conv_metrics[0])
                biological_ci_range = sample_result.biological_ci_range
                biological_ci_typical = sample_result.biological_ci_typical
                literature_source = sample_result.literature_source
                # Check if post-convergence CI falls within biological range
                matches_biology = (
                    biological_ci_range[0]
                    <= post_convergence_chemotaxis_index
                    <= biological_ci_range[1]
                )

    return ResultsMetadata(
        total_runs=total_runs,
        success_rate=success_rate,
        avg_steps=avg_steps,
        avg_reward=avg_reward,
        avg_foods_collected=avg_foods_collected,
        avg_distance_efficiency=avg_distance_efficiency,
        completed_all_food=completed_all_food,
        starved=starved,
        max_steps_reached=max_steps_reached,
        goal_reached=goal_reached,
        predator_deaths=predator_deaths,
        avg_predator_encounters=avg_predator_encounters,
        avg_successful_evasions=avg_successful_evasions,
        # Convergence-based metrics
        converged=convergence_metrics.converged,
        convergence_run=convergence_metrics.convergence_run,
        runs_to_convergence=convergence_metrics.runs_to_convergence,
        post_convergence_success_rate=convergence_metrics.post_convergence_success_rate,
        post_convergence_avg_steps=convergence_metrics.post_convergence_avg_steps,
        post_convergence_avg_foods=convergence_metrics.post_convergence_avg_foods,
        post_convergence_variance=convergence_metrics.post_convergence_variance,
        post_convergence_distance_efficiency=convergence_metrics.distance_efficiency,
        composite_benchmark_score=convergence_metrics.composite_score,
        # Chemotaxis validation metrics
        avg_chemotaxis_index=avg_chemotaxis_index,
        avg_time_in_attractant=avg_time_in_attractant,
        avg_approach_frequency=avg_approach_frequency,
        avg_path_efficiency=avg_path_efficiency,
        post_convergence_chemotaxis_index=post_convergence_chemotaxis_index,
        post_convergence_time_in_attractant=post_convergence_time_in_attractant,
        post_convergence_approach_frequency=post_convergence_approach_frequency,
        post_convergence_path_efficiency=post_convergence_path_efficiency,
        chemotaxis_validation_level=chemotaxis_validation_level,
        biological_ci_range=biological_ci_range,
        biological_ci_typical=biological_ci_typical,
        matches_biology=matches_biology,
        literature_source=literature_source,
    )


def capture_experiment_metadata(
    config_path: Path,
    env: DynamicForagingEnvironment | StaticEnvironment,
    brain_type: str,
    config: dict,
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,  # noqa: ARG001
    device_type: DeviceType,
    qpu_backend: str | None = None,
    exports_path: str | None = None,
    session_id: str | None = None,
) -> ExperimentMetadata:
    """Capture complete experiment metadata.

    Parameters
    ----------
    config_path : Path
        Path to configuration file.
    env : DynamicForagingEnvironment | StaticEnvironment
        Environment instance.
    brain_type : str
        Brain type string.
    config : dict
        Full configuration dictionary.
    all_results : list[SimulationResult]
        List of all simulation results.
    metrics : PerformanceMetrics
        Performance metrics.
    device_type : DeviceType
        Device type used.
    qpu_backend : str | None, optional
        QPU backend name if using quantum hardware.
    exports_path : str | None, optional
        Path to exports directory.
    session_id : str | None, optional
        Session ID to use as experiment ID. If not provided, generates new timestamp.

    Returns
    -------
    ExperimentMetadata
        Complete experiment metadata.
    """
    # Use session ID as experiment ID if provided, otherwise generate new timestamp
    timestamp = datetime.now(UTC)
    experiment_id = session_id if session_id is not None else timestamp.strftime("%Y%m%d_%H%M%S")

    # Capture git context
    git_context = capture_git_context()

    # Get relative config path
    relative_config_path = get_relative_config_path(config_path)

    # Compute config hash
    config_hash = compute_config_hash(config_path)

    # Extract metadata from components
    environment_metadata = extract_environment_metadata(env, config.get("satiety", {}))
    brain_metadata = extract_brain_metadata(
        brain_type=brain_type,
        config=config.get("brain", {}),
        parameter_initializer_config=config.get("parameter_initializer"),
    )
    reward_metadata = extract_reward_metadata(config)
    learning_rate_metadata = extract_learning_rate_metadata(config)
    gradient_metadata = extract_gradient_metadata(config)
    results_metadata = aggregate_results_metadata(all_results)
    system_metadata_dict = capture_system_info(device_type, qpu_backend)

    # Create SystemMetadata from captured info
    system_metadata = SystemMetadata(
        python_version=str(system_metadata_dict["python_version"]),
        qiskit_version=str(system_metadata_dict["qiskit_version"]),
        torch_version=system_metadata_dict.get("torch_version"),
        device_type=str(system_metadata_dict["device_type"]),
        qpu_backend=system_metadata_dict.get("qpu_backend"),
    )

    # Extract git context with proper types
    git_commit = git_context["git_commit"]
    git_branch = git_context["git_branch"]
    git_dirty = git_context["git_dirty"]

    return ExperimentMetadata(
        experiment_id=experiment_id,
        timestamp=timestamp,
        config_file=relative_config_path,
        config_hash=config_hash,
        git_commit=git_commit if isinstance(git_commit, str) else None,
        git_branch=git_branch if isinstance(git_branch, str) else None,
        git_dirty=bool(git_dirty),
        environment=environment_metadata,
        brain=brain_metadata,
        reward=reward_metadata,
        learning_rate=learning_rate_metadata,
        gradient=gradient_metadata,
        results=results_metadata,
        system=system_metadata,
        exports_path=exports_path,
    )
