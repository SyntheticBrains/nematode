"""Data models for experiment and benchmark metadata."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class PerRunResult(BaseModel):
    """Per-run result data for scientific reproducibility and analysis.

    This lightweight model captures essential per-run metrics without
    heavy data like paths or food histories. It enables:
    - Reproducibility verification (comparing individual run outcomes)
    - Learning curve analysis (plotting performance over time)
    - Custom statistical analysis by researchers
    - Debugging (identifying problematic runs)

    Attributes
    ----------
    run : int
        1-indexed run number.
    seed : int
        Random seed used for this run.
    success : bool
        Whether the run was successful.
    steps : int
        Number of steps taken.
    total_reward : float
        Total reward achieved.
    termination_reason : str
        How the run ended (e.g., "completed_all_food", "starved", "max_steps").
    foods_collected : int | None
        Foods collected (dynamic environments only).
    distance_efficiency : float | None
        Navigation efficiency for this run (dynamic environments only).
    """

    run: int
    seed: int
    success: bool
    steps: int
    total_reward: float
    termination_reason: str
    foods_collected: int | None = None
    distance_efficiency: float | None = None


class EnvironmentMetadata(BaseModel):
    """Metadata about the simulation environment.

    Attributes
    ----------
    type : str
        Environment type ("static" or "dynamic").
    grid_size : int
        Size of the grid environment.
    num_foods : int | None
        Number of food items on grid (constant supply, dynamic environments only).
    target_foods_to_collect : int | None
        Target foods to collect for victory (dynamic environments only).
    initial_satiety : float | None
        Initial satiety level (dynamic environments only).
    satiety_decay_rate : float | None
        Satiety decay per step (dynamic environments only).
    viewport_size : list[int] | None
        Viewport dimensions [height, width] (dynamic environments only).
    predators_enabled : bool
        Whether predators are enabled (dynamic environments only).
    num_predators : int | None
        Number of predators (dynamic environments with predators only).
    predator_speed : float | None
        Predator movement speed (predator environments only).
    predator_detection_radius : int | None
        Predator detection radius (predator environments only).
    predator_kill_radius : int | None
        Predator kill radius (predator environments only).
    predator_gradient_decay : float | None
        Predator gradient decay constant (predator environments only).
    predator_gradient_strength : float | None
        Predator gradient strength (predator environments only).
    """

    type: str
    grid_size: int
    num_foods: int | None = None
    target_foods_to_collect: int | None = None
    initial_satiety: float | None = None
    satiety_decay_rate: float | None = None
    viewport_size: list[int] | None = None
    predators_enabled: bool = False
    num_predators: int | None = None
    predator_speed: float | None = None
    predator_detection_radius: int | None = None
    predator_kill_radius: int | None = None
    predator_gradient_decay: float | None = None
    predator_gradient_strength: float | None = None


class ParameterInitializer(BaseModel):
    """Parameter initialization configuration.

    Attributes
    ----------
    type : str
        Type of parameter initializer ("manual", "random_pi", "random_small", "zero", etc.).
    manual_parameter_values : dict[str, float] | None
        Manual parameter initialization values (only when type is "manual").
    """

    type: str
    manual_parameter_values: dict[str, float] | None = None


class RewardMetadata(BaseModel):
    """Metadata about reward function configuration.

    Attributes
    ----------
    reward_goal : float
        Reward for reaching the goal.
    reward_distance_scale : float
        Scale factor for distance-based rewards.
    reward_exploration : float
        Bonus reward for visiting new cells.
    penalty_step : float
        Penalty applied per step.
    penalty_anti_dithering : float
        Penalty for oscillating back to previous position.
    penalty_stuck_position : float
        Penalty for staying in same position.
    stuck_position_threshold : int
        Number of steps before stuck penalty applies.
    penalty_starvation : float
        Penalty when satiety reaches zero (dynamic environments only).
    penalty_predator_death : float
        Penalty when caught by predator (predator environments only).
    penalty_predator_proximity : float
        Penalty per step within predator detection radius (predator environments only).
    """

    reward_goal: float
    reward_distance_scale: float
    reward_exploration: float
    penalty_step: float
    penalty_anti_dithering: float
    penalty_stuck_position: float
    stuck_position_threshold: int
    penalty_starvation: float
    penalty_predator_death: float
    penalty_predator_proximity: float


class LearningRateMetadata(BaseModel):
    """Metadata about learning rate configuration.

    Attributes
    ----------
    method : str
        Learning rate method ("adam", "dynamic", "performance_based").
    initial_learning_rate : float
        Initial learning rate value.
    decay_type : str | None
        Type of decay ("exponential", "step", etc.) for dynamic method.
    decay_rate : float | None
        Decay rate parameter for dynamic method.
    decay_factor : float | None
        Decay factor for step decay.
    step_size : int | None
        Step size for step decay.
    min_lr : float | None
        Minimum learning rate for dynamic method.
    max_steps : int | None
        Maximum number of steps for decay.
    power : float | None
        Power parameter for polynomial decay.
    """

    method: str
    initial_learning_rate: float
    decay_type: str | None = None
    decay_rate: float | None = None
    decay_factor: float | None = None
    min_lr: float | None = None
    step_size: int | None = None
    max_steps: int | None = None
    power: float | None = None


class GradientMetadata(BaseModel):
    """Metadata about gradient calculation method.

    Attributes
    ----------
    method : str | None
        Gradient calculation method ("raw", "clip", "normalize").
    max_norm : float | None
        Maximum norm for gradient clipping (only for "norm_clip" method).
    """

    method: str | None = None
    max_norm: float | None = None


class BrainMetadata(BaseModel):
    """Metadata about the brain architecture.

    Attributes
    ----------
    type : str
        Brain architecture type ("modular", "mlp", "qmodular", "qmlp", "spiking").
    qubits : int | None
        Number of qubits (quantum brains only).
    shots : int | None
        Number of measurement shots (quantum brains only).
    num_layers : int | None
        Number of circuit layers (quantum brains only).
    hidden_dim : int | None
        Hidden layer dimension (classical brains only).
    num_hidden_layers : int | None
        Number of hidden layers (classical brains only).
    learning_rate : float
        Initial learning rate.
    modules : dict[str, list[int]] | None
        Module configuration (modular brains only).
    parameter_initializer : ParameterInitializer | None
        Parameter initialization configuration (quantum brains only).
    """

    type: str
    qubits: int | None = None
    shots: int | None = None
    num_layers: int | None = None
    hidden_dim: int | None = None
    num_hidden_layers: int | None = None
    learning_rate: float | None = None
    modules: dict[str, list[int]] | None = None
    parameter_initializer: ParameterInitializer | None = None


class ResultsMetadata(BaseModel):
    """Aggregated results from simulation runs.

    Attributes
    ----------
    total_runs : int
        Total number of runs completed.
    success_rate : float
        Proportion of successful runs.
    avg_steps : float
        Average steps per run.
    avg_reward : float
        Average total reward per run.
    avg_foods_collected : float | None
        Average foods collected (dynamic environments only).
    avg_distance_efficiency : float | None
        Average distance efficiency across all runs (dynamic environments only).
    completed_all_food : int
        Number of runs that collected all foods.
    starved : int
        Number of runs that ended in starvation.
    max_steps_reached : int
        Number of runs that reached max steps.
    goal_reached : int
        Number of runs that reached the goal.
    predator_deaths : int
        Number of runs that ended due to predator collision.
    avg_predator_encounters : float | None
        Average predator encounters per run (predator environments only).
    avg_successful_evasions : float | None
        Average successful evasions per run (predator environments only).
    converged : bool
        Whether the learning strategy converged within the session.
    convergence_run : int | None
        Zero-based index of run where convergence was detected (None if never converged).
    runs_to_convergence : int | None
        Number of runs required to reach convergence (None if never converged).
    post_convergence_success_rate : float | None
        Success rate after convergence point (or last N runs if not converged).
    post_convergence_avg_steps : float | None
        Average steps in successful runs after convergence.
    post_convergence_avg_foods : float | None
        Average foods collected after convergence (dynamic environments only).
    post_convergence_variance : float | None
        Variance in success rate after convergence (measures stability).
    post_convergence_distance_efficiency : float | None
        Average distance efficiency in successful runs
        after convergence (dynamic environments only).
        Range: 0.0 to 1.0, where 1.0 means perfect optimal navigation.
    composite_benchmark_score : float | None
        Weighted composite score combining success, efficiency, speed, and stability.
    learning_speed : float | None
        Normalized learning speed (0.0 to 1.0, higher = faster learning).
        Calculated as 1.0 - (episodes_to_80_success / max_episodes).
    learning_speed_episodes : int | None
        Number of episodes to reach 80% rolling success rate.
        None if 80% success was never achieved.
    stability : float | None
        Normalized stability metric (0.0 to 1.0, higher = more consistent).
        Calculated as 1.0 - coefficient_of_variation, clamped to [0, 1].
    per_run_results : list[PerRunResult] | None
        Per-run result data for full transparency and reproducibility verification.
        Contains essential metrics for each run including seeds.
    avg_chemotaxis_index : float | None
        Average chemotaxis index across all runs (dynamic environments only).
    avg_time_in_attractant : float | None
        Average fraction of time spent near food (dynamic environments only).
    avg_approach_frequency : float | None
        Average fraction of steps moving toward food (dynamic environments only).
    avg_path_efficiency : float | None
        Average path efficiency (optimal/actual distance) (dynamic environments only).
    post_convergence_chemotaxis_index : float | None
        Chemotaxis index for post-convergence runs only (trained behavior).
    post_convergence_time_in_attractant : float | None
        Fraction of time near food for post-convergence runs.
    post_convergence_approach_frequency : float | None
        Fraction of steps toward food for post-convergence runs.
    post_convergence_path_efficiency : float | None
        Path efficiency for post-convergence runs.
    chemotaxis_validation_level : str | None
        Biological validation level based on post-convergence CI: none/minimum/target/excellent.
    biological_ci_range : tuple[float, float] | None
        Expected CI range from C. elegans literature (min, max).
    biological_ci_typical : float | None
        Typical/median CI value from C. elegans literature.
    matches_biology : bool | None
        Whether post-convergence CI falls within biological range.
    literature_source : str | None
        Citation for the biological data used for comparison.
    """

    total_runs: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    avg_foods_collected: float | None = None
    avg_distance_efficiency: float | None = None
    completed_all_food: int = 0
    starved: int = 0
    max_steps_reached: int = 0
    goal_reached: int = 0
    predator_deaths: int = 0
    avg_predator_encounters: float | None = None
    avg_successful_evasions: float | None = None
    # Convergence-based metrics (added for benchmark v2)
    converged: bool = False
    convergence_run: int | None = None
    runs_to_convergence: int | None = None
    post_convergence_success_rate: float | None = None
    post_convergence_avg_steps: float | None = None
    post_convergence_avg_foods: float | None = None
    post_convergence_variance: float | None = None
    post_convergence_distance_efficiency: float | None = None
    composite_benchmark_score: float | None = None
    # Learning metrics (added for NematodeBench format)
    learning_speed: float | None = None
    learning_speed_episodes: int | None = None
    stability: float | None = None
    per_run_results: list[PerRunResult] | None = None
    # Chemotaxis validation metrics (added for biological validation)
    # All-run chemotaxis metrics (includes learning phase)
    avg_chemotaxis_index: float | None = None
    avg_time_in_attractant: float | None = None
    avg_approach_frequency: float | None = None
    avg_path_efficiency: float | None = None
    # Post-convergence chemotaxis metrics (trained behavior, used for biological validation)
    post_convergence_chemotaxis_index: float | None = None
    post_convergence_time_in_attractant: float | None = None
    post_convergence_approach_frequency: float | None = None
    post_convergence_path_efficiency: float | None = None
    # Validation against biological literature (uses post-convergence metrics)
    chemotaxis_validation_level: str | None = None
    biological_ci_range: tuple[float, float] | None = None
    biological_ci_typical: float | None = None
    matches_biology: bool | None = None
    literature_source: str | None = None


class SystemMetadata(BaseModel):
    """System and dependency version information.

    Attributes
    ----------
    python_version : str
        Python version (e.g., "3.12.0").
    qiskit_version : str
        Qiskit version.
    torch_version : str | None
        PyTorch version (if installed).
    device_type : str
        Device type ("cpu", "gpu", or "qpu").
    qpu_backend : str | None
        QPU backend name (if using quantum hardware).
    """

    python_version: str
    qiskit_version: str
    torch_version: str | None = None
    device_type: str
    qpu_backend: str | None = None


class BenchmarkMetadata(BaseModel):
    """Benchmark-specific metadata for curated submissions.

    Attributes
    ----------
    contributor : str
        Contributor display name.
    github_username : str | None
        GitHub username (optional).
    category : str
        Benchmark category (e.g., "dynamic_medium_quantum").
    notes : str | None
        Notes about optimization approach.
    verified : bool
        Whether the benchmark has been verified by maintainers.
    """

    contributor: str
    github_username: str | None = None
    category: str
    notes: str | None = None
    verified: bool = False


class ConfigSummary(BaseModel):
    """Minimal config summary for filtering/categorization without loading full config.

    This is stored in the experiment JSON to enable quick filtering without
    duplicating the entire config (which is saved as a separate YAML file).

    Attributes
    ----------
    brain_type : str
        Brain architecture type ("modular", "mlp", "ppo", "spiking", etc.).
    environment_type : str
        Environment type ("static" or "dynamic").
    grid_size : int
        Size of the grid environment.
    predators_enabled : bool
        Whether predators are enabled.
    """

    brain_type: str
    environment_type: str
    grid_size: int
    predators_enabled: bool = False


class ExperimentMetadata(BaseModel):
    """Complete metadata for a simulation experiment.

    Attributes
    ----------
    experiment_id : str
        Unique experiment identifier (timestamp-based).
    timestamp : datetime
        Experiment timestamp.
    config_file : str
        Relative path to configuration file.
    config_hash : str
        SHA256 hash of configuration content.
    git_commit : str | None
        Git commit hash (None if not in repository).
    git_branch : str | None
        Git branch name (None if not in repository).
    git_dirty : bool
        Whether there were uncommitted changes.
    environment : EnvironmentMetadata
        Environment metadata.
    brain : BrainMetadata
        Brain architecture metadata.
    reward : RewardMetadata
        Reward function configuration.
    learning_rate : LearningRateMetadata | None
        Learning rate configuration.
    gradient : GradientMetadata
        Gradient calculation method.
    results : ResultsMetadata
        Aggregated simulation results.
    system : SystemMetadata
        System and dependency information.
    exports_path : str | None
        Path to detailed exports directory.
    benchmark : BenchmarkMetadata | None
        Benchmark metadata (if this is a benchmark submission).
    """

    experiment_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_file: str
    config_hash: str
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    environment: EnvironmentMetadata
    brain: BrainMetadata
    reward: RewardMetadata
    learning_rate: LearningRateMetadata | None = None
    gradient: GradientMetadata
    results: ResultsMetadata
    system: SystemMetadata
    exports_path: str | None = None
    benchmark: BenchmarkMetadata | None = None

    def to_dict(self, *, exclude_config_details: bool = True) -> dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization.

        Parameters
        ----------
        exclude_config_details : bool, optional
            If True (default), excludes detailed config sections (environment,
            brain, reward, learning_rate, gradient) since they're duplicated
            in the config YAML file. A minimal config_summary is included for
            filtering/categorization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with ISO format timestamp.
        """
        if exclude_config_details:
            # Exclude detailed config sections - they're in the YAML file
            raw_data = self.model_dump(
                exclude={"environment", "brain", "reward", "learning_rate", "gradient"},
            )
            # Build ordered dict with config_summary before results
            config_summary = ConfigSummary(
                brain_type=self.brain.type,
                environment_type=self.environment.type,
                grid_size=self.environment.grid_size,
                predators_enabled=self.environment.predators_enabled,
            ).model_dump()
            # Reorder: put config_summary before results
            data: dict[str, Any] = {}
            for key, value in raw_data.items():
                if key == "results":
                    # Insert config_summary just before results
                    data["config_summary"] = config_summary
                data[key] = value
            # If results wasn't in raw_data, add config_summary at end
            if "config_summary" not in data:
                data["config_summary"] = config_summary
        else:
            # Include everything (for backward compatibility or special cases)
            data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentMetadata":
        """Create metadata from dictionary.

        Handles both old format (full config details) and new lean format
        (with config_summary). For lean format, creates minimal stub metadata
        objects from the config_summary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with experiment metadata.

        Returns
        -------
        ExperimentMetadata
            Parsed metadata object.
        """
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        # Check if this is the new lean format (has config_summary, missing full config)
        if "config_summary" in data and "environment" not in data:
            summary = data.pop("config_summary")
            # Create minimal stub metadata from config_summary
            data["environment"] = EnvironmentMetadata(
                type=summary["environment_type"],
                grid_size=summary["grid_size"],
                predators_enabled=summary.get("predators_enabled", False),
            )
            data["brain"] = BrainMetadata(type=summary["brain_type"])
            data["reward"] = RewardMetadata(
                reward_goal=0.0,
                reward_distance_scale=0.0,
                reward_exploration=0.0,
                penalty_step=0.0,
                penalty_anti_dithering=0.0,
                penalty_stuck_position=0.0,
                stuck_position_threshold=0,
                penalty_starvation=0.0,
                penalty_predator_death=0.0,
                penalty_predator_proximity=0.0,
            )
            data["gradient"] = GradientMetadata()

        return cls(**data)
