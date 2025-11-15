"""Data models for experiment and benchmark metadata."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EnvironmentMetadata(BaseModel):
    """Metadata about the simulation environment.

    Attributes
    ----------
    type : str
        Environment type ("static" or "dynamic").
    grid_size : int
        Size of the grid environment.
    num_foods : int | None
        Number of food items (dynamic environments only).
    max_active_foods : int | None
        Maximum active foods (dynamic environments only).
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
    """

    type: str
    grid_size: int
    num_foods: int | None = None
    max_active_foods: int | None = None
    initial_satiety: float | None = None
    satiety_decay_rate: float | None = None
    viewport_size: list[int] | None = None
    predators_enabled: bool = False
    num_predators: int | None = None


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
    """

    type: str
    qubits: int | None = None
    shots: int | None = None
    num_layers: int | None = None
    hidden_dim: int | None = None
    num_hidden_layers: int | None = None
    learning_rate: float | None = None
    modules: dict[str, list[int]] | None = None


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
        Average distance efficiency (dynamic environments only).
    completed_all_food : int
        Number of runs that collected all foods.
    starved : int
        Number of runs that ended in starvation.
    max_steps_reached : int
        Number of runs that reached max steps.
    goal_reached : int
        Number of runs that reached the goal.
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
    results: ResultsMetadata
    system: SystemMetadata
    exports_path: str | None = None
    benchmark: BenchmarkMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with ISO format timestamp.
        """
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentMetadata":
        """Create metadata from dictionary.

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
        return cls(**data)
