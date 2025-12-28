"""Data models for NematodeBench submissions.

This module defines the schema for official NematodeBench submissions that aggregate
multiple independent training sessions for scientific rigor.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from quantumnematode.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    StatValue,
)


class SessionReference(BaseModel):
    """Reference to a source experiment session.

    Attributes
    ----------
    experiment_id : str
        Unique identifier for the experiment session (timestamp-based).
    file_path : str
        Relative path to the experiment folder in artifacts/experiments/.
    session_seed : int
        Master seed used for this session (derived from config or auto-generated).
    num_runs : int
        Number of runs (episodes) in this session.
    """

    experiment_id: str
    file_path: str
    session_seed: int
    num_runs: int


class AggregateMetrics(BaseModel):
    """Aggregated metrics across multiple sessions using StatValue.

    All metrics are aggregated across sessions (not runs within a session).
    This captures the variance in algorithm performance across independent
    training runs, which is critical for scientific reproducibility.

    Attributes
    ----------
    success_rate : StatValue
        Success rate aggregated across sessions.
    composite_score : StatValue
        Composite benchmark score aggregated across sessions.
    distance_efficiency : StatValue | None
        Distance efficiency aggregated across sessions (dynamic environments only).
    learning_speed : StatValue
        Normalized learning speed aggregated across sessions.
    learning_speed_episodes : StatValue
        Episodes to reach 80% success rate aggregated across sessions.
    stability : StatValue
        Stability metric aggregated across sessions.
    """

    success_rate: StatValue
    composite_score: StatValue
    distance_efficiency: StatValue | None = None
    learning_speed: StatValue
    learning_speed_episodes: StatValue
    stability: StatValue


class NematodeBenchSubmission(BaseModel):
    """Official NematodeBench submission format.

    This model represents an aggregated benchmark submission that combines
    results from multiple independent training sessions. This ensures
    scientific rigor by capturing algorithm reproducibility variance.

    Requirements for valid submissions:
    - Minimum 10 independent sessions
    - All seeds unique across all runs in all sessions
    - Each session must have full experiment metadata

    Attributes
    ----------
    submission_id : str
        Unique identifier for the submission (generated at submission time).
    timestamp : datetime
        Timestamp when the submission was created.
    brain_type : str
        Type of brain architecture (e.g., "ppo", "mlp", "qmodular").
    brain_config : BrainMetadata
        Brain configuration metadata (from first session, validated consistent).
    environment : EnvironmentMetadata
        Environment configuration metadata (from first session, validated consistent).
    category : str
        Benchmark category (e.g., "foraging_small/classical").
    sessions : list[SessionReference]
        References to source experiment sessions.
    total_sessions : int
        Total number of sessions included.
    total_runs : int
        Total number of runs across all sessions.
    metrics : AggregateMetrics
        Aggregated metrics across sessions.
    all_seeds_unique : bool
        Whether all seeds are unique across all runs in all sessions.
    contributor : str
        Contributor display name.
    github_username : str | None
        GitHub username (optional).
    notes : str | None
        Additional notes about the submission.
    """

    submission_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    brain_type: str
    brain_config: BrainMetadata
    environment: EnvironmentMetadata
    category: str

    # Session references (not duplicated data)
    sessions: list[SessionReference]
    total_sessions: int
    total_runs: int

    # Aggregated metrics (StatValue across sessions)
    metrics: AggregateMetrics

    # Validation
    all_seeds_unique: bool

    # Contributor info
    contributor: str
    github_username: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert submission to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with ISO format timestamp.
            Excludes null values for cleaner output.
        """
        data = self.model_dump(exclude_none=True)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NematodeBenchSubmission":
        """Create submission from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with submission data.

        Returns
        -------
        NematodeBenchSubmission
            Parsed submission object.
        """
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# Minimum requirements for valid NematodeBench submissions
MIN_SESSIONS_REQUIRED = 10
MIN_RUNS_PER_SESSION = 50
