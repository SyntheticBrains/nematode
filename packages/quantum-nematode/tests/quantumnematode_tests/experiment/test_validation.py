"""Tests for experiment validation utilities.

This module tests the validation functions used for NematodeBench submissions,
including seed uniqueness, minimum sessions/runs requirements, and config consistency.
"""

from datetime import UTC, datetime

import pytest
from quantumnematode.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    GradientMetadata,
    LearningRateMetadata,
    PerRunResult,
    ResultsMetadata,
    RewardMetadata,
    SystemMetadata,
)
from quantumnematode.experiment.validation import (
    _validate_config_consistency,
    _validate_minimum_runs_per_session,
    _validate_minimum_sessions,
    _validate_seed_uniqueness,
    validate_submission,
)


@pytest.fixture
def base_experiment(reward_metadata: RewardMetadata) -> ExperimentMetadata:
    """Create a base experiment metadata for testing."""
    return ExperimentMetadata(
        experiment_id="20250101_120000",
        timestamp=datetime.now(UTC),
        config_file="configs/test.yml",
        config_hash="abc123",
        git_commit="def456",
        git_branch="main",
        git_dirty=False,
        environment=EnvironmentMetadata(type="dynamic", grid_size=20, num_foods=10),
        brain=BrainMetadata(type="ppo", learning_rate=0.001),
        reward=reward_metadata,
        learning_rate=LearningRateMetadata(
            method="static",
            initial_learning_rate=0.001,
        ),
        gradient=GradientMetadata(method="raw"),
        results=ResultsMetadata(
            total_runs=50,
            success_rate=0.9,
            avg_steps=40.0,
            avg_reward=120.0,
            per_run_results=[
                PerRunResult(
                    run=i + 1,
                    seed=1000 + i,
                    success=True,
                    steps=40,
                    total_reward=120.0,
                    termination_reason="completed_all_food",
                )
                for i in range(50)
            ],
        ),
        system=SystemMetadata(
            python_version="3.12.0",
            qiskit_version="1.0.0",
            device_type="cpu",
        ),
    )


def create_experiment_with_seeds(
    base: ExperimentMetadata,
    experiment_id: str,
    seeds: list[int],
) -> ExperimentMetadata:
    """Create an experiment with specific seeds for testing."""
    return ExperimentMetadata(
        experiment_id=experiment_id,
        timestamp=base.timestamp,
        config_file=base.config_file,
        config_hash=base.config_hash,
        git_commit=base.git_commit,
        git_branch=base.git_branch,
        git_dirty=base.git_dirty,
        environment=base.environment,
        brain=base.brain,
        reward=base.reward,
        learning_rate=base.learning_rate,
        gradient=base.gradient,
        results=ResultsMetadata(
            total_runs=len(seeds),
            success_rate=0.9,
            avg_steps=40.0,
            avg_reward=120.0,
            per_run_results=[
                PerRunResult(
                    run=i + 1,
                    seed=seed,
                    success=True,
                    steps=40,
                    total_reward=120.0,
                    termination_reason="completed_all_food",
                )
                for i, seed in enumerate(seeds)
            ],
        ),
        system=base.system,
    )


class TestValidateSeedUniqueness:
    """Test seed uniqueness validation."""

    def test_unique_seeds_pass(self, base_experiment):
        """Test validation passes when all seeds are unique."""
        # Create 3 experiments with non-overlapping seeds
        experiments = [
            create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3]),
            create_experiment_with_seeds(base_experiment, "exp2", [4, 5, 6]),
            create_experiment_with_seeds(base_experiment, "exp3", [7, 8, 9]),
        ]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is True
        assert errors == []

    def test_duplicate_seeds_within_experiment(self, base_experiment):
        """Test validation fails when seeds duplicate within single experiment."""
        experiments = [
            create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 2, 3]),
        ]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is False
        assert len(errors) == 1
        assert "Seed 2 appears 2 times" in errors[0]

    def test_duplicate_seeds_across_experiments(self, base_experiment):
        """Test validation fails when seeds duplicate across experiments."""
        experiments = [
            create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3]),
            create_experiment_with_seeds(base_experiment, "exp2", [3, 4, 5]),  # 3 is duplicate
        ]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is False
        assert len(errors) == 1
        assert "Seed 3 appears 2 times" in errors[0]
        assert "exp1" in errors[0]
        assert "exp2" in errors[0]

    def test_multiple_duplicates(self, base_experiment):
        """Test validation reports all duplicate seeds."""
        experiments = [
            create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3]),
            create_experiment_with_seeds(base_experiment, "exp2", [2, 3, 4]),  # 2 and 3 duplicate
        ]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is False
        assert len(errors) == 2

    def test_no_per_run_results(self, base_experiment):
        """Test validation fails when no per_run_results present."""
        base_experiment.results.per_run_results = None
        experiments = [base_experiment]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is False
        assert "No seeds found" in errors[0]

    def test_empty_per_run_results(self, base_experiment):
        """Test validation fails with empty per_run_results."""
        base_experiment.results.per_run_results = []
        experiments = [base_experiment]

        is_valid, errors = _validate_seed_uniqueness(experiments)

        assert is_valid is False
        assert "No seeds found" in errors[0]


class TestValidateMinimumSessions:
    """Test minimum sessions validation."""

    def test_sufficient_sessions(self, base_experiment):
        """Test validation passes with sufficient sessions."""
        experiments = [base_experiment] * 10

        is_valid, errors = _validate_minimum_sessions(experiments)

        assert is_valid is True
        assert errors == []

    def test_exactly_minimum_sessions(self, base_experiment):
        """Test validation passes with exactly minimum sessions."""
        experiments = [base_experiment] * 10

        is_valid, errors = _validate_minimum_sessions(experiments, min_sessions=10)

        assert is_valid is True
        assert errors == []

    def test_insufficient_sessions(self, base_experiment):
        """Test validation fails with insufficient sessions."""
        experiments = [base_experiment] * 5

        is_valid, errors = _validate_minimum_sessions(experiments)

        assert is_valid is False
        assert len(errors) == 1
        assert "Insufficient sessions: 5 provided" in errors[0]
        assert "minimum 10 required" in errors[0]

    def test_custom_minimum_sessions(self, base_experiment):
        """Test validation with custom minimum sessions."""
        experiments = [base_experiment] * 5

        is_valid, errors = _validate_minimum_sessions(experiments, min_sessions=5)

        assert is_valid is True
        assert errors == []

    def test_no_sessions(self):
        """Test validation fails with no sessions."""
        experiments: list[ExperimentMetadata] = []

        is_valid, errors = _validate_minimum_sessions(experiments)

        assert is_valid is False
        assert "Insufficient sessions: 0 provided" in errors[0]


class TestValidateMinimumRunsPerSession:
    """Test minimum runs per session validation."""

    def test_sufficient_runs(self, base_experiment):
        """Test validation passes when all sessions have sufficient runs."""
        base_experiment.results.total_runs = 50
        experiments = [base_experiment] * 3

        is_valid, errors = _validate_minimum_runs_per_session(experiments)

        assert is_valid is True
        assert errors == []

    def test_exactly_minimum_runs(self, base_experiment):
        """Test validation passes with exactly minimum runs."""
        base_experiment.results.total_runs = 50
        experiments = [base_experiment]

        is_valid, errors = _validate_minimum_runs_per_session(experiments, min_runs=50)

        assert is_valid is True
        assert errors == []

    def test_insufficient_runs_single_session(self, base_experiment):
        """Test validation fails when one session has insufficient runs."""
        base_experiment.results.total_runs = 30
        experiments = [base_experiment]

        is_valid, errors = _validate_minimum_runs_per_session(experiments)

        assert is_valid is False
        assert len(errors) == 1
        assert "has only 30 runs" in errors[0]
        assert "minimum 50 required" in errors[0]

    def test_insufficient_runs_multiple_sessions(self, base_experiment):
        """Test validation reports all sessions with insufficient runs."""
        exp1 = create_experiment_with_seeds(base_experiment, "exp1", list(range(30)))
        exp1.results.total_runs = 30
        exp2 = create_experiment_with_seeds(base_experiment, "exp2", list(range(100, 125)))
        exp2.results.total_runs = 25
        experiments = [exp1, exp2]

        is_valid, errors = _validate_minimum_runs_per_session(experiments)

        assert is_valid is False
        assert len(errors) == 2
        assert any("exp1" in e and "30 runs" in e for e in errors)
        assert any("exp2" in e and "25 runs" in e for e in errors)

    def test_custom_minimum_runs(self, base_experiment):
        """Test validation with custom minimum runs."""
        base_experiment.results.total_runs = 25
        experiments = [base_experiment]

        is_valid, errors = _validate_minimum_runs_per_session(experiments, min_runs=25)

        assert is_valid is True
        assert errors == []


class TestValidateConfigConsistency:
    """Test config consistency validation."""

    def test_consistent_config(self, base_experiment):
        """Test validation passes with consistent config across experiments."""
        experiments = [
            create_experiment_with_seeds(
                base_experiment,
                f"exp{i}",
                list(range(i * 10, i * 10 + 5)),
            )
            for i in range(3)
        ]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is True
        assert errors == []

    def test_inconsistent_brain_type(self, base_experiment):
        """Test validation fails with different brain types."""
        exp1 = create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3])
        exp2 = create_experiment_with_seeds(base_experiment, "exp2", [4, 5, 6])
        exp2.brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiments = [exp1, exp2]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is False
        assert len(errors) == 1
        assert "brain type 'mlp'" in errors[0]
        assert "expected 'ppo'" in errors[0]

    def test_inconsistent_environment_type(self, base_experiment):
        """Test validation fails with different environment types."""
        exp1 = create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3])
        exp2 = create_experiment_with_seeds(base_experiment, "exp2", [4, 5, 6])
        exp2.environment = EnvironmentMetadata(type="static", grid_size=20)
        experiments = [exp1, exp2]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is False
        assert "environment type 'static'" in errors[0]
        assert "expected 'dynamic'" in errors[0]

    def test_inconsistent_grid_size(self, base_experiment):
        """Test validation fails with different grid sizes."""
        exp1 = create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3])
        exp2 = create_experiment_with_seeds(base_experiment, "exp2", [4, 5, 6])
        exp2.environment = EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=10)
        experiments = [exp1, exp2]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is False
        assert "grid size 50" in errors[0]
        assert "expected 20" in errors[0]

    def test_multiple_inconsistencies(self, base_experiment):
        """Test validation reports all inconsistencies."""
        exp1 = create_experiment_with_seeds(base_experiment, "exp1", [1, 2, 3])
        exp2 = create_experiment_with_seeds(base_experiment, "exp2", [4, 5, 6])
        exp2.brain = BrainMetadata(type="mlp", learning_rate=0.001)
        exp2.environment = EnvironmentMetadata(type="static", grid_size=50)
        experiments = [exp1, exp2]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is False
        assert len(errors) == 3  # brain type, env type, grid size

    def test_empty_experiments(self):
        """Test validation fails with no experiments."""
        experiments: list[ExperimentMetadata] = []

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is False
        assert "No experiments provided" in errors[0]

    def test_single_experiment(self, base_experiment):
        """Test validation passes with single experiment (nothing to compare)."""
        experiments = [base_experiment]

        is_valid, errors = _validate_config_consistency(experiments)

        assert is_valid is True
        assert errors == []


class TestValidateSubmission:
    """Test the main validate_submission function."""

    def test_valid_submission(self, base_experiment):
        """Test a fully valid submission passes all checks."""
        # Create 10 experiments with unique seeds and sufficient runs
        experiments = [
            create_experiment_with_seeds(
                base_experiment,
                f"exp{i}",
                list(range(i * 100, i * 100 + 50)),
            )
            for i in range(10)
        ]
        for exp in experiments:
            exp.results.total_runs = 50

        is_valid, errors = validate_submission(experiments)

        assert is_valid is True
        assert errors == []

    def test_insufficient_sessions_only(self, base_experiment):
        """Test submission fails with only sessions error."""
        experiments = [
            create_experiment_with_seeds(
                base_experiment,
                f"exp{i}",
                list(range(i * 100, i * 100 + 50)),
            )
            for i in range(5)  # Only 5 sessions
        ]
        for exp in experiments:
            exp.results.total_runs = 50

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        assert any("Insufficient sessions" in e for e in errors)

    def test_insufficient_runs_only(self, base_experiment):
        """Test submission fails with only runs error."""
        experiments = [
            create_experiment_with_seeds(
                base_experiment,
                f"exp{i}",
                list(range(i * 100, i * 100 + 30)),
            )
            for i in range(10)
        ]
        for exp in experiments:
            exp.results.total_runs = 30  # Only 30 runs

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        assert any("has only 30 runs" in e for e in errors)

    def test_duplicate_seeds_only(self, base_experiment):
        """Test submission fails with only seed error."""
        experiments = [
            create_experiment_with_seeds(base_experiment, f"exp{i}", list(range(50)))  # Same seeds!
            for i in range(10)
        ]
        for exp in experiments:
            exp.results.total_runs = 50

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        assert any("Seed" in e and "appears" in e for e in errors)

    def test_inconsistent_config_only(self, base_experiment):
        """Test submission fails with only config error."""
        experiments = [
            create_experiment_with_seeds(
                base_experiment,
                f"exp{i}",
                list(range(i * 100, i * 100 + 50)),
            )
            for i in range(10)
        ]
        for exp in experiments:
            exp.results.total_runs = 50
        # Make one experiment have different brain type
        experiments[5].brain = BrainMetadata(type="mlp", learning_rate=0.001)

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        assert any("brain type" in e for e in errors)

    def test_multiple_errors(self, base_experiment):
        """Test submission reports all errors together."""
        experiments = [
            create_experiment_with_seeds(base_experiment, f"exp{i}", list(range(50)))  # Same seeds
            for i in range(5)  # Only 5 sessions
        ]
        for exp in experiments:
            exp.results.total_runs = 30  # Only 30 runs

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        # Should have errors from multiple validations
        assert len(errors) >= 2

    def test_empty_submission(self):
        """Test submission fails when empty."""
        experiments: list[ExperimentMetadata] = []

        is_valid, errors = validate_submission(experiments)

        assert is_valid is False
        assert len(errors) >= 1
