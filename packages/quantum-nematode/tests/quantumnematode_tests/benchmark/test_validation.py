"""Tests for benchmark validation."""

from datetime import UTC, datetime

import pytest
from quantumnematode.benchmark.validation import (
    BenchmarkValidationError,
    BenchmarkValidationRules,
    validate_benchmark,
    validate_git_state,
    validate_minimum_runs,
    validate_success_rate,
)
from quantumnematode.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    ResultsMetadata,
    SystemMetadata,
)


@pytest.fixture
def basic_experiment():
    """Create a basic experiment metadata for testing."""
    from quantumnematode.experiment.metadata import BenchmarkMetadata

    return ExperimentMetadata(
        experiment_id="20250101_120000",
        timestamp=datetime.now(UTC),
        config_file="configs/test.yml",
        config_hash="abc123",
        git_commit="def456",
        git_branch="main",
        git_dirty=False,
        environment=EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=20),
        brain=BrainMetadata(type="modular", qubits=4, learning_rate=0.01),
        results=ResultsMetadata(
            total_runs=50,  # Updated from 20 to 50 to match new default
            success_rate=0.9,
            avg_steps=40.0,
            avg_reward=120.0,
        ),
        system=SystemMetadata(
            python_version="3.12.0",
            qiskit_version="1.0.0",
            device_type="cpu",
        ),
        benchmark=BenchmarkMetadata(
            contributor="Test User",
            category="dynamic_medium_quantum",
        ),
    )


class TestBenchmarkValidationRules:
    """Test validation rules configuration."""

    def test_default_rules(self):
        """Test default validation rules."""
        rules = BenchmarkValidationRules()

        assert rules.min_runs == 50  # Updated from 20 to 50 for convergence detection
        assert rules.require_clean_git is True
        assert rules.min_success_rate is None

    def test_custom_rules(self):
        """Test custom validation rules."""
        rules = BenchmarkValidationRules(
            min_runs=50,
            require_clean_git=False,
            min_success_rate=0.8,
        )

        assert rules.min_runs == 50
        assert rules.require_clean_git is False
        assert rules.min_success_rate == 0.8


class TestValidateMinimumRuns:
    """Test minimum runs validation."""

    def test_validate_minimum_runs_pass(self, basic_experiment):
        """Test validation passes with sufficient runs."""
        rules = BenchmarkValidationRules(min_runs=20)
        # Should not raise
        validate_minimum_runs(basic_experiment, rules)

    def test_validate_minimum_runs_exactly_minimum(self, basic_experiment):
        """Test validation passes with exactly minimum runs."""
        rules = BenchmarkValidationRules(min_runs=20)
        basic_experiment.results.total_runs = 20
        # Should not raise
        validate_minimum_runs(basic_experiment, rules)

    def test_validate_minimum_runs_fail(self, basic_experiment):
        """Test validation fails with insufficient runs."""
        rules = BenchmarkValidationRules(min_runs=30)
        basic_experiment.results.total_runs = 20

        with pytest.raises(
            BenchmarkValidationError,
            match="Benchmark requires at least 30 runs",
        ):
            validate_minimum_runs(basic_experiment, rules)

    def test_validate_minimum_runs_zero(self, basic_experiment):
        """Test validation with zero runs."""
        rules = BenchmarkValidationRules(min_runs=10)
        basic_experiment.results.total_runs = 0

        with pytest.raises(BenchmarkValidationError):
            validate_minimum_runs(basic_experiment, rules)


class TestValidateSuccessRate:
    """Test success rate validation."""

    def test_validate_success_rate_pass(self, basic_experiment):
        """Test validation passes with good success rate."""
        rules = BenchmarkValidationRules(min_success_rate=0.8)
        basic_experiment.results.success_rate = 0.9
        # Should not raise (just warns)
        validate_success_rate(basic_experiment, rules)

    def test_validate_success_rate_no_minimum(self, basic_experiment):
        """Test validation with no minimum set."""
        rules = BenchmarkValidationRules(min_success_rate=None)
        basic_experiment.results.success_rate = 0.1
        # Should not raise
        validate_success_rate(basic_experiment, rules)

    def test_validate_success_rate_low(self, basic_experiment):
        """Test validation with low success rate (should warn but not fail)."""
        rules = BenchmarkValidationRules(min_success_rate=0.8)
        basic_experiment.results.success_rate = 0.5
        # Should only warn, not raise
        validate_success_rate(basic_experiment, rules)

    def test_validate_success_rate_zero(self, basic_experiment):
        """Test validation with zero success rate."""
        rules = BenchmarkValidationRules(min_success_rate=0.5)
        basic_experiment.results.success_rate = 0.0
        # Should only warn, not raise
        validate_success_rate(basic_experiment, rules)


class TestValidateGitState:
    """Test git state validation."""

    def test_validate_git_state_clean(self, basic_experiment):
        """Test validation passes with clean git state."""
        rules = BenchmarkValidationRules(require_clean_git=True)
        basic_experiment.git_dirty = False
        # Should not raise
        validate_git_state(basic_experiment, rules)

    def test_validate_git_state_dirty_required_clean(self, basic_experiment):
        """Test validation warns with dirty git when clean required."""
        rules = BenchmarkValidationRules(require_clean_git=True)
        basic_experiment.git_dirty = True

        # Should only warn, not raise
        validate_git_state(basic_experiment, rules)

    def test_validate_git_state_dirty_not_required_clean(self, basic_experiment):
        """Test validation passes with dirty git when not required clean."""
        rules = BenchmarkValidationRules(require_clean_git=False)
        basic_experiment.git_dirty = True
        # Should not raise
        validate_git_state(basic_experiment, rules)

    def test_validate_git_state_no_git_info(self, basic_experiment):
        """Test validation with missing git info."""
        rules = BenchmarkValidationRules(require_clean_git=True)
        basic_experiment.git_commit = None
        basic_experiment.git_branch = None
        # Should not raise - missing git info is acceptable
        validate_git_state(basic_experiment, rules)


class TestValidateBenchmark:
    """Test complete benchmark validation."""

    def test_validate_benchmark_pass_all(self, basic_experiment):
        """Test validation passes all checks."""
        rules = BenchmarkValidationRules(
            min_runs=50,
            require_clean_git=True,
            min_success_rate=0.8,
        )

        result = validate_benchmark(basic_experiment, rules)
        assert result is True

    def test_validate_benchmark_fail_runs(self, basic_experiment):
        """Test validation fails on insufficient runs."""
        rules = BenchmarkValidationRules(min_runs=50)
        basic_experiment.results.total_runs = 10

        # Should raise and return False
        with pytest.raises(BenchmarkValidationError):
            validate_benchmark(basic_experiment, rules)

    def test_validate_benchmark_pass_with_dirty_git(self, basic_experiment):
        """Test validation passes with dirty git (only warns)."""
        rules = BenchmarkValidationRules(require_clean_git=True)
        basic_experiment.git_dirty = True

        # Git validation only warns, doesn't fail
        result = validate_benchmark(basic_experiment, rules)
        assert result is True

    def test_validate_benchmark_pass_with_warnings(self, basic_experiment):
        """Test validation passes but with warnings."""
        rules = BenchmarkValidationRules(
            min_runs=50,
            require_clean_git=True,
            min_success_rate=0.95,  # Higher than actual
        )
        basic_experiment.results.success_rate = 0.85

        # Should pass despite warning about success rate
        result = validate_benchmark(basic_experiment, rules)
        assert result is True

    def test_validate_benchmark_default_rules(self, basic_experiment):
        """Test validation with default rules."""
        result = validate_benchmark(basic_experiment)
        assert result is True

    def test_validate_benchmark_multiple_failures(self, basic_experiment):
        """Test validation with multiple failures."""
        rules = BenchmarkValidationRules(
            min_runs=50,
            require_clean_git=True,
        )
        basic_experiment.results.total_runs = 10
        basic_experiment.git_dirty = True

        # Should raise on the first critical failure (min_runs)
        with pytest.raises(BenchmarkValidationError):
            validate_benchmark(basic_experiment, rules)

    def test_validate_benchmark_edge_cases(self, basic_experiment):
        """Test validation with edge case values."""
        rules = BenchmarkValidationRules(
            min_runs=1,
            require_clean_git=False,
            min_success_rate=0.0,
        )
        basic_experiment.results.total_runs = 1
        basic_experiment.results.success_rate = 0.0

        result = validate_benchmark(basic_experiment, rules)
        assert result is True
