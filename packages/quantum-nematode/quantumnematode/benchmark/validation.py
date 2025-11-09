"""Benchmark validation rules and checking."""

from dataclasses import dataclass

from quantumnematode.experiment.metadata import ExperimentMetadata
from quantumnematode.logging_config import logger


@dataclass
class BenchmarkValidationRules:
    """Validation rules for benchmark submissions.

    Attributes
    ----------
    min_runs : int
        Minimum number of runs required.
    min_success_rate : float | None
        Minimum success rate (None to skip check).
    require_clean_git : bool
        Whether to require clean git state.
    require_config_in_repo : bool
        Whether to require config file in repository.
    require_contributor_name : bool
        Whether to require contributor name.
    """

    min_runs: int = 20
    min_success_rate: float | None = None
    require_clean_git: bool = True
    require_config_in_repo: bool = True
    require_contributor_name: bool = True


class BenchmarkValidationError(Exception):
    """Exception raised when benchmark validation fails."""


def validate_minimum_runs(metadata: ExperimentMetadata, rules: BenchmarkValidationRules) -> None:
    """Validate minimum number of runs.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules
        Validation rules.

    Raises
    ------
    BenchmarkValidationError
        If validation fails.
    """
    if metadata.results.total_runs < rules.min_runs:
        msg = (
            f"Benchmark requires at least {rules.min_runs} runs, "
            f"but only {metadata.results.total_runs} were completed"
        )
        raise BenchmarkValidationError(
            msg,
        )


def validate_success_rate(metadata: ExperimentMetadata, rules: BenchmarkValidationRules) -> None:
    """Validate success rate threshold.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules
        Validation rules.
    """
    if rules.min_success_rate is None:
        return

    if metadata.results.success_rate < rules.min_success_rate:
        logger.warning(
            f"Success rate ({metadata.results.success_rate:.1%}) is below recommended "
            f"threshold ({rules.min_success_rate:.1%}). "
            "Consider improving results before submission.",
        )


def validate_git_state(metadata: ExperimentMetadata, rules: BenchmarkValidationRules) -> None:
    """Validate git repository state.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules
        Validation rules.
    """
    if not rules.require_clean_git:
        return

    if metadata.git_commit is None:
        logger.warning(
            "Experiment was not run in a git repository - reproducibility may be limited",
        )
    elif metadata.git_dirty:
        logger.warning(
            "Repository had uncommitted changes during experiment. "
            "Consider committing changes for full reproducibility.",
        )


def validate_config_file(metadata: ExperimentMetadata, rules: BenchmarkValidationRules) -> None:  # noqa: ARG001
    """Validate configuration file.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules
        Validation rules.
    """
    # Basic validation - config file path is stored
    if not metadata.config_file:
        logger.warning("No config file path recorded")


def validate_contributor_info(
    metadata: ExperimentMetadata,
    rules: BenchmarkValidationRules,
) -> None:
    """Validate contributor information.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules
        Validation rules.

    Raises
    ------
    BenchmarkValidationError
        If contributor name is required but missing.
    """
    if not rules.require_contributor_name:
        return

    if metadata.benchmark is None or not metadata.benchmark.contributor:
        msg = "Contributor name is required for benchmark submission"
        raise BenchmarkValidationError(msg)


def validate_benchmark(
    metadata: ExperimentMetadata,
    rules: BenchmarkValidationRules | None = None,
) -> bool:
    """Validate benchmark submission against rules.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to validate.
    rules : BenchmarkValidationRules | None, optional
        Validation rules (uses defaults if None).

    Returns
    -------
    bool
        True if validation passes, False otherwise.

    Raises
    ------
    BenchmarkValidationError
        If critical validation checks fail.
    """
    if rules is None:
        rules = BenchmarkValidationRules()

    try:
        # Critical validations (raise exceptions)
        validate_minimum_runs(metadata, rules)
        validate_contributor_info(metadata, rules)

        # Warning validations (log warnings but don't fail)
        validate_success_rate(metadata, rules)
        validate_git_state(metadata, rules)
        validate_config_file(metadata, rules)

        logger.info("Benchmark validation passed")
        return True

    except BenchmarkValidationError as e:
        logger.error(f"Benchmark validation failed: {e}")
        raise
