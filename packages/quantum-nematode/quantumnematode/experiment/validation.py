"""Validation utilities for experiment and benchmark submissions.

This module provides validation functions for ensuring data integrity
and reproducibility requirements are met for NematodeBench submissions.
"""

from collections import Counter

from quantumnematode.experiment.metadata import ExperimentMetadata
from quantumnematode.experiment.submission import MIN_RUNS_PER_SESSION, MIN_SESSIONS_REQUIRED


def _validate_seed_uniqueness(
    experiments: list[ExperimentMetadata],
) -> tuple[bool, list[str]]:
    """Validate that all seeds are unique across all runs in all sessions.

    This is a critical requirement for reproducibility - if any seeds are
    duplicated, the experiments cannot be considered independent.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata objects to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages if invalid)

    Examples
    --------
    >>> experiments = [exp1, exp2, exp3]
    >>> is_valid, errors = validate_seed_uniqueness(experiments)
    >>> if not is_valid:
    ...     for error in errors:
    ...         print(f"Error: {error}")
    """
    all_seeds: list[int] = []
    seeds_by_experiment: dict[str, list[int]] = {}

    for exp in experiments:
        if exp.results.per_run_results:
            seeds = [r.seed for r in exp.results.per_run_results]
            all_seeds.extend(seeds)
            seeds_by_experiment[exp.experiment_id] = seeds

    if not all_seeds:
        return False, ["No seeds found in any experiment. Ensure per_run_results is populated."]

    # Find duplicates
    seed_counts = Counter(all_seeds)
    duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}

    if duplicates:
        error_messages = []
        for seed, count in duplicates.items():
            # Find which experiments contain this seed
            containing_experiments = [
                exp_id for exp_id, seeds in seeds_by_experiment.items() if seed in seeds
            ]
            error_messages.append(
                f"Seed {seed} appears {count} times in experiments: {containing_experiments}",
            )
        return False, error_messages

    return True, []


def _validate_minimum_sessions(
    experiments: list[ExperimentMetadata],
    min_sessions: int = MIN_SESSIONS_REQUIRED,
) -> tuple[bool, list[str]]:
    """Validate that the minimum number of sessions is met.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata objects to validate.
    min_sessions : int, optional
        Minimum required sessions (default: MIN_SESSIONS_REQUIRED = 10).

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages if invalid)
    """
    if len(experiments) < min_sessions:
        return False, [
            f"Insufficient sessions: {len(experiments)} provided, "
            f"minimum {min_sessions} required for NematodeBench submission.",
        ]
    return True, []


def _validate_minimum_runs_per_session(
    experiments: list[ExperimentMetadata],
    min_runs: int = MIN_RUNS_PER_SESSION,
) -> tuple[bool, list[str]]:
    """Validate that each session has the minimum number of runs.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata objects to validate.
    min_runs : int, optional
        Minimum required runs per session (default: MIN_RUNS_PER_SESSION = 50).

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages if invalid)
    """
    errors = [
        f"Experiment {exp.experiment_id} has only {exp.results.total_runs} runs, "
        f"minimum {min_runs} required."
        for exp in experiments
        if exp.results.total_runs < min_runs
    ]
    return len(errors) == 0, errors


def _validate_config_consistency(
    experiments: list[ExperimentMetadata],
) -> tuple[bool, list[str]]:
    """Validate that all experiments use consistent configuration.

    Checks that brain type and environment type are consistent across
    all experiments. Minor config differences (like different seeds)
    are allowed.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata objects to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages if invalid)
    """
    if not experiments:
        return False, ["No experiments provided."]

    errors = []
    first_exp = experiments[0]
    reference_brain_type = first_exp.brain.type
    reference_env_type = first_exp.environment.type
    reference_grid_size = first_exp.environment.grid_size

    for exp in experiments[1:]:
        if exp.brain.type != reference_brain_type:
            errors.append(
                f"Experiment {exp.experiment_id} has brain type '{exp.brain.type}', "
                f"expected '{reference_brain_type}'.",
            )
        if exp.environment.type != reference_env_type:
            errors.append(
                f"Experiment {exp.experiment_id} has environment type '{exp.environment.type}', "
                f"expected '{reference_env_type}'.",
            )
        if exp.environment.grid_size != reference_grid_size:
            errors.append(
                f"Experiment {exp.experiment_id} has grid size {exp.environment.grid_size}, "
                f"expected {reference_grid_size}.",
            )

    return len(errors) == 0, errors


def validate_submission(
    experiments: list[ExperimentMetadata],
) -> tuple[bool, list[str]]:
    """Perform all validation checks for a NematodeBench submission.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata objects to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of all error messages if invalid)

    Examples
    --------
    >>> experiments = load_experiments_from_folders(experiment_paths)
    >>> is_valid, errors = validate_submission(experiments)
    >>> if is_valid:
    ...     print("Submission is valid!")
    ... else:
    ...     print("Validation failed:")
    ...     for error in errors:
    ...         print(f"  - {error}")
    """
    all_errors: list[str] = []

    # Check minimum sessions
    _, errors = _validate_minimum_sessions(experiments)
    all_errors.extend(errors)

    # Check minimum runs per session
    _, errors = _validate_minimum_runs_per_session(experiments)
    all_errors.extend(errors)

    # Check config consistency
    _, errors = _validate_config_consistency(experiments)
    all_errors.extend(errors)

    # Check seed uniqueness (most important for reproducibility)
    _, errors = _validate_seed_uniqueness(experiments)
    all_errors.extend(errors)

    return len(all_errors) == 0, all_errors
