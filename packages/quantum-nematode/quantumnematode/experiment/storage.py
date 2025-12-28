# pragma: no cover

"""Experiment metadata storage and retrieval using JSON files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quantumnematode.experiment.metadata import ExperimentMetadata
from quantumnematode.logging_config import logger

# Default storage directory
EXPERIMENTS_DIR = Path.cwd() / "experiments"


def save_experiment(
    metadata: ExperimentMetadata,
    base_dir: Path,
) -> Path:
    """Save experiment metadata to JSON file.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to save.
    base_dir : Path
        Directory to save the experiment. The directory will be created if it
        doesn't exist. The JSON file is saved as <experiment_id>.json within
        this directory.

    Returns
    -------
    Path
        Path to saved JSON file.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    filepath = base_dir / f"{metadata.experiment_id}.json"

    # Write atomically using temporary file
    temp_filepath = filepath.with_suffix(".tmp")
    try:
        with temp_filepath.open("w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        temp_filepath.replace(filepath)
        logger.info(f"Experiment metadata saved to {filepath}")
        return filepath
    except Exception as e:
        if temp_filepath.exists():
            temp_filepath.unlink()
        logger.error(f"Failed to save experiment metadata: {e}")
        raise


def load_experiment(experiment_id: str) -> ExperimentMetadata:
    """Load experiment metadata from JSON file.

    Parameters
    ----------
    experiment_id : str
        Experiment ID to load.

    Returns
    -------
    ExperimentMetadata
        Loaded experiment metadata.

    Raises
    ------
    FileNotFoundError
        If experiment file doesn't exist.
    """
    filepath = EXPERIMENTS_DIR / experiment_id / f"{experiment_id}.json"
    if not filepath.exists():
        msg = f"Experiment {experiment_id} not found"
        raise FileNotFoundError(msg)

    with filepath.open() as f:
        data = json.load(f)

    return ExperimentMetadata.from_dict(data)


def _find_experiment_ids() -> list[str]:
    """Find all experiment IDs in the experiments directory.

    Returns
    -------
    list[str]
        List of experiment IDs found.
    """
    if not EXPERIMENTS_DIR.exists():
        return []

    experiment_ids = []
    for subdir in EXPERIMENTS_DIR.iterdir():
        if subdir.is_dir():
            json_file = subdir / f"{subdir.name}.json"
            if json_file.exists():
                experiment_ids.append(subdir.name)

    return experiment_ids


def list_experiments(
    environment_type: str | None = None,
    brain_type: str | None = None,
    min_success_rate: float | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[ExperimentMetadata]:
    """List experiments with optional filtering.

    Parameters
    ----------
    environment_type : str | None, optional
        Filter by environment type ("static" or "dynamic").
    brain_type : str | None, optional
        Filter by brain type.
    min_success_rate : float | None, optional
        Minimum success rate threshold.
    since : datetime | None, optional
        Only include experiments after this date.
    limit : int | None, optional
        Maximum number of experiments to return.

    Returns
    -------
    list[ExperimentMetadata]
        List of matching experiments, sorted by timestamp (newest first).
    """
    experiments = []
    for experiment_id in _find_experiment_ids():
        try:
            metadata = load_experiment(experiment_id)

            # Apply filters
            if environment_type and metadata.environment.type != environment_type:
                continue
            if brain_type and metadata.brain.type != brain_type:
                continue
            if min_success_rate is not None and metadata.results.success_rate < min_success_rate:
                continue
            if since and metadata.timestamp < since:
                continue

            experiments.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load experiment {experiment_id}: {e}")

    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x.timestamp, reverse=True)

    # Apply limit
    if limit:
        experiments = experiments[:limit]

    return experiments


def compare_experiments(exp_id_1: str, exp_id_2: str) -> dict[str, Any]:
    """Compare two experiments and generate a diff.

    Parameters
    ----------
    exp_id_1 : str
        First experiment ID.
    exp_id_2 : str
        Second experiment ID.

    Returns
    -------
    dict[str, Any]
        Comparison structure with configuration diffs and results comparison.
    """
    exp1 = load_experiment(exp_id_1)
    exp2 = load_experiment(exp_id_2)

    comparison = {
        "experiment_ids": {"exp1": exp_id_1, "exp2": exp_id_2},
        "config_diff": {},
        "results_comparison": {},
        "performance_delta": {},
    }

    # Compare configurations
    config_diff = {}
    if exp1.environment.type != exp2.environment.type:
        config_diff["environment_type"] = {
            "exp1": exp1.environment.type,
            "exp2": exp2.environment.type,
        }
    if exp1.environment.grid_size != exp2.environment.grid_size:
        config_diff["grid_size"] = {
            "exp1": exp1.environment.grid_size,
            "exp2": exp2.environment.grid_size,
        }
    if exp1.brain.type != exp2.brain.type:
        config_diff["brain_type"] = {"exp1": exp1.brain.type, "exp2": exp2.brain.type}
    if exp1.brain.learning_rate != exp2.brain.learning_rate:
        config_diff["learning_rate"] = {
            "exp1": exp1.brain.learning_rate,
            "exp2": exp2.brain.learning_rate,
        }

    comparison["config_diff"] = config_diff

    # Compare results
    results_comparison = {
        "success_rate": {
            "exp1": exp1.results.success_rate,
            "exp2": exp2.results.success_rate,
        },
        "avg_steps": {"exp1": exp1.results.avg_steps, "exp2": exp2.results.avg_steps},
        "avg_reward": {"exp1": exp1.results.avg_reward, "exp2": exp2.results.avg_reward},
    }

    if (
        exp1.results.avg_foods_collected is not None
        and exp2.results.avg_foods_collected is not None
    ):
        results_comparison["avg_foods_collected"] = {
            "exp1": exp1.results.avg_foods_collected,
            "exp2": exp2.results.avg_foods_collected,
        }

    comparison["results_comparison"] = results_comparison

    # Calculate performance deltas
    success_rate_diff = exp2.results.success_rate - exp1.results.success_rate
    avg_steps_diff = exp2.results.avg_steps - exp1.results.avg_steps
    avg_reward_diff = exp2.results.avg_reward - exp1.results.avg_reward

    performance_delta: dict[str, float | str] = {
        "success_rate_diff": success_rate_diff,
        "avg_steps_diff": avg_steps_diff,
        "avg_reward_diff": avg_reward_diff,
        # Indicate which is better
        "better_success_rate": "exp2" if success_rate_diff > 0 else "exp1",
        "better_avg_steps": "exp2" if avg_steps_diff < 0 else "exp1",  # Lower is better
        "better_avg_reward": "exp2" if avg_reward_diff > 0 else "exp1",
    }

    comparison["performance_delta"] = performance_delta

    return comparison
