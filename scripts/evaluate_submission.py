#!/usr/bin/env python
"""Validate NematodeBench submissions.

This script validates benchmark submission JSON files to ensure they meet
the requirements for inclusion in the NematodeBench leaderboard.

Usage:
    uv run scripts/evaluate_submission.py --submission path/to/submission.json
    uv run scripts/evaluate_submission.py --submission submission.json --reproduce --reproduce-runs 5
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from quantumnematode.experiment import MIN_RUNS_PER_SESSION, MIN_SESSIONS_REQUIRED


@dataclass
class ValidationResult:
    """Result of submission validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict


# Required fields for NematodeBench submissions
REQUIRED_FIELDS = [
    "submission_id",
    "brain_type",
    "brain_config",
    "environment",
    "category",
    "sessions",
    "total_sessions",
    "total_runs",
    "metrics",
    "all_seeds_unique",
    "contributor",
]

REQUIRED_METRIC_FIELDS = [
    "success_rate",
    "composite_score",
    "learning_speed",
    "stability",
]

VALID_BRAIN_TYPES = ["mlp", "ppo", "modular", "qmodular", "qmlp", "spiking"]


def _validate_required_fields(data: dict) -> list[str]:
    """Check for missing required fields."""
    return [f"Missing required field: {field}" for field in REQUIRED_FIELDS if field not in data]


def _validate_metrics_structure(data: dict) -> list[str]:
    """Validate metrics field structure."""
    errors = []
    if "metrics" not in data:
        return errors

    metrics = data["metrics"]
    for field in REQUIRED_METRIC_FIELDS:
        if field not in metrics:
            errors.append(f"Missing required metric field: {field}")
        elif not isinstance(metrics[field], dict):
            errors.append(f"Metric field '{field}' must be an object with mean/std/min/max")
        elif "mean" not in metrics[field]:
            errors.append(f"Metric field '{field}' missing 'mean' value")
    return errors


def _validate_brain_type(data: dict) -> list[str]:
    """Validate brain type field."""
    if "brain_type" in data and data["brain_type"] not in VALID_BRAIN_TYPES:
        return [f"Invalid brain_type: {data['brain_type']}. Valid types: {VALID_BRAIN_TYPES}"]
    return []


def _validate_session_count(data: dict) -> list[str]:
    """Validate session count meets minimum."""
    if "total_sessions" in data and data["total_sessions"] < MIN_SESSIONS_REQUIRED:
        return [
            f"Insufficient sessions: {data['total_sessions']}. "
            f"Minimum required: {MIN_SESSIONS_REQUIRED}",
        ]
    return []


def _validate_sessions_list(data: dict) -> tuple[list[str], list[str]]:
    """Validate sessions list structure and content."""
    errors = []
    warnings = []

    if "sessions" not in data:
        return errors, warnings

    sessions = data["sessions"]
    if len(sessions) != data.get("total_sessions", 0):
        warnings.append(
            f"sessions list length ({len(sessions)}) doesn't match "
            f"total_sessions ({data.get('total_sessions', 0)})",
        )

    for i, session in enumerate(sessions):
        if "experiment_id" not in session:
            errors.append(f"Session {i}: missing experiment_id")
        if "file_path" not in session:
            errors.append(f"Session {i}: missing file_path")
        if "num_runs" not in session:
            warnings.append(f"Session {i}: missing num_runs")
        elif session["num_runs"] < MIN_RUNS_PER_SESSION:
            warnings.append(
                f"Session {i}: only {session['num_runs']} runs, "
                f"recommended minimum: {MIN_RUNS_PER_SESSION}",
            )

    return errors, warnings


def validate_structure(data: dict) -> tuple[list[str], list[str]]:
    """Validate the structure of a NematodeBench submission."""
    errors = []
    warnings = []

    errors.extend(_validate_required_fields(data))
    errors.extend(_validate_metrics_structure(data))
    errors.extend(_validate_brain_type(data))
    errors.extend(_validate_session_count(data))

    session_errors, session_warnings = _validate_sessions_list(data)
    errors.extend(session_errors)
    warnings.extend(session_warnings)

    # Check seed uniqueness flag
    if "all_seeds_unique" in data and not data["all_seeds_unique"]:
        errors.append("Seed uniqueness validation failed (all_seeds_unique is False)")

    return errors, warnings


def validate_session_references(data: dict, base_path: Path) -> tuple[list[str], list[str]]:
    """Validate that session references exist."""
    errors = []
    warnings = []

    if "sessions" not in data:
        return errors, warnings

    for session in data["sessions"]:
        file_path = session.get("file_path")
        if file_path:
            full_path = base_path / file_path
            if not full_path.exists():
                errors.append(f"Session reference not found: {file_path}")
            else:
                # Check for experiment JSON
                json_files = list(full_path.glob("*.json")) if full_path.is_dir() else []
                if full_path.is_dir() and not json_files:
                    errors.append(f"No JSON file in session folder: {file_path}")

    return errors, warnings


RATE_METRICS = {"success_rate", "stability", "distance_efficiency", "learning_speed"}


def _validate_metric_mean(metric_name: str, mean: object) -> list[str]:
    """Validate a single metric's mean value."""
    if mean is None:
        return []

    if not isinstance(mean, (int, float)):
        return [f"Invalid mean value for {metric_name}: {mean}"]

    if math.isnan(mean):
        return [f"NaN value in {metric_name}.mean"]

    if metric_name in RATE_METRICS and (mean < 0 or mean > 1):
        return [f"{metric_name}.mean out of bounds [0,1]: {mean}"]

    return []


def _validate_metric_std(metric_name: str, std: object) -> list[str]:
    """Validate a single metric's std value."""
    if std is None:
        return []

    if not isinstance(std, (int, float)):
        return [f"Invalid std value for {metric_name}: {std}"]

    if std < 0:
        return [f"Negative std for {metric_name}: {std}"]

    return []


def validate_metrics(data: dict) -> tuple[list[str], list[str]]:
    """Validate metric values are reasonable."""
    errors = []
    warnings = []

    for metric_name, metric_data in data.get("metrics", {}).items():
        if not isinstance(metric_data, dict):
            continue

        errors.extend(_validate_metric_mean(metric_name, metric_data.get("mean")))
        errors.extend(_validate_metric_std(metric_name, metric_data.get("std")))

    return errors, warnings


def generate_summary(data: dict) -> dict:
    """Generate a summary of a NematodeBench submission."""
    metrics = data.get("metrics", {})
    success_rate = metrics.get("success_rate", {})
    composite = metrics.get("composite_score", {})

    return {
        "brain_type": data.get("brain_type", "unknown"),
        "category": data.get("category", "unknown"),
        "total_sessions": data.get("total_sessions", 0),
        "total_runs": data.get("total_runs", 0),
        "success_rate": f"{success_rate.get('mean', 0) * 100:.1f}% ± {success_rate.get('std', 0) * 100:.1f}%",
        "composite_score": f"{composite.get('mean', 0):.3f} ± {composite.get('std', 0):.3f}",
        "contributor": data.get("contributor", "unknown"),
        "all_seeds_unique": data.get("all_seeds_unique", False),
    }


def validate_submission(
    submission_path: Path,
    base_path: Path | None = None,
) -> ValidationResult:
    """Validate a benchmark submission file.

    Args:
        submission_path: Path to submission JSON file
        base_path: Base path for resolving config files (default: repo root)

    Returns
    -------
        ValidationResult with validation outcome
    """
    errors = []
    warnings = []

    # Determine base path
    if base_path is None:
        base_path = Path(__file__).parent.parent

    # Load and parse JSON
    try:
        with submission_path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            errors=[f"Invalid JSON: {e}"],
            warnings=[],
            summary={},
        )
    except FileNotFoundError:
        return ValidationResult(
            valid=False,
            errors=[f"Submission file not found: {submission_path}"],
            warnings=[],
            summary={},
        )

    # Check for NematodeBench format
    if "submission_id" not in data or "sessions" not in data:
        return ValidationResult(
            valid=False,
            errors=[
                "Invalid submission format. Expected NematodeBench format with 'submission_id' and 'sessions' fields.",
                "See docs/nematodebench/SUBMISSION_GUIDE.md for the required format.",
            ],
            warnings=[],
            summary={},
        )

    # Run validations
    e, w = validate_structure(data)
    errors.extend(e)
    warnings.extend(w)

    e, w = validate_session_references(data, base_path)
    errors.extend(e)
    warnings.extend(w)

    e, w = validate_metrics(data)
    errors.extend(e)
    warnings.extend(w)

    summary = generate_summary(data)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def print_result(result: ValidationResult) -> None:
    """Print validation result to console."""
    if result.valid:
        print("✓ Submission is VALID.\n")
    else:
        print("✗ Submission is INVALID.\n")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  ✗ {error}")
        print()

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
        print()

    if result.summary:
        print("Summary:")
        for key, value in result.summary.items():
            print(f"  {key}: {value}")


def main() -> None:
    """Run the submission validation CLI."""
    parser = argparse.ArgumentParser(
        description="Validate NematodeBench submission files",
        epilog="""
Examples:
  uv run scripts/evaluate_submission.py -s benchmarks/foraging_small/classical/20251228.json
""",
    )
    parser.add_argument(
        "--submission",
        "-s",
        type=Path,
        required=True,
        help="Path to submission JSON file",
    )
    parser.add_argument(
        "--base-path",
        "-b",
        type=Path,
        default=None,
        help="Base path for resolving config files",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Attempt to reproduce a subset of runs",
    )
    parser.add_argument(
        "--reproduce-runs",
        type=int,
        default=5,
        help="Number of runs to reproduce (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    result = validate_submission(args.submission, args.base_path)

    if args.reproduce and result.valid:
        print("Reproduction verification not yet implemented.")
        print("This would re-run experiments and compare results.")

    if args.json:
        output = {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "summary": result.summary,
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
