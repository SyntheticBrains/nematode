"""CSV export functions for Quantum Nematode simulation data."""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult, TrackingData


def export_simulation_results_to_csv(  # pragma: no cover
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """
    Export simulation results to CSV files.

    Args:
        all_results (list[SimulationResult]): List of simulation results.
        data_dir (Path): Directory to save the CSV files.
        file_prefix (str): Prefix for the output file names.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Export main simulation results
    _export_main_results(all_results, data_dir, file_prefix)

    # Export run-specific metrics
    _export_run_metrics(all_results, data_dir, file_prefix)

    # Export path data for each run
    _export_path_data(all_results, data_dir, file_prefix)


def _export_main_results(
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str,
) -> None:
    """Export main simulation results to CSV."""
    filename = f"{file_prefix}simulation_results.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        fieldnames = [
            "run",
            "steps",
            "total_reward",
            "last_total_reward",
            "efficiency_score",
            "path_length",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(
                {
                    "run": result.run,
                    "steps": result.steps,
                    "total_reward": result.total_reward,
                    "last_total_reward": result.last_total_reward,
                    "efficiency_score": result.efficiency_score,
                    "path_length": len(result.path),
                },
            )


def _export_run_metrics(
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str,
) -> None:
    """Export derived metrics per run to CSV."""
    filename = f"{file_prefix}run_metrics.csv"
    filepath = data_dir / filename

    # Calculate cumulative metrics
    cumulative_success_count = 0
    cumulative_steps = 0
    max_steps = max(result.steps for result in all_results) if all_results else 100

    with filepath.open("w", newline="") as csvfile:
        fieldnames = [
            "run",
            "steps",
            "is_success",
            "cumulative_reward",
            "efficiency_score",
            "success_rate_to_date",
            "average_steps_to_date",
            "running_average_steps_5",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(all_results):
            # Calculate success (assuming success = completed in fewer than max steps)
            is_success = result.steps < max_steps
            if is_success:
                cumulative_success_count += 1
            cumulative_steps += result.steps

            # Calculate running average over last 5 runs
            window_size = min(5, i + 1)
            start_idx = max(0, i - window_size + 1)
            running_avg_steps = (
                sum(all_results[j].steps for j in range(start_idx, i + 1)) / window_size
            )

            writer.writerow(
                {
                    "run": result.run,
                    "steps": result.steps,
                    "is_success": 1 if is_success else 0,
                    "cumulative_reward": result.total_reward,
                    "efficiency_score": result.efficiency_score,
                    "success_rate_to_date": cumulative_success_count / (i + 1),
                    "average_steps_to_date": cumulative_steps / (i + 1),
                    "running_average_steps_5": running_avg_steps,
                },
            )


def _export_path_data(
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str,
) -> None:
    """Export path coordinates for each run to CSV."""
    filename = f"{file_prefix}paths.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["run", "step", "x", "y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            for step, (x, y) in enumerate(result.path):
                writer.writerow(
                    {
                        "run": result.run,
                        "step": step,
                        "x": x,
                        "y": y,
                    },
                )


def export_performance_metrics_to_csv(  # pragma: no cover
    metrics: PerformanceMetrics,
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """
    Export performance metrics to CSV.

    Args:
        metrics (PerformanceMetrics): Performance metrics object.
        data_dir (Path): Directory to save the CSV file.
        file_prefix (str): Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{file_prefix}performance_metrics.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({"metric": "success_rate", "value": metrics.success_rate})
        writer.writerow({"metric": "average_steps", "value": metrics.average_steps})
        writer.writerow({"metric": "average_reward", "value": metrics.average_reward})


def export_tracking_data_to_csv(  # pragma: no cover
    tracking_data: TrackingData,
    brain_type: BrainType,
    data_dir: Path,
    qubits: int | None = None,
    file_prefix: str = "",
) -> None:
    """
    Export tracking data to CSV files.

    Args:
        tracking_data (TrackingData): Tracking data containing brain history.
        brain_type (BrainType): Type of the brain.
        data_dir (Path): Directory to save the CSV files.
        qubits (int | None): Number of qubits if applicable.
        file_prefix (str): Prefix for the output file names.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(tracking_data.data.keys())
    if not runs:
        logger.warning("No runs found in tracking data. Skipping CSV export.")
        return

    # Get the structure from the first run
    first_run_data = tracking_data.data[runs[0]]
    keys = list(first_run_data.__dict__.keys())

    # Export session-level data (last value per run for each metric)
    _export_session_tracking_data(
        tracking_data,
        runs,
        keys,
        data_dir,
        file_prefix,
        brain_type,
        qubits,
    )

    # Export step-by-step data for each run
    _export_detailed_tracking_data(tracking_data, runs, keys, data_dir, file_prefix)


def _export_session_tracking_data(  # noqa: PLR0913
    tracking_data: TrackingData,
    runs: list[int],
    keys: list[str],
    data_dir: Path,
    file_prefix: str,
    brain_type: BrainType,
    qubits: int | None,
) -> None:
    """Export session-level tracking data (last value per run)."""
    # Create metadata file
    metadata_filename = f"{file_prefix}tracking_metadata.csv"
    metadata_filepath = data_dir / metadata_filename

    with metadata_filepath.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["attribute", "value"])
        writer.writerow(["brain_type", brain_type.value])
        if qubits is not None:
            writer.writerow(["qubits", qubits])
        writer.writerow(["total_runs", len(runs)])

    # Process each key and create appropriate CSV structure
    for key in keys:
        # Gather the last value for this key in each run
        last_values = []
        for run in runs:
            run_data = tracking_data.data[run]
            values = getattr(run_data, key, None)
            if isinstance(values, list) and values:
                last_values.append(values[-1])
            else:
                last_values.append(values)

        # Skip if all values are None
        if all(val is None for val in last_values):
            continue

        _export_key_data_to_csv(key, runs, last_values, data_dir, file_prefix)


def _export_key_data_to_csv(
    key: str,
    runs: list[int],
    values: list[Any],
    data_dir: Path,
    file_prefix: str,
) -> None:
    """Export data for a specific key to CSV."""
    filename = f"{file_prefix}tracking_{key}.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        # Handle different data types
        if isinstance(values[0], dict):
            # Dictionary data (e.g., parameters)
            param_keys = set()
            for d in values:
                if isinstance(d, dict):
                    param_keys.update(d.keys())

            fieldnames = ["run", *sorted(param_keys)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for run, d in zip(runs, values, strict=False):
                row = {"run": run}
                if isinstance(d, dict):
                    row.update(d)
                else:
                    # Fill with NaN for missing data
                    row.update(dict.fromkeys(param_keys, np.nan))
                writer.writerow(row)

        elif isinstance(values[0], ActionData):
            # ActionData objects
            fieldnames = ["run", "state", "action", "probability"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for run, action_data in zip(runs, values, strict=False):
                if isinstance(action_data, ActionData):
                    writer.writerow(
                        {
                            "run": run,
                            "state": action_data.state,
                            "action": action_data.action,
                            "probability": action_data.probability,
                        },
                    )
                else:
                    writer.writerow(
                        {
                            "run": run,
                            "state": "",
                            "action": "",
                            "probability": np.nan,
                        },
                    )

        else:
            # Scalar values
            fieldnames = ["run", key]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for run, value in zip(runs, values, strict=False):
                writer.writerow(
                    {
                        "run": run,
                        key: value if value is not None else np.nan,
                    },
                )


def _export_detailed_tracking_data(  # noqa: C901, PLR0912
    tracking_data: TrackingData,
    runs: list[int],
    keys: list[str],
    data_dir: Path,
    file_prefix: str,
) -> None:
    """Export detailed step-by-step tracking data for each run."""
    detailed_dir = data_dir / "detailed"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        filename = f"{file_prefix}detailed_{key}.csv"
        filepath = detailed_dir / filename

        with filepath.open("w", newline="") as csvfile:
            # Determine the structure based on the first non-empty data
            sample_data = None
            for run in runs:
                run_data = tracking_data.data[run]
                values = getattr(run_data, key, None)
                if values and isinstance(values, list) and values:
                    sample_data = values[0]
                    break

            if sample_data is None:
                continue

            if isinstance(sample_data, dict):
                # Dictionary data - include all possible keys
                all_param_keys = set()
                for run in runs:
                    run_data = tracking_data.data[run]
                    values = getattr(run_data, key, [])
                    for value in values:
                        if isinstance(value, dict):
                            all_param_keys.update(value.keys())

                fieldnames = ["run", "step", *sorted(all_param_keys)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for run in runs:
                    run_data = tracking_data.data[run]
                    values = getattr(run_data, key, [])
                    for step, value in enumerate(values):
                        row = {"run": run, "step": step}
                        if isinstance(value, dict):
                            row.update(value)
                        writer.writerow(row)

            elif isinstance(sample_data, ActionData):
                # ActionData objects
                fieldnames = ["run", "step", "state", "action", "probability"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for run in runs:
                    run_data = tracking_data.data[run]
                    values = getattr(run_data, key, [])
                    for step, value in enumerate(values):
                        if isinstance(value, ActionData):
                            writer.writerow(
                                {
                                    "run": run,
                                    "step": step,
                                    "state": value.state,
                                    "action": value.action,
                                    "probability": value.probability,
                                },
                            )
                        else:
                            writer.writerow(
                                {
                                    "run": run,
                                    "step": step,
                                    "state": "",
                                    "action": "",
                                    "probability": np.nan,
                                },
                            )

            else:
                # Scalar values
                fieldnames = ["run", "step", key]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for run in runs:
                    run_data = tracking_data.data[run]
                    values = getattr(run_data, key, [])
                    for step, value in enumerate(values):
                        writer.writerow(
                            {
                                "run": run,
                                "step": step,
                                key: value if value is not None else np.nan,
                            },
                        )


def export_run_data_to_csv(  # pragma: no cover  # noqa: C901
    tracking_data: TrackingData,
    run: int,
    timestamp: str,
) -> None:
    """
    Export tracking data for a single run to CSV files.

    Args:
        tracking_data (TrackingData): Tracking data containing brain history.
        run (int): Run number to export.
        timestamp (str): Timestamp for the export directory.
    """
    run_dir = Path.cwd() / "exports" / timestamp / f"run_{run}" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    current_run_data = tracking_data.data.get(run, None)
    if current_run_data is None:
        logger.warning(f"No tracking data available for run {run}. Skipping CSV export.")
        return

    for key, values in current_run_data.__dict__.items():
        if values is None or (isinstance(values, list) and len(values) == 0):
            continue

        filename = f"{key}.csv"
        filepath = run_dir / filename

        with filepath.open("w", newline="") as csvfile:
            if isinstance(values[0], ActionData):
                # ActionData objects
                fieldnames = ["step", "state", "action", "probability"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for step, action_data in enumerate(values):
                    writer.writerow(
                        {
                            "step": step,
                            "state": action_data.state,
                            "action": action_data.action,
                            "probability": action_data.probability,
                        },
                    )

            elif isinstance(values[0], dict):
                # Dictionary data
                all_keys = set()
                for value in values:
                    if isinstance(value, dict):
                        all_keys.update(value.keys())

                fieldnames = ["step", *sorted(all_keys)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for step, value in enumerate(values):
                    row = {"step": step}
                    if isinstance(value, dict):
                        row.update(value)
                    writer.writerow(row)

            else:
                # Scalar values
                fieldnames = ["step", key]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for step, value in enumerate(values):
                    writer.writerow(
                        {
                            "step": step,
                            key: value if value is not None else np.nan,
                        },
                    )
