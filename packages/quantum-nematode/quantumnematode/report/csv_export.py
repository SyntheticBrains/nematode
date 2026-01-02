"""CSV export functions for Quantum Nematode simulation data."""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult, TrackingData

if TYPE_CHECKING:
    from quantumnematode.experiment.metadata import ExperimentMetadata


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
            "termination_reason",
            "success",
            "foods_collected",
            "foods_available",
            "satiety_remaining",
            "avg_distance_efficiency",
            "predator_encounters",
            "successful_evasions",
            "died_to_predator",
            "died_to_health_depletion",
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
                    "efficiency_score": result.efficiency_score
                    if result.efficiency_score is not None
                    else np.nan,
                    "path_length": len(result.path),
                    "termination_reason": result.termination_reason.value,
                    "success": result.success,
                    "foods_collected": result.foods_collected
                    if result.foods_collected is not None
                    else np.nan,
                    "foods_available": result.foods_available
                    if result.foods_available is not None
                    else np.nan,
                    "satiety_remaining": result.satiety_remaining
                    if result.satiety_remaining is not None
                    else np.nan,
                    "avg_distance_efficiency": result.average_distance_efficiency
                    if result.average_distance_efficiency is not None
                    else np.nan,
                    "predator_encounters": result.predator_encounters
                    if result.predator_encounters is not None
                    else np.nan,
                    "successful_evasions": result.successful_evasions
                    if result.successful_evasions is not None
                    else np.nan,
                    "died_to_predator": result.died_to_predator
                    if result.died_to_predator is not None
                    else np.nan,
                    "died_to_health_depletion": result.died_to_health_depletion
                    if result.died_to_health_depletion is not None
                    else np.nan,
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


def export_convergence_metrics_to_csv(  # pragma: no cover
    experiment_metadata: "ExperimentMetadata",
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """
    Export convergence analysis metrics to CSV.

    This exports the convergence-based benchmark metrics including convergence
    detection results, post-convergence performance, and composite scores.

    Args:
        experiment_metadata (ExperimentMetadata): Complete experiment metadata
            with convergence analysis.
        data_dir (Path): Directory to save the CSV file.
        file_prefix (str): Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{file_prefix}convergence_analysis.csv"
    filepath = data_dir / filename

    results = experiment_metadata.results

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Convergence status
        writer.writerow({"metric": "converged", "value": results.converged})
        writer.writerow({"metric": "convergence_run", "value": results.convergence_run or "N/A"})
        writer.writerow(
            {"metric": "runs_to_convergence", "value": results.runs_to_convergence or "N/A"},
        )

        # Post-convergence performance
        writer.writerow(
            {
                "metric": "post_convergence_success_rate",
                "value": (
                    f"{results.post_convergence_success_rate:.4f}"
                    if results.post_convergence_success_rate is not None
                    else "N/A"
                ),
            },
        )
        writer.writerow(
            {
                "metric": "post_convergence_avg_steps",
                "value": (
                    f"{results.post_convergence_avg_steps:.2f}"
                    if results.post_convergence_avg_steps is not None
                    else "N/A"
                ),
            },
        )
        writer.writerow(
            {
                "metric": "post_convergence_avg_foods",
                "value": (
                    f"{results.post_convergence_avg_foods:.2f}"
                    if results.post_convergence_avg_foods is not None
                    else "N/A"
                ),
            },
        )

        # Stability and efficiency
        writer.writerow(
            {
                "metric": "post_convergence_variance",
                "value": (
                    f"{results.post_convergence_variance:.4f}"
                    if results.post_convergence_variance is not None
                    else "N/A"
                ),
            },
        )
        writer.writerow(
            {
                "metric": "distance_efficiency",
                "value": (
                    f"{results.post_convergence_distance_efficiency:.4f}"
                    if results.post_convergence_distance_efficiency is not None
                    else "N/A"
                ),
            },
        )

        # Composite benchmark score
        writer.writerow(
            {
                "metric": "composite_benchmark_score",
                "value": (
                    f"{results.composite_benchmark_score:.4f}"
                    if results.composite_benchmark_score is not None
                    else "N/A"
                ),
            },
        )

        # Comparison metrics (all-run vs post-convergence)
        writer.writerow({"metric": "all_run_success_rate", "value": f"{results.success_rate:.4f}"})
        writer.writerow(
            {
                "metric": "success_rate_improvement",
                "value": (
                    f"{(results.post_convergence_success_rate - results.success_rate):.4f}"
                    if results.post_convergence_success_rate is not None
                    else "N/A"
                ),
            },
        )


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

    runs = sorted(tracking_data.brain_data.keys())
    if not runs:
        logger.warning("No runs found in tracking data. Skipping CSV export.")
        return

    # Get the structure from the first run
    first_run_data = tracking_data.brain_data[runs[0]]
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
            run_data = tracking_data.brain_data[run]
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
                run_data = tracking_data.brain_data[run]
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
                    run_data = tracking_data.brain_data[run]
                    values = getattr(run_data, key, [])
                    for value in values:
                        if isinstance(value, dict):
                            all_param_keys.update(value.keys())

                fieldnames = ["run", "step", *sorted(all_param_keys)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for run in runs:
                    run_data = tracking_data.brain_data[run]
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
                    run_data = tracking_data.brain_data[run]
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
                    run_data = tracking_data.brain_data[run]
                    values = getattr(run_data, key, [])
                    for step, value in enumerate(values):
                        writer.writerow(
                            {
                                "run": run,
                                "step": step,
                                key: value if value is not None else np.nan,
                            },
                        )


def export_run_data_to_csv(  # pragma: no cover  # noqa: C901, PLR0912, PLR0915
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

    current_brain_run_data = tracking_data.brain_data.get(run, None)
    if current_brain_run_data is None:
        logger.warning(f"No brain tracking data available for run {run}. Skipping CSV export.")
        return

    for key, values in current_brain_run_data.__dict__.items():
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

    # Export episode tracking data (foraging environments)
    current_episode_run_data = tracking_data.episode_data.get(run, None)
    if current_episode_run_data is not None:
        # Export satiety history
        if current_episode_run_data.satiety_history:
            filepath = run_dir / "satiety_history.csv"
            with filepath.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["step", "satiety"])
                writer.writeheader()
                for step, satiety in enumerate(current_episode_run_data.satiety_history):
                    writer.writerow({"step": step, "satiety": satiety})

        # Export health history (if health system was enabled)
        if current_episode_run_data.health_history:
            filepath = run_dir / "health_history.csv"
            with filepath.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["step", "health"])
                writer.writeheader()
                for step, health in enumerate(current_episode_run_data.health_history):
                    writer.writerow({"step": step, "health": health})

        # Export temperature history (if thermotaxis was enabled)
        if current_episode_run_data.temperature_history:
            filepath = run_dir / "temperature_history.csv"
            with filepath.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["step", "temperature"])
                writer.writeheader()
                for step, temperature in enumerate(current_episode_run_data.temperature_history):
                    writer.writerow({"step": step, "temperature": temperature})

        # Export distance efficiencies
        if current_episode_run_data.distance_efficiencies:
            filepath = run_dir / "distance_efficiencies.csv"
            with filepath.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["food_number", "distance_efficiency"])
                writer.writeheader()
                for food_num, dist_eff in enumerate(
                    current_episode_run_data.distance_efficiencies,
                    start=1,
                ):
                    writer.writerow({"food_number": food_num, "distance_efficiency": dist_eff})

        # Export foraging summary for this run
        filepath = run_dir / "foraging_summary.csv"
        with filepath.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerow(
                {"metric": "foods_collected", "value": current_episode_run_data.foods_collected},
            )
            if current_episode_run_data.distance_efficiencies:
                avg_dist_eff = sum(current_episode_run_data.distance_efficiencies) / len(
                    current_episode_run_data.distance_efficiencies,
                )
                writer.writerow(
                    {"metric": "avg_distance_efficiency", "value": f"{avg_dist_eff:.4f}"},
                )
            if current_episode_run_data.satiety_history:
                final_satiety = current_episode_run_data.satiety_history[-1]
                max_satiety = max(current_episode_run_data.satiety_history)
                writer.writerow({"metric": "final_satiety", "value": f"{final_satiety:.2f}"})
                writer.writerow({"metric": "max_satiety", "value": f"{max_satiety:.2f}"})
            if current_episode_run_data.health_history:
                final_health = current_episode_run_data.health_history[-1]
                max_health = max(current_episode_run_data.health_history)
                writer.writerow({"metric": "final_health", "value": f"{final_health:.2f}"})
                writer.writerow({"metric": "max_health", "value": f"{max_health:.2f}"})
            if current_episode_run_data.temperature_history:
                temps = current_episode_run_data.temperature_history
                final_temp = temps[-1]
                mean_temp = sum(temps) / len(temps)
                min_temp = min(temps)
                max_temp = max(temps)
                writer.writerow({"metric": "final_temperature", "value": f"{final_temp:.2f}"})
                writer.writerow({"metric": "mean_temperature", "value": f"{mean_temp:.2f}"})
                writer.writerow({"metric": "min_temperature", "value": f"{min_temp:.2f}"})
                writer.writerow({"metric": "max_temperature", "value": f"{max_temp:.2f}"})


# Dynamic Foraging Environment Specific Exports


def export_foraging_results_to_csv(  # pragma: no cover
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """Export foraging-specific results to CSV.

    Parameters
    ----------
    all_results : list[SimulationResult]
        List of simulation results with foraging data.
    data_dir : Path
        Directory to save the CSV file.
    file_prefix : str, optional
        Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_prefix}foraging_results.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        fieldnames = [
            "run",
            "foods_collected",
            "foods_available",
            "satiety_remaining",
            "avg_distance_efficiency",
            "foraging_efficiency",  # foods per step
            "predator_encounters",
            "successful_evasions",
            "died_to_predator",
            "died_to_health_depletion",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            # Only export if this is a dynamic foraging environment result
            if result.foods_collected is not None:
                # Calculate foraging efficiency (foods per step)
                foraging_efficiency = (
                    result.foods_collected / result.steps if result.steps > 0 else 0.0
                )

                writer.writerow(
                    {
                        "run": result.run,
                        "foods_collected": result.foods_collected,
                        "foods_available": result.foods_available
                        if result.foods_available is not None
                        else np.nan,
                        "satiety_remaining": result.satiety_remaining
                        if result.satiety_remaining is not None
                        else np.nan,
                        "avg_distance_efficiency": result.average_distance_efficiency
                        if result.average_distance_efficiency is not None
                        else np.nan,
                        "foraging_efficiency": foraging_efficiency,
                        "predator_encounters": result.predator_encounters
                        if result.predator_encounters is not None
                        else np.nan,
                        "successful_evasions": result.successful_evasions
                        if result.successful_evasions is not None
                        else np.nan,
                        "died_to_predator": result.died_to_predator
                        if result.died_to_predator is not None
                        else np.nan,
                        "died_to_health_depletion": result.died_to_health_depletion
                        if result.died_to_health_depletion is not None
                        else np.nan,
                    },
                )

    logger.info(f"Foraging results exported to {filepath}")


def export_distance_efficiencies_to_csv(  # pragma: no cover
    tracking_data: TrackingData,
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """Export all individual distance efficiencies to CSV.

    Parameters
    ----------
    tracking_data : TrackingData
        Tracking data containing episode data for all runs.
    data_dir : Path
        Directory to save the CSV file.
    file_prefix : str, optional
        Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_prefix}distance_efficiencies.csv"
    filepath = data_dir / filename

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["run", "food_number", "distance_efficiency"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through episode data and export individual distance efficiencies
        for run_num in sorted(tracking_data.episode_data.keys()):
            episode_data = tracking_data.episode_data[run_num]
            for food_num, dist_eff in enumerate(episode_data.distance_efficiencies, start=1):
                writer.writerow(
                    {
                        "run": run_num,
                        "food_number": food_num,
                        "distance_efficiency": dist_eff,
                    },
                )

    logger.info(f"Distance efficiencies exported to {filepath}")


def export_foraging_session_metrics_to_csv(  # pragma: no cover
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """Export session-level foraging metrics to CSV.

    Parameters
    ----------
    all_results : list[SimulationResult]
        List of all simulation results.
    metrics : PerformanceMetrics
        Session performance metrics.
    data_dir : Path
        Directory to save the CSV file.
    file_prefix : str, optional
        Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_prefix}foraging_session_metrics.csv"
    filepath = data_dir / filename

    # Calculate session-level foraging statistics
    foraging_results = [r for r in all_results if r.foods_collected is not None]

    if not foraging_results:
        logger.warning("No foraging results to export")
        return

    total_foods_collected = sum(
        r.foods_collected for r in foraging_results if r.foods_collected is not None
    )
    total_runs = len(foraging_results)
    avg_foods_per_run = total_foods_collected / total_runs if total_runs > 0 else 0.0

    # Get all distance efficiencies
    all_distance_effs = [
        r.average_distance_efficiency
        for r in foraging_results
        if r.average_distance_efficiency is not None
    ]

    overall_distance_eff = (
        sum(all_distance_effs) / len(all_distance_effs) if all_distance_effs else 0.0
    )

    # Count termination reasons
    starvation_count = sum(
        1 for r in foraging_results if "starv" in r.termination_reason.value.lower()
    )
    max_steps_count = sum(
        1 for r in foraging_results if "max_steps" in r.termination_reason.value.lower()
    )
    health_depleted_count = sum(
        1 for r in foraging_results if "health" in r.termination_reason.value.lower()
    )
    success_count = sum(1 for r in foraging_results if r.success)

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({"metric": "total_runs", "value": total_runs})
        writer.writerow({"metric": "total_foods_collected", "value": total_foods_collected})
        writer.writerow({"metric": "avg_foods_per_run", "value": f"{avg_foods_per_run:.2f}"})
        writer.writerow(
            {"metric": "overall_distance_efficiency", "value": f"{overall_distance_eff:.4f}"},
        )
        writer.writerow({"metric": "success_count", "value": success_count})
        writer.writerow({"metric": "starvation_count", "value": starvation_count})
        writer.writerow({"metric": "health_depleted_count", "value": health_depleted_count})
        writer.writerow({"metric": "max_steps_count", "value": max_steps_count})
        writer.writerow({"metric": "success_rate", "value": f"{metrics.success_rate:.4f}"})

        # Add foraging-specific metrics from PerformanceMetrics
        if metrics.foraging_efficiency is not None:
            writer.writerow(
                {
                    "metric": "foraging_efficiency_foods_per_step",
                    "value": f"{metrics.foraging_efficiency:.6f}",
                },
            )
        if metrics.average_distance_efficiency is not None:
            writer.writerow(
                {
                    "metric": "avg_distance_efficiency_from_metrics",
                    "value": f"{metrics.average_distance_efficiency:.4f}",
                },
            )
        if metrics.average_foods_collected is not None:
            writer.writerow(
                {
                    "metric": "avg_foods_collected_from_metrics",
                    "value": f"{metrics.average_foods_collected:.2f}",
                },
            )

    logger.info(f"Foraging session metrics exported to {filepath}")


def export_predator_results_to_csv(  # pragma: no cover
    all_results: list[SimulationResult],
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """Export predator-specific results to CSV.

    Parameters
    ----------
    all_results : list[SimulationResult]
        List of simulation results with predator data.
    data_dir : Path
        Directory to save the CSV file.
    file_prefix : str, optional
        Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_prefix}predator_results.csv"
    filepath = data_dir / filename

    # Filter for results with predator data
    predator_results = [r for r in all_results if r.predator_encounters is not None]

    if not predator_results:
        logger.warning("No predator results to export")
        return

    with filepath.open("w", newline="") as csvfile:
        fieldnames = [
            "run",
            "predator_encounters",
            "successful_evasions",
            "evasion_rate",
            "died_to_predator",
            "foods_collected",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in predator_results:
            # Calculate evasion rate
            evasion_rate = (
                result.successful_evasions / result.predator_encounters
                if result.predator_encounters
                and result.predator_encounters > 0
                and result.successful_evasions is not None
                else 0.0
            )

            writer.writerow(
                {
                    "run": result.run,
                    "predator_encounters": result.predator_encounters,
                    "successful_evasions": result.successful_evasions,
                    "evasion_rate": evasion_rate,
                    "died_to_predator": result.died_to_predator,
                    "foods_collected": result.foods_collected
                    if result.foods_collected is not None
                    else np.nan,
                },
            )

    logger.info(f"Predator results exported to {filepath}")


def export_predator_session_metrics_to_csv(  # pragma: no cover
    all_results: list[SimulationResult],
    metrics: PerformanceMetrics,
    data_dir: Path,
    file_prefix: str = "",
) -> None:
    """Export session-level predator metrics to CSV.

    Parameters
    ----------
    all_results : list[SimulationResult]
        List of all simulation results.
    metrics : PerformanceMetrics
        Session performance metrics.
    data_dir : Path
        Directory to save the CSV file.
    file_prefix : str, optional
        Prefix for the output file name.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_prefix}predator_session_metrics.csv"
    filepath = data_dir / filename

    # Filter for results with predator data
    predator_results = [r for r in all_results if r.predator_encounters is not None]

    if not predator_results:
        logger.warning("No predator results to export")
        return

    total_runs = len(predator_results)
    total_encounters = sum(
        r.predator_encounters for r in predator_results if r.predator_encounters is not None
    )
    total_evasions = sum(
        r.successful_evasions for r in predator_results if r.successful_evasions is not None
    )
    total_deaths = sum(1 for r in predator_results if r.died_to_predator)
    survival_rate = 1.0 - (total_deaths / total_runs) if total_runs > 0 else 0.0
    overall_evasion_rate = total_evasions / total_encounters if total_encounters > 0 else 0.0

    with filepath.open("w", newline="") as csvfile:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({"metric": "total_runs", "value": total_runs})
        writer.writerow({"metric": "total_predator_encounters", "value": total_encounters})
        writer.writerow({"metric": "total_successful_evasions", "value": total_evasions})
        writer.writerow({"metric": "total_predator_deaths", "value": total_deaths})
        writer.writerow({"metric": "survival_rate", "value": f"{survival_rate:.4f}"})
        writer.writerow({"metric": "overall_evasion_rate", "value": f"{overall_evasion_rate:.4f}"})

        # Add metrics from PerformanceMetrics
        if metrics.average_predator_encounters is not None:
            writer.writerow(
                {
                    "metric": "avg_predator_encounters_per_run",
                    "value": f"{metrics.average_predator_encounters:.2f}",
                },
            )
        if metrics.average_successful_evasions is not None:
            writer.writerow(
                {
                    "metric": "avg_successful_evasions_per_run",
                    "value": f"{metrics.average_successful_evasions:.2f}",
                },
            )

    logger.info(f"Predator session metrics exported to {filepath}")
