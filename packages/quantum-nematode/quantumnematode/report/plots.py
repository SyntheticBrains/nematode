"""Plotting functions for Quantum Nematode."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.report.dtypes import BrainDataSnapshot, PerformanceMetrics, TrackingData


def plot_success_rate_over_time(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    success_rates: list[float],
) -> None:
    """
    Plot success rates over time and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        plot_dir (Path): Directory to save the plot.
        success_rates (list[float]): List of success rates.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, success_rates, marker="o", label="Success Rate Over Time")
    plt.title("Success Rate Over Time")
    plt.xlabel("Run")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}success_rate_over_time.png")
    plt.close()


def plot_last_cumulative_rewards(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    last_cumulative_rewards: list[float],
) -> None:
    """
    Plot last cumulative rewards over time and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        plot_dir (Path): Directory to save the plot.
        last_cumulative_rewards (list[float]): List of last cumulative rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, last_cumulative_rewards, marker="o", label="Last Cumulative Rewards")
    plt.title("Last Cumulative Reward Over Time")
    plt.xlabel("Run")
    plt.ylabel("Last Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_last_reward_over_time.png")
    plt.close()


def plot_cumulative_reward_per_run(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    cumulative_rewards: list[float],
) -> None:
    """
    Plot cumulative rewards per run and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int): List of run indices.
        plot_dir (Path): Directory to save the plot.
        cumulative_rewards (list[float]): List of cumulative rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, cumulative_rewards, marker="o", label="Cumulative Reward per Run")
    plt.title("Cumulative Reward per Run")
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_reward_per_run.png")
    plt.close()


def plot_steps_per_run(  # pragma: no cover
    metrics: PerformanceMetrics,
    file_prefix: str,
    runs: list[int],
    steps: list[int],
    plot_dir: Path,
) -> None:
    """
    Plot steps per run and save the plot.

    Args:
        metrics (PerformanceMetrics): An object containing metrics like average steps.
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        steps (list[int]): List of steps per run.
        plot_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, steps, marker="o", label="Steps per Run")
    plt.axhline(y=metrics.average_steps, color="r", linestyle="--", label="Average Steps")
    plt.title("Steps per Run")
    plt.xlabel("Run")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}steps_per_run.png")
    plt.close()


def plot_running_average_steps(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    steps: list[int],
    plot_dir: Path,
    window_size: int = 5,
) -> None:
    """
    Plot running/moving average of steps per run and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        steps (list[int]): List of steps per run.
        plot_dir (Path): Directory to save the plot.
        window_size (int): Window size for moving average calculation (default: 5).
    """
    if len(steps) < window_size:
        logger.warning(
            f"Not enough data points ({len(steps)}) for moving average "
            f"with window size {window_size}. Using smaller window.",
        )
        window_size = max(1, len(steps))

    # Calculate running average
    running_avg = []
    for i in range(len(steps)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        avg = sum(steps[start_idx:end_idx]) / (end_idx - start_idx)
        running_avg.append(avg)

    # Calculate overall session average
    session_avg = sum(steps) / len(steps)

    plt.figure(figsize=(12, 8))

    # Plot actual steps per run
    plt.plot(runs, steps, marker="o", alpha=0.6, color="lightblue", label="Actual Steps per Run")

    # Plot running average
    plt.plot(
        runs,
        running_avg,
        marker="s",
        linewidth=2,
        color="blue",
        label=f"Running Average (window={window_size})",
    )

    # Plot session average line
    plt.axhline(
        y=session_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Session Average ({session_avg:.1f})",
    )

    plt.title(f"Running Average Steps per Run (Window Size: {window_size})")
    plt.xlabel("Run")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(alpha=0.3)

    # Add text annotation with session stats
    plt.text(
        0.02,
        0.98,
        f"Total Runs: {len(runs)}\nSession Avg: {session_avg:.2f}\nWindow Size: {window_size}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}running_average_steps.png")
    plt.close()


def plot_tracking_data_by_session(  # pragma: no cover  # noqa: C901, PLR0912, PLR0915
    tracking_data: TrackingData,
    brain_type: BrainType,
    plot_dir: Path,
    qubits: int | None = None,
    file_prefix: str = "",
) -> None:
    """
    Generate and save plots for the last data point of each tracked variable per run.

    Args:
        tracking_data (TrackingData): Tracking data containing brain history.
        brain_type (BrainType): Type of the brain (e.g., "quantum", "classical").
        plot_dir (Path): Directory to save the plots.
        qubits (int | None): Number of qubits if applicable, otherwise None.
        file_prefix (str): Prefix for the output file names.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_postfix: str = (
        f" [{brain_type.value} {qubits}Q]" if isinstance(qubits, int) else f" [{brain_type.value}]"
    )

    runs = sorted(tracking_data.brain_data.keys())
    if not runs:
        logger.warning("No runs found in tracking data. Skipping plots.")
        return
    first_run_data = tracking_data.brain_data[runs[0]]
    if isinstance(first_run_data, BrainDataSnapshot):
        keys = list(first_run_data.last_values.keys())
    else:
        keys = list(first_run_data.__dict__.keys())

    for key in keys:
        # Gather the last value for this key in each run
        last_values = []
        for run in runs:
            run_data = tracking_data.brain_data[run]
            if isinstance(run_data, BrainDataSnapshot):
                last_values.append(run_data.last_values.get(key))
            else:
                v = getattr(run_data, key, None)
                if isinstance(v, list) and v:
                    last_values.append(v[-1])
                else:
                    last_values.append(v)
        # Skip if all values are None or empty
        if all(val is None for val in last_values):
            logger.warning(f"No data available for {key} across runs. Skipping plot.")
            continue
        # Handle last value is list or contains lists (e.g., satiety_history) - skip plotting
        if any(isinstance(val, list) for val in last_values):
            logger.warning(
                f"Skipping plot for {key} - list data type not supported for session plots.",
            )
            continue
        # Handle last value is dict (e.g., parameter dicts)
        if isinstance(last_values[0], dict):
            param_keys = set()
            for d in last_values:
                if isinstance(d, dict):
                    param_keys.update(d.keys())
            plt.figure(figsize=(10, 6))
            for param in param_keys:
                param_values = [
                    d.get(param, np.nan) if isinstance(d, dict) else np.nan for d in last_values
                ]
                plt.plot(runs, param_values, marker="o", label=param)
            plt.title(f"{key}{title_postfix}")
            plt.xlabel("Run")
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        # Handle last value is ActionData
        elif isinstance(last_values[0], ActionData):
            probs = [a.probability if isinstance(a, ActionData) else np.nan for a in last_values]
            actions = [a.action if isinstance(a, ActionData) else "" for a in last_values]
            plt.figure(figsize=(10, 6))
            plt.plot(runs, probs, marker="o", color="orange", label="Probability (last step)")
            for run, prob, action in zip(runs, probs, actions, strict=False):
                plt.text(
                    run,
                    prob + 0.02,
                    action,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            plt.title(f"{key} (last step per run){title_postfix}")
            plt.xlabel("Run")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        # Handle last value is scalar
        elif isinstance(last_values[0], (int, float)) or all(
            (v is None or isinstance(v, (int, float))) for v in last_values
        ):
            plot_values = [v if v is not None else np.nan for v in last_values]
            plt.figure(figsize=(10, 6))
            plt.plot(runs, plot_values, marker="o", label=key)
            plt.title(f"{key}{title_postfix}")
            plt.xlabel("Run")
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        else:
            logger.warning(f"Unrecognized data type for {key}. Skipping plot.")


def plot_tracking_data_by_latest_run(  # pragma: no cover  # noqa: C901, PLR0912, PLR0915
    tracking_data: TrackingData,
    timestamp: str,
    run: int,
) -> None:
    """
    Generate and save plots for tracked agent data for a single run.

    Args:
        timestamp (str): Timestamp for the plot directory.
        agent (QuantumNematodeAgent): Agent containing brain with tracking histories.
        run (int): Index of the run.
    """
    run_dir = Path.cwd() / "exports" / timestamp / f"run_{run}" / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    current_run_data = tracking_data.brain_data.get(run, None)

    if current_run_data is None:
        logger.warning(f"No tracking data available for run {run}. Skipping plots.")
        return

    if isinstance(current_run_data, BrainDataSnapshot):
        logger.warning(
            f"Brain data for run {run} has been flushed to snapshot. Skipping per-run plots.",
        )
        return

    for key, values in current_run_data.__dict__.items():
        if values is None or len(values) == 0:
            logger.warning(f"No data available for {key} in run {run}. Skipping plot.")
            continue

        plt.figure(figsize=(8, 4))
        # values is always a list of the type
        first = values[0]
        if isinstance(first, ActionData):
            # Plot probability as a line, annotate actions at each step
            actions = [a.action for a in values]
            probs = [a.probability for a in values]
            steps = list(range(len(values)))
            plt.plot(steps, probs, color="orange", marker="o", label="Probability")
            for _i, (step, prob, action) in enumerate(zip(steps, probs, actions, strict=False)):
                plt.text(
                    step,
                    prob + 0.02,
                    action,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            plt.title(f"Action Probabilities with Actions (run {run})")
            plt.ylabel("probability")
            plt.legend()
        elif isinstance(first, dict):
            # Plot each parameter in the dict as a line over steps
            param_keys = list(first.keys())
            steps = list(range(len(values)))
            for param in param_keys:
                param_values = [v.get(param, np.nan) for v in values]
                plt.plot(steps, param_values, marker="o", label=param)
            plt.title(f"{key} (run {run})")
            plt.ylabel(key)
            plt.legend()
        else:
            # Scalar or list of floats/ints
            steps = list(range(len(values)))
            plt.plot(steps, values, marker="o")
            plt.ylabel(key)
        plt.title(f"{key} (run {run})")
        plt.xlabel("step")
        plt.tight_layout()
        plt.savefig(run_dir / f"{key}.png")
        plt.close()

    # Plot episode tracking data (foraging environments)
    episode_data = tracking_data.episode_data.get(run, None)
    if episode_data is not None:
        # Plot satiety progression
        if episode_data.satiety_history:
            satiety_history = episode_data.satiety_history
            max_satiety = max(satiety_history) if satiety_history else 100.0
            steps = list(range(len(satiety_history)))

            plt.figure(figsize=(14, 6))
            plt.plot(steps, satiety_history, linewidth=2, label="Satiety Level", color="blue")
            plt.axhline(
                y=0,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Starvation Threshold",
            )
            plt.axhline(
                y=max_satiety * 0.2,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="Low Satiety (20%)",
            )
            plt.axhline(
                y=max_satiety,
                color="green",
                linestyle=":",
                linewidth=1.5,
                alpha=0.5,
                label=f"Max Satiety ({max_satiety:.0f})",
            )
            plt.fill_between(steps, 0, satiety_history, alpha=0.2, color="blue")
            plt.title(f"Satiety Progression (run {run})")
            plt.xlabel("Step")
            plt.ylabel("Satiety Level")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "satiety_progression.png")
            plt.close()

        # Plot health progression (if health system was enabled)
        if episode_data.health_history:
            health_history = episode_data.health_history
            max_health = max(health_history) if health_history else 100.0
            steps = list(range(len(health_history)))

            plt.figure(figsize=(14, 6))
            plt.plot(steps, health_history, linewidth=2, label="Health (HP)", color="purple")
            plt.axhline(
                y=0,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Death Threshold",
            )
            plt.axhline(
                y=max_health * 0.3,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="Low Health (30%)",
            )
            plt.axhline(
                y=max_health,
                color="green",
                linestyle=":",
                linewidth=1.5,
                alpha=0.5,
                label=f"Max Health ({max_health:.0f})",
            )
            plt.fill_between(steps, 0, health_history, alpha=0.2, color="purple")
            plt.title(f"Health Progression (run {run})")
            plt.xlabel("Step")
            plt.ylabel("Health (HP)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "health_progression.png")
            plt.close()

        # Plot temperature progression (if thermotaxis was enabled)
        if episode_data.temperature_history:
            temperature_history = episode_data.temperature_history
            steps = list(range(len(temperature_history)))

            plt.figure(figsize=(14, 6))
            plt.plot(
                steps,
                temperature_history,
                linewidth=2,
                label="Temperature",
                color="red",
            )
            # Calculate and show mean temperature
            mean_temp = float(np.mean(temperature_history))
            plt.axhline(
                y=mean_temp,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_temp:.1f}°C",
            )
            plt.title(f"Temperature Progression (run {run})")
            plt.xlabel("Step")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "temperature_progression.png")
            plt.close()

        # Plot distance efficiencies for this run
        if episode_data.distance_efficiencies:
            dist_effs = episode_data.distance_efficiencies
            food_numbers = list(range(1, len(dist_effs) + 1))

            plt.figure(figsize=(10, 6))
            plt.bar(food_numbers, dist_effs, alpha=0.7, color="skyblue", edgecolor="black")
            mean_eff = float(np.mean(dist_effs))
            plt.axhline(
                y=mean_eff,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Average: {mean_eff:.3f}",
            )
            plt.axhline(
                y=1.0,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.5,
                label="Optimal (1.0)",
            )
            plt.title(f"Distance Efficiency per Food (run {run})")
            plt.xlabel("Food Number")
            plt.ylabel("Distance Efficiency")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "distance_efficiencies.png")
            plt.close()


# Dynamic Foraging Environment Specific Plots


def plot_foods_collected_per_run(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    foods_collected: list[int],
    foods_available: int,
) -> None:
    """Plot foods collected per run with foods available reference line.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    foods_collected : list[int]
        Number of foods collected in each run.
    foods_available : int
        Maximum number of foods available per run.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(runs, foods_collected, alpha=0.7, label="Foods Collected")
    plt.axhline(
        y=foods_available,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Foods Available ({foods_available})",
    )
    average_foods = sum(foods_collected) / len(foods_collected)
    plt.axhline(
        y=average_foods,
        color="g",
        linestyle=":",
        linewidth=2,
        label=f"Average ({average_foods:.1f})",
    )
    plt.title("Foods Collected Per Run")
    plt.xlabel("Run")
    plt.ylabel("Foods Collected")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}foods_collected_per_run.png")
    plt.close()


def plot_distance_efficiency_trend(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    avg_distance_efficiencies: list[float],
) -> None:
    """Plot average distance efficiency trend over runs.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    avg_distance_efficiencies : list[float]
        Average distance efficiency for each run.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        runs,
        avg_distance_efficiencies,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="Avg Distance Efficiency",
    )
    overall_avg = sum(avg_distance_efficiencies) / len(avg_distance_efficiencies)
    plt.axhline(
        y=overall_avg,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Overall Average ({overall_avg:.3f})",
    )
    plt.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    plt.title("Distance Efficiency Trend Over Runs")
    plt.xlabel("Run")
    plt.ylabel("Average Distance Efficiency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}distance_efficiency_trend.png")
    plt.close()


def plot_foraging_efficiency_per_run(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    foods_collected: list[int],
    steps: list[int],
) -> None:
    """Plot foraging efficiency (foods per step) over runs.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    foods_collected : list[int]
        Number of foods collected in each run.
    steps : list[int]
        Number of steps taken in each run.
    """
    if len(foods_collected) != len(steps):
        logger.error(
            "Length of foods_collected and steps must be the same "
            "to calculate foraging efficiency.",
        )
        return
    # Calculate foods per step ratio
    foraging_efficiency = [
        foods / step if step > 0 else 0.0
        for foods, step in zip(foods_collected, steps, strict=True)
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(
        runs,
        foraging_efficiency,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=6,
        color="purple",
        label="Foraging Efficiency",
    )
    avg_efficiency = sum(foraging_efficiency) / len(foraging_efficiency)
    plt.axhline(
        y=avg_efficiency,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_efficiency:.4f} foods/step)",
    )
    plt.title("Foraging Efficiency (Foods per Step) Over Runs")
    plt.xlabel("Run")
    plt.ylabel("Foods per Step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}foraging_efficiency_per_run.png")
    plt.close()


def plot_satiety_at_episode_end(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    satiety_remaining: list[float],
    max_satiety: float,
) -> None:
    """Plot satiety levels at episode end per run.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    satiety_remaining : list[float]
        Final satiety level at end of each run.
    max_satiety : float
        Maximum satiety level.
    """
    plt.figure(figsize=(12, 6))
    colors = [
        "red" if s <= 0 else "orange" if s < max_satiety * 0.2 else "green"
        for s in satiety_remaining
    ]
    plt.bar(runs, satiety_remaining, alpha=0.7, color=colors, label="Satiety Remaining")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Starvation Threshold",
    )
    plt.axhline(
        y=max_satiety * 0.2,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Low Satiety (20%)",
    )
    avg_satiety = sum(satiety_remaining) / len(satiety_remaining)
    plt.axhline(
        y=avg_satiety,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_satiety:.1f})",
    )
    plt.title("Satiety at Episode End Per Run")
    plt.xlabel("Run")
    plt.ylabel("Satiety Level")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}satiety_at_episode_end.png")
    plt.close()


def plot_all_distance_efficiencies_distribution(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    all_distance_efficiencies: list[float],
) -> None:
    """Plot distribution of all distance efficiencies across all runs.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    all_distance_efficiencies : list[float]
        All distance efficiency values from all food collections across all runs.
    """
    if not all_distance_efficiencies:
        logger.warning("No distance efficiencies to plot")
        return

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(all_distance_efficiencies, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    mean_eff = np.mean(all_distance_efficiencies)
    median_eff = np.median(all_distance_efficiencies)
    ax1.axvline(mean_eff, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_eff:.3f}")
    ax1.axvline(
        median_eff,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_eff:.3f}",
    )
    ax1.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Optimal (1.0)")
    ax1.set_title("Distribution of Distance Efficiencies")
    ax1.set_xlabel("Distance Efficiency")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Box plot
    ax2.boxplot(all_distance_efficiencies, vert=True, patch_artist=True)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Optimal (1.0)")
    ax2.set_title("Distance Efficiency Box Plot")
    ax2.set_ylabel("Distance Efficiency")
    ax2.set_xticklabels(["All Foods"])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}distance_efficiency_distribution.png")
    plt.close()


def plot_termination_reasons_breakdown(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    termination_counts: dict[str, int],
) -> None:
    """Plot breakdown of termination reasons across all runs.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    termination_counts : dict[str, int]
        Dictionary mapping termination reason names to counts.
    """
    if not termination_counts:
        logger.warning("No termination data to plot")
        return

    reasons = list(termination_counts.keys())
    counts = list(termination_counts.values())
    colors = [
        "green"
        if "goal" in r.lower() or "food" in r.lower()
        else "red"
        if "starved" in r.lower()
        else "darkred"
        if "predator" in r.lower()
        else "purple"
        if "health" in r.lower()
        else "orange"
        if "max_steps" in r.lower()
        else "gray"
        for r in reasons
    ]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    ax1.bar(reasons, counts, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_title("Termination Reasons Breakdown")
    ax1.set_xlabel("Termination Reason")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)
    for i, (_reason, count) in enumerate(zip(reasons, counts, strict=False)):
        ax1.text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

    # Pie chart
    ax2.pie(counts, labels=reasons, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Termination Reasons Distribution")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}termination_reasons_breakdown.png")
    plt.close()


def plot_foods_vs_reward_correlation(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    foods_collected: list[int],
    total_rewards: list[float],
) -> None:
    """Plot correlation between foods collected and total reward.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    foods_collected : list[int]
        Number of foods collected in each run.
    total_rewards : list[float]
        Total reward for each run.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(foods_collected, total_rewards, alpha=0.6, s=100, edgecolors="black")

    # Add trend line only if there's variation in the data
    if (
        len(foods_collected) > 1
        and len(total_rewards) == len(foods_collected)
        and len(set(foods_collected)) > 1
        and len(set(total_rewards)) > 1
    ):
        try:
            z = np.polyfit(foods_collected, total_rewards, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(foods_collected), max(foods_collected), 100)
            plt.plot(
                x_line,
                p(x_line),
                "r--",
                linewidth=2,
                label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
            )

            # Calculate correlation
            correlation = np.corrcoef(foods_collected, total_rewards)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
        except (np.linalg.LinAlgError, ValueError):
            logger.warning(
                "Insufficient variation for trend line calculation; skipping trend line.",
            )

    plt.title("Foods Collected vs Total Reward")
    plt.xlabel("Foods Collected")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}foods_vs_reward_correlation.png")
    plt.close()


def plot_foods_vs_steps_correlation(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    foods_collected: list[int],
    steps: list[int],
) -> None:
    """Plot correlation between foods collected and steps taken.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    foods_collected : list[int]
        Number of foods collected in each run.
    steps : list[int]
        Number of steps taken in each run.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(foods_collected, steps, alpha=0.6, s=100, color="purple", edgecolors="black")

    # Add trend line only if there's variation in the data
    if (
        len(foods_collected) > 1
        and len(steps) == len(foods_collected)
        and len(set(foods_collected)) > 1
        and len(set(steps)) > 1
    ):
        try:
            z = np.polyfit(foods_collected, steps, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(foods_collected), max(foods_collected), 100)
            plt.plot(
                x_line,
                p(x_line),
                "r--",
                linewidth=2,
                label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
            )

            # Calculate correlation
            correlation = np.corrcoef(foods_collected, steps)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
        except (np.linalg.LinAlgError, ValueError):
            logger.warning(
                "Insufficient variation for trend line calculation; skipping trend line.",
            )

    plt.title("Foods Collected vs Steps Taken")
    plt.xlabel("Foods Collected")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}foods_vs_steps_correlation.png")
    plt.close()


def plot_satiety_progression_single_run(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    run_number: int,
    satiety_history: list[float],
    max_satiety: float,
) -> None:
    """Plot satiety progression throughout a single run.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    run_number : int
        The run number being plotted.
    satiety_history : list[float]
        Satiety levels at each step.
    max_satiety : float
        Maximum satiety level.
    """
    if not satiety_history:
        logger.warning(f"No satiety history for run {run_number}")
        return

    steps = list(range(len(satiety_history)))

    plt.figure(figsize=(14, 6))
    plt.plot(steps, satiety_history, linewidth=2, label="Satiety Level")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Starvation Threshold",
    )
    plt.axhline(
        y=max_satiety * 0.2,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Low Satiety (20%)",
    )
    plt.axhline(
        y=max_satiety,
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label=f"Max Satiety ({max_satiety:.0f})",
    )
    plt.fill_between(steps, 0, satiety_history, alpha=0.2)
    plt.title(f"Satiety Progression - Run {run_number}")
    plt.xlabel("Step")
    plt.ylabel("Satiety Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}satiety_progression_run_{run_number}.png")
    plt.close()


# Health System Specific Plots


def plot_health_at_episode_end(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    health_remaining: list[float],
    max_health: float,
) -> None:
    """Plot health levels at episode end per run.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    health_remaining : list[float]
        Final health level at end of each run.
    max_health : float
        Maximum health level.
    """
    if not runs or not health_remaining:
        logger.warning("No health data to plot")
        return

    plt.figure(figsize=(12, 6))
    colors = [
        "red" if h <= 0 else "orange" if h < max_health * 0.3 else "green" for h in health_remaining
    ]
    plt.bar(runs, health_remaining, alpha=0.7, color=colors, label="Health Remaining")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Death Threshold",
    )
    plt.axhline(
        y=max_health * 0.3,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Low Health (30%)",
    )
    avg_health = sum(health_remaining) / len(health_remaining)
    plt.axhline(
        y=avg_health,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_health:.1f})",
    )
    plt.title("Health at Episode End Per Run")
    plt.xlabel("Run")
    plt.ylabel("Health (HP)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}health_at_episode_end.png")
    plt.close()


def plot_health_progression_single_run(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    run_number: int,
    health_history: list[float],
    max_health: float,
) -> None:
    """Plot health progression throughout a single run.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    run_number : int
        The run number being plotted.
    health_history : list[float]
        Health levels at each step.
    max_health : float
        Maximum health level.
    """
    if not health_history:
        logger.warning(f"No health history for run {run_number}")
        return

    steps = list(range(len(health_history)))

    plt.figure(figsize=(14, 6))
    plt.plot(steps, health_history, linewidth=2, color="purple", label="Health (HP)")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Death Threshold",
    )
    plt.axhline(
        y=max_health * 0.3,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Low Health (30%)",
    )
    plt.axhline(
        y=max_health,
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label=f"Max Health ({max_health:.0f})",
    )
    plt.fill_between(steps, 0, health_history, alpha=0.2, color="purple")
    plt.title(f"Health Progression - Run {run_number}")
    plt.xlabel("Step")
    plt.ylabel("Health (HP)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}health_progression_run_{run_number}.png")
    plt.close()


# Predator Evasion Environment Specific Plots


def plot_predator_encounters_over_time(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    predator_encounters: list[int],
) -> None:
    """Plot predator encounters over time.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    predator_encounters : list[int]
        Number of predator encounters in each run.
    """
    # Guard against empty inputs
    if not runs or not predator_encounters:
        logger.warning("No predator encounter data to plot")
        return

    # Validate input lengths match
    if len(runs) != len(predator_encounters):
        logger.warning(
            f"Length mismatch: runs ({len(runs)}) vs predator_encounters "
            f"({len(predator_encounters)}). Skipping plot.",
        )
        return

    plt.figure(figsize=(12, 6))
    plt.bar(
        runs,
        predator_encounters,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Encounters",
    )
    avg_encounters = sum(predator_encounters) / len(predator_encounters)
    plt.axhline(
        y=avg_encounters,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_encounters:.1f})",
    )
    plt.title("Predator Encounters Per Run")
    plt.xlabel("Run")
    plt.ylabel("Number of Encounters")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}predator_encounters_over_time.png")
    plt.close()


def plot_evasion_success_rate_over_time(  # pragma: no cover
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    predator_encounters: list[int],
    successful_evasions: list[int],
) -> None:
    """Plot evasion success rate over time.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    runs : list[int]
        List of run indices.
    plot_dir : Path
        Directory to save the plot.
    predator_encounters : list[int]
        Number of predator encounters in each run.
    successful_evasions : list[int]
        Number of successful evasions in each run.
    """
    # Guard against empty inputs
    if not runs or not predator_encounters or not successful_evasions:
        logger.warning("No evasion data to plot")
        return

    # Validate input lengths match
    if len(runs) != len(predator_encounters) or len(runs) != len(successful_evasions):
        logger.warning(
            f"Length mismatch: runs ({len(runs)}) vs predator_encounters "
            f"({len(predator_encounters)}) vs successful_evasions "
            f"({len(successful_evasions)}). Skipping plot.",
        )
        return

    # Calculate evasion success rate per run
    evasion_rates = [
        evasions / encounters if encounters > 0 else 0.0
        for encounters, evasions in zip(predator_encounters, successful_evasions, strict=True)
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(
        runs,
        evasion_rates,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        color="green",
        label="Evasion Success Rate",
    )

    # Calculate overall average (only for runs with encounters)
    valid_rates = [
        rate
        for rate, encounters in zip(evasion_rates, predator_encounters, strict=True)
        if encounters > 0
    ]
    if valid_rates:
        avg_rate = sum(valid_rates) / len(valid_rates)
        plt.axhline(
            y=avg_rate,
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Average ({avg_rate:.2%})",
        )

    plt.ylim(-0.05, 1.05)
    plt.title("Evasion Success Rate Over Runs")
    plt.xlabel("Run")
    plt.ylabel("Success Rate (Evasions / Encounters)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}evasion_success_rate_over_time.png")
    plt.close()


def plot_survival_vs_food_collection(  # pragma: no cover
    file_prefix: str,
    plot_dir: Path,
    foods_collected: list[int],
    predator_deaths: list[bool],
) -> None:
    """Plot survival rate vs food collection scatter plot.

    Parameters
    ----------
    file_prefix : str
        Prefix for the output file name.
    plot_dir : Path
        Directory to save the plot.
    foods_collected : list[int]
        Number of foods collected in each run.
    predator_deaths : list[bool]
        Whether each run ended in predator death (True) or survival (False).
    """
    # Guard against empty inputs
    if not foods_collected or not predator_deaths:
        logger.warning("No survival/food collection data to plot")
        return

    # Validate input lengths match
    if len(foods_collected) != len(predator_deaths):
        logger.warning(
            f"Length mismatch: foods_collected ({len(foods_collected)}) vs "
            f"predator_deaths ({len(predator_deaths)}). Skipping plot.",
        )
        return

    # Separate data by survival status
    survived_foods = [
        foods for foods, died in zip(foods_collected, predator_deaths, strict=True) if not died
    ]
    died_foods = [
        foods for foods, died in zip(foods_collected, predator_deaths, strict=True) if died
    ]

    plt.figure(figsize=(10, 8))

    # Plot survived runs
    if survived_foods:
        plt.scatter(
            survived_foods,
            [1] * len(survived_foods),
            alpha=0.6,
            s=100,
            color="green",
            edgecolors="black",
            label=f"Survived ({len(survived_foods)} runs)",
        )

    # Plot died runs
    if died_foods:
        plt.scatter(
            died_foods,
            [0] * len(died_foods),
            alpha=0.6,
            s=100,
            color="red",
            edgecolors="black",
            label=f"Killed by Predator ({len(died_foods)} runs)",
        )

    # Add statistics
    total_runs = len(predator_deaths)
    survival_rate = len(survived_foods) / total_runs if total_runs > 0 else 0.0
    avg_foods_survived = sum(survived_foods) / len(survived_foods) if survived_foods else 0.0
    avg_foods_died = sum(died_foods) / len(died_foods) if died_foods else 0.0
    avg_foods_died_str = f"{avg_foods_died:.1f}" if died_foods else "N/A"

    stats_text = (
        f"Survival Rate: {survival_rate:.1%}\n"
        f"Avg Foods (Survived): {avg_foods_survived:.1f}\n"
        f"Avg Foods (Died): {avg_foods_died_str}"
    )
    plt.text(
        0.05,
        0.5,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.yticks([0, 1], ["Killed", "Survived"])
    plt.title("Survival vs Food Collection")
    plt.xlabel("Foods Collected")
    plt.ylabel("Outcome")
    plt.legend()
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{file_prefix}survival_vs_food_collection.png")
    plt.close()
