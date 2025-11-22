"""Convergence detection and composite scoring for benchmark evaluation.

This module implements adaptive convergence detection to identify when a learning
strategy has stabilized, enabling more accurate performance assessment based on
converged behavior rather than all-run averages.
"""

import numpy as np
from pydantic import BaseModel

from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import SimulationResult


class ConvergenceMetrics(BaseModel):
    """Metrics describing learning convergence and post-convergence performance."""

    converged: bool
    """Whether the learning strategy converged within the session."""

    convergence_run: int | None
    """Zero-based index into results where convergence was detected (None if never converged)."""

    runs_to_convergence: int | None
    """
    Number of runs before converged region starts
    (equal to convergence_run, None if never converged).
    """

    post_convergence_success_rate: float | None
    """Success rate after convergence point (or last 10 runs if not converged)."""

    post_convergence_avg_steps: float | None
    """Average steps in successful runs after convergence."""

    post_convergence_avg_foods: float | None
    """Average foods collected after convergence (dynamic environments only)."""

    post_convergence_variance: float | None
    """Variance in success rate after convergence (measures stability)."""

    distance_efficiency: float | None
    """Average distance efficiency in successful runs (dynamic environments only)."""

    composite_score: float
    """Weighted composite benchmark score (0.0 to 1.0)."""


def detect_convergence(
    results: list[SimulationResult],
    variance_threshold: float = 0.05,
    stability_runs: int = 10,
    min_runs: int = 20,
) -> int | None:
    """
    Detect when learning strategy converges using adaptive algorithm.

    Convergence is detected when the success rate variance falls below a threshold
    for a sustained number of runs, indicating the strategy has stabilized.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    variance_threshold : float, optional
        Maximum variance to consider converged (default: 0.05 = 5%).
    stability_runs : int, optional
        Number of consecutive low-variance runs required (default: 10).
    min_runs : int, optional
        Minimum runs before convergence can be declared (default: 20).

    Returns
    -------
    int | None
        Zero-based index into results where convergence was detected, or None if never converged.

    Notes
    -----
    The algorithm checks if success rate variance in a sliding window of
    `stability_runs` remains below the threshold. This prevents declaring
    convergence during temporary plateaus and ensures genuine stabilization.
    """
    if len(results) < min_runs + stability_runs:
        return None

    # Extract binary success indicators (1.0 for success, 0.0 for failure)
    successes = [1.0 if r.success else 0.0 for r in results]

    # Search for convergence point starting after minimum runs
    for i in range(min_runs, len(successes) - stability_runs + 1):
        # Check variance in stability window
        window = successes[i : i + stability_runs]
        variance = float(np.var(window))

        if variance < variance_threshold:
            # Found stable region - convergence detected
            return i

    # Never converged within the session
    return None


def calculate_post_convergence_metrics(
    results: list[SimulationResult],
    convergence_run: int | None,
    fallback_window: int = 10,
) -> dict[str, float | None]:
    """
    Calculate performance metrics after convergence point.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    convergence_run : int | None
        Run number where convergence was detected (None triggers fallback).
    fallback_window : int, optional
        Number of final runs to use if never converged (default: 10).

    Returns
    -------
    dict[str, float | None]
        Dictionary containing:
        - success_rate: Post-convergence success rate
        - avg_steps: Average steps in successful runs
        - avg_foods: Average foods collected (dynamic env only)
        - variance: Variance in success indicators
        - distance_efficiency: Average distance efficiency in successful runs (dynamic only)
    """
    # Determine which runs to analyze
    if convergence_run is not None:
        analysis_runs = results[convergence_run:]
    else:
        # Fallback: use last N runs
        analysis_runs = results[-fallback_window:] if len(results) >= fallback_window else results

    if not analysis_runs:
        # Edge case: no runs to analyze
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_foods": None,
            "variance": 0.0,
            "distance_efficiency": None,
        }

    # Calculate success rate
    successes = [1.0 if r.success else 0.0 for r in analysis_runs]
    success_rate = float(np.mean(successes))
    variance = float(np.var(successes))

    # Calculate average steps from SUCCESSFUL runs only
    successful_runs = [r for r in analysis_runs if r.success]
    if successful_runs:
        avg_steps = float(np.mean([r.steps for r in successful_runs]))
    else:
        # No successful runs - use average of all runs
        avg_steps = float(np.mean([r.steps for r in analysis_runs]))

    # Dynamic environment metrics (foods collected and distance efficiency)
    avg_foods = None
    distance_efficiency = None

    # Calculate average foods collected if available
    if any(r.foods_collected is not None for r in analysis_runs):
        foods_values = [r.foods_collected for r in analysis_runs if r.foods_collected is not None]
        if foods_values:
            avg_foods = float(np.mean(foods_values))

    # Calculate distance efficiency if available (has average_distance_efficiency field)
    if any(r.average_distance_efficiency is not None for r in analysis_runs):
        # Calculate average distance efficiency from SUCCESSFUL runs only
        efficiencies = [
            r.average_distance_efficiency
            for r in successful_runs
            if r.average_distance_efficiency is not None
        ]
        distance_efficiency = float(np.mean(efficiencies)) if efficiencies else None

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_foods": avg_foods,
        "variance": variance,
        "distance_efficiency": distance_efficiency,
    }


def calculate_composite_score(  # noqa: PLR0913
    success_rate: float,
    distance_efficiency: float | None,
    runs_to_convergence: int | None,
    variance: float,
    total_runs: int = 50,
    max_variance: float = 0.2,
) -> float:
    """
    Calculate weighted composite benchmark score.

    The composite score combines multiple aspects of learning performance:
    - Success rate (40%): How often the strategy succeeds
    - Distance efficiency (30%): How efficiently it navigates (optimal path / actual path)
    - Learning speed (20%): How quickly it converges
    - Stability (10%): How consistent it is after convergence

    Parameters
    ----------
    success_rate : float
        Post-convergence success rate (0.0 to 1.0).
    distance_efficiency : float | None
        Average distance efficiency in successful runs (None for static environments).
        Range: 0.0 to 1.0, where 1.0 means perfect optimal navigation.
    runs_to_convergence : int | None
        Number of runs to reach convergence (None if never converged).
    variance : float
        Post-convergence variance in success rate.
    total_runs : int, optional
        Total number of runs in the session (default: 50).
    max_variance : float, optional
        Maximum variance for normalization (default: 0.2).

    Returns
    -------
    float
        Composite score from 0.0 to 1.0, where higher is better.

    Notes
    -----
    All components are normalized to [0, 1] before weighting:
    - Success rate: Already normalized
    - Distance efficiency: Already normalized (0.0 to 1.0)
    - Speed: Inverted (faster = better)
    - Stability: Inverted variance (lower variance = better)
    """
    # Component 1: Success rate (already 0-1)
    norm_success = success_rate

    # Component 2: Distance efficiency (already normalized 0-1)
    # Static environment (None): use success rate as efficiency proxy
    norm_efficiency = distance_efficiency if distance_efficiency is not None else success_rate

    # Component 3: Learning speed (faster convergence = better)
    if runs_to_convergence is not None:
        # Normalize: fewer runs = higher score
        norm_speed = max(0.0, 1.0 - (runs_to_convergence / total_runs))
    else:
        # Never converged: assign zero learning speed score
        norm_speed = 0.0

    # Component 4: Stability (lower variance = better)
    norm_stability = max(0.0, 1.0 - (variance / max_variance))

    # Weighted composite score
    return 0.40 * norm_success + 0.30 * norm_efficiency + 0.20 * norm_speed + 0.10 * norm_stability


def analyze_convergence(
    results: list[SimulationResult],
    total_runs: int,
) -> ConvergenceMetrics:
    """
    Perform complete convergence analysis on session results.

    This is the main entry point for convergence-based benchmark evaluation.
    It detects convergence, calculates post-convergence metrics, and computes
    the composite benchmark score.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    total_runs : int
        Total number of runs in the session.

    Returns
    -------
    ConvergenceMetrics
        Complete convergence analysis including composite score.

    Examples
    --------
    >>> results = [...]  # List of SimulationResult objects
    >>> metrics = analyze_convergence(results, total_runs=50)
    >>> print(f"Converged: {metrics.converged}")
    >>> print(f"Composite Score: {metrics.composite_score:.3f}")
    """
    if total_runs == 0:
        error_message = "Total runs must be greater than zero for convergence analysis."
        logger.error(error_message)
        raise ValueError(error_message)

    if total_runs != len(results):
        error_message = (
            f"Total runs ({total_runs}) does not match number of results ({len(results)})."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Step 1: Detect convergence
    convergence_run = detect_convergence(results)

    converged = convergence_run is not None
    runs_to_convergence = convergence_run if converged else None

    # Step 2: Calculate post-convergence metrics
    post_metrics = calculate_post_convergence_metrics(results, convergence_run)

    # Step 3: Calculate composite score
    # Use defaults for None values (but keep distance_efficiency as None for static environments)
    composite_score = calculate_composite_score(
        success_rate=post_metrics["success_rate"]
        if post_metrics["success_rate"] is not None
        else 0.0,
        distance_efficiency=post_metrics[
            "distance_efficiency"
        ],  # Keep None for static environments
        runs_to_convergence=runs_to_convergence,
        variance=post_metrics["variance"] if post_metrics["variance"] is not None else 1.0,
        total_runs=total_runs,
    )

    # Step 4: Build convergence metrics object
    return ConvergenceMetrics(
        converged=converged,
        convergence_run=convergence_run,
        runs_to_convergence=runs_to_convergence,
        post_convergence_success_rate=post_metrics["success_rate"],
        post_convergence_avg_steps=post_metrics["avg_steps"],
        post_convergence_avg_foods=post_metrics["avg_foods"],
        post_convergence_variance=post_metrics["variance"],
        distance_efficiency=post_metrics["distance_efficiency"],
        composite_score=composite_score,
    )
