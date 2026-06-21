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
    """1-indexed run number where convergence was detected (None if never converged)."""

    runs_to_convergence: int | None
    """
    Number of runs before converged region starts
    (equal to convergence_run - 1, None if never converged).
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

    learning_speed: float
    """
    Normalized learning speed (0.0 to 1.0, higher = faster learning).
    Calculated as 1.0 - (episodes_to_80_success / max_episodes).
    A value of 0.0 means 80% success was never reached.
    """

    learning_speed_episodes: int | None
    """
    Number of episodes required to reach 80% rolling success rate.
    None if 80% success was never achieved.
    """

    stability: float
    """
    Normalized stability metric (0.0 to 1.0, higher = more consistent).
    Calculated as 1.0 - coefficient_of_variation, clamped to [0, 1].
    Measures consistency of success rates across runs.
    """

    # Multi-objective metrics (added for thermotaxis/survival tasks)
    avg_survival_score: float | None = None
    """Average survival score (final_hp / max_hp) across all runs. None if health disabled."""

    avg_temperature_comfort_score: float | None = None
    """Average temperature comfort score across all runs. None if thermotaxis disabled."""

    avg_oxygen_comfort_score: float | None = None
    """
    Average oxygen comfort score from post-convergence analysis runs.
    None if aerotaxis disabled.
    """


def detect_convergence(
    results: list[SimulationResult],
    *,
    stability_runs: int = 10,
    min_total_runs: int = 30,
    band: float = 0.10,
    tail_frac: float = 0.25,
) -> int | None:
    """
    Detect when a learning strategy reaches its performance plateau.

    Convergence here means *the policy has stopped improving* — it is decoupled
    from the absolute success level, so a stable 40% plateau and a stable 95%
    plateau are both detected. This is deliberately NOT a "reaches near-100%
    streaks" test: on binary success data a low-variance high-mean window is
    reachable only near 100%, so a variance-based gate mis-classifies stable
    intermediate plateaus (e.g. a policy that completes the task ~45% of episodes
    from a fully-converged policy) as "never converged". See the
    ``architecture-comparison-protocol`` capability spec.

    Two steps:

    1. **No-trend gate.** Compare the final trailing block of ``tail_frac`` of the
       runs (the converged level ``L``) against the immediately-preceding block of
       equal size. They are large blocks, so per-episode binary noise is averaged
       out; convergence is declared only when their mean success rates agree within
       ``band``. A run still climbing/declining at the end fails this and returns
       ``None`` (flagged for an extend-and-rerun, not mis-ranked).
    2. **Onset.** The earliest ``stability_runs``-window whose mean success reaches
       within ``band`` of ``L``. During warm-up the rate sits away from ``L`` (a
       flat-at-zero prefix is below ``L`` by more than ``band``, so it is excluded);
       the metric averages raw success from this onset to the end.

    ``band`` is calibrated to block sampling noise: block means over
    ``t = tail_frac * N`` runs have std ~ sqrt(p(1-p)/t); at N~1200, t~300 that is
    ~0.03, so ``band = 0.10`` (~2.5 sigma between two blocks) passes a truly-flat
    plateau ~99% of the time while flagging a residual trend > 0.10 per block.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    stability_runs : int, optional
        Width of the onset window (default: 10).
    min_total_runs : int, optional
        Minimum total runs before detection is attempted (default: 30).
    band : float, optional
        Absolute success-rate tolerance for the no-trend gate and onset, in [0, 1]
        (default: 0.10). Calibrated to two-block sampling noise (see above).
    tail_frac : float, optional
        Fraction of runs in each comparison block for the no-trend gate
        (default: 0.25).

    Returns
    -------
    int | None
        1-indexed run number where the plateau begins, or None if the run has
        not converged (still trending at its budget).

    Notes
    -----
    The onset is the start of the final at-plateau region, NOT the first
    fully-homogeneous (~100%) window the legacy detector required; for high-band
    runs this can differ by <=1 run, but the averaged plateau metric is unchanged
    within sampling noise.

    A flat run (e.g. one that never succeeds, or sits at a low level) satisfies
    the no-trend gate and is reported as converged at that level. This is correct
    for an architecture that genuinely plateaus low (a valid ranking outcome), but
    it cannot be distinguished from a slow-igniter cut off mid-warm-up purely from
    the run — both look flat. That ambiguity is a budget-sufficiency concern, not a
    detection one: callers MUST give every arm enough episodes to reach its plateau
    (the ``architecture-comparison-protocol`` budget requirement), and the
    full-window-mean cross-check flags a run whose tail still disagrees.
    """
    n = len(results)
    if n < min_total_runs:
        return None

    # Binary success indicators (1.0 success / 0.0 failure).
    successes = np.array([1.0 if r.success else 0.0 for r in results], dtype=float)

    # No-trend gate: the final block vs the equal-size preceding block. Large
    # blocks average out per-episode noise, so this tests for a residual trend
    # (still learning), NOT for low per-episode variance.
    tail = max(int(tail_frac * n), stability_runs)
    final_level = float(successes[-tail:].mean())
    prev_block = successes[-2 * tail : -tail] if n >= 2 * tail else successes[:-tail]
    prev_level = float(prev_block.mean()) if prev_block.size else final_level
    if abs(final_level - prev_level) > band:
        # Still climbing/declining at the budget -> not yet converged.
        return None

    # Onset: earliest run after which the smoothed success rate STAYS within `band`
    # of the converged level for the rest of the run. Requiring it to stay (rather
    # than the first window to merely touch the band) avoids a premature onset
    # during a warm-up climb, which would drag the averaged metric below the true
    # plateau. The smoothing window scales with run length so long runs are robust
    # to per-episode noise while short runs still resolve an early onset; a
    # flat-at-zero warm-up is excluded because its smoothed rate is below
    # `final_level` by more than `band`.
    # The band is one-sided (below `final_level`): RL arms warm up from below the
    # plateau, so the warm-up is excluded by `rolling < final_level - band`, while a
    # noisy excursion ABOVE the plateau does not reset the onset (it is still
    # converged, just noisy-high). The converged-from-above case does not arise for
    # learning-from-scratch and is guarded by the no-trend gate above.
    window = max(stability_runs, n // 20)
    csum = np.cumsum(np.insert(successes, 0, 0.0))
    rolling = (csum[window:] - csum[:-window]) / window  # rolling[k] = mean(successes[k:k+window])
    below_band = np.flatnonzero(rolling < final_level - band)
    if below_band.size == 0:
        onset = 0
    elif int(below_band[-1]) == rolling.size - 1:
        # Smoothed rate is still below the plateau at the very end -> not converged.
        return None
    else:
        onset = int(below_band[-1]) + 1  # start of the final region at/above the plateau

    return onset + 1


def calculate_learning_speed_episodes(
    results: list[SimulationResult],
    target_success_rate: float = 0.80,
    window_size: int = 10,
) -> int | None:
    """
    Calculate the number of episodes to reach a target rolling success rate.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    target_success_rate : float, optional
        Target success rate to reach (default: 0.80 = 80%).
    window_size : int, optional
        Size of rolling window for success rate calculation (default: 10).

    Returns
    -------
    int | None
        Number of episodes to reach target success rate, or None if never reached.

    Examples
    --------
    >>> results = [...]  # List of SimulationResult objects
    >>> episodes = calculate_learning_speed_episodes(results)
    >>> if episodes:
    ...     print(f"Reached 80% success after {episodes} episodes")
    """
    if len(results) < window_size:
        return None

    # Extract binary success indicators
    successes = [1.0 if r.success else 0.0 for r in results]

    # Check rolling success rate at each window position
    for i in range(window_size, len(successes) + 1):
        window = successes[i - window_size : i]
        rolling_rate = float(np.mean(window))

        if rolling_rate >= target_success_rate:
            # Reached target - return the episode index (end of window)
            return i

    # Never reached target success rate
    return None


def calculate_learning_speed(
    learning_speed_episodes: int | None,
    max_episodes: int,
) -> float:
    """
    Calculate normalized learning speed from episodes to target.

    Parameters
    ----------
    learning_speed_episodes : int | None
        Number of episodes to reach 80% success, or None if never reached.
    max_episodes : int
        Total number of episodes in the session.

    Returns
    -------
    float
        Normalized learning speed in range [0, 1] where 1.0 = instant learning.
        Returns 0.0 if target was never reached.

    Examples
    --------
    >>> speed = calculate_learning_speed(20, 100)
    >>> print(f"Learning speed: {speed:.2f}")  # 0.80
    """
    if learning_speed_episodes is None or max_episodes <= 0:
        return 0.0

    # Normalize: fewer episodes = higher speed
    # If it takes 0 episodes, speed = 1.0
    # If it takes all episodes, speed = 0.0
    return max(0.0, 1.0 - (learning_speed_episodes / max_episodes))


def calculate_stability(
    results: list[SimulationResult],
    convergence_run: int | None = None,
    fallback_window: int = 10,
) -> float:
    """
    Calculate stability metric from coefficient of variation.

    Stability measures how consistent the success rate is across runs.
    Uses coefficient of variation (CV = std/mean) inverted and clamped.

    Parameters
    ----------
    results : list[SimulationResult]
        Ordered list of simulation results from a session.
    convergence_run : int | None, optional
        If provided (1-indexed), calculate stability only for post-convergence runs.
    fallback_window : int, optional
        Number of final runs to use if never converged (default: 10).

    Returns
    -------
    float
        Stability metric in range [0, 1] where 1.0 = perfectly consistent.

    Examples
    --------
    >>> results = [...]  # List of SimulationResult objects
    >>> stability = calculate_stability(results)
    >>> print(f"Stability: {stability:.2f}")
    """
    # Determine which runs to analyze
    if convergence_run is not None:
        # Convert 1-indexed convergence_run to 0-indexed for slicing
        start_idx = convergence_run - 1
        analysis_runs = results[start_idx:]
    else:
        # Fallback: use last N runs
        analysis_runs = results[-fallback_window:] if len(results) >= fallback_window else results

    if len(analysis_runs) < 2:  # noqa: PLR2004
        # Not enough runs to calculate variance - assume stable
        return 1.0

    # Extract binary success indicators
    successes = [1.0 if r.success else 0.0 for r in analysis_runs]
    mean_success = float(np.mean(successes))
    std_success = float(np.std(successes))

    # Handle edge case: zero mean (all failures)
    if mean_success == 0.0:
        # All failures = unstable (CV would be undefined)
        return 0.0

    # Coefficient of variation (CV = std / mean)
    cv = std_success / mean_success

    # Stability = 1 - CV, clamped to [0, 1]
    # Low CV = high stability
    return max(0.0, min(1.0, 1.0 - cv))


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
        1-indexed run number where convergence was detected (None triggers fallback).
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
        # Convert 1-indexed convergence_run to 0-indexed for slicing
        start_idx = convergence_run - 1
        analysis_runs = results[start_idx:]
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

    # Multi-objective metrics (survival and temperature comfort)
    avg_survival_score = None
    avg_temperature_comfort_score = None

    # Calculate average survival score if available
    survival_scores = [r.survival_score for r in analysis_runs if r.survival_score is not None]
    if survival_scores:
        avg_survival_score = float(np.mean(survival_scores))

    # Calculate average temperature comfort score if available
    comfort_scores = [
        r.temperature_comfort_score
        for r in analysis_runs
        if r.temperature_comfort_score is not None
    ]
    if comfort_scores:
        avg_temperature_comfort_score = float(np.mean(comfort_scores))

    # Calculate average oxygen comfort score if available
    avg_oxygen_comfort_score = None
    o2_comfort_scores = [
        r.oxygen_comfort_score for r in analysis_runs if r.oxygen_comfort_score is not None
    ]
    if o2_comfort_scores:
        avg_oxygen_comfort_score = float(np.mean(o2_comfort_scores))

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_foods": avg_foods,
        "variance": variance,
        "distance_efficiency": distance_efficiency,
        "avg_survival_score": avg_survival_score,
        "avg_temperature_comfort_score": avg_temperature_comfort_score,
        "avg_oxygen_comfort_score": avg_oxygen_comfort_score,
    }


def calculate_composite_score(  # noqa: PLR0913
    success_rate: float,
    distance_efficiency: float | None,
    runs_to_convergence: int | None,
    variance: float,
    total_runs: int = 50,
    max_variance: float = 0.2,
    survival_score: float | None = None,
    temperature_comfort_score: float | None = None,
) -> float:
    """
    Calculate weighted composite benchmark score with multi-objective support.

    The composite score uses hierarchical weighting where survival acts as a gate:
    - If agent died (survival_score provided but low): score is capped
    - Secondary objectives (temperature comfort) only count if agent survives

    For single-objective tasks (no survival/thermotaxis), uses the original formula:
    - Success rate (40%): How often the strategy succeeds
    - Distance efficiency (30%): How efficiently it navigates
    - Learning speed (20%): How quickly it converges
    - Stability (10%): How consistent it is after convergence

    For multi-objective tasks, the formula adapts:
    - If died (survival_score < 0.1): composite capped at 0.3 * primary_completion
    - If survived: weights redistribute to include secondary objectives

    Parameters
    ----------
    success_rate : float
        Post-convergence success rate (0.0 to 1.0).
    distance_efficiency : float | None
        Average distance efficiency in successful runs (None for no food collected).
        Range: 0.0 to 1.0, where 1.0 means perfect optimal navigation.
    runs_to_convergence : int | None
        Number of runs to reach convergence (None if never converged).
    variance : float
        Post-convergence variance in success rate.
    total_runs : int, optional
        Total number of runs in the session (default: 50).
    max_variance : float, optional
        Maximum variance for normalization (default: 0.2).
    survival_score : float | None, optional
        Average survival score (final_hp / max_hp). None if health system disabled.
    temperature_comfort_score : float | None, optional
        Average temperature comfort score. None if thermotaxis disabled.

    Returns
    -------
    float
        Composite score from 0.0 to 1.0, where higher is better.

    Notes
    -----
    Multi-objective scoring hierarchy:
    1. Survival is a gate - dying significantly caps the score
    2. Primary objective (food collection) is the main metric
    3. Secondary objectives (temperature comfort, efficiency) matter only if survived

    This mirrors biological fitness where a dead organism gets no credit for
    almost completing a task.
    """
    # Determine if this is a multi-objective task
    is_multi_objective = survival_score is not None or temperature_comfort_score is not None

    # Component 1: Success rate (already 0-1)
    norm_success = success_rate

    # Component 2: Distance efficiency (already normalized 0-1)
    norm_efficiency = distance_efficiency if distance_efficiency is not None else 0.0

    # Component 3: Learning speed (faster convergence = better)
    if runs_to_convergence is not None:
        # Normalize: fewer runs = higher score
        norm_speed = max(0.0, 1.0 - (runs_to_convergence / total_runs))
    else:
        # Never converged: assign zero learning speed score
        norm_speed = 0.0

    # Component 4: Stability (lower variance = better)
    norm_stability = max(0.0, 1.0 - (variance / max_variance))

    if not is_multi_objective:
        # Original single-objective formula
        return (
            0.40 * norm_success + 0.30 * norm_efficiency + 0.20 * norm_speed + 0.10 * norm_stability
        )

    # Multi-objective scoring with hierarchical weighting
    # Survival acts as a gate
    effective_survival = survival_score if survival_score is not None else 1.0
    effective_comfort = temperature_comfort_score if temperature_comfort_score is not None else 0.0

    # Check if agent "died" (survival score below threshold)
    survival_threshold = 0.1
    if effective_survival < survival_threshold:
        # Died: cap score at 30% of primary completion
        return 0.30 * norm_success

    # Survived: use full multi-objective formula
    # Weights:
    # - Primary objective (success rate): 50%
    # - Survival quality: 15% (how healthy agent remained)
    # - Temperature comfort: 10% (if thermotaxis enabled)
    # - Distance efficiency: 15%
    # - Learning speed + stability: 10%

    if temperature_comfort_score is not None:
        # Full multi-objective with thermotaxis
        return (
            0.50 * norm_success
            + 0.15 * effective_survival
            + 0.10 * effective_comfort
            + 0.15 * norm_efficiency
            + 0.05 * norm_speed
            + 0.05 * norm_stability
        )

    # Multi-objective without thermotaxis (just health)
    return (
        0.50 * norm_success
        + 0.20 * effective_survival
        + 0.20 * norm_efficiency
        + 0.05 * norm_speed
        + 0.05 * norm_stability
    )


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
    # runs_to_convergence = number of runs before convergence (convergence_run - 1)
    # e.g., if convergence_run=5 (1-indexed), there were 4 runs before convergence
    runs_to_convergence = (convergence_run - 1) if converged else None

    # Step 2: Calculate post-convergence metrics
    post_metrics = calculate_post_convergence_metrics(results, convergence_run)

    # Step 3: Calculate learning speed metrics
    learning_speed_episodes = calculate_learning_speed_episodes(results)
    learning_speed = calculate_learning_speed(learning_speed_episodes, total_runs)

    # Step 4: Calculate stability metric
    stability = calculate_stability(results, convergence_run)

    # Step 5: Calculate composite score
    # Use defaults for None values (but keep distance_efficiency as None for uncollected food)
    composite_score = calculate_composite_score(
        success_rate=post_metrics["success_rate"]
        if post_metrics["success_rate"] is not None
        else 0.0,
        distance_efficiency=post_metrics["distance_efficiency"],
        runs_to_convergence=runs_to_convergence,
        variance=post_metrics["variance"] if post_metrics["variance"] is not None else 1.0,
        total_runs=total_runs,
        survival_score=post_metrics["avg_survival_score"],
        temperature_comfort_score=post_metrics["avg_temperature_comfort_score"],
    )

    # Step 6: Build convergence metrics object
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
        learning_speed=learning_speed,
        learning_speed_episodes=learning_speed_episodes,
        stability=stability,
        avg_survival_score=post_metrics["avg_survival_score"],
        avg_temperature_comfort_score=post_metrics["avg_temperature_comfort_score"],
        avg_oxygen_comfort_score=post_metrics["avg_oxygen_comfort_score"],
    )
