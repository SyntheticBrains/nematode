"""Metrics computation for plasticity evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantumnematode.plasticity.dtypes import EvalResult, SeedResult


def _get_eval_score(
    eval_results: list[EvalResult],
    objective: str,
    transition: str,
) -> float | None:
    """Find a specific eval result by objective and transition point."""
    for r in eval_results:
        if r.objective_name == objective and r.transition_point == transition:
            return r.mean_success_rate
    return None


def compute_convergence_episode(
    successes: list[bool],
    threshold: float,
    window: int = 20,
) -> int | None:
    """Find first episode where trailing-window mean success rate exceeds threshold.

    Returns None if convergence is not reached.
    """
    for i in range(window - 1, len(successes)):
        trailing = successes[i - window + 1 : i + 1]
        if np.mean(trailing) >= threshold:
            return i + 1  # 1-indexed episode number
    return None


def compute_seed_metrics(
    seed_result: SeedResult,
    convergence_threshold: float,
) -> None:
    """Compute BF, FT, PR for a single seed and store on the SeedResult."""
    evals = seed_result.eval_results

    # Backward Forgetting: post_A_score - post_C_score_on_A
    post_a = _get_eval_score(evals, "foraging", "post_A")
    post_c_on_a = _get_eval_score(evals, "foraging", "post_C")
    if post_a is not None and post_c_on_a is not None:
        seed_result.backward_forgetting = post_a - post_c_on_a

    # Forward Transfer: post_A_eval_on_B - random_baseline_on_B
    post_a_on_b = _get_eval_score(evals, "pursuit_predators", "post_A")
    baseline_on_b = _get_eval_score(evals, "pursuit_predators", "pre_training")
    if post_a_on_b is not None and baseline_on_b is not None:
        seed_result.forward_transfer = post_a_on_b - baseline_on_b

    # Plasticity Retention: convergence_episodes_A / convergence_episodes_A'
    phase_a = next(
        (t for t in seed_result.training_results if t.phase_name == "foraging"),
        None,
    )
    phase_a_prime = next(
        (t for t in seed_result.training_results if t.phase_name == "foraging_return"),
        None,
    )
    if phase_a and phase_a_prime:
        conv_a = compute_convergence_episode(
            phase_a.episode_successes,
            convergence_threshold,
        )
        conv_a_prime = compute_convergence_episode(
            phase_a_prime.episode_successes,
            convergence_threshold,
        )
        if conv_a is not None and conv_a_prime is not None:
            seed_result.plasticity_retention = conv_a / conv_a_prime
