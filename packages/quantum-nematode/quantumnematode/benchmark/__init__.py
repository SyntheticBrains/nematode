"""Convergence detection and composite scoring for Quantum Nematode."""

from quantumnematode.benchmark.convergence import (
    ConvergenceMetrics,
    analyze_convergence,
    calculate_learning_speed,
    calculate_learning_speed_episodes,
    calculate_stability,
)

__all__ = [
    "ConvergenceMetrics",
    "analyze_convergence",
    "calculate_learning_speed",
    "calculate_learning_speed_episodes",
    "calculate_stability",
]
