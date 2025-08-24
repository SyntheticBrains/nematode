"""
Overfitting Detection and Monitoring for Reinforcement Learning.

This module provides tools to detect various forms of overfitting and generalization
issues in RL agents, specifically for the quantum nematode navigation task.
"""

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantumnematode.logging_config import logger

MIN_RECENT_STEPS = 10
MIN_LONG_TERM_STEPS = 20
MIN_ACTION_SEQUENCES = 5
MIN_ACTION_NGRAM = 3
MIN_POSITION_TRAJECTORIES = 5
MIN_POLICY_OUTPUTS = 10
# NOTE: Increase this once we have more diverse start positions
MIN_START_POSITIONS = 4
MIN_LOSSES = 10
RISK_LOW_THRESHOLD = 0.3
RISK_MEDIUM_THRESHOLD = 0.5
RISK_HIGH_THRESHOLD = 0.7
ACTION_ENTROPY_LOW = 0.3
POSITION_DIVERSITY_LOW = 0.3
START_POSITION_SENSITIVITY_HIGH = 0.7
POLICY_CONSISTENCY_LOW = 0.3


@dataclass
class OverfittingMetrics:
    """Metrics for detecting overfitting in RL agents."""

    # Performance variance metrics
    performance_variance: float
    recent_performance_std: float
    long_term_performance_std: float

    # Generalization metrics
    position_diversity_score: float
    action_entropy: float
    policy_consistency_score: float

    # Learning stability metrics
    loss_variance: float
    gradient_variance: float | None
    parameter_stability: float | None

    # Environment dependency metrics
    start_position_sensitivity: float
    maze_configuration_sensitivity: float

    # Overfitting risk level
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: float  # 0-1 composite score


class OverfittingDetector:
    """Detects overfitting and generalization issues in RL agents."""

    def __init__(self, window_size: int = 50, sensitivity_threshold: float = 0.8) -> None:
        """Initialize the OverfittingDetector with specified parameters."""
        self.window_size = window_size
        self.sensitivity_threshold = sensitivity_threshold

        # Performance tracking
        self.recent_steps = deque(maxlen=window_size)
        self.long_term_steps = deque(maxlen=window_size * 4)
        self.recent_rewards = deque(maxlen=window_size)

        # Behavioral tracking
        self.action_sequences = deque(maxlen=window_size)
        self.position_trajectories = deque(maxlen=window_size)
        self.start_positions = deque(maxlen=window_size)

        # Learning tracking
        self.losses = deque(maxlen=window_size)
        self.policy_outputs = deque(maxlen=window_size)

        # Environment variation tracking
        self.maze_configs = deque(maxlen=window_size)

    def update_performance_metrics(self, steps: int, total_reward: float) -> None:
        """Update performance tracking metrics."""
        self.recent_steps.append(steps)
        self.long_term_steps.append(steps)
        self.recent_rewards.append(total_reward)

    def update_behavioral_metrics(
        self,
        action_sequence: list[str],
        position_trajectory: list[tuple[int, int]],
        start_position: tuple[int, int],
    ) -> None:
        """Update behavioral pattern tracking."""
        self.action_sequences.append(action_sequence)
        self.position_trajectories.append(position_trajectory)
        self.start_positions.append(start_position)

    def update_learning_metrics(self, loss: float | None, policy_probs: np.ndarray) -> None:
        """Update learning stability tracking."""
        if loss is not None:
            self.losses.append(loss)
        self.policy_outputs.append(policy_probs.copy())

    def update_environment_metrics(self, maze_config_hash: str) -> None:
        """Update environment variation tracking."""
        self.maze_configs.append(maze_config_hash)

    def compute_performance_variance(self) -> tuple[float, float]:
        """Compute performance variance metrics."""
        if len(self.recent_steps) < MIN_RECENT_STEPS:
            return 0.0, 0.0

        recent_std = float(np.std(list(self.recent_steps)))
        long_term_std = (
            float(np.std(list(self.long_term_steps)))
            if len(self.long_term_steps) >= MIN_LONG_TERM_STEPS
            else recent_std
        )

        # Normalize by mean to get coefficient of variation
        recent_mean = float(np.mean(list(self.recent_steps)))
        recent_cv = float(recent_std / (recent_mean + 1e-8))

        return recent_cv, long_term_std

    def compute_action_entropy(self) -> float:
        """Compute entropy of action sequences to detect policy diversity."""
        if len(self.action_sequences) < MIN_ACTION_SEQUENCES:
            return 1.0  # High entropy initially

        # Compute entropy over action n-grams
        action_ngrams = []
        for seq in self.action_sequences:
            if len(seq) >= MIN_ACTION_NGRAM:
                for i in range(len(seq) - (MIN_ACTION_NGRAM - 1)):
                    action_ngrams.extend(
                        [
                            tuple(seq[i : i + MIN_ACTION_NGRAM])
                            for i in range(len(seq) - (MIN_ACTION_NGRAM - 1))
                        ],
                    )

        if not action_ngrams:
            return 1.0

        # Calculate frequency distribution
        unique_ngrams, counts = np.unique(action_ngrams, return_counts=True, axis=0)
        probs = counts / np.sum(counts)

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(unique_ngrams) + 1e-8)

        return entropy / (max_entropy + 1e-8)  # Normalized entropy

    def compute_position_diversity(self) -> float:
        """Compute diversity of position trajectories."""
        if len(self.position_trajectories) < MIN_POSITION_TRAJECTORIES:
            return 1.0

        # Compute unique positions visited across trajectories
        all_positions = set()
        total_positions = 0

        for traj in self.position_trajectories:
            all_positions.update(traj)
            total_positions += len(traj)

        if total_positions == 0:
            return 0.0

        # Diversity score: unique positions / total positions
        return len(all_positions) / total_positions

    def compute_policy_consistency(self) -> float:
        """Compute consistency of policy outputs for similar states."""
        if len(self.policy_outputs) < MIN_POLICY_OUTPUTS:
            return 1.0  # Assume consistent initially

        policy_arrays = np.array(list(self.policy_outputs))

        # Compute pairwise policy distances
        distances = []
        for i in range(len(policy_arrays) - 1):
            for j in range(
                i + 1,
                min(i + MIN_POLICY_OUTPUTS, len(policy_arrays)),
            ):  # Compare with recent policies
                dist = np.linalg.norm(policy_arrays[i] - policy_arrays[j])
                distances.append(dist)

        if not distances:
            return 1.0

        # Lower distance = higher consistency
        mean_distance = np.mean(distances)
        return np.exp(-mean_distance)  # Convert to 0-1 scale

    def compute_start_position_sensitivity(self) -> float:
        """Compute sensitivity to starting positions."""
        if (
            len(self.start_positions) < MIN_START_POSITIONS
            or len(self.recent_steps) < MIN_START_POSITIONS
        ):
            return 0.0

        # Group performance by start position
        position_performance = {}
        for pos, steps in zip(list(self.start_positions), list(self.recent_steps), strict=False):
            if pos not in position_performance:
                position_performance[pos] = []
            position_performance[pos].append(steps)

        # If only one start position, sensitivity is high
        if len(position_performance) <= 1:
            return 1.0

        # Compute variance across different start positions
        position_means = [float(np.mean(perf)) for perf in position_performance.values()]
        if len(position_means) <= 1:
            return 0.0

        overall_mean = float(np.mean(position_means))
        sensitivity = float(np.std(position_means) / (overall_mean + 1e-8))

        return min(sensitivity, 1.0)  # Cap at 1.0

    def compute_loss_variance(self) -> float:
        """Compute variance in loss values."""
        if len(self.losses) < MIN_LOSSES:
            return 0.0

        losses_array = np.array(list(self.losses))
        if np.all(losses_array == 0):
            return 0.0

        loss_std = float(np.std(losses_array))
        loss_mean = float(np.mean(losses_array))

        return float(loss_std / (loss_mean + 1e-8))

    def compute_overfitting_metrics(self) -> OverfittingMetrics:
        """Compute comprehensive overfitting metrics."""
        # Performance variance
        perf_var, long_term_var = self.compute_performance_variance()

        # Behavioral metrics
        position_diversity = float(self.compute_position_diversity())
        action_entropy = float(self.compute_action_entropy())
        policy_consistency = float(self.compute_policy_consistency())

        # Learning stability
        loss_variance = float(self.compute_loss_variance())

        # Environment sensitivity
        start_sensitivity = float(self.compute_start_position_sensitivity())
        maze_sensitivity = 0.0  # Placeholder - would need maze variation data

        # Compute composite risk score
        risk_factors = [
            float((1.0 - action_entropy) * 0.25),  # Low action diversity = risk
            float((1.0 - position_diversity) * 0.20),  # Low position diversity = risk
            float(start_sensitivity * 0.20),  # High start sensitivity = risk
            float(loss_variance * 0.15),  # High loss variance = risk
            float((1.0 - policy_consistency) * 0.20),  # Low policy consistency = risk
        ]

        risk_score = float(np.sum(risk_factors))

        # Determine risk level
        if risk_score < RISK_LOW_THRESHOLD:
            risk_level = "LOW"
        elif risk_score < RISK_MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        elif risk_score < RISK_HIGH_THRESHOLD:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return OverfittingMetrics(
            performance_variance=perf_var,
            recent_performance_std=float(np.std(list(self.recent_steps)))
            if self.recent_steps
            else 0.0,
            long_term_performance_std=long_term_var,
            position_diversity_score=position_diversity,
            action_entropy=action_entropy,
            policy_consistency_score=policy_consistency,
            loss_variance=loss_variance,
            gradient_variance=None,  # NOTE: Would need gradient tracking
            parameter_stability=None,  # NOTE: Would need parameter tracking
            start_position_sensitivity=start_sensitivity,
            maze_configuration_sensitivity=maze_sensitivity,
            risk_level=risk_level,
            risk_score=risk_score,
        )

    def log_overfitting_analysis(self) -> None:
        """Log comprehensive overfitting analysis."""
        metrics = self.compute_overfitting_metrics()

        logger.info("=== OVERFITTING ANALYSIS ===")
        logger.info(f"Risk Level: {metrics.risk_level} (Score: {metrics.risk_score:.3f})")
        logger.info(f"Performance Variance: {metrics.performance_variance:.3f}")
        logger.info(f"Action Entropy: {metrics.action_entropy:.3f}")
        logger.info(f"Position Diversity: {metrics.position_diversity_score:.3f}")
        logger.info(f"Policy Consistency: {metrics.policy_consistency_score:.3f}")
        logger.info(f"Start Position Sensitivity: {metrics.start_position_sensitivity:.3f}")
        logger.info(f"Loss Variance: {metrics.loss_variance:.3f}")

        # Provide recommendations
        if metrics.risk_level in ["HIGH", "CRITICAL"]:
            logger.warning("OVERFITTING DETECTED - Recommendations:")
            if metrics.action_entropy < ACTION_ENTROPY_LOW:
                logger.warning("- Low action diversity - consider increasing exploration")
            if metrics.position_diversity_score < POSITION_DIVERSITY_LOW:
                logger.warning("- Low position diversity - vary start positions and mazes")
            if metrics.start_position_sensitivity > START_POSITION_SENSITIVITY_HIGH:
                logger.warning("- High start position sensitivity - train on more diverse starts")
            if metrics.policy_consistency_score < POLICY_CONSISTENCY_LOW:
                logger.warning("- Low policy consistency - check for training instability")


def create_overfitting_detector_for_brain(brain_type: str) -> OverfittingDetector:
    """Create appropriate overfitting detector for brain type."""
    if brain_type in ["mlp", "qmlp"]:
        # Classical ML-style overfitting detection
        return OverfittingDetector(window_size=30, sensitivity_threshold=0.8)
    if brain_type in ["modular", "qmodular"]:
        # Quantum parameter-focused detection
        return OverfittingDetector(window_size=50, sensitivity_threshold=0.7)
    # Default detector
    return OverfittingDetector()
