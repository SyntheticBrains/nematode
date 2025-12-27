"""Chemotaxis index calculation and metrics for biological validation.

This module implements the standard chemotaxis index (CI) formula from C. elegans
literature, enabling comparison of simulated agent behavior against real worm data.

The chemotaxis index measures how strongly an organism is attracted to or avoids
a chemical stimulus:
    CI = (N_attractant - N_control) / N_total

Where:
    - N_attractant = steps spent in the attractant zone (near food)
    - N_control = steps spent in the control zone (away from food)
    - N_total = total episode steps

CI ranges from -1 (perfect avoidance) to +1 (perfect attraction).
"""

import math
from dataclasses import dataclass
from enum import Enum

# Validation thresholds for biological plausibility
CI_THRESHOLD_MINIMUM = 0.4
CI_THRESHOLD_TARGET = 0.6
CI_THRESHOLD_EXCELLENT = 0.75

# Minimum positions required for trajectory calculations
MIN_POSITIONS_FOR_TRAJECTORY = 2

# Tolerance for floating point comparisons
DISTANCE_EPSILON = 1e-6


class ValidationLevel(Enum):
    """Validation levels for biological comparison."""

    NONE = "none"  # CI < 0.4, not biologically plausible
    MINIMUM = "minimum"  # CI >= 0.4, minimally biologically plausible
    TARGET = "target"  # CI >= 0.6, target for good biological match
    EXCELLENT = "excellent"  # CI >= 0.75, excellent biological match


@dataclass
class ChemotaxisMetrics:
    """Chemotaxis metrics computed from an episode trajectory.

    Attributes
    ----------
        chemotaxis_index: CI in range [-1, 1] where 1 = perfect attraction
        time_in_attractant: Fraction of time spent near food [0, 1]
        approach_frequency: Fraction of steps moving toward food [0, 1]
        path_efficiency: Optimal distance / actual distance (0, 1]
        total_steps: Total number of steps in the episode
        steps_in_attractant: Steps within attractant zone
        steps_in_control: Steps in control zone (outside attractant)
        reliable: Whether the result is statistically reliable (enough steps)
    """

    chemotaxis_index: float
    time_in_attractant: float
    approach_frequency: float
    path_efficiency: float
    total_steps: int
    steps_in_attractant: int
    steps_in_control: int
    reliable: bool = True


@dataclass
class ValidationResult:
    """Result of validating agent behavior against biological data.

    Attributes
    ----------
        agent_ci: The agent's chemotaxis index
        biological_ci_range: The expected CI range from literature (min, max)
        biological_ci_typical: The typical/median CI from literature
        matches_biology: Whether agent CI falls within biological range
        validation_level: How well the agent matches (none/minimum/target/excellent)
        literature_source: Citation for the biological data used
        agent_metrics: Full chemotaxis metrics for the agent
    """

    agent_ci: float
    biological_ci_range: tuple[float, float]
    biological_ci_typical: float
    matches_biology: bool
    validation_level: ValidationLevel
    literature_source: str
    agent_metrics: ChemotaxisMetrics


def calculate_chemotaxis_index(
    positions: list[tuple[float, float]],
    food_positions: list[tuple[float, float]],
    attractant_zone_radius: float = 5.0,
) -> tuple[float, int, int]:
    """Calculate chemotaxis index from a trajectory.

    Uses the standard formula: CI = (N_attractant - N_control) / N_total

    Args:
        positions: List of (x, y) agent positions for each step
        food_positions: List of (x, y) food source positions
        attractant_zone_radius: Distance threshold for attractant zone

    Returns
    -------
        Tuple of (chemotaxis_index, steps_in_attractant, steps_in_control)
        Returns (0.0, 0, 0) for empty trajectories.
    """
    if not positions:
        return 0.0, 0, 0

    n_total = len(positions)
    n_attractant = 0
    n_control = 0

    for pos in positions:
        in_attractant = False
        for food_pos in food_positions:
            distance = math.sqrt(
                (pos[0] - food_pos[0]) ** 2 + (pos[1] - food_pos[1]) ** 2,
            )
            if distance <= attractant_zone_radius:
                in_attractant = True
                break

        if in_attractant:
            n_attractant += 1
        else:
            n_control += 1

    ci = (n_attractant - n_control) / n_total
    return ci, n_attractant, n_control


def _calculate_approach_frequency(
    positions: list[tuple[float, float]],
    food_positions: list[tuple[float, float]],
) -> float:
    """Calculate fraction of steps moving toward food.

    Args:
        positions: List of (x, y) agent positions
        food_positions: List of (x, y) food positions

    Returns
    -------
        Fraction of steps that moved closer to nearest food [0, 1]
    """
    if len(positions) < MIN_POSITIONS_FOR_TRAJECTORY or not food_positions:
        return 0.0

    approaching_steps = 0
    total_moves = len(positions) - 1

    for i in range(total_moves):
        # Find nearest food to current and next position
        pos_current = positions[i]
        pos_next = positions[i + 1]

        dist_current = min(
            math.sqrt((pos_current[0] - f[0]) ** 2 + (pos_current[1] - f[1]) ** 2)
            for f in food_positions
        )
        dist_next = min(
            math.sqrt((pos_next[0] - f[0]) ** 2 + (pos_next[1] - f[1]) ** 2) for f in food_positions
        )

        if dist_next < dist_current:
            approaching_steps += 1

    return approaching_steps / total_moves if total_moves > 0 else 0.0


def _calculate_path_efficiency(
    positions: list[tuple[float, float]],
    food_positions: list[tuple[float, float]],
) -> float:
    """Calculate path efficiency (optimal / actual distance).

    Args:
        positions: List of (x, y) agent positions
        food_positions: List of (x, y) food positions

    Returns
    -------
        Ratio of optimal to actual distance (0, 1], or 1.0 if no movement
    """
    if len(positions) < MIN_POSITIONS_FOR_TRAJECTORY or not food_positions:
        return 1.0

    start_pos = positions[0]

    # Find nearest food to start position
    optimal_distance = min(
        math.sqrt((start_pos[0] - f[0]) ** 2 + (start_pos[1] - f[1]) ** 2) for f in food_positions
    )

    # Calculate actual distance traveled
    actual_distance = 0.0
    for i in range(len(positions) - 1):
        dx = positions[i + 1][0] - positions[i][0]
        dy = positions[i + 1][1] - positions[i][1]
        actual_distance += math.sqrt(dx * dx + dy * dy)

    if actual_distance < DISTANCE_EPSILON:
        return 1.0

    # Efficiency capped at 1.0 (can't be more efficient than optimal)
    return min(1.0, optimal_distance / actual_distance) if optimal_distance > 0 else 1.0


def calculate_chemotaxis_metrics(
    positions: list[tuple[float, float]],
    food_positions: list[tuple[float, float]],
    attractant_zone_radius: float = 5.0,
    minimum_reliable_steps: int = 10,
) -> ChemotaxisMetrics:
    """Calculate comprehensive chemotaxis metrics from a trajectory.

    Args:
        positions: List of (x, y) agent positions for each step
        food_positions: List of (x, y) food source positions
        attractant_zone_radius: Distance threshold for attractant zone
        minimum_reliable_steps: Minimum steps for reliable statistics

    Returns
    -------
        ChemotaxisMetrics with all computed values
    """
    total_steps = len(positions)

    # Handle empty trajectory
    if total_steps == 0:
        return ChemotaxisMetrics(
            chemotaxis_index=0.0,
            time_in_attractant=0.0,
            approach_frequency=0.0,
            path_efficiency=1.0,
            total_steps=0,
            steps_in_attractant=0,
            steps_in_control=0,
            reliable=False,
        )

    # Calculate core CI
    ci, n_attractant, n_control = calculate_chemotaxis_index(
        positions,
        food_positions,
        attractant_zone_radius,
    )

    # Calculate additional metrics
    time_in_attractant = n_attractant / total_steps if total_steps > 0 else 0.0
    approach_frequency = _calculate_approach_frequency(positions, food_positions)
    path_efficiency = _calculate_path_efficiency(positions, food_positions)

    return ChemotaxisMetrics(
        chemotaxis_index=ci,
        time_in_attractant=time_in_attractant,
        approach_frequency=approach_frequency,
        path_efficiency=path_efficiency,
        total_steps=total_steps,
        steps_in_attractant=n_attractant,
        steps_in_control=n_control,
        reliable=total_steps >= minimum_reliable_steps,
    )


def get_validation_level(ci: float) -> ValidationLevel:
    """Determine validation level based on chemotaxis index.

    Args:
        ci: Chemotaxis index value

    Returns
    -------
        ValidationLevel enum value
    """
    if ci >= CI_THRESHOLD_EXCELLENT:
        return ValidationLevel.EXCELLENT
    if ci >= CI_THRESHOLD_TARGET:
        return ValidationLevel.TARGET
    if ci >= CI_THRESHOLD_MINIMUM:
        return ValidationLevel.MINIMUM
    return ValidationLevel.NONE
