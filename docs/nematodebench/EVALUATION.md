# NematodeBench Evaluation Methodology

This document describes the scoring methodology used in NematodeBench.

## Composite Score

The benchmark uses a weighted composite score to evaluate brain architectures:

```
score = 0.40 × success_rate +
        0.30 × distance_efficiency +
        0.20 × learning_speed +
        0.10 × stability
```

Each component is normalized to [0, 1] before weighting.

## Component Metrics

### Success Rate (40% weight)

**Definition:** Fraction of episodes that reach the food collection target.

```python
success_rate = num_successful_episodes / total_episodes
```

An episode is successful if:
- Agent collects `target_foods_to_collect` items, OR
- Agent is still alive at `max_steps` with food collected

**Rationale:** Primary goal is foraging success. This metric directly measures task completion.

### Distance Efficiency (30% weight)

**Definition:** How directly the agent navigates toward food.

```python
efficiency = optimal_distance / actual_distance_traveled
```

Where:
- `optimal_distance` = straight-line distance from start to first food
- `actual_distance_traveled` = sum of step distances

**Normalization:** Capped at 1.0 (can't be more efficient than optimal path).

**Rationale:** Efficient navigation indicates good gradient following and learned spatial awareness.

### Learning Speed (20% weight)

**Definition:** How quickly the agent learns to succeed.

```python
learning_speed = 1.0 - (episodes_to_80_success / max_episodes)
```

Where:
- `episodes_to_80_success` = first episode where rolling success rate ≥ 80%
- `max_episodes` = maximum training episodes (typically 500)

If 80% success is never reached, learning_speed = 0.

**Rationale:** Faster learning means more efficient optimization and better inductive biases.

### Stability (10% weight)

**Definition:** Consistency of performance across training runs.

```python
stability = 1.0 - coefficient_of_variation(final_success_rates)
stability = 1.0 - (std / mean)
```

Capped at [0, 1].

**Rationale:** Stable algorithms are more reliable for real-world deployment.

## Score Interpretation

| Score Range | Performance Level |
|-------------|-------------------|
| ≥ 0.90 | Exceptional |
| 0.80 - 0.89 | Excellent |
| 0.70 - 0.79 | Good |
| 0.60 - 0.69 | Acceptable |
| < 0.60 | Below threshold |

## Statistical Requirements

### Minimum Runs

- **Required:** 50 independent training runs
- **Recommended:** 100 runs for tight confidence intervals

### Confidence Intervals

Results are reported with 95% confidence intervals:

```python
ci_95 = mean ± 1.96 × (std / sqrt(n_runs))
```

## Biological Validation Bonus

Submissions that include chemotaxis validation receive bonus recognition:

| Validation Level | Description |
|-----------------|-------------|
| None | No chemotaxis validation |
| Minimum | CI ≥ 0.4 |
| Target | CI ≥ 0.6 |
| Excellent | CI ≥ 0.75 (wild-type level) |

Excellent biological validation is noted on the leaderboard but doesn't affect the composite score directly.

## Example Calculation

```python
# Example scores
success_rate = 0.92        # 92% success
distance_efficiency = 0.78  # 78% efficient paths
learning_speed = 0.85      # Reached 80% at episode 75/500
stability = 0.95           # Low variance across runs

# Composite calculation
score = (0.40 × 0.92) +    # 0.368
        (0.30 × 0.78) +    # 0.234
        (0.20 × 0.85) +    # 0.170
        (0.10 × 0.95)      # 0.095
      = 0.867              # Excellent

# With 95% CI
score = 0.867 ± 0.023
```

## Leaderboard Updates

The leaderboard is updated when:
1. A new submission passes validation
2. The submission's mean score exceeds existing entries for that brain/task combination
3. OR the submission's lower CI bound exceeds the existing entry's upper CI bound (significant improvement)

Equal scores are ordered by:
1. Lower variance (more reliable)
2. Better biological validation
3. Earlier submission date
