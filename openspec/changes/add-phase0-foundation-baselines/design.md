# Design: Phase 0 Foundation and Baselines

## Overview

This design document covers the technical approach for four Phase 0 deliverables: PPO brain implementation, chemotaxis validation system, NematodeBench documentation, and optimization method documentation.

## 1. PPO Brain Architecture

### Design Rationale

PPO is chosen over SAC or TD3 because:
- Most widely used modern RL algorithm
- Stable training with clipped surrogate objective
- Works well with discrete action spaces (our 4 actions: forward, left, right, stay)
- Extensively validated in literature

### Architecture

```text
PPOBrain (ClassicalBrain)
├── Actor Network (policy)
│   ├── Input: 2D state [gradient_strength, relative_angle]
│   ├── Hidden: 64 -> 64 (configurable)
│   └── Output: 4 action logits -> softmax
├── Critic Network (value function)
│   ├── Input: 2D state (same as actor)
│   ├── Hidden: 64 -> 64 (configurable)
│   └── Output: 1 scalar value
└── Rollout Buffer
    ├── States, actions, log_probs, values, rewards
    ├── Size: 2048 steps (configurable)
    └── Minibatch sampling for updates
```

### Key Implementation Details

**Preprocessing**: Match MLPBrain exactly
- `gradient_strength`: Normalized to [0, 1]
- `relative_angle`: Computed as `(gradient_direction - agent_facing_angle + π) mod 2π - π`, normalized to [-1, 1]

**Training Loop**:
1. Collect rollout_buffer_size steps
2. Compute GAE advantages
3. For num_epochs:
   - For each minibatch:
     - Compute new log_probs, values
     - Compute ratio = exp(new_log_prob - old_log_prob)
     - Clipped surrogate loss
     - Value loss (MSE)
     - Entropy bonus
     - Combined gradient update

**Hyperparameters** (defaults):
- `clip_epsilon`: 0.2
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `value_loss_coef`: 0.5
- `entropy_coef`: 0.01
- `learning_rate`: 0.0003
- `num_epochs`: 4
- `num_minibatches`: 4
- `rollout_buffer_size`: 2048

### Integration Points

1. **Brain Factory** (`brain/arch/__init__.py`): Add `PPOBrain`, `PPOBrainConfig` exports
2. **Benchmark Categorization**: PPO maps to "classical" (existing logic handles this)
3. **Configuration System**: Add PPO-specific YAML schema

## 2. Chemotaxis Validation System

### Design Rationale

Chemotaxis index (CI) is the standard metric in C. elegans research for quantifying attraction/avoidance behavior. Using published CI values enables direct comparison between simulated agents and real worms.

### Architecture

```text
quantumnematode/validation/
├── __init__.py
├── chemotaxis.py      # CI calculation and metrics
└── datasets.py        # Dataset loading and benchmark

data/chemotaxis/
├── literature_ci_values.json  # Published CI values
└── README.md                  # Dataset documentation
```

### Chemotaxis Index Formula

```text
CI = (N_attractant - N_control) / N_total

Where:
- N_attractant = steps in attractant zone (near food, radius ≤ 5)
- N_control = steps in control zone (away from food)
- N_total = total episode steps
```

### Additional Metrics

```python
@dataclass
class ChemotaxisMetrics:
    chemotaxis_index: float      # CI ∈ [-1, 1]
    time_in_attractant: float    # Fraction of time near food
    approach_frequency: float    # Steps moving toward food / total steps
    path_efficiency: float       # Direct distance / actual distance
```

### Literature Dataset Structure

```json
{
  "version": "1.0",
  "sources": [
    {
      "citation": "Bargmann & Horvitz (1991). Cell 65(5):837-847",
      "attractant": "diacetyl",
      "ci_wild_type": 0.75,
      "ci_range": [0.6, 0.9],
      "conditions": "standard assay, 1:1000 dilution"
    }
  ],
  "validation_thresholds": {
    "biological_match_minimum": 0.4,
    "biological_match_target": 0.6,
    "biological_match_excellent": 0.75
  }
}
```

### Validation Benchmark

```python
class ChemotaxisValidationBenchmark:
    def validate_agent(self, metrics: ChemotaxisMetrics) -> ValidationResult:
        """Compare agent CI to biological range."""
        matches = ci_range[0] <= metrics.chemotaxis_index <= ci_range[1]
        return ValidationResult(
            matches_biology=matches,
            agent_ci=metrics.chemotaxis_index,
            biological_ci_range=ci_range,
        )
```

### Integration Points

1. **Experiment Metadata** (`experiment/metadata.py`): Add `chemotaxis_index` to ResultsMetadata
2. **Simulation Script** (`run_simulation.py`): Add `--validate-chemotaxis` flag
3. **Benchmark Submission**: Include CI in benchmark JSON

## 3. NematodeBench Documentation

### Directory Structure

```text
docs/nematodebench/
├── README.md              # Overview and purpose
├── SUBMISSION_GUIDE.md    # Step-by-step process
├── EVALUATION.md          # Scoring methodology
└── REPRODUCIBILITY.md     # Requirements

scripts/
└── evaluate_submission.py # Validation script
```

### Submission Guide Outline

1. Prerequisites (50+ runs, clean git, config in repo)
2. Running experiments with tracking
3. Submitting via benchmark_submit.py
4. Creating PR with benchmark JSON
5. Verification process

### Evaluation Methodology

Composite score formula (already implemented):

```text
score = 0.40 * success_rate +
        0.30 * distance_efficiency +
        0.20 * learning_speed +
        0.10 * stability
```

### Evaluation Script

```python
def evaluate_submission(benchmark_path: Path, reproduce: bool = False) -> EvaluationResult:
    """Validate benchmark submission."""
    # 1. Validate JSON structure
    # 2. Check minimum 50 runs
    # 3. Verify config exists
    # 4. Optionally reproduce results
    # 5. Return pass/fail with details
```

## 4. Optimization Method Documentation

### Document Structure

```markdown
# Optimization Methods for Brain Architectures

## Summary Table
| Architecture | Primary Method | Secondary Method | Notes |

## Detailed Findings
### Quantum Architectures
- CMA-ES: 88% success (validated)
- Parameter-shift gradients: 22% success (high variance)

### Classical Architectures
- REINFORCE: 92% success (MLPBrain)
- PPO: Target >85% success (new)

### Spiking Architectures
- Surrogate gradients + REINFORCE: 63% on predator

## Configuration Examples
[YAML snippets for each approach]

## Selection Guidance
[Decision tree for choosing optimization method]
```

## Dependencies Between Deliverables

```text
                    ┌─────────────────┐
                    │   PPO Brain     │
                    │  (independent)  │
                    └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│   Chemotaxis    │     │  Optimization   │
│   Validation    │     │     Docs        │
│  (independent)  │     │  (independent)  │
└─────────────────┘     └─────────────────┘
        │                       │
        └───────┬───────────────┘
                │
        ┌───────▼───────┐
        │  NematodeBench │
        │     Docs       │
        │ (references    │
        │  other work)   │
        └────────────────┘
```

PPO, Chemotaxis, and Optimization Docs can be implemented in parallel. NematodeBench docs should reference them.

## Testing Strategy

### PPO Tests
1. Unit: Rollout buffer, GAE computation, clipped loss
2. Integration: Config loading, brain factory
3. Benchmark: 100 runs on dynamic_small targeting >85%

### Chemotaxis Tests
1. Unit: CI calculation, zone detection
2. Integration: Episode data extraction
3. Validation: Compare against known CI values

### Documentation Tests
1. Manual review of submission guide flow
2. Test evaluate_submission.py on existing benchmarks
3. Verify all code examples work
