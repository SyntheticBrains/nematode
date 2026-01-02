# 006: Unified Sensory Modules for PPO

**Status**: `complete`

**Branch**: `feature/unified-feature-extraction`

**Date Started**: 2026-01-01

**Date Completed**: 2026-01-02

## Objective

Implement a unified sensory module architecture that provides biologically-inspired feature extraction for PPO while matching legacy 2-feature performance. The architecture should:

1. Support multiple sensory modalities (food_chemotaxis, nociception, mechanosensation, thermotaxis)
2. Output consistent semantic features per module (strength in [0,1], angle in [-1,1])
3. Achieve parity with legacy combined-gradient preprocessing on stationary predator tasks

## Background

The legacy PPO brain uses 2 preprocessed features from a combined gradient:

- `gradient_strength`: [0, 1] - magnitude of combined food/predator signal
- `gradient_direction`: [-π, π] → normalized to [-1, 1]

This works well (~71% success, ~75% late-stage) but:

1. Pre-computes the optimal direction, limiting what the brain can learn
2. Cannot be extended to additional sensory modalities (thermotaxis, proprioception)
3. Doesn't match C. elegans neurobiology where sensory neurons are separated

The unified approach provides separated gradient signals:

- `food_chemotaxis`: Food-specific attraction (AWC, AWA neurons)
- `nociception`: Predator avoidance (ASH, ADL neurons)
- `mechanosensation`: Touch/contact signals (ALM, PLM neurons)

Each module outputs 2 features [strength, angle], giving 4-6 features total vs legacy's 2.

### The Challenge

More features = harder credit assignment. The network must independently learn:

1. Food strength high + angle aligned = move forward (good)
2. Predator strength high + angle aligned = turn away (bad)

Legacy mode pre-computes this relationship in the combined gradient.

## Hypothesis

With appropriate tuning, the unified 4-feature architecture can match legacy 2-feature performance by:

1. Using LR scheduling to stabilize early learning when credit assignment is uncertain
2. Reward shaping to make food/predator signals more distinguishable
3. Larger network capacity to handle the additional input complexity

## Method

### Architecture Evolution

| Version | Features | Description |
|---------|----------|-------------|
| Legacy | 2 | Combined gradient (strength, angle) |
| Unified v1 | 6 | Per-module quantum scaling (RX, RY, RZ) |
| Unified v2 | 4 | Classical scaling, skip zero RZ features |
| **Unified v3** | **4** | **Semantic scaling (0=no signal), 2 per module** |

### Key Implementation Changes

1. **Gradient normalization** (`env.py`): Applied tanh scaling to separated gradients

   ```python
   food_magnitude = float(np.tanh(food_magnitude_raw * GRADIENT_SCALING_TANH_FACTOR))
   ```

2. **Predator direction fix** (`env.py`): Inverted predator vector to point TOWARD danger

   ```python
   predator_direction = np.arctan2(-predator_vector_y, -predator_vector_x)
   ```

3. **Classical feature semantics** (`modules.py`): 0 = no signal (matching legacy)

   ```python
   # Core extractor outputs [0,1] for strength, [-1,1] for angle
   # Classical transform preserves these ranges
   ```

4. **LR scheduling** (`ppo.py`): Warmup + decay for stable learning

   ```python
   lr_warmup_episodes: 50      # 0.0001 → 0.001
   lr_decay_episodes: 200      # 0.001 → 0.0001 (episodes 300-500)
   ```

### Final Configuration

```yaml
brain:
  name: ppo
  config:
    actor_hidden_dim: 128
    critic_hidden_dim: 128
    num_hidden_layers: 2
    rollout_buffer_size: 512
    num_epochs: 20
    gae_lambda: 0.98
    learning_rate: 0.001
    lr_warmup_episodes: 50
    lr_warmup_start: 0.0001
    lr_decay_episodes: 200
    lr_decay_end: 0.0001
    sensory_modules:
      - food_chemotaxis
      - nociception

reward:
  reward_distance_scale: 0.3      # Weaker greedy food pull
  penalty_predator_proximity: 0.3  # Stronger avoidance signal
  penalty_health_damage: 1.5       # Make damage salient
```

______________________________________________________________________

## Results

> **Metrics**: Tables show two late-stage metrics:
>
> - **Post-Conv**: Success rate after convergence detection (variable window based on when model converges)
> - **Last 50**: Success rate for final 50 runs (fixed window: runs 451-500 for 500-run sessions)

### Stationary Predators: Progression Summary

| Configuration | Runs | Avg Success | Late Success | vs Legacy |
|---------------|------|-------------|--------------|-----------|
| Legacy (2 features) | 200 | ~71% | ~75% | - |
| Unified baseline | 200 | 34.5% | ~45% | -30% |
| + Gradient normalization | 200 | 29.6% | ~50% | -25% |
| + Predator direction fix | 200 | 34.5% | ~55% | -20% |
| + Reward shaping | 500 | 54.2% | 64.5% | -10.5% |
| + LR Warmup | 500 | 56.9% | 73.1% | -1.9% |
| **+ LR Decay** | **500** | **57.1%** | **75.0%** | **0%** |

### Stationary Predators: Final Results (LR Warmup + Decay)

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_052408 | 56.2% | 77.2% | 76.0% | 155 |
| 20260102_052410 | 49.8% | 73.3% | 74.0% | 150 |
| 20260102_052412 | **63.6%** | **76.2%** | **80.0%** | 122 |
| 20260102_052415 | 58.8% | 75.5% | 68.0% | 152 |
| **Average** | **57.1%** | **75.5%** | **74.5%** | **145** |

### Ablation Studies

#### 200 Runs (Shorter Training)

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_063518 | 27.0% | 30.0% | 42.0% | 100 |
| 20260102_063522 | 44.0% | 70.0% | 60.0% | 70 |
| 20260102_063525 | 27.0% | 20.0% | 60.0% | 78 |
| 20260102_063528 | 45.0% | 76.9% | 72.0% | 60 |
| **Average** | **35.8%** | **49.2%** | **58.5%** | **77** |

**Conclusion**: 500 runs required for this configuration to converge reliably.

#### Input Layer Norm (500 runs)

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_065108 | 41.4% | 60.0% | 40.0% | 231 |
| 20260102_065112 | 31.6% | 10.0% | 28.0% | 235 |
| 20260102_065114 | 41.2% | 60.0% | 36.0% | 172 |
| 20260102_065117 | 38.6% | 70.0% | 54.0% | 220 |
| **Average** | **38.2%** | **50.0%** | **39.5%** | **214** |

**Conclusion**: LayerNorm degrades learning. Post-conv 50% / Last-50 39.5% vs baseline 75.5% / 74.5%.

#### Entropy Scheduling (500 runs)

Config: `entropy_coef_start: 0.1`, `entropy_coef: 0.02`, `entropy_schedule_episodes: 100`

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_072933 | 60.6% | 74.9% | 72.0% | 129 |
| 20260102_072936 | 54.8% | 67.3% | 66.0% | 170 |
| 20260102_072939 | 53.8% | 69.9% | 62.0% | 186 |
| **Average** | **56.4%** | **70.7%** | **66.7%** | **162** |

**Conclusion**: Entropy scheduling hurts performance vs baseline (75.5% / 74.5%).

#### Mechanosensation Added (500 runs, 6 features)

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_060315 | 54.0% | 74.0% | 68.0% | 164 |
| 20260102_060318 | 64.0% | 78.1% | 84.0% | 107 |
| 20260102_060320 | 61.0% | 74.8% | 80.0% | 121 |
| 20260102_060323 | 52.0% | 64.9% | 62.0% | 188 |
| **Average** | **57.8%** | **73.0%** | **73.5%** | **145** |

**Conclusion**: Adding 3rd module (mechanosensation) maintains performance. Architecture scales.

### Pursuit Predators

Pursuit predators create different challenges than stationary: the agent is chased, making proximity unavoidable during escape sequences.

#### Legacy Baseline (Combined Gradient)

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_081941 | 82.5% | 93.8% | 94.0% | 35 |
| 20260102_081944 | 88.5% | 96.6% | 96.0% | 23 |
| 20260102_081946 | 83.0% | 95.5% | 100.0% | 34 |
| 20260102_081948 | 79.0% | 91.9% | 86.0% | 42 |
| **Average** | **83.2%** | **94.5%** | **94.0%** | **34** |

#### Unified with Stationary Rewards (FAILED)

Initial attempt using stationary predator reward config:

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_082832 | 37.4% | 50.5% | 46.0% | 236 |
| 20260102_082836 | 26.4% | 30.0% | 40.0% | 249 |
| 20260102_082839 | 12.6% | 10.0% | 8.0% | 289 |
| 20260102_082842 | 24.4% | 70.0% | 62.0% | 244 |
| **Average** | **25.2%** | **40.1%** | **39.0%** | **254** |

**Problem**: Stationary reward shaping backfires for pursuit - high proximity/damage penalties punish unavoidable chase behavior.

#### Unified with Pursuit-Tuned Rewards

Adjusted rewards to match legacy pursuit config:

- `reward_distance_scale: 0.5` (was 0.3) - stronger food incentive
- `penalty_predator_proximity: 0.15` (was 0.3) - lower chase punishment
- `penalty_health_damage: 0.5` (was 1.5) - less damage penalty

| Session ID | Success Rate | Post-Conv | Last 50 | Health Deaths |
|------------|--------------|-----------|---------|---------------|
| 20260102_093703 | 58.6% | 62.2% | 64.0% | 205 |
| 20260102_093709 | 60.4% | 81.3% | 76.0% | 197 |
| 20260102_093711 | 54.0% | 76.2% | 74.0% | 227 |
| 20260102_095905 | 59.6% | 75.3% | 74.0% | 196 |
| 20260102_095907 | 53.4% | 62.3% | 70.0% | 231 |
| 20260102_095909 | 48.6% | 74.2% | 74.0% | 257 |
| 20260102_095912 | 47.8% | 69.9% | 76.0% | 258 |
| **Average** | **54.6%** | **71.6%** | **72.6%** | **224** |

#### Pursuit Summary

| Configuration | Avg Success | Post-Conv | Last 50 | vs Legacy |
|---------------|-------------|-----------|---------|-----------|
| Legacy (combined gradient) | 83.2% | 94.5% | 94.0% | - |
| Unified (stationary rewards) | 25.2% | 40.1% | 39.0% | -55% |
| **Unified (pursuit rewards)** | **54.6%** | **71.6%** | **72.6%** | **-21%** |

**Key finding**: Pursuit requires different reward tuning than stationary. The remaining ~21-23% gap is likely due to temporal credit assignment challenges with moving predators.

______________________________________________________________________

## Analysis

### What Worked

| Improvement | Impact | Why It Helped |
|-------------|--------|---------------|
| Gradient normalization (tanh) | Fixed scaling | Raw magnitudes were >1, causing feature saturation |
| Predator direction fix | +5-10% success | Both gradients now point TOWARD their source |
| Classical feature semantics | Matched legacy | 0 = no signal (not -1) |
| Reward shaping | +20% success, -30% deaths | Balanced food/predator signal strength |
| Larger network (128/512) | Better at 500 runs | More capacity for 4-feature credit assignment |
| More epochs + GAE (20/0.98) | Better credit assignment | Longer temporal credit propagation |
| LR warmup (50 episodes) | +8% late-stage | Stable early feature learning |
| LR decay (200 episodes) | +2% late-stage | Fine-tunes converged policy |

### What Didn't Work

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Slower learning rate (0.0003) | No convergence | Too slow for credit assignment |
| Skip zero RZ features | Worse | Zero features may provide regularization |
| Input LayerNorm | Much worse | Interferes with gradient signal magnitudes |
| Entropy scheduling (high→low) | Worse | High early entropy prevents exploitation of sparse successes |
| Inverted predator magnitude | Slightly worse | Direction fix was correct approach, not magnitude |

### Core Challenge Explained

```text
LEGACY (2 features):
  Combined gradient → pre-computed optimal direction
  Network learns: "higher strength = move in gradient direction"
  Simple linear correlation

UNIFIED (4 features):
  Separate food + predator gradients
  Network must learn:
    "food high + aligned → move forward"
    "predator high + aligned → turn away"
  Requires learning anti-correlated responses to similar inputs
```

The 4-feature architecture requires the network to independently discover that:

1. Food strength correlates positively with reward
2. Predator strength correlates negatively with reward
3. These correlations depend on the agent's response (approach vs avoid)

This is fundamentally harder credit assignment, requiring ~2.5x more training (500 vs 200 runs).

______________________________________________________________________

## Conclusions

1. **Goal achieved for stationary**: Unified 4-feature architecture matches legacy 75% late-stage performance on stationary predators

2. **Pursuit gap remains**: Unified achieves 72% late-stage vs legacy's 95% on pursuit predators (-23% gap)

3. **Reward tuning is scenario-specific**: Stationary needs high penalties; pursuit needs low penalties (unavoidable chase)

4. **LR scheduling is critical**: Warmup prevents early instability; decay fine-tunes converged policy

5. **Architecture scales**: Adding mechanosensation (3rd module, 6 features) maintains performance

6. **Some techniques hurt**: LayerNorm, entropy scheduling, and slower LR all degraded performance

7. **Training cost**: 500 runs required vs legacy's 200 runs (2.5x more training)

______________________________________________________________________

## Next Steps

- [x] Run pursuit predator benchmarks (legacy vs unified)
- [x] Document pursuit predator results
- [ ] Test thermotaxis integration (4th module)
- [ ] Consider attention mechanisms if 4+ modules struggle

## Future Work

### LSTM/Memory for Pursuit Gap

The remaining ~23% pursuit gap is likely due to temporal credit assignment - the network sees instantaneous gradients but pursuit predators require tracking trajectory over time. Adding LSTM/GRU could help by:

1. **Remembering predator trajectory**: "It was approaching from the left" → continue evasion
2. **Anticipating pursuit**: Learning that proximity → sustained chase
3. **Better credit assignment**: Connecting current actions to future outcomes

**Deferred because**: 73% late-stage is a solid result for the unified architecture. LSTM adds significant complexity (hidden state management, sequence-based rollouts, longer training). Recommend pursuing thermotaxis integration first to validate the modular architecture scales, then revisit memory if needed.

______________________________________________________________________

## Data References

### Session IDs

**Stationary Predators (Final Config)**:

- LR Warmup: `20260102_043506`, `20260102_043509`, `20260102_043512`, `20260102_043515`
- LR Warmup + Decay: `20260102_052408`, `20260102_052410`, `20260102_052412`, `20260102_052415`

**Ablation Studies**:

- 200 runs: `20260102_063518`, `20260102_063522`, `20260102_063525`, `20260102_063528`
- Input LayerNorm: `20260102_065108`, `20260102_065112`, `20260102_065114`, `20260102_065117`
- Entropy scheduling: `20260102_072933`, `20260102_072936`, `20260102_072939`
- Mechanosensation: `20260102_060315`, `20260102_060318`, `20260102_060320`, `20260102_060323`

**Pursuit Predators**:

- Legacy: `20260102_081941`, `20260102_081944`, `20260102_081946`, `20260102_081948`
- Unified (stationary rewards - failed): `20260102_082832`, `20260102_082836`, `20260102_082839`, `20260102_082842`
- Unified (pursuit rewards): `20260102_093703`, `20260102_093709`, `20260102_093711`, `20260102_095905`, `20260102_095907`, `20260102_095909`, `20260102_095912`

> **Note**: Only the top sessions are saved in `artifacts/experiments`.

### Config Files

- **Baseline (unified)**: `configs/examples/ppo_stationary_predators_sensory_small.yml`
- **Pursuit (unified)**: `configs/examples/ppo_pursuit_predators_sensory_small.yml`
- **Legacy (stationary)**: `configs/examples/ppo_stationary_predators_small.yml`
- **Legacy (pursuit)**: `configs/examples/ppo_pursuit_predators_small.yml`
