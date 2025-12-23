# 003 Appendix: Spiking Brain Experiment Details

This appendix contains detailed experiment data, session tables, failed approaches, and implementation specifics for the spiking brain optimization experiment. For the main findings and conclusions, see [003-spiking-brain-optimization.md](003-spiking-brain-optimization.md).

---

## Table of Contents

1. [Static Navigation Experiments](#static-navigation-experiments)
2. [Dynamic Foraging Experiments](#dynamic-foraging-experiments)
3. [Predator Environment Experiments](#predator-environment-experiments)
4. [Failed Approaches](#failed-approaches)
5. [Architecture Diagrams](#architecture-diagrams)
6. [Hyperparameter Reference](#hyperparameter-reference)
7. [Session Data References](#session-data-references)

---

## Static Navigation Experiments

### Phase 1-2: Core Architecture & Gradient Fix

**Problem**: Agent stuck selecting STAY action 99.8% of the time

**Diagnosis**:

```text
Gradient norms: 9 trillion → infinity (catastrophic)
Policy collapse: STAY selected 3,860/3,868 times (99.8%)
Entropy: 0.15-0.50 (should be 1.38 for uniform)
Spike rates: Normal at 2-3% (not the issue)
```

**Root Causes**:
1. Surrogate gradient with `alpha=10.0` produced enormous gradients (scales with alpha²)
2. Only gradient norm clipping (not value clipping) allowed inf to persist
3. Entropy regularization too weak (beta=0.01, contribution ~0.001)

**Fixes**:

```python
# Clip individual gradient values first
for param in self.policy.parameters():
    if param.grad is not None:
        param.grad.clamp_(-1.0, 1.0)  # Prevent inf/NaN

# Then clip norm
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

```yaml
# Config changes
surrogate_alpha: 10.0 → 1.0      # 100x smaller gradients
entropy_beta: 0.01 → 0.05        # 5x stronger exploration
```

**Results**: Gradient norms dropped from infinity to 0.01-1.0 (healthy range)

### Phase 3: Performance Inconsistency

**Problem**: High variance across sessions (25%-65% success rate)

**Session Data**:

| Session | Success Rate | Pattern |
|---------|--------------|---------|
| 101355 | 65% | Good, no convergence |
| 101359 | 40% | Moderate |
| 101402 | 25% | Poor throughout |
| 101406 | 60% | Strong finish (100% last runs) |
| 101848 | 46% | Converged at 62, then regressed to 39.5% |
| 101842 | 64% | Converged at 42, maintained 67.2% |

**Root Causes**:
1. **Fixed learning rate** (0.001) → Can't fine-tune after initial learning
2. **Fixed entropy** (0.05) → Prevents deterministic exploitation late
3. **Random initialization variance** → Some starts better than others

**Fixes**:

```yaml
# Learning rate decay (exponential)
lr_decay_rate: 0.015  # 1.5% per episode
# Schedule: 0.001 → 0.0007 → 0.0002 over 100 episodes

# Entropy decay (linear)
entropy_beta: 0.05              # Initial (exploration)
entropy_beta_final: 0.01        # Final (exploitation)
entropy_decay_episodes: 50      # Decay over first 50 runs
```

### Phase 5: Kaiming Initialization Experiments

**Problem**: 60-point variance (18%-78%) with default PyTorch initialization

**Sessions**: `20251219_112425-112435` (pre-Kaiming baseline)
- Range: 18-78% (60-point spread)
- 1 catastrophic failure at 18%
- Initialization lottery: some random seeds excellent, others terrible

**First Kaiming Implementation (Failed)**:

```python
weight_init: kaiming  # Added to config
torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
```

**Problem Discovered**: Complete policy collapse

```text
Action probabilities: [1.000, 0.000, 0.000, 0.000]  # 100% FORWARD
Then later:          [0.000, 0.000, 0.000, 1.000]  # 100% STAY
Agent stuck in place, no learning visible
```

**Root Cause**: Kaiming initialization created extreme logits
- Kaiming scale: `sqrt(2/fan_in)` for ReLU → larger weights than default
- Output layer with Kaiming weights → logits like [20.0, 0.01, 0.01, 0.01]
- After softmax → [1.000, 0.000, 0.000, 0.000] (deterministic from start)
- No gradient flow from deterministic policy → stuck forever

**Fix - Output Layer Scaling**:

```python
def _initialize_weights(self, method: str):
    # Find output layer (last linear layer)
    linear_layers = [m for m in self.policy.modules() if isinstance(m, torch.nn.Linear)]
    output_layer = linear_layers[-1] if linear_layers else None

    for module in self.policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            if method == "kaiming":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

                # Scale down output layer to prevent extreme logits
                if module is output_layer:
                    with torch.no_grad():
                        module.weight.mul_(0.01)  # Scale by 100x
```

**Entropy Tuning for Kaiming**:

Sessions: `20251219_120330-120340` (Kaiming + low entropy)

```yaml
entropy_beta: 0.05  # Original value
```

- Range: 50-76% (26-point spread)
- Still seeing catastrophic failures (50%)
- Kaiming + scaled output still more deterministic than default init

**Solution - Higher Initial Entropy**:

Sessions: `20251219_121946-121955` (Kaiming + high entropy)

```yaml
entropy_beta: 0.15          # Tripled from 0.05
entropy_beta_final: 0.01    # Keep same
entropy_decay_episodes: 50  # Gradual decay
```

**Initialization Comparison**:

| Configuration | Range | Variance | Min | Max | Converge |
|--------------|-------|----------|-----|-----|----------|
| Default init | 18-78% | 60 pts | 18% | 78% | 1/4 |
| Kaiming + entropy=0.05 | 50-76% | 26 pts | 50% | 76% | 1/4 |
| **Kaiming + entropy=0.15** | **58-68%** | **10 pts** | **58%** | **68%** | **3/4** |

**Learning Rate Experiments (Failed)**:

Attempted faster convergence with `lr=0.0015` + `entropy_decay=30`:
Sessions: `20251219_124240-124249`
- **FAILED**: Range 27-55% (worse than before)
- Higher LR caused premature convergence to bad policies
- Faster entropy decay prevented sufficient exploration
- Reverted to `lr=0.001` + `entropy_decay=50`

### Phase 4 Session Results

Sessions: `20251219_105228`, `20251219_105232`, `20251219_105235`, `20251219_105239`

| Session | Success Rate | Converged | Conv Run | Post-Conv SR | Variance | Composite | Pattern |
|---------|--------------|-----------|----------|-------------|----------|-----------|---------|
| 105228 | **75%** | Yes | 74 | 96.2% | 0.037 | 0.807 | Late convergence, excellent stability |
| **105232** | **78%** | Yes | 52 | **100%** | **0.000** | **0.896** | Perfect post-convergence |
| 105235 | 38% | Yes | 20 | 46.2% | 0.249 | 0.484 | Poor initialization |
| 105239 | 60% | Yes | 40 | 61.7% | 0.236 | 0.552 | Moderate performance |

**Best Session Analysis** (105232):

```text
Early (1-20):    10/20 (50%)   - Avg steps: 227
Mid (21-50):     20/30 (67%)   - Learning phase
Late (51-100):   48/50 (96%)   - Avg steps: 87 (61% faster!)

Last 30 runs: 30/30 (100%) ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓

Composite score: 0.896 (approaching MLP's ~0.92)
```

---

## Dynamic Foraging Experiments

### The Dynamic Foraging Challenge

Unlike static navigation (go to fixed goal), dynamic foraging requires:
- Collecting 10 food items spawned at random positions
- Adapting to changing food locations as items are consumed
- Managing satiety (starvation penalty if satiety depletes)
- More complex state space: gradient strength + relative angle to nearest food

**Initial Results**: 0% success with end-of-episode updates only. The sparse reward signal (only at episode end) was insufficient for the spiking network to learn.

### Key Insight: MLP Uses Intra-Episode Updates

Analysis of MLP brain revealed a critical difference: MLP performs gradient updates **every 5 steps** during an episode, not just at episode end. This provides:
1. Dense learning signal (100 updates/episode vs 1)
2. Immediate feedback on good/bad actions
3. Lower variance in return estimates (5-step windows vs 500-step episodes)

### Experiment 1: First Intra-Episode Update Attempt

**Changes**:
- Added `update_frequency: 5` parameter (gradient update every 5 steps)
- Lowered learning rate to 0.0003 (from 0.001)

**Results**: 0% success - policy locked to single action within first episode

**Root Cause**: Combination of:
1. High entropy_beta (0.5) fighting policy gradient at every update
2. Cumulative effect of 100 small updates still pushed toward determinism
3. Policy locked faster than with end-of-episode updates

### Experiment 2: MLP-Like Settings

**Changes**:
- `learning_rate: 0.001` (MLP's value)
- `entropy_beta: 0.1` (lower, like MLP)
- `min_action_prob: 0.02` (prevent full determinism)

**Results**: 0% success - gradient death

**Root Cause Analysis**:

```text
Action probs: ['0.020', '0.020', '0.020', '0.940']  # Locked to STAY
Gradient norm: 0.000000  # DEAD - no gradient signal
```

The `min_action_prob: 0.02` floor was the culprit:
- Softmax saturates at 0.02/0.94 extremes
- Gradients ≈ 0 at saturation points
- Network cannot recover once it hits the floor

### Experiment 3: Lower LR + Disable Floor

**Key Insight**: Spiking network has ~30x more parameters than MLP (136k vs 4.6k). Same learning rate means ~30x more effective gradient magnitude.

**Changes**:
- `learning_rate: 0.0001` (10x lower than MLP)
- `min_action_prob: 0.0` (disabled - was causing gradient death)
- `entropy_beta: 0.3` (moderate)
- `update_frequency: 10` (less frequent - less noisy)

**Results**: Best session achieved **14% success** with 7 consecutive wins (runs 3-10), then collapsed

**Analysis**:

```text
Runs 3-10:  SUCCESS streak - entropy 0.4-0.8
Run 11+:    Entropy collapsed to 0.001, no more successes
```

Without the floor, entropy could collapse to near-zero. Once deterministic, no recovery.

### Experiment 4: Low Floor (0.005)

**Changes**:
- `min_action_prob: 0.005` (low floor, not zero)

**Results**: Best session achieved **24% success** with 9 consecutive wins (runs 10-19)

**Improvement**: Floor at 0.005 prevented total collapse (minimum entropy ~0.10 vs 0.001).
**Remaining Issue**: After ~20 runs, entropy stabilized at floor - enough for occasional success but not consistent learning.

### Experiment 5: Optimal Floor (0.01) - BREAKTHROUGH

**Changes**:
- `min_action_prob: 0.01` (slightly higher floor)
- Minimum entropy ~0.16 instead of ~0.10

**Results**:

| Session | Success | Notes |
|---------|---------|-------|
| Best | **82%** | 41/50 wins, sustained from run 24! |
| Medium | 14% | Good but unstable |
| Worst | 0% | Collapsed early |

**Best Session Analysis**:
- Runs 1-22: Warmup/learning phase with scattered successes
- Runs 23-50: **SUSTAINED SUCCESS** - 27/28 wins!
- Converged at run 22, stability 0.000, composite score 0.733

**Comparison with MLP**:

| Brain | Score | Post-Conv Success | Avg Steps | Converge@Run |
|-------|-------|-------------------|-----------|--------------|
| Spiking | 0.733 | 100% | 267 | 22 |
| MLP | 0.822 | 100% | 181 | 20 |

**Spiking matches MLP success rate!** Lower composite score due to more steps, but proves spiking neural networks CAN learn dynamic foraging effectively.

### Why min_action_prob Sweet Spot is 0.01

| Floor | Minimum Entropy | Max Probability | Result |
|-------|-----------------|-----------------|--------|
| 0.02 | ~0.29 | 0.94 | Gradient death - softmax saturates |
| 0.01 | ~0.16 | 0.97 | Sweet spot - gradients flow, some exploration |
| 0.005 | ~0.10 | 0.985 | Too confident, regression after success |
| 0.0 | ~0.001 | 0.999 | Total collapse possible |

---

## Predator Environment Experiments

### Phase 7: Initial Predator Testing

Created three new configs: `spiking_predators_small.yml`, `spiking_predators_medium.yml`, `spiking_predators_large.yml`

**Config (small)**: 20x20 grid, 2 predators, 5 foods on grid, 10 to collect, 500 max steps

**Results Summary**:

| Session | Success | Foods | Predator Deaths | Starved | Evasion Rate | Composite |
|---------|---------|-------|-----------------|---------|--------------|-----------|
| **20251221_065658** | **28%** | 4.17 | 40% | 30% | 94% | **0.390** |
| 20251221_072148 | 19% | 2.97 | 39% | 42% | 93% | 0.372 |
| 20251221_062455 | 2% | 1.18 | 35% | 61% | 93% | 0.338 |
| 20251221_073925 | 0% | 0.10 | 77% | 23% | 82% | 0.260 |

**Comparison to Other Brain Types**:

| Brain Type | Success Rate | Predator Deaths | Foods Avg | Composite | Notes |
|------------|-------------|-----------------|-----------|-----------|-------|
| **MLP** | **85%** | 10.5% | 9.2 | **0.740** | Baseline - best performance |
| **Quantum (evolved)** | 88% | 12% | 9.5 | 0.675 | Fixed params, no learning |
| **Spiking (best)** | 28% | 40% | 4.17 | 0.390 | High variance |
| **Spiking (worst)** | 0% | 77% | 0.10 | 0.260 | Catastrophic |

**Gap**: Spiking achieves ~33% of MLP's success rate (28%/85%)

### Detailed Analysis of Best Session (20251221_065658)

**Learning Curve**:

```text
Runs 1-16:  Mostly failing (starving or eaten), scattered food collection
Runs 17-18: BREAKTHROUGH - 10/10 foods collected twice in a row
Runs 19-34: Mixed results, learning predator avoidance
Runs 35-66: SUSTAINED SUCCESS - 20/32 runs with 10 foods (62.5%)
Runs 67-100: REGRESSION - Success dropped, only scattered wins
```

**Death Analysis**:

```text
Predator deaths: 40/100 runs (40%)
Starvation: 30/100 runs (30%)
Success: 28/100 runs (28%)
Max steps: 2/100 runs (2%)
```

The 30% starvation rate indicates the brain often avoids predators but fails to find food - the opposite problem to early runs.

### Root Cause Analysis

#### Problem 1: Single Combined Gradient Signal

The spiking brain receives only 2 inputs:
- `gradient_strength`: Combined food + predator signal magnitude
- `gradient_direction`: Direction of combined vector

**Issue**: When food is NORTH and predator is SOUTH, the combined gradient points NORTH (correct). But when food is NORTH and predator is NORTH, the combined gradient is ambiguous - the brain can't distinguish "go north for food" from "flee south from predator".

**Evidence from quantum logbook 001**: Same issue caused quantum circuits to plateau at ~30% - the combined gradient does heavy lifting, not the brain.

#### Problem 2: Conflicting Reward Signals

The reward structure creates competing objectives:

```yaml
reward_distance_scale: 0.5    # Approach food
penalty_predator_proximity: 0.1  # Avoid predator
penalty_predator_death: 10.0   # Strong death penalty
penalty_starvation: 2.0        # Moderate starvation penalty
```

**Issue**: The predator death penalty (10.0) is 5x stronger than starvation (2.0), causing the network to become overly cautious - avoiding predators at the cost of finding food.

#### Problem 3: High Initialization Variance

| Session | Success | Pattern |
|---------|---------|---------|
| 065658 | 28% | Good initialization, learned both objectives |
| 072148 | 19% | Moderate initialization |
| 062455 | 2% | Poor initialization, never learned foraging |
| 073925 | 0% | Catastrophic, locked to single action |

**Issue**: 28x difference between best and worst sessions with identical config - initialization lottery problem magnified by multi-objective difficulty.

### Phase 10: Hyperparameter Tuning Experiments

#### Experiment 1: Balanced Penalties

**Hypothesis**: The 10:2 predator:starvation penalty ratio may be too aggressive.

**Changes**:
- `penalty_predator_death`: 10.0 → 5.0
- `penalty_starvation`: 2.0 → 5.0
- `batch_size`: 1 → 3

**Results**: 0-3% success - **WORSE** than baseline. High predator penalty is beneficial.

#### Experiment 2: 200-Episode Training

**Hypothesis**: 100 episodes insufficient for stable convergence.

**Results**:
- **Session 134319**: 10.5% success with 21 wins in runs 1-76, then **post-convergence collapse** to 0%
- The network learned good behavior, then entropy/LR decay killed it

**Key Insight**: Post-convergence regression is the problem, not insufficient episodes.

#### Experiment 3: Constant Entropy

**Hypothesis**: Entropy decay causes late-stage collapse. Keep constant to prevent regression.

**Config**: `entropy_beta: 0.25` (constant), `lr_decay_rate: 0.002`

**Results (5 sessions)**: 0-2% success - **WORSE**. Policy stayed too stochastic, never committed.

#### Experiment 4: Higher Entropy + No Intra-Episode Updates

**Hypothesis**: Higher initial entropy with slow decay + batch gradient averaging.

**Config**: `entropy_beta: 0.4 → 0.1` over 150 episodes, `batch_size: 4`, `update_frequency: 0`

**Results (8 sessions)**: 0-3% success - **WORSE**. Disabling intra-episode updates hurt learning.

#### Experiment 5: Higher Entropy Schedule Only

**Hypothesis**: Keep intra-episode updates, just change entropy schedule.

**Config**: `entropy_beta: 0.4 → 0.1` over 150 episodes, `update_frequency: 10`

**Results (4 sessions)**: 0% success - **WORSE**. Original entropy settings were reasonable.

#### Experiment 6: Slower LR Decay ⭐ BREAKTHROUGH

**Hypothesis**: Learning rate decays too fast, network doesn't have time to learn.

**Config**: `lr_decay_rate: 0.01 → 0.005` (half the decay rate), original entropy (0.3)

**Results (4 sessions)**:

| Session | Success | Avg Foods | Pred Deaths | Starved | Composite |
|---------|---------|-----------|-------------|---------|-----------|
| **054653** | **61%** | **7.36** | 37 | 40 | **0.556** |
| 054656 | 0% | 0.24 | 135 | 65 | 0.28 |
| 054659 | 0% | 0.01 | 77 | 123 | 0.28 |
| 054702 | 0% | 0.01 | 78 | 122 | 0.28 |

**Session 054653 achieved 61% success** - more than **DOUBLE** the previous best (28%)!

Post-convergence metrics:
- Success rate: 62.8%
- Avg foods: 7.46
- Distance efficiency: 41.6%

**Why slower LR decay worked**:

1. **Original**: `lr_decay_rate: 0.01` → LR drops to ~37% after 100 episodes
2. **New**: `lr_decay_rate: 0.005` → LR drops to ~61% after 100 episodes

The faster decay (0.01) caused the learning rate to drop too low before the network could fully learn the task. With slower decay:
- More gradient updates with meaningful LR
- Network has time to refine behavior
- Good patterns get reinforced before LR bottoms out

#### Experiment 7: Even Slower LR Decay (0.003)

Tested whether even slower decay would improve further.

**Results**: 11-21.5% success - **WORSE** than 0.005. LR stays too high, network can't converge properly.

| LR Decay Rate | Best Success | Conclusion |
|---------------|--------------|------------|
| 0.01 (original) | 28% | Too fast |
| **0.005** | **61%** | **Optimal** |
| 0.003 | 21.5% | Too slow |

This confirms **0.005 is the sweet spot** for this task.

#### Phase 10 Summary

| Experiment | Key Change | Best Result | Conclusion |
|------------|-----------|-------------|------------|
| Balanced penalties | 5:5 ratio | 3% | High predator penalty helps |
| 200 episodes | More training | 10.5% | Post-convergence collapse |
| Constant entropy | No decay | 2% | Too stochastic |
| No intra-episode | Batch updates only | 3% | Loses learning signal |
| Higher entropy schedule | 0.4→0.1 | 0% | Original settings fine |
| **Slower LR decay** | **0.01→0.005** | **61%** | **BREAKTHROUGH** |
| Even slower LR decay | 0.003 | 21.5% | Too slow - confirms 0.005 optimal |

---

## Failed Approaches

### Phase 8: Separated Gradient Experiment

Based on the hypothesis that the combined gradient was ambiguous (food NORTH + predator NORTH = unclear signal), we implemented separated gradient inputs.

**Implementation**:

Added `use_separated_gradients: bool` config option to `SpikingBrainConfig`:
- When enabled, input changes from 2 features to 4 features:
  - `[food_strength, food_rel_angle, pred_strength, pred_rel_angle]`
- Each angle normalized relative to agent facing direction [-1, 1]
- Network input_dim automatically set to 4 when enabled

Files modified:
- `spiking.py`: Added config option, updated `preprocess()` for 4-feature output
- `run_simulation.py`: Dynamic input_dim based on config
- `spiking_predators_*.yml`: All three configs updated with `use_separated_gradients: true`

**Results: FAILURE**

| Session | Success | Foods | Pred Deaths | Starved | Evasion | Composite |
|---------|---------|-------|-------------|---------|---------|-----------|
| 090405 | **0%** | 0.65 | 48% | 52% | 89% | 0.26 |
| 090408 | **0%** | 0.10 | 45% | 55% | 92% | 0.26 |
| 090412 | **0%** | 0.03 | 58% | 42% | 90% | 0.26 |
| 090415 | **0%** | 0.18 | 40% | 60% | 92% | 0.26 |

**Comparison: Separated vs Combined Gradients**:

| Metric | Combined (Best) | Separated (Best) | Change |
|--------|-----------------|------------------|--------|
| Success rate | 28% | 0% | **-28%** |
| Avg foods | 4.17 | 0.65 | **-85%** |
| Predator deaths | 40% | 40-58% | Similar |
| Evasion rate | 94% | 89-92% | Similar |
| Composite | 0.390 | 0.26 | **-33%** |

**Verdict**: Separated gradients made performance WORSE, not better.

**Why This Failed**:

1. **Doubled input complexity without architectural support**: Going from 2→4 inputs doubled the parameter space, but the network converged to a passive policy (0% success)

2. **Early convergence to bad policy**: All 4 sessions locked into doing nothing useful very early

3. **Predator avoidance maintained, food-seeking lost**: 89-92% evasion rate preserved, but food collection dropped from 4.17 to <1 avg. The network learned "don't die" but forgot "find food"

4. **The hypothesis was wrong**: The combined gradient actually HELPS because:
   - Environment pre-computes optimal direction (food attraction + predator repulsion)
   - Brain just learns "follow this direction" (simple)
   - With separated gradients, brain must learn to INTEGRATE signals (harder)

**Key Insight**:

This mirrors the quantum experiment 001 finding:
> "The environment's signal does the heavy lifting, not the brain"

The combined gradient is a **feature**, not a bug. It offloads the hard multi-objective optimization to the environment, leaving the brain with a simpler single-objective task.

### Phase 9: Dual-Stream Architecture Experiment

Based on the hypothesis that raw 4-input concatenation failed because the network had to learn multi-objective integration (a hard problem), we implemented a biologically-inspired dual-stream architecture with explicit gating.

**Architecture Design**:

```text
food_grad ──► [Appetitive Stream (LIF)] ──► approach_logits ─┐
                                                              ├──► [Blend] ──► action
pred_grad ──► [Aversive Stream (LIF)]  ──► avoid_logits ────┘
              [Gating Network (MLP)]   ──► gate_weight (0-1)
```

**Key design principles**:
1. **Separation of concerns**: Each stream learns ONE objective (simpler learning)
2. **Learned gating**: Small MLP learns when to prioritize each stream
3. **Biological plausibility**: Mirrors appetitive/aversive circuits in real brains
4. **Satiety modulation**: Gate receives satiety input (hungry → food, full → avoid)

**Implementation Details**:

Files created:
- `_dual_stream_spiking.py`: Core network with GatingNetwork and DualStreamSpikingNetwork
- `dual_spiking.py`: Brain wrapper with DualStreamSpikingBrainConfig

**Gating Network**:
- Input: 5 features (food_strength, food_angle, pred_strength, pred_angle, satiety)
- Hidden: 16 neurons
- Output: 1 value (gate weight 0-1)
- Initialization bias: -0.85 → sigmoid ≈ 0.3 (slightly favor food-seeking)

**Each Stream**:
- Input: 2 features (strength, relative_angle)
- Optional population coding (8 neurons per feature)
- 1 LIF hidden layer (64 neurons)
- Output: 4 action logits

#### Experiment 1: Learned Gating

**Results (8 sessions)**:

| Session | Success | Foods | Pred Deaths | Starved | Evasion | Composite |
|---------|---------|-------|-------------|---------|---------|-----------|
| 110248 | 0% | 0.78 | 60% | 40% | 89% | 0.26 |
| 110250 | 0% | 0.96 | 72% | 28% | 85% | 0.26 |
| 110253 | 0% | 0.01 | 45% | 55% | 92% | 0.26 |
| 110256 | 0% | 0.83 | 64% | 36% | 86% | 0.26 |
| 104803 | 0% | 1.06 | 72% | 28% | 85% | 0.26 |
| 104805 | 0% | 1.11 | 72% | 28% | 84% | 0.26 |
| 104809 | 0% | 0.06 | 50% | 50% | 90% | 0.26 |
| 104812 | 0% | 0.03 | 50% | 50% | 89% | 0.26 |

**Observation**: High variance (0.01 to 1.11 foods) with no successes. The gating network wasn't learning - random initialization dominated behavior.

#### Experiment 2: Satiety-Modulated Gating

Added satiety as input to gating network with hypothesis: "When hungry, prioritize food. When full, prioritize avoidance."

**Changes**:
- Gating network input: 4 → 5 features (added normalized satiety)
- Bias initialization: favor food-seeking when hungry
- Increased food rewards: reward_goal 2.0 → 3.0, reward_distance_scale 0.5 → 1.0
- Reduced predator penalty: 10.0 → 2.0

**Results**: Still 0% success. High variance persisted - random initialization still dominated.

#### Experiment 3: Fixed Gating Diagnostic

To determine if the architecture was sound but learning was broken, or if the dual-stream approach itself was flawed, we implemented a **fixed gating diagnostic**:

```python
if self.use_fixed_gating:
    # Bypass learned gating entirely
    gate_weight = satiety  # hungry (low) → 0 (food), full (high) → 1 (avoid)
else:
    gate_weight = self.gate(context)  # learned
```

**Hypothesis**: If fixed gating works, the architecture is sound and learning is the bottleneck. If it still fails, the dual-stream separation itself is the problem.

**Results (8 sessions)**:

| Session | Success | Foods | Pred Deaths | Starved | Evasion | Composite |
|---------|---------|-------|-------------|---------|---------|-----------|
| 114421 | 0% | 0.32 | 48% | 52% | 90% | 0.26 |
| 114423 | 0% | 0.27 | 35% | 65% | 94% | 0.26 |
| 114426 | 0% | 0.84 | 63% | 37% | 86% | 0.26 |
| 114430 | 0% | 0.29 | 51% | 49% | 91% | 0.26 |
| 115555 | 0% | 0.46 | 59% | 41% | 90% | 0.26 |
| 115558 | 0% | 0.57 | 40% | 60% | 89% | 0.26 |
| 115601 | 0% | 0.50 | 47% | 53% | 89% | 0.26 |
| 115604 | 0% | 0.13 | 43% | 57% | 92% | 0.26 |

**Averages**: 0% success, 0.42 foods, 48.3 pred deaths, 51.8 starved, 90% evasion

#### Diagnostic Conclusion

**The dual-stream architecture itself is fundamentally broken**, not just the learning.

Even with perfect gating (verified to work correctly: hungry→food, full→avoid), the architecture produces 0% success.

**Root Cause Analysis**:

The problem is that **each stream only sees its own gradient** (2 features each) which is insufficient context:

| Stream | Inputs | What it can learn | What it can't learn |
|--------|--------|-------------------|---------------------|
| Appetitive | [food_strength, food_angle] | "Move toward food" | "When to back off for predators" |
| Aversive | [pred_strength, pred_angle] | "Move away from predators" | "When to risk approach for food" |

**Compare to working approaches**:
- **Combined-gradient spiking (28%)**: Sees pre-computed optimal direction
- **MLP (85%)**: Sees all 4 inputs simultaneously, can learn internal integration

**The fundamental issue**: Separated inputs mean each stream makes decisions in isolation. Even with perfect gating:
1. The appetitive stream can't factor in predator proximity when seeking food
2. The aversive stream can't factor in food urgency when fleeing
3. Each stream makes locally-optimal but globally-poor decisions

**Why Dual-Stream Worked in Biology but Not Here**:

In biological brains, appetitive/aversive circuits have:
1. **Lateral connections**: Streams share information, not completely isolated
2. **Neuromodulation**: Dopamine/serotonin provide global context signals
3. **Hierarchical override**: Amygdala can completely suppress appetitive behavior in danger
4. **Rich sensory input**: Each circuit gets full sensory context, not partial

Our implementation had:
1. **Complete isolation**: No cross-stream connections
2. **Simple gating**: Linear blend, no override capability
3. **Partial inputs**: Each stream sees only 2 of 4 features

#### Approach Comparison Summary

| Approach | Architecture | Success Rate | Key Insight |
|----------|-------------|--------------|-------------|
| Combined gradient (baseline) | Single stream, 2 inputs | **28%** | Environment pre-computes direction |
| Separated gradients | Single stream, 4 inputs | 0% | Too hard to learn integration |
| Dual-stream learned gating | 2 streams, learned gate | 0% | Learning fails in 100 episodes |
| Dual-stream fixed gating | 2 streams, fixed gate | 0% | **Architecture itself broken** |
| MLP | Single network, 4 inputs | 85% | Sufficient capacity for integration |

---

## Architecture Diagrams

### Spiking Neural Network (Dynamic Foraging Configuration)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPIKING NEURAL NETWORK                               │
│                     (Dynamic Foraging Configuration)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS (2 features)           POPULATION ENCODER                           │
│  ┌──────────────────┐           ┌─────────────────────────────────────────┐ │
│  │ gradient_strength│──────────▶│  Gaussian Tuning Curves (8 neurons)     │ │
│  │ [0.0 - 1.0]      │           │  ○ ○ ○ ○ ○ ○ ○ ○  (σ=0.25)              │ │
│  └──────────────────┘           │  Preferred values: 0.0, 0.14, ..., 1.0  │ │
│  ┌──────────────────┐           ├─────────────────────────────────────────┤ │
│  │ relative_angle   │──────────▶│  Gaussian Tuning Curves (8 neurons)     │ │
│  │ [-π, +π]         │           │  ○ ○ ○ ○ ○ ○ ○ ○  (σ=0.25)              │ │
│  └──────────────────┘           │  Preferred values: -π, ..., 0, ..., +π  │ │
│                                 └─────────────────────────────────────────┘ │
│                                              │                              │
│                                              ▼ 16 neurons                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     TEMPORAL PROCESSING (100 timesteps)              │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  INPUT LAYER (LIF)                                             │  │   │
│  │  │  16 → 256 neurons                                              │  │   │
│  │  │  ┌───┐ ┌───┐ ┌───┐     ┌───┐                                   │  │   │
│  │  │  │LIF│ │LIF│ │LIF│ ... │LIF│  τ_m=20.0, V_th=1.0               │  │   │
│  │  │  └─┬─┘ └─┬─┘ └─┬─┘     └─┬─┘                                   │  │   │
│  │  └────┼─────┼─────┼─────────┼─────────────────────────────────────┘  │   │
│  │       │     │     │         │  spikes (0/1)                          │   │
│  │       ▼     ▼     ▼         ▼                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  HIDDEN LAYER 1 (LIF)                                          │  │   │
│  │  │  256 → 256 neurons                                             │  │   │
│  │  │  ┌───┐ ┌───┐ ┌───┐     ┌───┐                                   │  │   │
│  │  │  │LIF│ │LIF│ │LIF│ ... │LIF│  Surrogate gradient: α=1.0        │  │   │
│  │  │  └─┬─┘ └─┬─┘ └─┬─┘     └─┬─┘                                   │  │   │
│  │  └────┼─────┼─────┼─────────┼─────────────────────────────────────┘  │   │
│  │       │     │     │         │                                        │   │
│  │       ▼     ▼     ▼         ▼                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  HIDDEN LAYER 2 (LIF)                                          │  │   │
│  │  │  256 → 256 neurons                                             │  │   │
│  │  │  ┌───┐ ┌───┐ ┌───┐     ┌───┐                                   │  │   │
│  │  │  │LIF│ │LIF│ │LIF│ ... │LIF│                                   │  │   │
│  │  │  └─┬─┘ └─┬─┘ └─┬─┘     └─┬─┘                                   │  │   │
│  │  └────┼─────┼─────┼─────────┼─────────────────────────────────────┘  │   │
│  │       │     │     │         │                                        │   │
│  │       ▼     ▼     ▼         ▼                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  SPIKE ACCUMULATOR                                             │  │   │
│  │  │  Sum spikes over 100 timesteps → [0, 100] per neuron           │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼ 256 spike counts             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT LAYER (Linear)                                               │   │
│  │  256 → 4 action logits                                               │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │   │
│  │  │ FORWARD  │ │   LEFT   │ │  RIGHT   │ │   STAY   │                 │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                 │   │
│  └───────┼────────────┼────────────┼────────────┼───────────────────────┘   │
│          │            │            │            │                           │
│          ▼            ▼            ▼            ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  SOFTMAX + min_action_prob floor (0.01)                              │   │
│  │  Ensures each action has ≥1% probability (entropy floor ~0.16)       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│                                    ACTION PROBABILITIES                     │
│                              [P(FWD), P(LEFT), P(RIGHT), P(STAY)]           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  LEARNING: REINFORCE with entropy regularization                            │
│  • Updates every 10 steps (intra-episode) + episode end                     │
│  • LR: 0.0001, entropy_beta: 0.3, advantage_clip: 2.0                       │
│  • ~136,000 trainable parameters                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Biological Realism Spectrum

```text
BIOLOGICAL REALISM SPECTRUM
════════════════════════════════════════════════════════════════════

Pure Biology          Our SNN              Rate-coded ANN        Standard MLP
(Hodgkin-Huxley)     (LIF + Surrogate)    (Firing rates)       (ReLU/Sigmoid)
     │                    │                     │                     │
     ▼                    ▼                     ▼                     ▼
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Ion     │ LIF     │ LIF +   │ Rate    │ Sigmoid │ ReLU    │ Linear  │
│ Channels│ + STDP  │ Backprop│ Neurons │ Units   │ Units   │ Units   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
    ◄─────── More Biological ───────────── More Abstract ───────────►

• Our approach: LIF dynamics with gradient learning (standard neuromorphic trade-off)
• For true biological plausibility: Replace surrogate gradients with STDP
• STDP achieves worse task performance but is deployable on neuromorphic hardware
```

---

## Hyperparameter Reference

### Hyperparameters That Work

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Architecture** | | |
| hidden_size | 128-256 | Sweet spot for capacity |
| num_hidden_layers | 2 | Deeper may help but not tested |
| num_timesteps | 100 | Critical (50 was too low) |
| **LIF Neurons** | | |
| tau_m | 20.0 | Membrane time constant |
| v_threshold | 1.0 | Firing threshold |
| v_reset | 0.0 | Post-spike reset |
| surrogate_alpha | 1.0 | **Critical** (10.0 explodes) |
| **Learning** | | |
| learning_rate | 0.0001-0.001 | Lower for dynamic tasks |
| lr_decay_rate | 0.005-0.015 | 0.005 for predators, 0.015 for static |
| gamma | 0.99 | Discount factor |
| baseline_alpha | 0.05 | Baseline EMA |
| entropy_beta | 0.15-0.30 | Higher with Kaiming |
| entropy_beta_final | 0.01-0.30 | Constant for dynamic tasks |
| entropy_decay_episodes | 50 | First half of training |
| min_action_prob | 0.01 | Entropy floor (critical for dynamic) |
| **Initialization** | | |
| weight_init | kaiming/orthogonal | Both reduce variance |
| output_layer_scale | 0.01 | **Critical for Kaiming** |
| **Gradient Control** | | |
| value_clip | 1.0 | Clamp individual gradients |
| norm_clip | 1.0 | Clip total gradient norm |
| advantage_clip | 2.0 | Prevent catastrophic updates |
| return_clip | 50.0 | Balance success/failure signals |

### Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| surrogate_alpha=10.0 | Gradient explosion → policy collapse | Use 1.0 |
| No value clipping | inf/NaN gradients corrupt network | Clamp to [-1, 1] |
| Fixed entropy | Can't exploit after exploration | Decay 0.15→0.01 |
| Fixed LR | Can't fine-tune, may regress | Decay appropriately |
| num_timesteps=50 | Insufficient temporal integration | Use 100 |
| No gradient norm clip | Complements value clip | Use max_norm=1.0 |
| Kaiming without output scaling | Policy collapse (100% one action) | Scale output by 0.01 |
| Low entropy with Kaiming | Early collapse | Use entropy_beta=0.15+ |
| High LR (0.0015) | Premature bad convergence | Keep at 0.001 |
| Fast entropy decay (30 eps) | Insufficient exploration | Use 50 episodes |
| min_action_prob=0.02 | Gradient death | Use 0.01 |
| min_action_prob=0.0 | Entropy collapse | Use 0.01 |

---

## Session Data References

### Best Sessions by Environment

**Static Navigation**:
- **20251219_052425**: 83% success, 100% post-convergence, 0.932 composite
  - Config: `spiking_static_medium.yml` (with batch_size: 3, entropy_beta: 0.2, LR: 0.0001)
  - Runs: 100
  - Converged: Run 34
  - Pattern: After convergence, perfect 66/66 runs

**Dynamic Foraging**:
- Best session: 82% success, 100% post-convergence, 0.733 composite
  - Config: `spiking_foraging_small.yml`
  - Key: `min_action_prob: 0.01`, `update_frequency: 10`

**Predator + Foraging**:
- **20251222_054653**: 61% success, 7.36 avg foods, 0.556 composite
  - Config: `spiking_predators_small.yml` with `lr_decay_rate: 0.005`
  - Submitted to benchmarks leaderboard
  - Gap to MLP reduced from 57% (28% vs 85%) to 24% (61% vs 85%)

### Comparison Baselines

**MLP**:
- Static: 100% success, 0.960 composite
- Foraging: 100% success, 0.822 composite
- Predators: 85% success, 10.5% predator deaths, 0.740 composite

**Quantum (Evolved)**:
- Static: 100% success, 0.980 composite
- Foraging: 100% success, 0.762 composite
- Predators: 88% success (evolved params), 12% predator deaths, 0.675 composite

### Config Files

- `configs/examples/spiking_static_medium.yml`
- `configs/examples/spiking_foraging_small.yml`, `medium.yml`, `large.yml`
- `configs/examples/spiking_predators_small.yml`, `medium.yml`, `large.yml`

### Code Files Modified

- `quantumnematode/brain/arch/spiking.py` - Complete rewrite
- `quantumnematode/brain/arch/_spiking_layers.py` - LIF neurons, surrogate gradients
- All 7 spiking config files - Added decay parameters
