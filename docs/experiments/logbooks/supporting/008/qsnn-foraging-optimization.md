# 008 Appendix: QSNN Optimization History

This appendix documents the QSNN optimization journey from 0% to 73.9% success on foraging. For main findings, see [008-quantum-brain-evaluation.md](../../008-quantum-brain-evaluation.md). For predator optimization history, see [qsnn-predator-optimization.md](qsnn-predator-optimization.md).

______________________________________________________________________

## Table of Contents

1. [Optimization Summary](#optimization-summary)
2. [Phase 1: Hebbian Learning (Rounds 0-11)](#phase-1-hebbian-learning-rounds-0-11)
3. [Phase 2: Surrogate Gradient Foundation (Rounds 12-12e)](#phase-2-surrogate-gradient-foundation-rounds-12-12e)
4. [Phase 3: Stabilization and Tuning (Rounds 12f-12o)](#phase-3-stabilization-and-tuning-rounds-12f-12o)
5. [Phase 4: Validation (Rounds 12p-12q)](#phase-4-validation-rounds-12p-12q)
6. [Key Technical Decisions](#key-technical-decisions)
7. [Final Configuration](#final-configuration)

______________________________________________________________________

## Optimization Summary

| Round | Key Change | Success | Finding |
|-------|-----------|---------|---------|
| 0 | Baseline (Hebbian, 6→4→4) | 0% | LR too low |
| 1 | Higher LR, advantage, centering | 0% | Centering amplified noise 83x |
| 2 | Fixed action selection, intra-ep updates | 0% | W_hm symmetry collapse |
| 3 | Noise + column repulsion | 0% | Noise too weak AND destructive |
| 4 | Eligibility noise + L2 decay | 0% | Learning algorithm is bottleneck |
| 5 | tanh + theta training + small weights | 0% | Weights grow past tanh saturation |
| 6 | Strong weight decay | 0% | Decay can't counter update magnitude |
| 7 | Eligibility trace normalization | 0% | Policy collapse cycles |
| 8 | Exploration floor (epsilon=0.1) | 0% | Sensory weight death |
| 9 | Centered eligibility traces | 0% | Best Hebbian round (CI=-0.055) |
| 10 | W_hm column normalization | 0% | Magnitude norm can't prevent directional collapse |
| 11 | Action-specific eligibility | 0% | Update starvation on unchosen columns |
| **12** | **Surrogate gradient (replace Hebbian)** | 0% | Theta frozen (zero grad bug) |
| 12b | Theta gradient fix + entropy boost | 0% | Weight collapse from Adam weight_decay |
| 12c | Remove weight_decay, lower entropy | 0% | Unbounded weight growth |
| 12d | LR 0.1→0.001, alpha 10→1 | 0% | LR too conservative |
| **12e** | **LR 0.001→0.01** | **9%** | **First success ever, +CI** |
| **12f** | **Advantage clip + logit scale + exploration decay** | **44.6%** | **Best session 70%, surpasses SNN avg** |
| 12g | Replication of R12f | 16.6% | Extreme seed variance, catastrophic forgetting |
| **12h** | **LR decay (cosine annealing)** | **34.3%** | **Best session 76%, eliminated late collapse** |
| 12i | Orthogonal init + theta=pi/2 | 6.5% | Entropy collapse from theta=pi/2 |
| 12j | Orthogonal init + theta=0 | 0% | Symmetry trap from uniform column norms |
| 12k | Revert to randn + adaptive entropy | 29.9% | Adaptive bonus can't rescue failing seeds |
| **12l** | **Multi-timestep integration (10 steps)** | **41.8%** | **100% convergence, solved seed sensitivity** |
| 12m | Warm-start init (scale 0.3, theta=pi/4) | 37.3% | Scale 0.3 too aggressive |
| 12n | Stabilized warm-start (scale 0.15, clip 3.0) | 35.4% | Entropy rescue too weak |
| **12o** | **Stronger entropy rescue (20x) + extended exploration** | **73.9%** | **Matches SpikingReinforce, 100% convergence** |
| 12p | Lower LR floor (0.1→0.05) | 4.1% | Catastrophic regression, reverted |
| 12q | Reduced integration steps (10→5) | 52.6% | Noisier gradients, 50% convergence |

______________________________________________________________________

## Phase 1: Hebbian Learning (Rounds 0-11)

### The Problem

3-factor Hebbian learning (`delta_w = lr * pre_spike * post_spike * reward`) could not solve the foraging task after 12 rounds of intensive tuning. The fundamental issue was a tension between two failure modes:

1. **All-column updates** (Rounds 2-10): Updating all W_hm columns each step causes them to receive correlated updates and converge to the same direction, collapsing action diversity
2. **Chosen-column-only updates** (Round 11): Updating only the chosen action's column causes unchosen columns to atrophy, eventually starving the agent

### Key Findings

- **Best Hebbian result**: Round 9 achieved CI=-0.055 and 0.97 avg foods (1 successful episode across all sessions) using centered eligibility traces
- **Symmetry collapse** is inherent to Hebbian learning on this task: with 4 actions and outer-product eligibility traces, columns receive highly correlated updates
- **Weight dynamics** were extensively studied: L2 decay, column normalization, eligibility normalization, and exploration floors all failed to resolve the core issue

### Conclusion

Local Hebbian learning lacks the credit assignment precision needed for RL. Surrogate gradients (providing `d(loss)/d(weight)` per parameter) were required.

______________________________________________________________________

## Phase 2: Surrogate Gradient Foundation (Rounds 12-12e)

### The Approach

Replace Hebbian learning with REINFORCE policy gradient using `QLIFSurrogateSpike` — a custom `torch.autograd.Function` that:

- **Forward**: Returns the actual quantum-measured spike probability
- **Backward**: Uses sigmoid surrogate gradient centered at pi/2 (the RY gate transition point)

### Key Debugging Steps

| Round | Issue | Fix |
|-------|-------|-----|
| 12 | Theta parameters had zero gradient (not in autograd graph) | Connected theta through `ry_angle = theta + tanh(w*x)*pi` |
| 12b | Adam `weight_decay` drained parameters to zero | Removed weight_decay, lowered entropy_coef 0.05→0.02 |
| 12c | Unbounded weight growth (230-370% over 50 episodes) | Reduced LR 0.1→0.001 |
| 12d | LR=0.001 too conservative, weights frozen | Increased LR to 0.01 |
| **12e** | **LR=0.01 + alpha=1.0** | **First successful episodes (9% success, CI=+0.121)** |

### Why alpha=1.0 Matters

The surrogate gradient sharpness parameter controls gradient noise:

- `alpha=10.0`: Sharp sigmoid → large gradient magnitude → amplifies quantum shot noise
- `alpha=1.0`: Smooth sigmoid → moderate gradients → compatible with noisy quantum measurements

______________________________________________________________________

## Phase 3: Stabilization and Tuning (Rounds 12f-12o)

### Round 12f: Advantage Clipping + Logit Scaling (44.6%)

Three changes that together produced 44.6% average success (best session 70%):

- **Advantage clipping** ([-2, +2]): Prevents outlier returns from producing catastrophically large updates
- **Logit scaling** (20.0): Maps spike probability differences to meaningful action differentiation
- **Exploration decay** (30 episodes): Linearly decays epsilon and temperature for explore→exploit transition

### Round 12h: LR Decay (34.3%, best session 76%)

Cosine annealing LR decay (0.01→0.001 over 200 episodes) eliminated late-episode catastrophic forgetting. Without decay, converged policies were destroyed by continued large weight updates.

### Round 12l: Multi-Timestep Integration (41.8%, 100% convergence)

Averaging spike probabilities across 10 QLIF timesteps per decision reduced quantum shot noise variance by 10x. This solved the seed sensitivity problem: convergence rate went from 25-50% to 100%.

### Round 12o: Adaptive Entropy + Extended Exploration (73.9%)

Two changes that together achieved classical SNN parity:

- **ENTROPY_BOOST_MAX 5→20**: When entropy drops below 0.5 nats, entropy_coef scales up to 20x (effective_coef = 0.02 * 20 = 0.40), competitive with the REINFORCE gradient force (~1.0)
- **EXPLORATION_DECAY_EPISODES 30→80**: Keeps epsilon at 0.1 during the critical early window when premature policy commitment is most dangerous

______________________________________________________________________

## Phase 4: Validation (Rounds 12p-12q)

### Round 12p: LR Floor Sensitivity (4.1% - reverted)

Reducing LR_MIN_FACTOR from 0.1 to 0.05 caused catastrophic regression (3/4 sessions at 0%). Cosine annealing reaches sub-0.001 LR by episode ~100, so halving the floor starves gradient signal during mid-training refinement. LR_MIN_FACTOR=0.1 is at or near the minimum viable learning rate.

### Round 12q: Integration Steps Sensitivity (52.6%)

Reducing num_integration_steps from 10 to 5 showed clear regression (52.6% vs 73.9%, 50% convergence vs 100%). The noise reduction from averaging 10 timesteps is essential for stable REINFORCE training.

______________________________________________________________________

## Key Technical Decisions

### 1. QLIFSurrogateSpike Autograd Function

The hybrid quantum-classical training approach:

- Forward pass runs quantum QLIF circuits for spike probabilities (preserves quantum dynamics)
- Backward pass uses sigmoid surrogate gradient on the RY angle (avoids parameter-shift rule cost)
- Both theta (membrane bias) and weights receive gradients through `ry_angle = theta + tanh(w·x) * pi`

### 2. Two-Sided Entropy Regulation

```text
entropy < 0.5 nats → scale up entropy_coef (floor boost, up to 20x)
0.5 < entropy < 0.95*max → no adjustment
entropy > 0.95*max → scale down entropy_coef (ceiling suppression)
```

This prevents both failure modes: entropy collapse (policy becomes deterministic on one action) and entropy explosion (policy drifts to uniform random).

### 3. Multi-Timestep Integration

10 QLIF timesteps per decision with 1024 shots each = 10,240 effective samples. This reduces spike probability measurement variance by 10x, giving REINFORCE cleaner gradient signal. Classical SpikingReinforceBrain uses 100 timesteps with deterministic dynamics; QSNN needs fewer because each timestep already has 1024 measurement samples.

### 4. Weight Initialization

- **W_sh, W_hm**: `randn * 0.15` (breaks symmetry without amplifying gradient noise)
- **theta_hidden**: `pi/4` (warm start at ~60% gradient sensitivity)
- **theta_motor**: `0` (no initial action bias)

______________________________________________________________________

## Final Configuration

```yaml
brain:
  name: qsnn
  config:
    num_sensory_neurons: 6
    num_hidden_neurons: 8
    num_motor_neurons: 4
    membrane_tau: 0.9
    threshold: 0.5
    refractory_period: 0
    use_local_learning: false
    shots: 1024
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.02
    weight_clip: 3.0
    update_interval: 20
    num_integration_steps: 10
```

**Constants** (in `qsnn.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| WEIGHT_INIT_SCALE | 0.15 | Random Gaussian weight scale |
| MAX_ELIGIBILITY_NORM | 1.0 | Caps eligibility trace magnitude (Hebbian mode) |
| EXPLORATION_EPSILON | 0.1 | Uniform mixing floor for action probabilities |
| DEFAULT_SURROGATE_ALPHA | 1.0 | Sigmoid surrogate sharpness |
| SURROGATE_GRAD_CLIP | 1.0 | Max gradient norm |
| ADVANTAGE_CLIP | 2.0 | Caps normalized advantages |
| LOGIT_SCALE | 20.0 | Spike prob → action logit scaling |
| EXPLORATION_DECAY_EPISODES | 80 | Exploration schedule length |
| LR_DECAY_EPISODES | 200 | Cosine annealing LR schedule length |
| LR_MIN_FACTOR | 0.1 | Minimum LR = initial_lr * factor |
| DEFAULT_NUM_INTEGRATION_STEPS | 10 | QLIF timesteps per decision |
| ENTROPY_FLOOR | 0.5 | Adaptive entropy boost threshold (nats) |
| ENTROPY_BOOST_MAX | 20.0 | Max entropy_coef multiplier |
| ENTROPY_CEILING_FRACTION | 0.95 | Entropy suppression threshold (fraction of max) |

______________________________________________________________________

### Key Sessions

| Round | Sessions | Result |
|-------|----------|--------|
| R12e (first success) | 20260207_180812-180827 | 9% avg, first +CI |
| R12f (breakthrough) | 20260208_011057-011121 | 44.6% avg, best 70% |
| R12h (LR decay) | 20260208_073028-073037 | 34.3% avg, best 76% |
| R12l (multi-timestep) | 20260209_071045-071103 | 41.8% avg, 100% convergence |
| **R12o (foraging baseline)** | **20260208_234508-234524** | **73.9% avg, matches SNN** |
| R12p (LR floor test) | 20260209_040610-040628 | 4.1% avg (reverted) |
| R12q (integration steps test) | 20260209_080259-080317 | 52.6% avg |
