# 008 Appendix: QSNN-PPO Optimization History

This appendix documents the QSNN-PPO hybrid architecture evaluation across 4 rounds (16 sessions, 1,000 episodes). For main findings, see [008-quantum-brain-evaluation.md](../../008-quantum-brain-evaluation.md). For QSNN standalone predator history, see [qsnn-predator-optimization.md](qsnn-predator-optimization.md).

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Optimization Summary](#optimization-summary)
3. [Round PPO-0: Initial Architecture](#round-ppo-0-initial-architecture)
4. [Round PPO-1: Param Tuning + Code Fixes](#round-ppo-1-param-tuning--code-fixes)
5. [Round PPO-2: Break Policy Gradient Collapse](#round-ppo-2-break-policy-gradient-collapse)
6. [Round PPO-3: Fix Motor Spike Suppression](#round-ppo-3-fix-motor-spike-suppression)
7. [Root Cause: Architectural Incompatibility](#root-cause-architectural-incompatibility)
8. [Lessons Learned](#lessons-learned)
9. [Session References](#session-references)

______________________________________________________________________

## Architecture Overview

QSNN-PPO pairs the proven QSNN actor (QLIF circuits with surrogate gradients) with a classical MLP critic and PPO training algorithm. This was designed to address the three diagnosed root causes of standalone QSNN's predator failure: no value function, high REINFORCE variance, and insufficient gradient passes.

```text
QSNN Actor (212 params)          Classical Critic (5,569 params)
8 sensory → 16 hidden → 4 motor  Input: 8 sensory + 16 hidden spikes = 24-dim
QLIF circuits, surrogate grad    Linear(24,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)

Training: PPO with quantum caching
1. Collect rollout buffer (256–512 steps)
2. Compute GAE advantages using critic V(s)
3. For each of 2–4 epochs:
   a. Epoch 0: run quantum circuits, cache spike probs
   b. Epochs 1+: reuse cached probs, recompute ry_angles
   c. PPO clipped surrogate loss on actor
   d. Huber loss on critic
4. Clear buffer, repeat
```

**Task**: Pursuit predators (count 2, speed 0.5, detection_radius 6), health system (max_hp 100, predator_damage 20, food_healing 10), 20x20 grid. Classical MLP PPO baseline: ~93.5% success.

______________________________________________________________________

## Optimization Summary

| Round | Key Changes | Success | Avg Food | Evasion | PPO Updates | Key Finding |
|-------|-----------|---------|----------|---------|-------------|-------------|
| PPO-0 | Initial architecture | 0% | 0.52 | 43.5% | 200/sess | Buffer never fills; entropy collapse; theta_h collapse |
| PPO-1 | Cross-ep buffer, entropy=0.08, LR=0.003 | 0% | 0.56 | 35.1% | 77/sess | **100% policy_loss=0** — PPO completely inert |
| PPO-2 | logit_scale=20, entropy decay, motor init | 0% | 0.48 | 25.1% | ~20/sess | Motor probs at 0.02 not 0.5 — wrong hypothesis |
| PPO-3 | theta_h=pi/2, theta_m=linspace(pi/4,3pi/4) | 0% | 0.77 | 42.5% | ~20/sess | Motor probs fixed but **policy_loss still 0** |

**Outcome**: HALTED after 4 rounds. PPO fundamentally incompatible with surrogate gradient architecture.

______________________________________________________________________

## Round PPO-0: Initial Architecture

**Config**: `qsnnppo_pursuit_predators_small.yml`
**Sessions**: 20260215_040128, 20260215_040143, 20260215_040149, 20260215_040155
**Commit**: `3ec5790`
**Episodes**: 200 per session

### Initial Config

```yaml
brain:
  config:
    num_sensory_neurons: 8
    num_hidden_neurons: 16
    num_motor_neurons: 4
    shots: 1024
    num_integration_steps: 10
    logit_scale: 5.0
    weight_clip: 3.0
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    entropy_coef: 0.02
    num_epochs: 4
    num_minibatches: 4
    rollout_buffer_size: 512
    actor_lr: 0.01
    critic_lr: 0.001
    actor_weight_decay: 0.001
```

### Results

| Session | Success | Avg Food | Max Food | Avg Steps | Avg Reward | Evasion Rate | Converged |
|---------|---------|----------|----------|-----------|------------|-------------|-----------|
| 040128 | 0% | 0.57 | 5 | 112.8 | -13.67 | 39.7% | No |
| 040143 | 0% | 0.45 | 6 | 110.4 | -13.26 | 44.6% | No |
| 040149 | 0% | 0.81 | 6 | 124.0 | -12.73 | 48.3% | No |
| 040155 | 0% | 0.25 | 5 | 98.2 | -13.72 | 41.6% | No |
| **Avg** | **0%** | **0.52** | **6** | **111.3** | **-13.34** | **43.5%** | **No** |

**Termination**: ~96% health_depleted, ~3% starved, \<1% max_steps

### Training Metrics Trajectory (Session 040128, representative)

| Episode | policy_loss | value_loss | entropy | W_sh_norm | theta_h_norm | actor_lr |
|---------|------------|------------|---------|-----------|-------------|----------|
| 0 | 0.2091 | 1.4229 | 1.3861 | 1.767 | 3.016 | 0.0100 |
| 10 | 0.0525 | 0.6579 | 0.6769 | 2.552 | 2.755 | 0.0100 |
| 50 | -0.0921 | 1.4181 | 0.4171 | 3.309 | 0.718 | 0.0097 |
| 100 | 0.0000 | 1.6764 | 1.2438 | 1.049 | 0.373 | 0.0088 |
| 150 | 0.1579 | 1.0995 | 0.6749 | 2.272 | 0.877 | 0.0075 |
| 199 | -0.0000 | 0.7933 | 1.0563 | 1.243 | 0.333 | 0.0060 |

### Root Cause Analysis — 5 Critical Issues

**1. Buffer Never Fills (Critical)**

`rollout_buffer_size=512` never reached. Average episode length ~111 steps → PPO updates trigger at episode end with only ~111 samples (22% of buffer). Each minibatch has ~28 samples instead of intended ~128. This makes advantage estimates noisy and PPO multi-epoch updates overfit to tiny datasets.

Evidence: 200 PPO updates across 200 episodes = exactly 1 per episode. No buffer-full triggers.

**2. Entropy Collapse-Recovery Cycles (Critical)**

Entropy oscillates between near-maximum (~1.38 = uniform) and near-collapsed (~0.25-0.42 = deterministic). `entropy_coef=0.02` is too weak to prevent collapse but strong enough to force recovery, creating destructive cycles. During collapse, one action gets 75-91% probability.

**3. theta_hidden Collapse (Critical)**

`theta_hidden` decays from initial ~3.0 to as low as 0.14 across training, eliminating hidden layer representational capacity. QLIF neurons with theta near zero produce ~0.5 spike probability regardless of input.

**4. Policy Loss Frequently Zero (High)**

50%+ of PPO updates produce `policy_loss=0.0000`, meaning PPO ratio `exp(new_log_prob - old_log_prob)` equals ~1.0. The actor receives no training signal for half its updates.

**5. Weight Oscillation (High)**

Weight norms swing dramatically (W_sh: 0.83 to 4.50) with no convergence. `actor_lr=0.01` is too aggressive for 212-parameter quantum network.

### Recommendations Applied to PPO-1

| Priority | Change | From | To |
|----------|--------|------|-----|
| 1 | Cross-episode buffer | Reset per episode | Persist until full |
| 2 | entropy_coef | 0.02 | 0.08 |
| 3 | actor_lr | 0.01 | 0.003 |
| 4 | theta_hidden_min_norm | None | 0.5 |
| 5 | rollout_buffer_size | 512 | 256 |
| 6 | num_epochs | 4 | 2 |

______________________________________________________________________

## Round PPO-1: Param Tuning + Code Fixes

**Sessions**: 20260215_063301, 20260215_063308, 20260215_063313, 20260215_063319
**Episodes**: 200 per session

### Changes Applied (vs PPO-0)

| Change | Category | From | To |
|--------|----------|------|-----|
| Cross-episode buffer | code | Reset per episode | Persist until 256 full |
| entropy_coef | param | 0.02 | 0.08 |
| actor_lr | param | 0.01 | 0.003 |
| rollout_buffer_size | param | 512 | 256 |
| num_epochs | param | 4 | 2 |
| theta_hidden_min_norm | code | None | 0.5 |
| Enhanced PPO logging | code | debug, basic | info, comprehensive |

### Results

| Session | Success | Avg Food | Max Food | Avg Steps | Avg Reward | Evasion Rate | PPO Updates | Converged |
|---------|---------|----------|----------|-----------|------------|-------------|-------------|-----------|
| 063301 | 0% | 0.56 | 4 | 83.7 | -12.80 | 27.5% | 65 | No |
| 063308 | 0% | 0.46 | 7 | 88.7 | -13.19 | 37.7% | 69 | No |
| 063313 | 0% | 0.66 | 5 | 113.2 | -12.02 | 32.9% | 88 | No |
| 063319 | 0% | 0.58 | 4 | 112.4 | -12.40 | 42.1% | 87 | No |
| **Avg** | **0%** | **0.56** | **5** | **99.5** | **-12.60** | **35.1%** | **77** | **No** |

### What Improved vs PPO-0

| Issue | PPO-0 | PPO-1 | Status |
|-------|-------|-------|--------|
| Buffer never fills | Only 22% capacity used | Always full at 256 | **FIXED** |
| Entropy collapse-recovery | Oscillates 0.22-1.38 | Stable at 1.35-1.39 | **FIXED** (over-corrected) |
| theta_hidden collapse to 0.14 | Reached 0.14-0.17 | Stabilized at 1.37-1.55 | **FIXED** |
| Weight oscillation | Chaotic W_sh: 0.83-4.50 | Smooth W_sh: 1.2-1.8 | **FIXED** |
| Policy loss frequently zero | ~50% of updates | **100% of updates** | **WORSE** |

### Root Cause: Total Policy Gradient Collapse (NEW — Critical)

**100% of PPO updates across all 4 sessions have policy_loss=0.0000 and clip_frac=0.000.**

The PPO surrogate objective produces zero gradient in every single update (309/309 updates). The actor receives **no training signal** from the PPO objective at all. Only gradient comes from entropy bonus.

**Mechanism**: QLIF circuits produce motor spike probabilities near 0.5. After `logit_scale * (p - 0.5)`, logits are near 0. After softmax, action probs are near 0.25 (uniform). When the same near-uniform distribution is recomputed during PPO update, `old_log_prob ≈ new_log_prob ≈ -1.386`, so PPO ratio is 1.0 everywhere.

**The core dilemma**: entropy_coef=0.02 causes destructive collapse/recovery cycles. entropy_coef=0.08 locks policy at uniform and kills all policy gradient. Neither produces learning.

______________________________________________________________________

## Round PPO-2: Break Policy Gradient Collapse

**Sessions**: 20260215_082646, 20260215_082651, 20260215_082657, 20260215_082702
**Commit**: `8bb66ca`
**Episodes**: 50 per session (reduced for faster iteration)

### Changes Applied (vs PPO-1)

| Change | Category | From | To | Rationale |
|--------|----------|------|----|-----------|
| logit_scale | param | 5.0 | 20.0 | Amplify spike prob differences |
| entropy_coef decay | code | 0.08 fixed | 0.05 → 0.005 over 100 eps | Allow exploration then specialization |
| actor_weight_decay | param | 0.001 | 0.0 | Stop theta erosion without policy gradient |
| theta_hidden_min_norm | param | 0.5 | 2.0 | Higher floor for hidden capacity |
| theta_motor init | code | zeros | linspace(-0.3, 0.3) | Break motor neuron symmetry |
| motor spike prob logging | code | not logged | per-update mean | Confirm spike prob hypothesis |

### Results

| Session | Success | Avg Food | Avg Steps | Avg Reward | Evasion Rate | Converged |
|---------|---------|----------|-----------|------------|-------------|-----------|
| 082646 | 0% | 0.56 | 79.2 | -13.41 | 31.5% | No |
| 082651 | 0% | 0.32 | 87.6 | -14.14 | 19.7% | No |
| 082657 | 0% | 0.52 | 93.1 | -12.95 | 32.4% | No |
| 082702 | 0% | 0.50 | 90.0 | -13.49 | 16.9% | No |
| **Avg** | **0%** | **0.48** | **87.5** | **-13.50** | **25.1%** | **No** |

### Critical Discovery: Motor Spike Probabilities

**The PPO-1 analysis hypothesised that motor spike probs were "near 0.5" creating a logit bottleneck. PPO-2's new motor spike logging reveals this was WRONG.**

Motor spike probabilities are **near 0.02-0.04** (2-4%), not 0.5. The QLIF motor neurons are barely firing at all.

**Root cause**: With `theta_motor` at `linspace(-0.3, 0.3)` and small weighted_input, `ry_angle ≈ 0.3 radians`, giving `sin²(0.15) ≈ 0.022`. For spike probability to reach 0.5, `ry_angle ≈ pi/2 ≈ 1.57 rad` — the current architecture produces ry_angles 5x too small.

**logit_scale=20.0 made things worse**: With motor probs at 0.03, `logits = (0.03 - 0.5) * 20 = -9.4`. All 4 logits in extreme negative saturation where softmax gradients vanish.

### Comparison vs PPO-1

| Metric | PPO-0 | PPO-1 | PPO-2 | Trend |
|--------|-------|-------|-------|-------|
| Success rate | 0% | 0% | 0% | No change |
| Avg food | 0.52 | 0.56 | 0.48 | Declining |
| Avg reward | -13.34 | -12.60 | -13.50 | **Regressed** |
| Avg steps | 111.3 | 99.5 | 87.5 | **Declining** |
| Evasion rate | 43.5% | 35.1% | 25.1% | **Severe regression** |

______________________________________________________________________

## Round PPO-3: Fix Motor Spike Suppression

**Sessions**: 20260215_085929, 20260215_085937, 20260215_085944, 20260215_085951
**Commit**: `024baac`
**Episodes**: 50 per session

### Changes Applied (vs PPO-2)

| Change | Category | From | To | Rationale |
|--------|----------|------|----|-----------|
| theta_hidden init | code | `torch.full(pi/4)` | `torch.full(pi/2)` | Max sensitivity spike prob 0.5 |
| theta_motor init | code | `linspace(-0.3, 0.3)` | `linspace(pi/4, 3*pi/4)` | Responsive spike prob range 0.15-0.85 |
| logit_scale | param | 20.0 | 5.0 | Revert: scale=20 caused extreme saturation |
| theta_motor_max_norm | param | 2.0 | 5.0 | Accommodate larger theta_motor init |
| entropy_decay_episodes | param | 100 | 200 | Slower decay for early exploration |

### Results

| Session | Success | Avg Food | Avg Steps | Avg Reward | Evasion Rate | Converged |
|---------|---------|----------|-----------|------------|-------------|-----------|
| 085929 | 0% | 1.14 | 106.5 | -12.48 | 43.2% | No |
| 085937 | 0% | 0.80 | 117.0 | -12.25 | 41.9% | No |
| 085944 | 0% | 0.74 | 90.8 | -13.59 | 51.0% | No |
| 085951 | 0% | 0.40 | 108.9 | -12.80 | 33.8% | No |
| **Avg** | **0%** | **0.77** | **105.8** | **-12.78** | **42.5%** | **No** |

**Termination**: 197/200 health_depleted (98.5%), 2 starved (1%), 1 max_steps (0.5%)

### Expected vs Actual Diagnostics

| Diagnostic | Expected | Actual | Status |
|------------|----------|--------|--------|
| Motor spike probs in 0.15-0.85 | Move from 0.02 to responsive range | **0.15-0.91 from first update** | **FIXED** |
| policy_loss > 0 | Non-zero policy gradient | **0.0000 in 100% of updates** | **FAILED** |
| clip_frac > 0 | PPO clipping active | **0.000 in 100% of updates** | **FAILED** |
| Entropy healthy | Maintained in useful range | **0.68-1.37**, around 1.20 | **PASSED** |
| Weight norms growing | Gradients flowing | **W_sh +6-23%, W_hm +12-17%** | **PASSED** |

### What PPO-3 Fixed

1. **Motor spike suppression completely resolved.** Motor neurons now in 0.15-0.91 spike probability range from first update.
2. **Evasion rate recovered** from 25.1% (PPO-2) to 42.5% — back to PPO-0 baseline.
3. **Late-episode learning trends** in 3/4 sessions. Session 085951 Episode 50: **7 food, 500 steps (max_steps!), reward +31.31** — best single episode across all QSNN-PPO rounds.
4. **Gradients are flowing** through all parameter groups. Weight matrices grew 6-23%.

### What PPO-3 Did NOT Fix

**policy_loss = 0.0000 in every single PPO update across all 4 sessions.** This led to the identification of the fundamental architectural incompatibility.

______________________________________________________________________

## Root Cause: Architectural Incompatibility

After 4 rounds (16 sessions, 1,000 episodes), the conclusion is definitive:

### How QLIFSurrogateSpike Works

```python
class QLIFSurrogateSpike(torch.autograd.Function):
    def forward(ctx, ry_angle, quantum_spike_prob, alpha):
        ctx.save_for_backward(ry_angle, ...)
        return torch.tensor(quantum_spike_prob)  # <-- CONSTANT in forward

    def backward(ctx, grad_output):
        ry_angle, alpha = ctx.saved_tensors
        shifted = alpha * (ry_angle - pi/2)
        grad_surrogate = alpha * sigmoid(shifted) * (1 - sigmoid(shifted))
        return grad_output * grad_surrogate, None, None
```

- **Forward pass**: Returns quantum-measured spike probability (a Python float wrapped in a tensor). This is a **constant** that does not change when weights/theta change.
- **Backward pass**: Approximates `d(spike_prob)/d(ry_angle)` using sigmoid surrogate. Parameter-dependent.

### Why PPO Fails

PPO computes `ratio = exp(log_pi_new - log_pi_old)`. During PPO re-evaluation, the forward pass runs `QLIFSurrogateSpike.forward()` which returns the **cached quantum_spike_prob** — same value regardless of weight updates:

- `pi_new(a|s) == pi_old(a|s)` always
- `ratio == 1.0` always
- `policy_loss == 0.0` always

### Why REINFORCE Works

REINFORCE only needs `d(log_prob)/d(theta)` — the backward pass gradient — which the surrogate provides correctly. It never re-evaluates the forward pass to compare old vs new policies.

### Why Modest Learning Occurred Despite Zero Policy Loss

The combined actor loss is `policy_loss - entropy_coef * entropy`. With `policy_loss = 0`, the entire actor gradient comes from the **entropy bonus**, which IS differentiable through the surrogate backward pass. This creates a weak form of exploration-driven learning — enough for late-episode food improvements but far too slow for convergence.

______________________________________________________________________

## Lessons Learned

| Lesson | Detail |
|--------|--------|
| Surrogate gradients are backward-only | The forward pass is a stochastic quantum measurement, not a differentiable function. Fine for REINFORCE but breaks importance sampling (PPO/TRPO). |
| Motor spike prob logging was invaluable | Added in PPO-2, it corrected the wrong hypothesis (probs at 0.02, not 0.5) and confirmed PPO-3's fix (probs at 0.15-0.85). |
| Theta init dominates early dynamics | With small weights (WEIGHT_INIT_SCALE=0.15), theta is the dominant component of ry_angle. Init near pi/2 is critical. |
| Entropy gradient provides weak learning | Even without policy gradient, entropy regularisation can drive exploration-based improvement, but far too slowly for practical convergence. |
| PPO infrastructure is reusable | The critic MLP, advantage estimation, and training loop code directly informed the QSNNReinforce A2C implementation. |
| Cross-episode buffer essential | Buffer-per-episode dramatically underutilises data. Accumulating across episodes is standard and necessary. |

______________________________________________________________________

## Cross-Round Performance Trajectory

| Round | Success | Avg Food | Avg Steps | Evasion | Key Delta |
|-------|---------|----------|-----------|---------|-----------|
| PPO-0 | 0% | 0.52 | 111.3 | 43.5% | Baseline |
| PPO-1 | 0% | 0.56 | 99.5 | 35.1% | Policy_loss 100% zero (was 50%) |
| PPO-2 | 0% | 0.48 | 87.5 | 25.1% | Worst round; wrong logit_scale |
| PPO-3 | 0% | 0.77 | 105.8 | 42.5% | Motor probs fixed; root cause identified |

______________________________________________________________________

## Session References

| Round | Sessions | Episodes | Commit | Result |
|-------|----------|----------|--------|--------|
| PPO-0 | 20260215_040128, 040143, 040149, 040155 | 200 | `3ec5790` | 0%, buffer never fills, entropy collapse |
| PPO-1 | 20260215_063301, 063308, 063313, 063319 | 200 | — | 0%, 100% policy_loss=0 |
| PPO-2 | 20260215_082646, 082651, 082657, 082702 | 50 | `8bb66ca` | 0%, motor probs 0.02, logit_scale backfired |
| PPO-3 | 20260215_085929, 085937, 085944, 085951 | 50 | `024baac` | 0%, motor probs fixed, root cause confirmed |

### Best Single Episode

Session 085951, Episode 50: 7 food, 500 steps (max_steps reached), reward +31.31. Driven by entropy gradient exploration, not policy learning.

### Final Config State

```yaml
brain:
  name: qsnnppo
  config:
    num_sensory_neurons: 8
    num_hidden_neurons: 16
    num_motor_neurons: 4
    shots: 1024
    num_integration_steps: 10
    logit_scale: 5.0
    weight_clip: 3.0
    theta_motor_max_norm: 5.0
    theta_hidden_min_norm: 2.0
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    entropy_coef: 0.05
    entropy_coef_end: 0.005
    entropy_decay_episodes: 200
    num_epochs: 2
    num_minibatches: 4
    rollout_buffer_size: 256
    actor_lr: 0.003
    critic_lr: 0.001
    actor_weight_decay: 0.0
    critic_hidden_dim: 64
    critic_num_layers: 2
    sensory_modules:
      - food_chemotaxis
      - nociception
```
