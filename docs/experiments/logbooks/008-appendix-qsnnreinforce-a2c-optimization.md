# 008 Appendix: QSNNReinforce A2C Optimization History

This appendix documents the QSNNReinforce A2C evaluation across 4 rounds (16 sessions, 3,200 episodes). For main findings, see [008-quantum-brain-evaluation.md](008-quantum-brain-evaluation.md). For QSNN standalone predator history, see [008-appendix-qsnn-predator-optimization.md](008-appendix-qsnn-predator-optimization.md). For QSNN-PPO history, see [008-appendix-qsnnppo-optimization.md](008-appendix-qsnnppo-optimization.md).

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Optimization Summary](#optimization-summary)
3. [Round A2C-0: Initial A2C](#round-a2c-0-initial-a2c)
4. [Round A2C-1: More Data + Simplified Critic](#round-a2c-1-more-data--simplified-critic)
5. [Round A2C-2: Bug Fixes](#round-a2c-2-bug-fixes)
6. [Round A2C-3: Sensory-Only Critic Input](#round-a2c-3-sensory-only-critic-input)
7. [Root Cause: Fundamental Environment Mismatch](#root-cause-fundamental-environment-mismatch)
8. [Lessons Learned](#lessons-learned)
9. [Session References](#session-references)

______________________________________________________________________

## Architecture Overview

QSNNReinforce A2C adds a classical MLP critic to the existing QSNNReinforce REINFORCE backbone. This was the pivot after QSNN-PPO failed — A2C preserves REINFORCE's reliance on backward-pass gradients only (compatible with surrogate gradients) while adding GAE advantage estimation via a learned value function.

```text
QSNN Actor (212 params, unchanged)     Classical MLP Critic (353–5,569 params)
8 sensory → 16 hidden → 4 motor QLIF   Input: sensory features ± hidden spike rates
Surrogate gradient REINFORCE backbone  Huber loss, separate Adam optimizer

Training loop (per 20-step window):
1. Compute GAE advantages using critic V(s) and bootstrap V(s')
2. For each of 2 REINFORCE epochs:
   a. Epoch 0: run quantum circuits, cache spike probs
   b. Epoch 1: reuse cached probs, recompute ry_angles
   c. REINFORCE loss = -(log_probs × GAE_advantages) - entropy_coef × entropy
3. Train critic on same window (critic_epochs gradient steps)
4. Log diagnostics (value_loss, explained_variance, etc.)
```

**Task**: Pursuit predators (count 2, speed 0.5, detection_radius 6), health system (max_hp 100, predator_damage 20, food_healing 10), 20x20 grid. Classical MLP PPO baseline: ~93.5% success.

______________________________________________________________________

## Optimization Summary

| Round | Key Changes | Success | Avg Food | Q4 Food | EV (Q4) | Critic Updates | Key Finding |
|-------|-----------|---------|----------|---------|---------|---------------|-------------|
| A2C-0 | Initial A2C (50 eps) | 0% | 0.62 | — | ~0 | 246 | Critic not learning; EV oscillates near zero |
| A2C-1 | 200 eps, smaller critic, lower LR | 0.13% | 1.52 | 2.05 | -0.008 | 898 | Actor improves via REINFORCE; critic still fails; 4 bugs found |
| A2C-2 | Bug fixes (multi-step, bootstrap, clip, EV) | 0.63% | 1.32 | 2.28 | -0.295 | ~900 | Bugs fixed but EV **worse**; overfitting-crash cycle |
| A2C-3 | Sensory-only critic input | 0.50% | 1.50 | 2.05 | -0.620 | 954 | Non-stationarity hypothesis disproved; A2C abandoned |

**Outcome**: HALTED after 4 rounds. The A2C critic cannot learn V(s) in this environment due to partial observability, policy non-stationarity, high return variance, and short GAE windows. All actor improvement driven by REINFORCE backbone.

______________________________________________________________________

## Round A2C-0: Initial A2C

**Sessions**: 20260215_103816, 20260215_103823, 20260215_103830, 20260215_103835
**Episodes**: 50 per session
**Baseline comparison**: PP8 (vanilla REINFORCE, best pursuit predator round)

### Config

```yaml
brain:
  name: qsnnreinforce
  config:
    num_sensory_neurons: 8
    num_hidden_neurons: 16
    num_motor_neurons: 4
    shots: 1024
    num_integration_steps: 10
    logit_scale: 5.0
    weight_clip: 3.0
    theta_motor_max_norm: 2.0
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.02
    update_interval: 20
    num_reinforce_epochs: 2
    advantage_clip: 2.0
    use_critic: true
    critic_hidden_dim: 64
    critic_num_layers: 2
    critic_lr: 0.001
    gae_lambda: 0.95
    value_loss_coef: 0.5
```

### Results

| Metric | Session 103816 | Session 103823 | Session 103830 | Session 103835 | **A2C-0 Avg** | **PP8 Avg** |
|--------|---|---|---|---|---|---|
| Success Rate | 0% | 0% | 0% | 0% | **0%** | **1.25%** |
| Avg Steps | 105.9 | 76.7 | 91.2 | 86.6 | **90.1** | **100.9** |
| Avg Reward | -11.37 | -13.05 | -11.64 | -13.37 | **-12.36** | **-11.55** |
| Avg Foods | 0.86 | 0.30 | 0.86 | 0.46 | **0.62** | **1.11** |
| Avg Pred Encounters | 1.24 | 1.38 | 1.38 | 1.48 | **1.37** | **1.55** |
| Avg Evasions | 0.28 | 0.38 | 0.38 | 0.50 | **0.39** | **0.56** |
| Health Depleted | 48 | 50 | 50 | 49 | **98.5%** | **94-98%** |
| Critic Updates | 288 | 212 | 249 | 236 | **246** | **—** |

**Termination**: ~98.5% health_depleted, ~1.5% starved.

### Key Findings

1. **A2C is worse than vanilla REINFORCE (PP8)** across every metric: food collection 44% worse (0.62 vs 1.11), evasion rate 28.5% vs 36.1%, dies 10% faster.

2. **Critic not learning**: `explained_variance` oscillates between -0.5 and +0.3 with no upward trend. One extreme outlier at -374.8 (numerical instability). EV≈0 means critic predictions are random noise.

3. **Short episodes starve the critic**: Median episode ~50-70 steps. With update_interval=20, the critic gets only 1-3 updates per episode. Total: ~250 updates across 50 episodes.

4. **Critic noise degrades advantages**: With EV≈0, GAE advantages are noisier than normalized-returns REINFORCE, actively hurting the policy gradient.

### Recommendations Applied to A2C-1

- Increase episodes to 200 (4x more critic training time)
- Reduce critic: 32 hidden, 1 layer (1.1K params vs 5.4K)
- Lower critic LR: 0.0003 (reduce oscillation)
- Lower GAE lambda: 0.8 (lower variance for short episodes)

______________________________________________________________________

## Round A2C-1: More Data + Simplified Critic

**Sessions**: 20260215_121727, 20260215_121735, 20260215_121742, 20260215_121748
**Episodes**: 200 per session

### Changes (vs A2C-0)

| Parameter | A2C-0 | A2C-1 | Rationale |
|-----------|-------|-------|-----------|
| Episodes per session | 50 | 200 | 4x more critic training time |
| critic_hidden_dim | 64 | 32 | Reduce capacity to match data |
| critic_num_layers | 2 | 1 | 5.4K → 1.1K params |
| critic_lr | 0.001 | 0.0003 | Reduce oscillation with noisy features |
| gae_lambda | 0.95 | 0.8 | Lower variance for short episodes |

### Results

| Metric | Session 121727 | Session 121735 | Session 121742 | Session 121748 | **A2C-1 Avg** | **A2C-0 Avg** |
|--------|---|---|---|---|---|---|
| Success Rate | 0.5% | 0% | 0% | 0% | **0.13%** | **0%** |
| Avg Steps | 88.9 | 75.2 | 83.8 | 77.7 | **81.4** | **90.1** |
| Avg Reward | -9.74 | -10.88 | -10.13 | -8.97 | **-9.93** | **-12.36** |
| Avg Foods | 1.58 | 1.29 | 1.45 | 1.78 | **1.52** | **0.62** |
| Critic Updates | 966 | 842 | 920 | 862 | **898** | **246** |

**Termination**: ~99.6% health_depleted, one success (session 121727, run 142: 10 foods, 424 steps, +30.1 reward).

### Explained Variance by Phase (Cross-Session)

| Phase | 121727 | 121735 | 121742 | 121748 | **Avg** |
|-------|--------|--------|--------|--------|---------|
| Ep 0-49 | -0.032 | -0.145 | -0.046 | -0.040 | **-0.066** |
| Ep 50-99 | -0.007 | -0.088 | -0.030 | -0.025 | **-0.037** |
| Ep 100-149 | +0.016 | -0.030 | -0.016 | -0.008 | **-0.010** |
| Ep 150-199 | -0.004 | +0.001 | -0.004 | -0.024 | **-0.008** |

EV improves from -0.066 (early) to -0.008 (late) — marginal, never breaching positive. Only 2-7% of individual updates exceed EV > 0.2.

### Actor Learning (Independent of Critic)

| Quartile | Avg Foods | Zero-Food % | Avg Reward |
|----------|----------|------------|------------|
| Q1 (ep 1-50) | 0.79 | 49% | -12.15 |
| Q2 (ep 51-100) | 1.48 | 27% | -10.04 |
| Q3 (ep 101-150) | 1.78 | 24% | -8.85 |
| Q4 (ep 151-200) | 2.05 | 19% | -8.57 |

Food collection improved 2.6x from Q1 to Q4 — driven entirely by the REINFORCE backbone with exploration decay, not by A2C advantage estimation.

### Key Findings

1. **Critic still fails** despite 4x more data (898 vs 246 updates) and reduced capacity (1.1K vs 5.4K params). Not a data or capacity problem.

2. **Actor improved substantially** (food: 0.62→1.52, reward: -12.36→-9.93) — driven by REINFORCE with 4x more training episodes.

3. **Code review identified 4 bugs** explaining critic failure:

   - **Bug 1 (Critical)**: Single gradient step per window (standard A2C uses 3-10)
   - **Bug 2 (Critical)**: Chimeric bootstrap input (features from state N+1, hidden spikes from state N)
   - **Bug 3 (Moderate)**: Shared gradient clip (actor's 1.0 norm applied to critic, reducing effective LR 5-10x)
   - **Bug 4 (Minor)**: Pre-update explained_variance (1-step diagnostic lag)

### Recommendations Applied to A2C-2

Fix all 4 bugs, revert capacity and LR to original values.

______________________________________________________________________

## Round A2C-2: Bug Fixes

**Sessions**: 20260215_135006, 20260215_135012, 20260215_135018, 20260215_135025
**Episodes**: 200 per session

### Changes (vs A2C-1)

| Parameter | A2C-1 | A2C-2 | Rationale |
|-----------|-------|-------|-----------|
| critic_hidden_dim | 32 | 64 | Revert — capacity wasn't the problem |
| critic_num_layers | 1 | 2 | Revert — capacity wasn't the problem |
| critic_lr | 0.0003 | 0.001 | Revert — was unnecessarily conservative |
| critic_epochs | 1 (implicit) | 5 | **Fix 1**: multi-step critic training |
| critic_grad_clip | 1.0 (shared) | 5.0 | **Fix 3**: separate critic gradient clip |
| Bootstrap ordering | features N+1, spikes N | features N+1, spikes N+1 | **Fix 2**: forward pass before bootstrap |
| Explained variance | Pre-update | Post-update | **Fix 4**: measure after optimizer step |

### Results

| Metric | Session 135006 | Session 135012 | Session 135018 | Session 135025 | **A2C-2 Avg** | **A2C-1 Avg** |
|--------|---|---|---|---|---|---|
| Success Rate | 0% | 0.5% | 0% | 2% | **0.63%** | **0.13%** |
| Avg Steps | 93.6 | 80.8 | 81.4 | 98.3 | **88.5** | **81.4** |
| Avg Reward | -13.39 | -9.61 | -9.65 | -10.01 | **-10.67** | **-9.93** |
| Avg Foods | 0.44 | 1.58 | 1.68 | 1.56 | **1.32** | **1.52** |
| Health Depleted | 196 | 199 | 199 | 193 | **98.4%** | **99.6%** |

**Termination**: 98.4% health_depleted, 5 successes across 800 episodes (0.63%). Session 135025 produced 4 of the 5 successes (episodes 154, 164, 167, 190).

### Explained Variance by Phase (Cross-Session)

| Phase | 135006 | 135012 | 135018 | 135025 | **A2C-2 Avg** | **A2C-1 Avg** |
|-------|--------|--------|--------|--------|--------------|--------------|
| Ep 0-49 | -0.220 | -0.607 | +0.003 | -0.074 | **-0.225** | **-0.066** |
| Ep 50-99 | -0.042 | -0.597 | -0.129 | -0.098 | **-0.217** | **-0.037** |
| Ep 100-149 | -0.375 | -0.173 | -0.107 | -0.503 | **-0.290** | **-0.010** |
| Ep 150-199 | -0.383 | -0.571 | -0.136 | -0.088 | **-0.295** | **-0.008** |

EV is **negative in every phase** and **trends downward** (-0.225 → -0.295) — the opposite of expected. A2C-2's Q4 EV (-0.295) is 37x worse than A2C-1's (-0.008).

### Key Findings

1. **Bug fixes made the critic worse, not better.** Multi-step training (5 epochs) on 20-step windows causes overfitting: the critic fits one window well, then crashes on the next. Value loss spikes escalate in Q4 (sessions 135018 and 135025).

2. **Session 135006: catastrophic regression.** Only session where food collection *declines* over training (Q1=0.84 → Q4=0.12, 7x worse). The critic's noisy advantages actively corrupted the actor gradient — a new failure mode.

3. **Q4 food collection improved** in non-outlier sessions: 3.00 avg (excluding 135006) vs A2C-1's 2.05 (+46%). Session 135025's Q4 of 4.04 foods is the best quartile in the entire A2C series.

4. **Best episode**: Session 135025, run 167 — 10 foods, 265 steps, reward +31.65 (highest in any A2C round).

### Root Cause: Architectural Mismatch

The root cause is not implementation bugs but a fundamental architectural mismatch:

- Non-stationary critic inputs (hidden spikes shift as W_sh grows 4-8x)
- 20-step windows too short for meaningful GAE with gamma=0.99
- Overfitting-crash cycle from multi-step training
- Critic-actor interference when EV < 0

### Recommendation Applied to A2C-3

**Option 1: Sensory-Only Critic Input** — drop hidden spikes (the non-stationary component) from critic input. If EV improves with stationary 8-dim input, non-stationarity was the root cause. If not, abandon A2C.

______________________________________________________________________

## Round A2C-3: Sensory-Only Critic Input

**Sessions**: 20260215_221154, 20260215_221202, 20260215_221207, 20260215_221213
**Episodes**: 200 per session

### Changes (vs A2C-2)

| Parameter | A2C-2 | A2C-3 | Rationale |
|-----------|-------|-------|-----------|
| critic_use_hidden_spikes | true (implicit) | **false** | Drop non-stationary hidden spike features |
| critic_hidden_dim | 64 | 32 | Smaller input (8-dim) needs less capacity |
| critic_num_layers | 2 | 1 | Match reduced input dimensionality |

The critic now sees only the 8 raw sensory features (4 food_chemotaxis + 4 nociception) — stationary inputs not affected by actor weight changes.

### Results

| Metric | Session 221154 | Session 221202 | Session 221207 | Session 221213 | **A2C-3 Avg** | **A2C-2 Avg** |
|--------|---|---|---|---|---|---|
| Success Rate | 0% | 2% | 0% | 0% | **0.50%** | **0.63%** |
| Avg Steps | 88.1 | 83.9 | 82.6 | 92.6 | **86.8** | **88.5** |
| Avg Reward | -10.01 | -9.77 | -10.89 | -9.55 | **-10.06** | **-10.67** |
| Avg Foods | 1.57 | 1.55 | 1.25 | 1.61 | **1.50** | **1.32** |
| Critic Updates | 967 | 923 | 916 | 1009 | **954** | **~900** |

**Termination**: 98.9% health_depleted. Session 221202 produced all 4 successes (episodes 129, 136, 140, 198).

### Explained Variance by Phase (Cross-Session)

| Phase | 221154 | 221202 | 221207 | 221213 | **A2C-3 Avg** | **A2C-2 Avg** |
|-------|--------|--------|--------|--------|--------------|--------------|
| Ep 0-49 | -0.096 | -0.038 | -0.070 | -0.073 | **-0.069** | **-0.225** |
| Ep 50-99 | -0.342 | -0.226 | -0.558 | -0.750 | **-0.469** | **-0.217** |
| Ep 100-149 | -0.513 | -0.399 | -1.466 | -1.735 | **-1.028** | **-0.290** |
| Ep 150-199 | -0.620 | -0.068 | -0.954 | -0.838 | **-0.620** | **-0.295** |

EV is **2.1x worse than A2C-2** in Q4 (-0.620 vs -0.295). The Q3 mean (-1.028) is catastrophically negative. Session 221202 achieved Q4 EV of -0.068 — the best Q4 EV of any A2C session — and the only session with successes.

### Hypothesis Verdict: Non-Stationarity Was NOT the Root Cause

**The A2C-2 hypothesis** — "the critic cannot learn because hidden spike features change every time W_sh updates" — **is disproved.** Removing hidden spikes made the critic *worse*, not better. The failures are more fundamental:

1. **State aliasing**: 8 sensory gradient features don't uniquely identify states. Many distinct positions/health/food states produce identical feature vectors.
2. **Policy non-stationarity persists**: Even with stationary inputs, the *targets* (returns) shift as the actor improves. The optimal V(s) is always a moving target.
3. **Partial observability**: The critic sees local viewport gradients, not global state (position, HP, food count, predator positions). Return prediction is ill-posed.

### Q4 Regression Pattern (New Failure Mode)

2 of 4 sessions show performance peaking in Q3, then declining in Q4:

| Session | Q3 Foods | Q4 Foods | Change |
|---------|---------|---------|--------|
| 221154 | 2.26 | 1.62 | **-28%** |
| 221202 | 2.04 | 3.36 | +65% |
| 221207 | 1.34 | 1.78 | +33% |
| 221213 | 2.06 | 1.42 | **-31%** |

This pattern was absent in A2C-1 (Q4 always best). The critic's worsening EV in late training (Q3-Q4) injects noise into the policy gradient, overriding the actor's own REINFORCE signal.

### Decision: Abandon A2C

The A2C-2 analysis stated: "If [sensory-only critic] still fails, Option 3 (abandon A2C) is warranted." **It failed. A2C is abandoned.**

______________________________________________________________________

## Root Cause: Fundamental Environment Mismatch

After 4 rounds (16 sessions, 3,200 episodes total), the diagnosis is conclusive. The A2C critic cannot learn V(s) in the pursuit predator environment.

### Systematic Elimination of Hypotheses

| Round | Hypothesis | Test | Result |
|-------|-----------|------|--------|
| A2C-0→1 | Insufficient data | 4x episodes (50→200) | Still fails |
| A2C-0→1 | Critic too large | 5.4K→1.1K params | Still fails |
| A2C-1→2 | Implementation bugs | Fixed all 4 | Made it **worse** |
| A2C-2→3 | Non-stationary hidden spikes | Removed from input | Made it **worse** |

### Root Causes (Fundamental, Not Fixable)

1. **Partial observability**: The critic sees local gradient features from a viewport, not global state (agent position, predator positions, remaining HP, food count). Without these, predicting episode returns is fundamentally ill-posed.

2. **Policy non-stationarity**: The optimal V(s) changes every time the actor updates. With 2 actor epochs per 20-step window, the actor's policy shifts faster than the critic can track.

3. **High return variance**: Returns span [-20, +30] depending on stochastic predator encounters and food spawns. A small MLP cannot capture this variance from limited gradient features.

4. **Short training windows**: 20-step windows with gamma=0.99 poorly approximate the true discounted return over episodes lasting 50-500 steps.

### Why the Actor Learns Without the Critic

The REINFORCE backbone uses normalized returns within each update window: `A_t = (G_t - mean(G)) / std(G)`. This is self-normalizing and doesn't require learning a separate value function. The critic's GAE advantages add noise on top of this signal — worse than no critic at all when EV < 0.

______________________________________________________________________

## Lessons Learned

| Lesson | Detail |
|--------|--------|
| A2C critics fail under partial observability | With only local gradient features, V(s) prediction is ill-posed. Classical MLP PPO succeeds because it sees full state information. |
| Non-functional critics harm the actor | When EV < 0, GAE advantages inject noise worse than normalized-returns baseline. 2/4 A2C-3 sessions showed Q4 regression absent in pure REINFORCE. |
| Bug fixes can expose deeper problems | A2C-2's 4 bug fixes were correct but made EV worse — they gave the critic more capacity to overfit small windows. |
| Non-stationarity was not the primary cause | Removing hidden spikes (A2C-3) made things worse. Policy non-stationarity (moving V(s) targets) dominates over input non-stationarity. |
| REINFORCE is surprisingly effective alone | Q1→Q4 food improvement of 2-4x across all rounds is driven entirely by REINFORCE + exploration decay. |
| Systematic hypothesis testing is essential | Each round tested one hypothesis. Without this discipline, we might have spent more rounds on dead-end fixes. |
| 20-step GAE windows are too short for gamma=0.99 | With high discount factor and episodes up to 500 steps, 20-step windows capture a small fraction of the return horizon. |

______________________________________________________________________

## Cross-Round Performance Trajectory

### Task Metrics

| Round | Success | Avg Food | Q4 Food | Avg Reward | Avg Steps |
|-------|---------|----------|---------|------------|-----------|
| A2C-0 (50 ep) | 0% | 0.62 | — | -12.36 | 90.1 |
| A2C-1 (200 ep) | 0.13% | 1.52 | 2.05 | -9.93 | 81.4 |
| A2C-2 (200 ep) | 0.63% | 1.32 | 2.28 | -10.67 | 88.5 |
| A2C-3 (200 ep) | 0.50% | 1.50 | 2.05 | -10.06 | 86.8 |

### Critic Health

| Round | EV (Q1) | EV (Q4) | Trend | Critic Updates |
|-------|---------|---------|-------|---------------|
| A2C-0 | ~0 | ~0 | flat | 246 |
| A2C-1 | -0.066 | -0.008 | slightly improving | 898 |
| A2C-2 | -0.225 | -0.295 | **worsening** | ~900 |
| A2C-3 | -0.069 | -0.620 | **worsening** | 954 |

The critic EV progressively worsens across rounds (Q4: 0 → -0.008 → -0.295 → -0.620) while the actor holds steady at ~1.5 avg foods and ~2.0 Q4 foods. This divergence confirms the actor learns independently of the critic.

______________________________________________________________________

## Session References

| Round | Sessions | Episodes | Config Changes | Result |
|-------|----------|----------|---------------|--------|
| A2C-0 | 20260215_103816, 103823, 103830, 103835 | 50 | Initial A2C | 0%, critic EV≈0 |
| A2C-1 | 20260215_121727, 121735, 121742, 121748 | 200 | Simplified critic, 4x data | 0.13%, 4 bugs found |
| A2C-2 | 20260215_135006, 135012, 135018, 135025 | 200 | Bug fixes | 0.63%, EV worse (-0.295) |
| A2C-3 | 20260215_221154, 221202, 221207, 221213 | 200 | Sensory-only critic | 0.50%, EV worst (-0.620) |

### Best Individual Episodes

| Session | Round | Run | Foods | Reward | Steps | Outcome |
|---------|-------|-----|-------|--------|-------|---------|
| 135025 | A2C-2 | 167 | **10** | **+31.65** | 265 | success |
| 121727 | A2C-1 | 142 | **10** | +30.10 | 424 | success |
| 221202 | A2C-3 | 129 | **10** | +26.37 | 245 | success |
| 135012 | A2C-2 | 150 | **10** | +18.35 | 326 | success |

### Final Config State (A2C-3)

```yaml
brain:
  name: qsnnreinforce
  config:
    num_sensory_neurons: 8
    num_hidden_neurons: 16
    num_motor_neurons: 4
    shots: 1024
    num_integration_steps: 10
    logit_scale: 5.0
    weight_clip: 3.0
    theta_motor_max_norm: 2.0
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.02
    update_interval: 20
    num_reinforce_epochs: 2
    advantage_clip: 2.0
    exploration_decay_episodes: 150
    lr_decay_episodes: 400
    lr_min_factor: 0.2
    use_critic: true
    critic_use_hidden_spikes: false
    critic_hidden_dim: 32
    critic_num_layers: 1
    critic_lr: 0.001
    critic_epochs: 5
    critic_grad_clip: 5.0
    gae_lambda: 0.8
    value_loss_coef: 0.5
    sensory_modules:
      - food_chemotaxis
      - nociception
```

### Log Files

| Round | Log Files |
|-------|-----------|
| A2C-0 | `logs/simulation_20260215_103815.log`, `simulation_20260215_103822.log`, `simulation_20260215_103829.log`, `simulation_20260215_103834.log` |
| A2C-1 | `logs/simulation_20260215_121726.log`, `simulation_20260215_121734.log`, `simulation_20260215_121741.log`, `simulation_20260215_121747.log` |
| A2C-2 | `logs/simulation_20260215_135005.log`, `simulation_20260215_135011.log`, `simulation_20260215_135017.log`, `simulation_20260215_135024.log` |
| A2C-3 | `logs/simulation_20260215_221153.log`, `simulation_20260215_221201.log`, `simulation_20260215_221206.log`, `simulation_20260215_221212.log` |
