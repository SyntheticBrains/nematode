# 008 Appendix: QSNN Predator Optimization History

This appendix documents the QSNN predator evasion optimization journey across two phases: random predators (Rounds P0–P3a, 10 rounds, 40 sessions) and pursuit predators (Rounds PP4–PP9, 6 rounds, 24 sessions). For main findings, see [008-quantum-brain-evaluation.md](008-quantum-brain-evaluation.md). For foraging optimization history, see [008-appendix-qsnn-foraging-optimization.md](008-appendix-qsnn-foraging-optimization.md).

______________________________________________________________________

## Table of Contents

1. [Optimization Summary](#optimization-summary)
2. [Phase 1: Random Predators — Config Tuning (Rounds P0–P2a)](#phase-1-random-predators--config-tuning-rounds-p0p2a)
3. [Phase 2: Random Predators — Architectural Changes (Rounds P2b–P3a)](#phase-2-random-predators--architectural-changes-rounds-p2bp3a)
4. [Phase 3: Pursuit Predators (Rounds PP4–PP9)](#phase-3-pursuit-predators-rounds-pp4pp9)
5. [Key Technical Decisions](#key-technical-decisions)
6. [Best Configurations](#best-configurations)
7. [Session References](#session-references)

______________________________________________________________________

## Optimization Summary

### Random Predators (P0–P3a)

| Round | Key Change | Success | Sessions | Finding |
|-------|-----------|---------|----------|---------|
| P0 | Direct transfer from foraging config | 14.5% | 4 | Extreme seed variance (0–44%), 1/4 converge |
| P1 | Sensory fix + entropy + reward rebalancing | 1.25% | 4 | REGRESSION: proximity penalty (0.3) overwhelmed food signal |
| P2a | Revert proximity, match gradient decay, LR schedule | 0.0% | 4 | Config tuning insufficient; episode-end REINFORCE dilutes death signal |
| **P2b** | **Intra-episode REINFORCE (code change)** | **3.0%** | 4 | Improved credit assignment but weight explosion → entropy collapse |
| **P2c** | **Weight clip 3.0→2.0, 300 episodes** | **22.3%** | 4 | **Best avg; 74.3% post-convergence (best session). 2-3/4 converge** |
| **P2d** | **Evasion shaping + entropy_coef 0.05→0.08** | **25.1%** | 4 | **Best reliability (3-4/4 converge). Proximity penalty didn't help evasion** |
| P2e | Faster exploration/LR decay | 1.9% | 4 | REGRESSION: premature policy crystallization |
| P2f | Cherry-pick proximity revert | 8.5% | 4 | Config tuning exhausted; stay-lock trap dominant |
| **P3a** | **PPO clipping + logit_scale 20→5** | **11.6%** | 4 | Fixed action death and stay-lock; evasion now bottleneck |

### Pursuit Predators (PP4–PP9)

| Round | Key Change | Success | Sessions | Finding |
|-------|-----------|---------|----------|---------|
| PP4 | Classical critic (QSNN-AC) | 0% | 4 | Critic failed to learn V(s) |
| PP5 | Fix 3 critic bugs + tune params | 0% | 4 | Critic fundamentally broken |
| PP6 | Disable critic, 40 hidden, 3-epoch REINFORCE | 0% | 4 | Weight clamping bug inside epoch loop → gradient death |
| PP7 | Fan-in-aware tanh scaling + lower entropy | 0% | 4 | Gradients survive but fan_in=40 makes weights irrelevant |
| PP8 | Holistic parameter overhaul (16 hidden, 8 sensory) | 1.25% | 4 | **First-ever pursuit predator success** (1 session, 5%) |
| PP9 | Stabilise learning (advantage clip, degenerate skip) | 0% | 4 | PP8 breakthrough was lucky seed, not reproducible |

______________________________________________________________________

## Phase 1: Random Predators — Config Tuning (Rounds P0–P2a)

**Task**: Collect 10 foods while surviving 2 random predators (speed 1.0, detection_radius 8) on 20x20 grid. Separated sensory modules: food_chemotaxis + nociception.

### Round P0: Direct Transfer from Foraging Config (Baseline)

**Sessions**: 20260209_101857, 20260209_101904, 20260209_101910, 20260209_101915

Config: Same as foraging R12o (6→8→4, 92 params) with predator penalties added.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 101857 | 0% | 1.84 | +0.010 | 30% | 69.5% | 93.9% | No |
| **101904** | **44%** | **6.45** | **+0.403** | **49.5%** | **6.5%** | **89.0%** | **Yes (ep 140)** |
| 101910 | 14% | 3.20 | +0.101 | 55.5% | 30.5% | 91.6% | No |
| 101915 | 0% | 0.21 | -0.287 | 49.5% | 50.5% | 89.9% | No |
| **Average** | **14.5%** | **2.93** | **+0.057** | **46.1%** | **39.3%** | **91.1%** | **1/4** |

Key observations:

- Extreme seed variance (0–44%). Session 101904 post-convergence: **72.1% success** (outperforms SpikingReinforce's 62.8% post-convergence)
- Per-encounter evasion consistently strong (89–94%) from nociception module, but cumulative risk (0.91^7 ≈ 50%) caps episode success
- Convergence 2.5x slower than foraging (ep 140 vs ep 45–63)
- Session 101915: entropy collapse → negative chemotaxis → complete failure

### Round P1: Sensory Fix + Entropy + Reward Rebalancing (REGRESSION)

**Sessions**: 20260209_130641, 20260209_130649, 20260209_130656, 20260209_130703

Changes: num_sensory 6→4, num_hidden 8→10, entropy_coef 0.02→0.05, penalty_death 10→5, penalty_proximity 0.1→0.3, exploration_decay 80→120.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 130641 | 4.5% | 2.12 | -0.012 | 57.0% | 38.5% | 90.6% | No |
| 130649 | 0.0% | 0.33 | -0.193 | 68.5% | 31.5% | 87.1% | No |
| 130656 | 0.0% | 0.60 | -0.145 | 55.5% | 44.5% | 89.8% | No |
| 130703 | 0.5% | 1.14 | +0.143 | 60.0% | 39.0% | 88.8% | No |
| **Average** | **1.25%** | **1.05** | **-0.052** | **60.3%** | **38.4%** | **89.1%** | **0/4** |

**Root cause of regression**: `penalty_predator_proximity=0.3` overwhelmed food reward signal. With detection_radius=8 on 20x20 grid, ~50-60% of cells trigger the -0.3/step penalty, accumulating ~-75/episode vs +20 max food reward. 3/4 sessions learned negative chemotaxis ("avoid everything"). The entropy_coef increase (0.02→0.05) was the one positive change — prevented permanent entropy collapse.

**Pre-P1 investigation also identified:**

1. Sensory neuron duplication bias: num_sensory=6 with 4 features gives food 4/6 neurons, predator 2/6
2. Episode-end-only REINFORCE dilutes death signal (gamma=0.99 over 500 steps → death at step 200 reaches step 1 as -0.67, indistinguishable from noise after normalization)

### Round P2a: Config-Only Fixes

**Sessions**: 20260209_235324, 20260209_235331, 20260209_235337, 20260209_235355

Changes: penalty_proximity 0.3→0.05, penalty_death 5→3, predator gradient_decay 12→8, lr_decay_episodes 200→400, lr_min_factor 0.1→0.2.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 235324 | 0.0% | 1.08 | -0.118 | 67.5% | 32.5% | 87.2% | No |
| 235331 | 0.0% | 0.34 | -0.124 | 51.0% | 49.0% | 90.4% | No |
| 235337 | 0.0% | 2.24 | +0.077 | 50.0% | 50.0% | 91.5% | No |
| 235355 | 0.0% | 0.59 | -0.066 | 57.0% | 43.0% | 88.7% | No |
| **Average** | **0.0%** | **1.06** | **-0.058** | **56.4%** | **43.6%** | **89.5%** | **0/4** |

**Conclusion**: Three rounds of config tuning (P0→P1→P2a, 12 sessions) produced only 1 convergence (P0's lucky seed). Config tuning alone cannot solve the credit assignment problem. Code change needed: intra-episode REINFORCE updates.

______________________________________________________________________

## Phase 2: Random Predators — Architectural Changes (Rounds P2b–P3a)

### Round P2b: Intra-Episode REINFORCE (Code Change)

**Sessions**: 20260210_024248, 20260210_024257, 20260210_024304, 20260210_024311

Code change: Added REINFORCE updates every `update_interval` (20) steps in surrogate gradient mode. Death signal now within 20-step window at near-full strength (25x stronger than episode-end).

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| **024248** | **5.0%** | **2.83** | **+0.242** | 67.5% | 26.5% | 89.0% | No |
| **024257** | **6.0%** | 1.61 | -0.202 | 47.0% | 47.0% | 88.2% | No |
| 024304 | 1.0% | 1.35 | +0.015 | 66.5% | 32.5% | 88.5% | No |
| 024311 | 0.0% | 0.63 | -0.266 | 53.5% | 46.5% | 90.2% | No |
| **Average** | **3.0%** | **1.60** | **-0.053** | **58.6%** | **38.1%** | **88.9%** | **0/4** |

Key finding: Food-seeking improved (avg foods 1.06→1.60) but rise-and-collapse pattern emerged. Session 024257 had 12 successes in eps 25-63 (~31% local rate) then entropy collapsed as W_sh grew from 0.94 to 3.26 unbounded. Weight explosion → entropy collapse is the new bottleneck.

### Round P2c: Weight Clip + Longer Training (BEST AVERAGE)

**Sessions**: 20260210_064304, 20260210_064313, 20260210_064320, 20260210_064327

Changes: weight_clip 3.0→2.0, training extended to 300 episodes.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| **064304** | **48.3%** | **6.84** | **+0.410** | 50.3% | 1.3% | 86.5% | **Yes (ep 266)** |
| 064313 | 33.0% | 6.10 | +0.400 | 64.0% | 3.0% | 86.4% | Borderline |
| 064320 | 0.0% | 0.01 | ~0 | 35.7% | 64.3% | 91.9% | No |
| **064327** | **8.0%** | 2.50 | +0.056 | 54.7% | 36.7% | 89.7% | **Late (~ep 287)** |
| **Average** | **22.3%** | **3.86** | **+0.217** | **51.2%** | **26.3%** | **88.6%** | **2-3/4** |

**Best QSNN predator result overall.** Session 064304 achieved 74.3% post-convergence — surpassing SpikingReinforce's 62.8% with 1,400x fewer parameters. Weight clip made entropy collapse recoverable rather than terminal. Session 064327 showed 180-episode collapse then late renaissance to 40% in last 20 episodes — demonstrating the weight clip keeps the network recoverable.

### Round P2d: Evasion Shaping + Entropy Stabilization (BEST RELIABILITY)

**Sessions**: 20260210_111131, 20260210_111139, 20260210_111144, 20260210_111149

Changes: penalty_predator_proximity 0.05→0.15, entropy_coef 0.05→0.08.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 111131 | 24.3% | 4.93 | +0.348 | 66.0% | 9.7% | 87.6% | Yes (~ep 100) |
| 111139 | 6.3% | 1.51 | -0.102 | 49.3% | 44.3% | 91.3% | Late (~ep 241) |
| **111144** | **35.3%** | **6.19** | **+0.386** | 61.7% | 3.0% | 85.0% | **Yes (~ep 141)** |
| **111149** | **34.3%** | **5.87** | **+0.366** | 61.0% | 4.3% | 84.8% | **Yes (~ep 151)** |
| **Average** | **25.1%** | **4.63** | **+0.250** | **59.5%** | **15.3%** | **87.2%** | **3-4/4** |

**Best session reliability (3-4/4 converge).** entropy_coef=0.08 was clearly positive. But the proximity penalty increase was harmful — evasion dropped (87.2% vs 88.6% in P2c), predator deaths increased. The agent learns evasion from the death signal, not distance shaping.

### Round P2e: Faster Convergence Attempt (CATASTROPHIC REGRESSION)

**Sessions**: 20260210_205259, 20260210_205306, 20260210_205312, 20260210_205320

Changes: penalty_proximity 0.15→0.05 (revert), exploration_decay 120→80, lr_decay 400→300.

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 205259 | 7.7% | 2.42 | ~+0.10 | 57.7% | 34.7% | 87.8% | No |
| 205306 | 0.0% | 0.08 | ~-0.20 | 35.7% | 64.3% | 93.2% | No |
| 205312 | 0.0% | 0.04 | ~-0.20 | 38.3% | 61.7% | 92.8% | No |
| 205320 | 0.0% | 0.45 | ~-0.17 | 35.0% | 65.0% | 90.5% | No |
| **Average** | **1.9%** | **0.75** | ~**-0.12** | **41.7%** | **56.4%** | **91.1%** | **0/4** |

**Root cause**: Faster exploration/LR decay caused premature policy crystallization. The dual-objective task needs longer exploration (120+ episodes) than foraging (80). Three sessions became passive non-moving agents (>90% starvation). High evasion rates are an artifact of not moving.

### Round P2f: Cherry-Pick Proximity Revert

**Sessions**: 20260211_024747, 20260211_024759, 20260211_024806, 20260211_024813

Config: P2d config with proximity=0.05 instead of 0.15 (the last untested combination).

| Session | Success | Avg Foods | CI | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-----|-------------|------------|-------------|-----------|
| 024747 | 0.3% | 0.52 | ~+0.01 | ~55% | ~44% | 88.1% | No |
| 024759 | 0.0% | 0.29 | ~-0.15 | ~45% | ~55% | 91.0% | No |
| 024806 | 0.0% | 0.08 | ~-0.20 | ~35% | ~65% | 92.9% | No |
| **024813** | **33.7%** | **5.92** | **~+0.35** | ~60% | ~6% | **86.2%** | **Yes** |
| **Average** | **8.5%** | **1.70** | **~0** | **~49%** | **~43%** | **89.6%** | **1/4** |

**Config tuning exhausted.** 3/4 sessions fell into the stay-lock trap (death penalty > starvation penalty → immobility is rational). Session 024813's U-shaped learning (collapse → recovery to 75% peak) shows the architecture *can* learn, but only ~25% of seeds find the right trajectory. Architectural limitations identified: no value function, separated gradient inputs, 92 params for dual-objective, no policy clipping, logit_scale=20.0 causing action death.

### Round P3a: PPO-Style Clipping + Reduced logit_scale

**Sessions**: 20260211_083903, 20260211_083911, 20260211_083917, 20260211_083928

Code changes: logit_scale 20.0→5.0, clip_epsilon=0.2 (PPO-style policy ratio clipping). 300 episodes.

| Session | Success | Avg Foods | Pred Deaths | Starvation | Evasion Rate | Converged |
|---------|---------|-----------|-------------|------------|-------------|-----------|
| 083903 | 3.0% | 2.75 | 81.0% | 15.0% | 88.1% | No |
| **083911** | **17.0%** | **4.45** | 79.0% | 4.0% | 87.0% | No |
| 083917 | 10.7% | 3.68 | 70.7% | 16.3% | 90.0% | Yes (late) |
| **083928** | **15.7%** | **4.36** | 76.7% | 7.0% | 89.1% | No (18%) |
| **Average** | **11.6%** | **3.81** | **76.9%** | **10.6%** | **88.6%** | **1/4** |

**Fixed action death and stay-lock** (starvation dropped from 43% to 10.6%, best ever). But predator death became dominant failure mode at 76.9%. The critical finding: **per-encounter evasion rate was unchanged at ~88% across all 10 rounds (P0–P3a, 40 sessions)**. The architecture learns food-seeking but has never improved evasion through training.

______________________________________________________________________

## Phase 3: Pursuit Predators (Rounds PP4–PP9)

After P3a, the task was changed from random predators to **pursuit predators** (speed 0.5, detection_radius 6, movement_pattern pursuit) to match the MLP PPO baseline environment. Health system enabled (max_hp 100, predator_damage 20, food_healing 10). Classical MLP PPO baseline: ~93.5% avg success with 34,949 params.

### Round PP4: QSNN-AC — Classical Critic

**Sessions**: 20260211_234430, 20260211_234438, 20260211_234444, 20260211_234451

Code change: Added classical MLP critic (4→32→32→1, ~1,249 params) for GAE advantage estimation. Config: num_hidden 10→20, penalty_death 3.0→10.0.

**Result: 0% across all 4 sessions.** Critic failed to learn V(s) — predictions stuck near zero, unable to track the extreme reward landscape (-10 to +2). Reward normalization (enabled by default) made critic targets non-stationary. Pursuit predators also fundamentally harder: evasion dropped to ~30% (from ~88% on random predators).

### Round PP5: Fix 3 Critic Bugs + Parameter Tuning

**Sessions**: 20260212_212054, 20260212_212100, 20260212_212111, 20260212_212117

Bug fixes: (1) skip reward normalization in AC mode, (2) Huber loss replaces MSE for critic, (3) deferred bootstrap at window boundaries. Parameter changes: critic_hidden 32→64, critic_lr 0.001→0.003, gae_lambda 0.98→0.95.

**Result: 0% across all 4 sessions.** Critic still not learning (value loss 0.2–11.0 with no trend). W_sh grew 1.3–3.8x. The ~4K-param critic cannot track the extreme reward landscape, and the 94-param actor with single-pass REINFORCE cannot extract enough learning from each experience. **Decision: abandon critic approach.**

### Round PP6: Disable Critic + Increase Capacity + Multi-Epoch REINFORCE

**Sessions**: 20260213_024236, 20260213_024243, 20260213_024249, 20260213_024255

Code changes: Disable critic, num_hidden 20→40 (94→364 actor params), add 3-epoch REINFORCE with quantum output caching (epoch 0 runs circuits, epochs 1-2 reuse cached spike probs with updated weights).

**Result: 0% across all 4 sessions.** A critical bug was discovered: `_apply_gradients_and_log()` was called inside the multi-epoch loop, causing weight clamping 3x per update. This created gradient death: W_sh norms grew to 4.1–5.65 (elements at ±2.0 boundaries), then grad_sh and grad_hm collapsed to exactly 0.0000 by ep 50–150. Only theta_motor (4 params) survived.

**Fix applied**: Separated into `_step_optimizer_and_log()` (per-epoch) and `_clamp_weights()` (once after all epochs).

### Round PP7: Fan-In-Aware tanh Scaling

**Sessions**: 20260213_094420, 20260213_094428, 20260213_094434, 20260213_094439

Code change: `tanh(w*x)` → `tanh(w*x / sqrt(fan_in))` in all QLIF layer methods. entropy_coef 0.08→0.04.

**Motivation**: PP6 analysis revealed the true root cause of gradient death across all predator experiments: `tanh(weighted_input)` saturates when `fan_in * avg_spike * |w| > ~2`. With 40 hidden neurons, `weighted_input = 10*|w|`, so tanh gradient dies at |w| = 0.18 — barely above init scale. The network was born dead.

| Session | Success | Avg Food | Evasion Rate | Health Depleted | Entropy Min |
|---------|---------|----------|-------------|-----------------|-------------|
| 094420 | 0% | 0.74 | 33.7% | 97.5% | 0.98 |
| 094428 | 0% | 0.56 | 30.6% | 98.5% | 1.00 |
| 094434 | 0% | 0.54 | 37.7% | 97.5% | 0.78 |
| 094439 | 0% | 0.69 | 34.0% | 97.0% | 0.99 |
| **Average** | **0%** | **0.63** | **34.0%** | **97.6%** | — |

**Fan-in fix confirmed correct** — gradients survived all 200 episodes (previously died by ep 10–20). Entropy broke free (dropped to 0.78 vs locked at 1.35+). But gradients through weights still weak (dominated by theta_motor), and W_sh/W_hm grew monotonically (1.7→13.0). Root cause: fan_in=40 makes individual weights irrelevant to motor output.

### Round PP8: Holistic Parameter Overhaul (FIRST PURSUIT SUCCESS)

**Sessions**: 20260213_150430, 20260213_150438, 20260213_150444, 20260213_150449

Changes: num_hidden 40→16, num_sensory 4→8, update_interval 10→20, weight_clip 2.0→3.0, entropy_coef 0.04→0.02, num_reinforce_epochs 3→1, advantage_clip 2.0→4.0. All aligned with foraging baseline values.

| Session | Success | Avg Food | Evasion | grad_sh ep150 | theta_m max | Degenerate Batches |
|---------|---------|----------|---------|---------------|-------------|-------------------|
| 150430 | 0% | 0.49 | 31.7% | 0.0013 (dead) | 1.00 (capped) | 18 |
| 150438 | 0% | 0.96 | 28.2% | 0.037 | 1.00 (capped) | 7 |
| **150444** | **5%** | **2.14** | **47.8%** | **0.096** | 0.61 | 5 |
| 150449 | 0% | 0.86 | 34.6% | 0.038 | 0.75 | 8 |
| **Average** | **1.25%** | **1.11** | **35.6%** | — | — | — |

**Milestone**: Session 150444 achieved the **first-ever non-zero QSNN success on pursuit predators**, breaking the 0% ceiling across PP4–PP7 (16 sessions). W_hm breakthrough at ep 136–138; evasion improved to 47.8% (first time evasion improved through training on pursuit predators).

3/4 sessions failed from: (1) degenerate gradient bombs from single-step death episodes (returns_std=0, advantage clamped to -4.0), (2) theta_motor saturation at 1.0 cap, (3) seed-dependent grad_sh collapse.

### Round PP9: Stabilise Learning Across Sessions

**Sessions**: 20260214_060126, 20260214_060132, 20260214_060138, 20260214_060144

Changes: advantage_clip 4.0→2.0, skip degenerate batches (num_steps < 2), num_reinforce_epochs 1→2, theta_motor_max_norm 1.0→2.0, exploration_decay 120→150.

| Session | Success | Avg Food | Evasion Rate | Health Depleted | Converged |
|---------|---------|----------|-------------|-----------------|-----------|
| 060126 | 0% | 1.05 | 29.6% | ~97% | No |
| 060132 | 0% | 1.035 | 29.3% | ~97% | No |
| 060138 | 0% | 0.84 | 48.8% | ~95% | No |
| 060144 | 0% | 0.395 | 41.0% | ~97% | No |
| **Average** | **0%** | **0.83** | **37.2%** | **~96.5%** | **0/4** |

**PP8's 5% success was a lucky seed, not reproducible.** PP9 fixes worked as designed (degenerate batches skipped, theta_m not saturated, advantage_clip limited gradient bombs), but the fundamental architecture cannot reliably learn pursuit predator evasion. All sessions showed unbounded weight growth (3–6x), entropy ceiling suppression zeroing entropy_coef for first ~30 episodes, and direction-lock as local optima.

______________________________________________________________________

## Key Technical Decisions

### 1. Intra-Episode REINFORCE (Round P2b)

The most impactful code change. Episode-end REINFORCE dilutes the death signal to noise (gamma=0.99 over 500 steps → death at step 200 contributes -0.40 to step 1 return, indistinguishable after normalization). With 20-step windows, death signal is 25x stronger at the decision point.

### 2. Weight Clipping (Round P2c)

Tightening weight_clip from 3.0 to 2.0 made entropy collapse recoverable rather than terminal. Without clipping, W_sh grew monotonically from 0.94 to 3.26 (P2b), overwhelming entropy regularization. With clipping, sessions could recover from entropy dips — enabling the 74.3% post-convergence result.

### 3. PPO-Style Policy Clipping (Round P3a)

Adding clip_epsilon=0.2 constrained policy ratios to [0.8, 1.2], preventing catastrophic shifts from single bad episodes. Combined with logit_scale reduction (20→5), this eliminated the stay-lock and action death failure modes that plagued earlier rounds.

### 4. Fan-In-Aware tanh Scaling (Round PP7)

The true root cause of gradient death: `tanh(weighted_input)` saturates when `fan_in * avg_spike * |w| > ~2`. With 40 hidden neurons, tanh gradient dies at |w| = 0.18. The fix `tanh(w*x / sqrt(fan_in))` normalizes input regardless of layer width. This was the last fundamental bug; after PP7, all remaining failures were capacity/architectural limitations.

### 5. Multi-Epoch REINFORCE with Quantum Caching (Round PP6)

Epoch 0 runs quantum circuits and caches spike probabilities; subsequent epochs reuse cached probs but recompute ry_angles from updated weights. PPO clipping constrains policy ratio across epochs. This bridges part of the gradient-pass gap with MLP PPO (3x vs 40x).

______________________________________________________________________

## Best Configurations

### Best Random Predator Config (P2c — 22.3% avg, 48.3% best)

```yaml
brain:
  name: qsnn
  config:
    num_sensory_neurons: 4
    num_hidden_neurons: 10
    num_motor_neurons: 4
    membrane_tau: 0.9
    threshold: 0.5
    use_local_learning: false
    shots: 1024
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.05
    weight_clip: 2.0
    update_interval: 20
    num_integration_steps: 10
    lr_decay_episodes: 400
    lr_min_factor: 0.2

# 300 episodes, random predators (speed 1.0, detection_radius 8)
```

### Best Session Reliability Config (P2d — 25.1% avg, 3-4/4 converge)

Same as P2c but: `penalty_predator_proximity: 0.15`, `entropy_coef: 0.08`.

### Best Pursuit Predator Config (PP8 — 1.25% avg, 5% best session)

```yaml
brain:
  name: qsnn
  config:
    num_sensory_neurons: 8
    num_hidden_neurons: 16
    num_motor_neurons: 4
    membrane_tau: 0.9
    threshold: 0.5
    use_local_learning: false
    shots: 1024
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.02
    weight_clip: 3.0
    update_interval: 20
    num_integration_steps: 10
    num_reinforce_epochs: 1
    advantage_clip: 4.0
    logit_scale: 5.0
    theta_motor_max_norm: 1.0
    sensory_modules:
      - food_chemotaxis
      - nociception

# 200 episodes, pursuit predators (speed 0.5, detection_radius 6)
```

______________________________________________________________________

## Cross-Round Performance Trajectory

| Round | Success | Avg Food | Evasion | Starvation | Convergence | Type |
|-------|---------|----------|---------|------------|-------------|------|
| P0 | 14.5% | 2.93 | 91.1% | 39.3% | 1/4 | Random |
| P1 | 1.25% | 1.05 | 89.1% | 38.4% | 0/4 | Random |
| P2a | 0.0% | 1.06 | 89.5% | 43.6% | 0/4 | Random |
| P2b | 3.0% | 1.60 | 88.9% | 38.1% | 0/4 | Random |
| **P2c** | **22.3%** | **3.86** | **88.6%** | **26.3%** | **2-3/4** | Random |
| **P2d** | **25.1%** | **4.63** | **87.2%** | **15.3%** | **3-4/4** | Random |
| P2e | 1.9% | 0.75 | 91.1% | 56.4% | 0/4 | Random |
| P2f | 8.5% | 1.70 | 89.6% | ~43% | 1/4 | Random |
| P3a | 11.6% | 3.81 | 88.6% | 10.6% | 1/4 | Random |
| PP4 | 0% | — | ~30% | — | 0/4 | Pursuit |
| PP5 | 0% | — | ~30% | — | 0/4 | Pursuit |
| PP6 | 0% | ~2 | ~30% | — | 0/4 | Pursuit |
| PP7 | 0% | 0.63 | 34.0% | — | 0/4 | Pursuit |
| PP8 | 1.25% | 1.11 | 35.6% | — | 0/4 | Pursuit |
| PP9 | 0% | 0.83 | 37.2% | — | 0/4 | Pursuit |

______________________________________________________________________

## Session References

### Random Predator Sessions

| Round | Sessions | Result |
|-------|----------|--------|
| P0 | 20260209_101857-101915 | 14.5% avg, best 44% (101904) |
| P1 | 20260209_130641-130703 | 1.25% avg, REGRESSION |
| P2a | 20260209_235324-235355 | 0% avg |
| P2b | 20260210_024248-024311 | 3.0% avg |
| **P2c** | **20260210_064304-064327** | **22.3% avg, 48.3% best (064304), 74.3% post-conv** |
| **P2d** | **20260210_111131-111149** | **25.1% avg, 3-4/4 converge** |
| P2e | 20260210_205259-205320 | 1.9% avg, REGRESSION |
| P2f | 20260211_024747-024813 | 8.5% avg |
| P3a | 20260211_083903-083928 | 11.6% avg |

### Pursuit Predator Sessions

| Round | Sessions | Result |
|-------|----------|--------|
| PP4 | 20260211_234430-234451 | 0%, critic failed |
| PP5 | 20260212_212054-212117 | 0%, critic bugs fixed but still failed |
| PP6 | 20260213_024236-024255 | 0%, weight clamping bug |
| PP7 | 20260213_094420-094439 | 0%, fan-in fix worked but weights irrelevant |
| PP8 | 20260213_150430-150449 | 1.25% avg, first pursuit success (150444) |
| PP9 | 20260214_060126-060144 | 0%, PP8 not reproducible |
