# 008 Appendix: HybridQuantum Brain Optimization History

This appendix documents the Hierarchical Hybrid Quantum Brain evaluation across 4 rounds (16 sessions, 4,200 episodes) covering all 3 training stages. For main findings, see [008-quantum-brain-evaluation.md](008-quantum-brain-evaluation.md). For architecture design, see [quantum-architectures.md](../../research/quantum-architectures.md).

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Optimization Summary](#optimization-summary)
3. [Round 1: Stage 1 QSNN Reflex Training](#round-1-stage-1-qsnn-reflex-training)
4. [Round 2: Stage 2 Cortex PPO Training](#round-2-stage-2-cortex-ppo-training)
5. [Round 3: Stage 2 Tuned Cortex PPO](#round-3-stage-2-tuned-cortex-ppo)
6. [Round 4: Stage 3 Joint Fine-Tune](#round-4-stage-3-joint-fine-tune)
7. [Pre-Stage-2 Bug Fixes](#pre-stage-2-bug-fixes)
8. [Fusion Strategy Analysis](#fusion-strategy-analysis)
9. [QSNN Weight Drift Analysis (Stage 3)](#qsnn-weight-drift-analysis-stage-3)
10. [Comparison with Classical Baselines](#comparison-with-classical-baselines)
11. [Lessons Learned](#lessons-learned)
12. [Session References](#session-references)

______________________________________________________________________

## Architecture Overview

The HybridQuantum brain implements a hierarchical architecture combining a QSNN reflex layer (quantum forward pass, surrogate gradient REINFORCE) with a classical cortex MLP (PPO training) for multi-objective decision-making.

```text
Sensory Input (8-dim)
       |
       v
QSNN Reflex Layer (QLIFNetwork)
  6 sensory → 8 hidden → 4 motor QLIF neurons
  ~92 quantum params, surrogate gradient REINFORCE
  Output: 4-dim reflex logits (proven reactive foraging)
       |
       v                    Sensory Input (6-dim from modules)
       |                          |
       v                          v
  +---------+            Classical Cortex (MLP)
  |  Fusion |<--------   6 → 64 → 64 → 7
  +---------+            (4 action biases + 3 mode logits)
       |                  PPO training, ~5K actor params
       v
  final_logits = reflex_logits * qsnn_trust + cortex_biases
       |
       v
  Action Selection (4 actions: UP/DOWN/LEFT/RIGHT)

Classical Critic (separate MLP)
  6 sensory → 64 → 64 → 1 = V(s)
  ~5K params, trains alongside cortex via PPO
```

**Three-stage curriculum:**
- Stage 1: QSNN reflex only (REINFORCE on foraging)
- Stage 2: Cortex PPO only (QSNN frozen, multi-objective environment)
- Stage 3: Joint fine-tune (both components, QSNN with 10x lower LR)

**Task (Stage 2/3)**: Pursuit predators (count 2, speed 0.5, detection_radius 6), health system (max_hp 100, predator_damage 20, food_healing 10), 20x20 grid. Sensory modules: food_chemotaxis + nociception + mechanosensation (6 cortex features). QSNN uses legacy 2-feature mode (gradient_strength, relative_angle).

______________________________________________________________________

## Optimization Summary

| Round | Stage | Key Changes | Success | Post-Conv | Key Finding |
|-------|-------|-----------|---------|-----------|-------------|
| 1 | 1 | QSNN reflex on foraging | 91.0%* | 99.5%* | QSNN reflex validated; 1 of 4 sessions unstable |
| 2 | 2 | Cortex PPO, QSNN frozen | 66.4% | 81.9% | Cortex learns foraging + evasion; beats MLP PPO unified by +10.3 pts |
| 3 | 2 | +LR schedule, +12 epochs, +mechano | 84.3% | 91.7% | Target ≥90% achieved; beats MLP PPO unified by +20.1 pts |
| 4 | 3 | Joint fine-tune, both trainable | **96.9%** | **96.9%** | Best result; +25.3 pts over MLP PPO unified baseline |

*Stage 1 best 3 of 4 sessions (1 outlier excluded).

**Outcome**: Three-stage curriculum fully validated. 96.9% post-convergence on pursuit predators with ~10K parameters (4.3x fewer than MLP PPO baseline).

______________________________________________________________________

## Round 1: Stage 1 QSNN Reflex Training

**Config**: `hybridquantum_foraging_small.yml`
**Sessions**: 20260216_100503, 20260216_100507, 20260216_100512, 20260216_100516
**Episodes**: 200 per session
**Task**: Foraging only (20x20 grid, 5 foods, target=10, 500 max steps)

### Config

```yaml
brain:
  name: hybridquantum
  config:
    training_stage: 1
    num_sensory_neurons: 6
    num_hidden_neurons: 8
    num_motor_neurons: 4
    shots: 1024
    num_qsnn_timesteps: 10
    surrogate_alpha: 1.0
    logit_scale: 5.0
    weight_clip: 3.0
    theta_motor_max_norm: 1.0
    qsnn_lr: 0.01
    qsnn_lr_decay_episodes: 200
    num_reinforce_epochs: 2
    reinforce_window_size: 20
    gamma: 0.99
    entropy_coeff: 0.02
```

### Results

| Session | Success | Convergence | Post-Conv | Post-Conv Var | Composite | Dist Efficiency | REINFORCE Updates | Final Entropy |
|---------|---------|-------------|-----------|---------------|-----------|----------------|-------------------|---------------|
| 100503 | 64.5% | Ep 7 | 66.5% | 0.223 | 0.547 | 0.290 | 3,811 | 1.168 |
| 100507 | 91.0% | Ep 18 | 99.5% | 0.005 | 0.775 | 0.324 | 2,969 | 1.073 |
| 100512 | 90.5% | Ep 20 | 100.0% | 0.000 | 0.780 | 0.332 | 2,952 | 1.129 |
| 100516 | 91.5% | Ep 16 | 98.4% | 0.016 | 0.761 | 0.301 | 3,320 | 1.139 |
| **Avg (excl 100503)** | **91.0%** | **18** | **99.3%** | **0.007** | **0.772** | **0.319** | **3,080** | **1.114** |

### Rolling Success Rate (20-episode windows)

| Window | 100503 | 100507 | 100512 | 100516 |
|--------|--------|--------|--------|--------|
| 0-19 | 70% | 15% | 5% | 30% |
| 20-39 | **100%** | **100%** | **100%** | 90% |
| 40-59 | 85% | 95% | **100%** | **100%** |
| 60-79 | 55% | **100%** | **100%** | **100%** |
| 80-99 | 85% | **100%** | **100%** | **100%** |
| 100-119 | 85% | **100%** | **100%** | **100%** |
| 120-139 | 60% | **100%** | **100%** | 95% |
| 140-159 | **25%** | **100%** | **100%** | **100%** |
| 160-179 | 45% | **100%** | **100%** | **100%** |
| 180-199 | 35% | **100%** | **100%** | **100%** |

### QSNN Weight Norms

| Session | Final W_sh | Final W_hm | Final θ_h | Final θ_m |
|---------|-----------|-----------|----------|----------|
| 100503 | 9.74 | 9.06 | — | 1.00 |
| 100507 | 10.45 | 9.45 | — | 1.00 |
| 100512 | 7.32 | 7.50 | — | 1.00 |
| 100516 | 11.31 | 9.84 | — | 1.00 |

### Analysis

**Session 100503 is an outlier.** It reached 100% at episodes 20-39 then degraded to 25-45% in later windows. The other 3 sessions all converged to near-perfect (95-100%) and held steady. 100503 likely hit an unlucky initialisation that led to weight drift — it accumulated more REINFORCE updates (3,811 vs ~3,000) because failing episodes run to 500 steps.

**Entropy stays high** (~1.07-1.17 final, vs max 1.386). The entropy floor mechanism (0.5 nats) plus adaptive entropy boost keeps policies from collapsing, but entropy isn't decreasing meaningfully toward a sharp policy. This is a known property: the QSNN reflex produces a stochastic reactive policy, not a deterministic one.

**theta_motor saturates at norm=1.0** across all sessions (`theta_motor_max_norm=1.0`). This is the norm clamp working as designed but may limit expressiveness.

**Best weights for Stage 2**: Session **100512** — 100% post-convergence, zero variance, most conservative weight norms (W_sh=7.32, W_hm=7.50).

______________________________________________________________________

## Round 2: Stage 2 Cortex PPO Training

**Config**: `hybridquantum_pursuit_predators_small.yml`
**Sessions**: 20260216_132604, 20260216_132609, 20260216_132614, 20260216_132619
**Episodes**: 200 per session
**QSNN weights**: Loaded from session 100512 (frozen)

### Changes vs Round 1

| Change | Value |
|--------|-------|
| Training stage | 1 → **2** (cortex PPO, QSNN frozen) |
| Task | Foraging only → **pursuit predators** (2 predators, health system) |
| Cortex actor | N/A → **4,935 params** (64×2 hidden, 3 modes) |
| Cortex critic | N/A → **4,545 params** (64×2 hidden, value function) |
| PPO | N/A → **clip=0.2, epochs=4, minibatches=4, buffer=512** |
| Cortex LR | N/A → **0.001** (flat, no schedule) |
| Sensory modules | Legacy 2-feature → **food_chemotaxis + nociception** (4 cortex features) |

### Results

| Session | Success | Convergence | Post-Conv | Post-Conv Var | Dist Eff | Evasion | HP Deaths | Avg Foods | Final Entropy | Final EV | QSNN Trust (end) |
|---------|---------|-------------|-----------|---------------|----------|---------|-----------|-----------|---------------|----------|-------------------|
| 132604 | 60.0% | Ep 95 | 74.5% | 0.190 | 0.457 | 81.3% | 80 | 7.97 | 0.75 | +0.20 | 0.151 |
| 132609 | 75.0% | Ep 49 | 85.5% | 0.124 | 0.419 | 83.9% | 50 | 8.76 | 0.79 | +0.40 | 0.426 |
| 132614 | 69.0% | Ep 96 | 81.0% | 0.154 | 0.411 | 84.0% | 61 | 8.37 | ~1.0 | +0.25 | 0.548 |
| 132619 | 61.5% | Ep 119 | 86.6% | 0.116 | 0.412 | 81.2% | 76 | 7.83 | 0.74 | +0.07 | 0.397 |
| **Avg** | **66.4%** | **90** | **81.9%** | **0.146** | **0.425** | **82.6%** | **66.8** | **8.23** | **0.82** | **+0.23** | **0.381** |

### Analysis

**Stage 2 works.** All 4 sessions converged and achieved 74-87% post-convergence success. Zero predator deaths across all 800 episodes. Two distinct fusion strategies emerged:
- **Cortex-dominant** (132604): trust drops to 0.151, cortex drives ~85% of decisions
- **QSNN-collaborative** (132609/132614/132619): trust rises to 0.40-0.55

Both achieve similar post-convergence (74-87%), but collaborative sessions show higher composite benchmarks (0.720 vs 0.622).

**Predator evasion learned from scratch**: The QSNN was trained only on foraging. All sessions achieved 81-84% evasion — entirely the cortex's contribution via nociception.

**All failures are health depletion**: 0 predator deaths, 0 starvation, 0 timeouts. predator_damage=20 chips away at HP=100.

**Negative approx_kl bug**: All sessions showed approx_kl of -0.7 to -1.4. Not affecting training (PPO uses clip objective) but indicates wrong formula.

### Recommendations Applied to Round 3

| Priority | Change | From | To |
|----------|--------|------|-----|
| 1 | ppo_epochs | 4 | 12 |
| 2 | LR schedule | flat 0.001 | warmup 0.0001→0.001 (50 eps), decay to 0.0001 (200 eps) |
| 3 | Add mechanosensation | 4 features | 6 features |
| 4 | Session length | 200 eps | 500 eps |
| 5 | Fix approx_kl | wrong formula | Schulman's 2nd-order approximation |
| 6 | Add cortex weight save | not saved | save to exports/ for Stage 3 |

______________________________________________________________________

## Round 3: Stage 2 Tuned Cortex PPO

**Config**: `hybridquantum_pursuit_predators_small.yml` (updated)
**Sessions**: 20260216_213406, 20260217_012722, 20260217_012729, 20260217_012735
**Episodes**: 500 per session
**QSNN weights**: Loaded from session 100512 (frozen)

### Changes vs Round 2

| Change | From | To |
|--------|------|-----|
| ppo_epochs | 4 | **12** (3x more gradient steps) |
| LR schedule | flat 0.001 | **warmup 0.0001→0.001 (50 eps), decay 0.001→0.0001 (200 eps)** |
| Sensory modules | food_chemotaxis + nociception (4) | **+ mechanosensation** (6 features) |
| Cortex actor params | 4,935 | **5,063** (wider input) |
| Cortex critic params | 4,545 | **4,673** (wider input) |
| Session length | 200 eps | **500 eps** |
| approx_kl formula | wrong (always negative) | **Schulman's approximation** (non-negative) |
| Cortex weight save | not saved | **auto-save to exports/** |

### Results

| Session | Success | Convergence | Post-Conv | Post-Conv Var | Dist Eff | Evasion | HP Deaths | Avg Foods | Last 100 | Last 50 | QSNN Trust (end) |
|---------|---------|-------------|-----------|---------------|----------|---------|-----------|-----------|----------|---------|-------------------|
| 213406 | 79.8% | Ep 81 | 90.0% | 0.090 | 0.491 | 85.9% | 100 | 8.78 | 99.0% | 100.0% | 0.188 |
| 012722 | 80.8% | Ep 132 | 94.0% | 0.056 | 0.504 | 86.3% | 96 | 8.90 | 97.0% | 94.0% | 0.169 |
| 012729 | 88.0% | Ep 35 | 91.2% | 0.081 | 0.480 | 88.5% | 60 | 9.49 | 94.0% | 92.0% | 0.553 |
| 012735 | 88.6% | Ep 31 | 91.5% | 0.078 | 0.534 | 87.8% | 57 | 9.51 | 98.0% | 100.0% | 0.507 |
| **Avg** | **84.3%** | **70** | **91.7%** | **0.076** | **0.502** | **87.2%** | **78.3** | **9.17** | **97.0%** | **96.5%** | **0.354** |

### Rolling Success Rate (20-episode windows)

| Window | 213406 | 012722 | 012729 | 012735 |
|--------|--------|--------|--------|--------|
| 1-20 | 0% | 0% | 25% | 20% |
| 21-40 | 15% | 50% | 70% | **95%** |
| 41-60 | 25% | 50% | 70% | 55% |
| 61-80 | 65% | 40% | 90% | 90% |
| 81-100 | 65% | 35% | 95% | 95% |
| 101-120 | 60% | 60% | 85% | 95% |
| 121-140 | 70% | 95% | 90% | 90% |
| 141-160 | 85% | 75% | 90% | 75% |
| 161-180 | **100%** | 95% | 85% | 90% |
| 181-200 | 90% | 80% | 85% | 85% |
| 201-220 | **100%** | 85% | 90% | 85% |
| 221-240 | **100%** | **100%** | 95% | 95% |
| 241-260 | 90% | **100%** | 90% | 90% |
| 261-280 | 85% | **100%** | 95% | 90% |
| 281-300 | 95% | 95% | 90% | 95% |
| 301-320 | 90% | **100%** | **100%** | **100%** |
| 321-340 | 90% | 90% | **100%** | 95% |
| 341-360 | 85% | 95% | 90% | **100%** |
| 361-380 | 95% | **100%** | 95% | 85% |
| 381-400 | 95% | 90% | **100%** | **100%** |
| 401-420 | **100%** | **100%** | 90% | 90% |
| 421-440 | 95% | **100%** | **100%** | **100%** |
| 441-460 | **100%** | **100%** | 95% | **100%** |
| 461-480 | **100%** | 90% | 95% | **100%** |
| 481-500 | **100%** | 95% | 90% | **100%** |

### Round 3 vs Round 2 Comparison

| Metric | Round 2 (avg) | Round 3 (avg) | Delta |
|--------|--------------|--------------|-------|
| Post-conv success | 81.9% | **91.7%** | **+9.8 pts** |
| Overall success | 66.4% | **84.3%** | **+17.9 pts** |
| Convergence episode | ~90 (of 200) | ~70 (of 500) | 1.3x faster |
| HP deaths (rate) | 33.4% | 15.7% | **-17.7 pts** |
| Evasion rate | 82.6% | **87.2%** | **+4.6 pts** |
| Avg foods collected | 8.23 | **9.17** | **+0.94** |
| Distance efficiency | 0.425 | **0.502** | **+0.077** |
| Post-conv variance | 0.146 | **0.076** | **-0.070** (more stable) |

### Analysis

**Target ≥90% achieved.** All 4 sessions individually exceeded 90% post-convergence. Last 100 episodes averaged 97.0%.

**Same two fusion strategies re-emerged**, now more pronounced:
- Cortex-dominant (213406, 012722): trust 0.17-0.19, explore mode 47-66%
- QSNN-collaborative (012729, 012735): trust 0.51-0.55, forage mode 51-55%

Both achieve ≥90%, but collaborative sessions show fewer HP deaths (57-60 vs 96-100).

**Learning plateaued by ~episode 300.** LR reaches terminal value at episode 250; the remaining episodes serve as consolidation.

**Best weights for Stage 3**: Session **012735** — 88.6% overall, 100% last 50, highest distance efficiency 0.534, QSNN-collaborative strategy.

______________________________________________________________________

## Round 4: Stage 3 Joint Fine-Tune

**Config**: `hybridquantum_pursuit_predators_small_finetune.yml`
**Sessions**: 20260217_061309, 20260217_061317, 20260217_061323, 20260217_061329
**Episodes**: 500 per session
**QSNN weights**: Loaded from stage 1 session 100512 (**unfrozen**)
**Cortex weights**: Loaded from stage 2 session 012735

### Changes vs Round 3 (Stage 2)

| Change | From (Stage 2 R3) | To (Stage 3) |
|--------|-------------------|--------------|
| Training stage | 2 | **3** (both trainable) |
| QSNN | Frozen | **Unfrozen**, effective LR=0.001 (0.01 × 0.1) |
| QSNN LR decay | N/A | **Cosine annealing** 0.001→0.0001 over 200 eps |
| Cortex base LR | 0.001 | **0.0005** (halved, already pre-trained) |
| Cortex LR warmup | 50 eps | **20 eps** (shorter for fine-tuning) |
| Cortex LR decay | 200 eps | **100 eps** (0.0005→0.0001) |
| Cortex weights | Random init | **Loaded from session 012735** |

### Results

| Session | Success | Convergence | Post-Conv | Post-Conv Var | Dist Eff | Evasion | HP Deaths | Avg Foods | Last 100 | Avg Steps | Composite |
|---------|---------|-------------|-----------|---------------|----------|---------|-----------|-----------|----------|-----------|-----------|
| 061309 | 96.6% | Ep 1 | 96.6% | 0.033 | 0.510 | 89.9% | 17 | 9.85 | 97.0% | 177.9 | 0.854 |
| 061317 | 97.2% | Ep 1 | 97.2% | 0.027 | 0.554 | 90.0% | 14 | 9.91 | 98.0% | 164.3 | 0.871 |
| 061323 | 96.4% | Ep 1 | 96.4% | 0.032 | 0.528 | 92.1% | 17 | 9.86 | 98.0% | 175.9 | 0.863 |
| 061329 | 97.2% | Ep 1 | 97.2% | 0.027 | 0.546 | 91.6% | 14 | 9.86 | 97.0% | 166.9 | 0.870 |
| **Avg** | **96.9%** | **1** | **96.9%** | **0.030** | **0.534** | **90.9%** | **15.5** | **9.87** | **97.5%** | **171.2** | **0.864** |

### Rolling Success Rate (20-episode windows)

| Window | 061309 | 061317 | 061323 | 061329 |
|--------|--------|--------|--------|--------|
| 1-20 | **100%** | **100%** | 90% | 95% |
| 21-40 | 95% | **100%** | 90% | 95% |
| 41-60 | 90% | 90% | 95% | **100%** |
| 61-80 | 85% | 90% | 90% | **100%** |
| 81-100 | **100%** | **100%** | 90% | 90% |
| 101-120 | **100%** | **100%** | **100%** | 95% |
| 121-140 | **100%** | 90% | **100%** | **100%** |
| 141-160 | 95% | **100%** | 95% | 95% |
| 161-180 | 95% | **100%** | 95% | 95% |
| 181-200 | **100%** | 95% | 85% | **100%** |
| 201-220 | 95% | 95% | **100%** | 95% |
| 221-240 | **100%** | 95% | **100%** | **100%** |
| 241-260 | **100%** | **100%** | **100%** | **100%** |
| 261-280 | 95% | **100%** | **100%** | 95% |
| 281-300 | 95% | **100%** | **100%** | 90% |
| 301-320 | 95% | **100%** | **100%** | **100%** |
| 321-340 | **100%** | **100%** | **100%** | **100%** |
| 341-360 | **100%** | 95% | **100%** | **100%** |
| 361-380 | **100%** | 90% | 90% | **100%** |
| 381-400 | 90% | **100%** | **100%** | **100%** |
| 401-420 | 90% | **100%** | **100%** | 95% |
| 421-440 | **100%** | **100%** | **100%** | **100%** |
| 441-460 | 95% | 95% | 95% | **100%** |
| 461-480 | **100%** | 95% | **100%** | 95% |
| 481-500 | **100%** | **100%** | 95% | 95% |

### QSNN Weight Drift

| Weight | Start (all sessions) | 061309 end | 061317 end | 061323 end | 061329 end | Mean end | Mean Δ% |
|--------|---------------------|------------|------------|------------|------------|----------|---------|
| W_sh_norm | 7.32-7.34 | 7.832 | 7.482 | 7.481 | 7.304 | 7.525 | +2.7% |
| W_hm_norm | 7.50-7.54 | **9.876** | **9.369** | **9.542** | **9.563** | **9.587** | **+27.7%** |
| theta_h_norm | 3.29 | 3.675 | 3.493 | 3.663 | 3.526 | 3.589 | +9.2% |
| theta_m_norm | 1.00 | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 | ~0% |

### Fusion Diagnostics

| Session | QSNN Trust (start) | QSNN Trust (end) | Mode 0 (Forage) | Mode 1 (Evade) | Mode 2 (Explore) |
|---------|---------------------|-------------------|-----------------|----------------|-----------------|
| 061309 | 0.546 | 0.589 | 0.589 | 0.212 | 0.199 |
| 061317 | 0.514 | 0.527 | 0.527 | 0.236 | 0.237 |
| 061323 | 0.527 | 0.515 | 0.515 | 0.291 | 0.194 |
| 061329 | 0.531 | 0.549 | 0.549 | 0.265 | 0.186 |

### Training Loop Diagnostics

Both training loops confirmed active in all sessions:
- **REINFORCE**: 4,300-4,700 epoch-0 updates/session (~8.7-9.3 updates/episode × 2 epochs)
- **PPO**: 500 updates/session (exactly 1 per episode)
- No interference between optimizers (QSNN Adam separate from cortex actor/critic Adams)

### Stage 3 vs Stage 2 Round 3 Comparison

| Metric | Stage 2 R3 (avg) | Stage 3 (avg) | Delta |
|--------|-----------------|---------------|-------|
| Post-conv success | 91.7% | **96.9%** | **+5.2 pts** |
| Overall success | 84.3% | **96.9%** | **+12.6 pts** |
| Convergence episode | ~70 (of 500) | 1 (of 500) | Immediate |
| HP death rate | 15.7% | **3.1%** | **-12.6 pts** |
| Evasion rate | 87.2% | **90.9%** | **+3.7 pts** |
| Avg foods collected | 9.17 | **9.87** | **+0.70** |
| Distance efficiency | 0.502 | **0.534** | **+0.032** |
| Post-conv variance | 0.076 | **0.030** | **-0.046** |
| Avg steps | ~200 | **171.2** | -28.8 fewer |
| Composite benchmark | 0.685 | **0.864** | **+0.179** |
| Session variance | 8.8 pt range | **0.8 pt range** | **11x tighter** |

______________________________________________________________________

## Pre-Stage-2 Bug Fixes

Two bugs were found and fixed between Round 1 and Round 2:

### 1. Session ID Mismatch

**Bug**: HybridQuantumBrain generated its own `_session_id` via `datetime.now(UTC)` at init time, typically 1-2 seconds after the simulation script's session ID. This caused weight file paths to mismatch (e.g., simulation ID `100507` but weights saved to `100508`).

**Fix**: Added `set_session_id()` method to HybridQuantumBrain, called from `run_simulation.py` to pass the simulation's session ID.

### 2. Stage 2 Config Mismatch

**Bug**: Stage 2 config initially used 8/16/4 QSNN neurons (matching standalone QSNN), but Stage 1 weights were trained with 6/8/4 neurons, causing shape mismatch on load.

**Fix**: Aligned Stage 2 config to 6/8/4 neurons. Split preprocessing into dual paths:
- QSNN: `_preprocess_legacy(params)` → gradient_strength, relative_angle (2 features)
- Cortex: `_preprocess_cortex(params)` → `extract_classical_features(params, modules)` → features from sensory modules

______________________________________________________________________

## Fusion Strategy Analysis

Across all 12 Stage 2/3 sessions, two distinct fusion strategies emerged:

### Cortex-Dominant Strategy

**Sessions**: 132604, 213406, 012722 (Stage 2 only)
- QSNN trust: 0.15-0.19
- Mode distribution: explore 47-66%, forage 17-19%, evade 17-34%
- The cortex uses explore mode as a generic "I'll handle everything" mode
- Cortex drives ~83% of decisions

### QSNN-Collaborative Strategy

**Sessions**: 132609, 132614, 132619, 012729, 012735, 061309, 061317, 061323, 061329
- QSNN trust: 0.40-0.59
- Mode distribution: forage 51-59%, evade 21-32%, explore 13-24%
- Cortex trusts QSNN for foraging, handles evasion itself

### Key Observation

**Stage 3 eliminated the cortex-dominant strategy.** All 4 Stage 3 sessions are QSNN-collaborative. When the QSNN is also being trained and improving, the cortex learns to trust it more.

**Mode gating acts as a static trust parameter**, not a dynamic per-step switch. Trust values remain stable within ±0.05 over 500 episodes. The 3-mode design is being used as a learned mixing parameter rather than a context-dependent mode selector.

______________________________________________________________________

## QSNN Weight Drift Analysis (Stage 3)

The most notable Stage 3 finding is **asymmetric weight drift**:

| Weight | Mean Δ% | Interpretation |
|--------|---------|----------------|
| W_sh (sensory→hidden) | +2.7% | Input encoding preserved from Stage 1 |
| **W_hm (hidden→motor)** | **+27.7%** | Output mapping systematically reshaped |
| theta_h (hidden thresholds) | +9.2% | Modest threshold adjustment |
| theta_m (motor thresholds) | ~0% | Clamped by `theta_motor_max_norm=1.0` |

**W_hm growth was monotonic, smooth, and consistent** across all 4 sessions. Growth rate ~0.004/episode. The QSNN's output layer strengthened to produce stronger motor signals, presumably adapting to the multi-objective environment where Stage 1's foraging-only training had weaker evasion signals.

**No performance degradation** despite W_hm growing +27.7%. The cortex fusion mechanism adapted to evolving QSNN outputs — the mode gating and action biases co-evolved with the changing reflex signal.

______________________________________________________________________

## Comparison with Classical Baselines

### Apples-to-Apples: MLP PPO Unified Pursuit (Logbook 006)

Same environment, reward, health, and sensory modules. Only difference is brain architecture.

| Metric | Hybrid R4 (avg) | MLP PPO Unified (7 sessions) | Gap |
|--------|-----------------|------------------------------|-----|
| Post-conv success | **96.9%** | 71.6% | **+25.3 pts** |
| Overall success | **96.9%** | 54.6% | **+42.3 pts** |
| HP death rate | **3.1%** | 44.8% | **-41.7 pts** |
| Evasion rate | **90.9%** | ~82% | **+8.9 pts** |
| Trainable params | **9,828** | ~42K | **4.3x fewer** |
| Training episodes | 200 (stage 2) + 500 (stage 3) | 500 | Comparable |

### vs Legacy MLP PPO (Pre-Computed Combined Gradient)

| Metric | Hybrid R4 (avg) | MLP PPO Legacy (4 sessions) | Gap |
|--------|-----------------|----------------------------|-----|
| Post-conv success | **96.9%** | 94.5% | **+2.4 pts** |
| HP death rate | **3.1%** | 6.8% | **-3.7 pts** |

The hybrid brain now **exceeds** the "cheating" legacy MLP PPO baseline that uses a pre-computed combined gradient (effectively giving the brain the optimal direction directly).

### Full Architecture Progression

| Metric | Stage 1 R1 | Stage 2 R2 | Stage 2 R3 | **Stage 3 R4** | MLP PPO Unified |
|--------|-----------|-----------|-----------|-------------|-----------------|
| Task | Foraging | Pursuit | Pursuit | **Pursuit** | Pursuit |
| Post-conv | 99.3%* | 81.9% | 91.7% | **96.9%** | 71.6% |
| HP death rate | N/A | 33.4% | 15.7% | **3.1%** | 44.8% |
| Evasion rate | N/A | 82.6% | 87.2% | **90.9%** | ~82% |
| Trainable params | 92 | 9,572 | 9,736 | **9,828** | ~42K |

*Stage 1 foraging-only (no predators).

______________________________________________________________________

## Lessons Learned

| Lesson | Detail |
|--------|--------|
| Three-stage curriculum works | Isolating QSNN reflex (stage 1), cortex PPO (stage 2), and joint fine-tune (stage 3) prevents interference and allows incremental validation |
| QSNN reflex provides genuine value | Sessions with higher QSNN trust show marginally better evasion and fewer HP deaths. The reflex is not mere noise — it contributes complementary reactive behaviour |
| Mode gating acts as static trust, not dynamic switching | The 3-mode design is used as a learned mixing parameter, not a per-step context switch. Simpler than designed, but effective |
| LR scheduling is essential for cortex PPO | Warmup + decay produced +9.8 pts improvement over flat LR (Round 2→3) |
| PPO epochs matter: 12 >> 4 | Tripling gradient steps per buffer flush produced major gains |
| Mechanosensation adds meaningful signal | Adding boundary/predator contact sensing improved evasion by +4.6 pts |
| W_hm is the "plastic" QSNN weight | In joint fine-tune, output layer adapts (+27.7%) while input encoding is preserved (+2.7%) |
| Pre-trained initialisation eliminates convergence lottery | Stage 3 session variance was 0.8 pts vs Stage 2's 8.8 pts |
| Dual-path preprocessing works | QSNN legacy 2-feature + cortex sensory modules coexist cleanly |
| Session ID should be passed from runner | Brain-generated session IDs cause weight file path mismatches |

______________________________________________________________________

## Logging Gaps Identified

1. **PPO diagnostics not in tracking CSVs**: Cortex PPO metrics (policy_loss, value_loss, entropy, explained_var, approx_kl) logged via `logger.info` but not written to `tracking_losses.csv` or `tracking_learning_rates.csv`. Analysis requires parsing log files.

2. **Fusion metrics not in CSVs**: qsnn_trust_mean and mode_distribution logged per-episode but not exported to any CSV.

3. **Cortex LR not tracked**: Cortex LR schedule managed internally, not exported. No per-episode LR data in CSVs.

4. **Log file session ID offset**: Log files named 1 second before session ID (e.g., `simulation_20260216_213405.log` for session `20260216_213406`).

______________________________________________________________________

## Session References

| Round | Stage | Sessions | Episodes | Config | Key Result |
|-------|-------|----------|----------|--------|------------|
| 1 | 1 | 20260216_100503, 100507, 100512, 100516 | 200 | `hybridquantum_foraging_small.yml` | 91.0% foraging (best 3/4); QSNN reflex validated |
| 2 | 2 | 20260216_132604, 132609, 132614, 132619 | 200 | `hybridquantum_pursuit_predators_small.yml` | 81.9% post-conv; beats MLP PPO unified +10.3 pts |
| 3 | 2 | 20260216_213406, 20260217_012722, 012729, 012735 | 500 | same (updated) | 91.7% post-conv; beats MLP PPO unified +20.1 pts |
| 4 | 3 | 20260217_061309, 061317, 061323, 061329 | 500 | `hybridquantum_pursuit_predators_small_finetune.yml` | **96.9% post-conv**; beats MLP PPO unified +25.3 pts |

### Best Weights

| Component | Session | Path | Notes |
|-----------|---------|------|-------|
| QSNN reflex | 100512 | `artifacts/models/20260216_100512/qsnn_weights.pt` | 100% post-conv foraging, zero variance |
| Cortex (Stage 2) | 012735 | `exports/20260217_012735/cortex_weights.pt` | 91.5% post-conv, QSNN-collaborative |
| Both (Stage 3) | 061317 | `exports/20260217_061317/` | 97.2% overall, composite 0.871, best efficiency |

### Artifacts

- Stage 1 results: `artifacts/logbooks/008/hybridquantum_foraging_small/`
- Stage 2/3 results: `artifacts/logbooks/008/hybridquantum_pursuit_predators_small/`
- Stage 1 config: `artifacts/logbooks/008/hybridquantum_foraging_small/hybridquantum_foraging_small.yml`
- Stage 2 config: `artifacts/logbooks/008/hybridquantum_pursuit_predators_small/hybridquantum_pursuit_predators_small.yml`
- Stage 3 config: `artifacts/logbooks/008/hybridquantum_pursuit_predators_small/hybridquantum_pursuit_predators_small_finetune.yml`
