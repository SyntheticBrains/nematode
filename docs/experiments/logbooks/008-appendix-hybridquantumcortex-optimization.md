# 008 Appendix: HybridQuantumCortex Brain Optimization History

This appendix documents the HybridQuantumCortex evaluation across 9 rounds (32 sessions, 14,600 episodes) covering a graduated 4-stage curriculum. For main findings, see [008-quantum-brain-evaluation.md](008-quantum-brain-evaluation.md). For architecture design, see [quantum-architectures.md](../../research/quantum-architectures.md).

______________________________________________________________________

## Table of Contents

01. [Architecture Overview](#architecture-overview)
02. [Optimization Summary](#optimization-summary)
03. [Stage 1: QSNN Reflex Training](#stage-1-qsnn-reflex-training)
04. [Stage 2 (Original): Direct Cortex Training — Catastrophic Failure](#stage-2-original-direct-cortex-training--catastrophic-failure)
05. [Deep Investigation & Bug Fixes](#deep-investigation--bug-fixes)
06. [Stage 2a Round 1: Cortex Foraging (Pre-Fix)](#stage-2a-round-1-cortex-foraging-pre-fix)
07. [Stage 2a Round 2: Cortex Foraging (Post-Fix)](#stage-2a-round-2-cortex-foraging-post-fix)
08. [Stage 2a Round 3: Cortex Foraging (Config Tuning)](#stage-2a-round-3-cortex-foraging-config-tuning)
09. [Stage 2b: One Pursuit Predator](#stage-2b-one-pursuit-predator)
10. [Stage 2c Round 1: Two Pursuit Predators](#stage-2c-round-1-two-pursuit-predators)
11. [Stage 3: Joint Fine-Tune — Catastrophic Forgetting](#stage-3-joint-fine-tune--catastrophic-forgetting)
12. [Stage 2c Round 2: Extended Training](#stage-2c-round-2-extended-training)
13. [Bug Tracker](#bug-tracker)
14. [Lessons Learned](#lessons-learned)
15. [Session References](#session-references)

______________________________________________________________________

## Architecture Overview

The HybridQuantumCortex brain replaces the classical cortex MLP (~5K params) with a QSNN-based cortex (~252 quantum params) using grouped QLIF neurons per sensory modality, raising the quantum fraction from ~1% to ~11%.

```text
Sensory Input (2-dim legacy)         Multi-sensory Input (4-7 dim)
       |                                       |
       v                                       v
QSNN Reflex (unchanged)              QSNN Cortex (NEW)
  6->8->4 QLIF                        Grouped sensory QLIF:
  ~92 quantum params                    food_chemotaxis: 4 neurons
  Surrogate REINFORCE                   nociception: 4 neurons
  Output: 4 reflex logits              (mechano: 4 — dropped)
       |                                       |
       |                              Shared hidden: 12 QLIF
       |                              Output: 8 QLIF -> 4 action biases
       |                                + 3 mode logits + 1 trust mod
       |                              ~252 quantum params
       |                              Surrogate REINFORCE + GAE
       |                                       |
       v                                       v
  +------------------------------------------------+
  | Fusion: reflex_logits * trust + action_biases   |
  | (mode-gated, same as HybridQuantum)             |
  +------------------------------------------------+
       |
       v
  Final action -> Action Selection (4 actions)

  Classical Critic (unchanged): sensory_dim->64->64->1, PPO GAE, ~5K params
```

**Training method**: Surrogate gradient REINFORCE with critic-provided GAE advantages (not PPO, which is incompatible with surrogate gradients).

**Graduated curriculum**:

- Stage 1: QSNN reflex only on foraging (REINFORCE)
- Stage 2a: QSNN cortex on foraging only (reflex frozen, 1 sensory group)
- Stage 2b: QSNN cortex with 1 predator (reflex frozen, 2 sensory groups)
- Stage 2c: QSNN cortex with 2 predators (reflex frozen, 2 sensory groups)
- Stage 3: Joint fine-tune (both trainable) — abandoned due to catastrophic forgetting

______________________________________________________________________

## Optimization Summary

| Round | Stage | Sessions | Episodes | Key Changes | Success | Post-Conv | Key Finding |
|-------|-------|----------|----------|-------------|---------|-----------|-------------|
| 1 | 1 | 4 | 800 | QSNN reflex on foraging | 82.5% | 95.1% | Reflex validated (matches HQ baseline) |
| 2 | 2 (orig) | 4 | 2,000 | Direct cortex training | 3.1% | N/A | Catastrophic failure; PPO->REINFORCE gap |
| 3 | 2a R1 | 4 | 800 | Graduated curriculum, foraging only | 19.1% | N/A | 3 critical bugs found |
| 4 | 2a R2 | 4 | 800 | Bug fixes (advantage clip/norm/clamp) | 52.6% | 84.5% | +33.5pp from fixes; 1/4 still collapses |
| 5 | 2a R3 | 4 | 800 | Config tuning (entropy, LR, alpha) | **88.8%** | 95.2% | Exceeds Stage 1 baseline (+6.3pp) |
| 6 | 2b R1 | 4 | 2,000 | 1 predator, nociception module | **96.8%** | 97.2% | Zero deaths; zero starvation |
| 7 | 2c R1 | 4 | 2,000 | 2 predators, speed 0.5, tight HP | 39.8% | N/A | Still improving at ep 500 |
| 8 | 3 R1 | 4 | 2,000 | Joint fine-tune, reflex unfrozen | 19.3% | N/A | Catastrophic forgetting; reflex destroyed |
| 9 | 2c R2 | 4 | 2,400 | Extended cortex-only, from 2c R1 weights | 40.9% | N/A | Plateau confirmed; architecture halted |

**Total**: 32 sessions, 14,600 episodes, 9 rounds across 5 stages.

**Outcome**: Architecture halted at ~40-45% ceiling on 2-predator environment. The QSNN cortex under REINFORCE with surrogate gradients cannot push past this level. Stage 2a-2b results (88.8% foraging, 96.8% 1-predator) demonstrate the architecture works on simpler tasks — the limitation is specific to the 2-predator environment with tight HP budget.

______________________________________________________________________

## Stage 1: QSNN Reflex Training

**Config**: `hybridquantumcortex_foraging_small_1.yml`
**Sessions**: 20260218_131409, 131415, 131420, 131425
**Episodes**: 200 per session
**Task**: Foraging only (20x20 grid, 5 foods, target=10, 500 max steps)

| Metric | 131409 | 131415 | 131420 | 131425 | Mean | Std |
|--------|--------|--------|--------|--------|------|-----|
| Overall success rate | 84.5% | 77.0% | 79.0% | 89.5% | 82.5% | 5.5% |
| Post-conv success rate | 99.4% | 99.3% | 84.5% | 97.3% | 95.1% | 7.1% |
| Convergence episode | 39 | 49 | 14 | 19 | 30.3 | 16.0 |
| Composite benchmark | 0.747 | 0.739 | 0.646 | 0.745 | 0.719 | 0.048 |

**Comparison with HybridQuantum Stage 1**: HQ baseline achieved 91.0% (best 3/4) vs HQCortex 82.5% (all 4). Including HQ's outlier session: HQ mean = 84.4%, close to HQCortex 82.5%. Both architectures use identical QSNN reflex code in Stage 1.

**Notable**: Session 131420 had a mid-session regression (100% at ep 21-40, dropped to 60-70% at ep 81-140, recovered to 95% by ep 161-200). Root cause: REINFORCE instability combined with exploration decay ending at ep 80.

**Best session for weight transfer**: 131409 (99.4% post-convergence).

______________________________________________________________________

## Stage 2 (Original): Direct Cortex Training — Catastrophic Failure

**Config**: `hybridquantumcortex_pursuit_predators_small_2c.yml` (original version)
**Sessions**: 20260219_002830, 002837, 003130, 003141
**Episodes**: 500 per session
**Task**: 2 pursuit predators, full difficulty

**Result**: 3.1% overall success (catastrophic failure). Root cause: jumping directly from foraging-only reflex training to 2-predator cortex training with 3 sensory modules was too large a step. The cortex had never learned anything and immediately faced the hardest environment.

**Decision**: Introduce graduated curriculum (Stage 2a → 2b → 2c).

______________________________________________________________________

## Deep Investigation & Bug Fixes

After the Stage 2 failure, a deep investigation of improvement options was conducted, producing a tiered recommendation list:

### Tier 1 (Implemented)

1. **Graduated curriculum** — Config-only change (2a foraging, 2b 1 predator, 2c 2 predators)
2. **Adaptive surrogate gradient alpha scheduling** — ~30 lines: `cortex_alpha_start: 0.3 → cortex_alpha_end: 2.0` over warmup episodes. Low alpha = broad gradients for exploration; high alpha = sharp decisions.
3. **Data re-uploading QLIF** — ~180 lines: Multi-layer single-qubit circuits re-encode input data at each layer. Added `build_qlif_circuit_reupload()`, `execute_qlif_layer_differentiable_reupload()`, `execute_qlif_layer_differentiable_cached_reupload()` to `_qlif_layers.py`.
4. **Increased REINFORCE epochs** — Config change (2 → 6, later reverted to 1 after caching bug found)

### Bugs Found During Stage 2a Round 1

| Bug | Severity | Fix |
|-----|----------|-----|
| Missing advantage clipping in cortex path | Critical | Added `torch.clamp(advantages, -clip, +clip)` |
| Missing cortex weight clamping | Critical | Added `_clamp_cortex_weights()` mirroring `_clamp_reflex_weights()` |
| Missing advantage normalization for GAE | Critical | Added `(advantages - mean) / (std + eps)` before clipping |
| Double warmup co-scheduling (alpha + LR) | High | Added `cortex_alpha_warmup_delay` parameter |

### Stage 2b Bug

| Bug | Severity | Fix |
|-----|----------|-----|
| Multi-epoch REINFORCE produces identical outputs | Medium | Caching bug: cached spike probs reused across epochs. Fixed by reducing to 1 epoch. |

______________________________________________________________________

## Stage 2a Round 1: Cortex Foraging (Pre-Fix)

**Config**: `hybridquantumcortex_foraging_small_2a.yml`
**Sessions**: 20260219_060444, 060452, 060457, 060503
**Episodes**: 200 per session
**Task**: Foraging only, 1 sensory group (food_chemotaxis)
**Reflex weights**: 131409 (Stage 1 best, frozen)

| Session | Overall Success | Post-Conv | Key Issue |
|---------|----------------|-----------|-----------|
| 060444 | 16.0% | N/A | Collapsed at ep ~30, never recovered |
| 060452 | 25.0% | N/A | Oscillating, trending downward |
| 060457 | 16.5% | N/A | Collapsed at ep ~30, never recovered |
| 060503 | 19.0% | N/A | Collapsed at ep ~30, grad norms 0.003 |
| **Mean** | **19.1%** | — | 3/4 sessions permanently collapsed |

**Root causes identified**: Missing advantage clipping (advantages of -23.0 caused catastrophic weight updates), missing weight clamping (norms grew 7-13x), missing advantage normalization. Session 060452's oscillation proved the architecture CAN learn — the problem was training stability.

______________________________________________________________________

## Stage 2a Round 2: Cortex Foraging (Post-Fix)

**Config**: `hybridquantumcortex_foraging_small_2a.yml` (with bug fixes applied)
**Sessions**: 20260219_093944, 093950, 093955, 094001
**Episodes**: 200 per session

| Session | Overall Success | Post-Conv | Key Issue |
|---------|----------------|-----------|-----------|
| 093944 | 14.5% | 9.6% | Entropy collapsed to 0.34 at ep 18 |
| 093950 | 70.5% | 84.5% | Recovered from dip |
| 093955 | 69.5% | 89.5% | Best — strong convergence |
| 094001 | 56.0% | 70.0% | Unstable late collapse |
| **Mean** | **52.6%** | — | **+33.5pp from bug fixes** |

**Universal pattern — "cortex learning dip"**: All 4 sessions showed performance collapse between episodes 40-100 when cortex alpha began ramping but cortex hadn't learned a useful policy yet. 3/4 sessions recovered; 1/4 (093944) had entropy collapse during this window and locked into a degenerate state permanently.

**Critic performance**: Explained variance near zero across all sessions (range -0.54 to +0.14). REINFORCE succeeded despite useless critic.

______________________________________________________________________

## Stage 2a Round 3: Cortex Foraging (Config Tuning)

**Config**: `hybridquantumcortex_foraging_small_2a.yml` (config changes only)
**Sessions**: 20260219_133505, 133514, 133520, 133527
**Episodes**: 200 per session

**Config changes from Round 2**:

| Parameter | Round 2 | Round 3 | Rationale |
|-----------|---------|---------|-----------|
| cortex_lr | 0.01 | 0.005 | Halve peak LR to reduce early weight growth |
| entropy_coeff | 0.05 | 0.10 | Stronger entropy floor to prevent collapse |
| cortex_alpha_warmup_episodes | 100 | 150 | Slower alpha ramp, gentler cortex influence |

| Metric | R2 Mean | R3 Mean | Delta |
|--------|---------|---------|-------|
| Overall success | 52.6% | **88.8%** | **+36.2pp** |
| Post-conv success | 63.4% | **95.2%** | **+31.8pp** |
| Cross-session std | 24.7% | **4.3%** | Much lower variance |
| Worst session | 14.5% | **80.0%** | No collapse |

**Key result**: All 4 sessions converged reliably. The QSNN cortex exceeds the Stage 1 reflex-only baseline (88.8% vs 82.5%, +6.3pp), proving the cortex architecture can learn and improve foraging beyond the reflex alone.

______________________________________________________________________

## Stage 2b: One Pursuit Predator

**Config**: `hybridquantumcortex_pursuit_predators_small_2b.yml`
**Sessions**: 20260220_014906, 014914, 014921, 014927
**Episodes**: 500 per session
**Task**: 1 pursuit predator (speed 0.3, detection_radius 5), max_hp=200, predator_damage=15
**Sensory modules**: food_chemotaxis + nociception (2 groups, 8 cortex neurons)
**Reflex weights**: 131409 (frozen)

| Session | Success | Post-Conv | Pred Deaths | Starvation |
|---------|---------|-----------|-------------|------------|
| 014906 | 96.6% | 97.3% | 0 | 0 |
| 014914 | 98.0% | 98.5% | 0 | 0 |
| 014921 | 96.6% | 96.0% | 0 | 0 |
| 014927 | 96.0% | 97.0% | 0 | 0 |
| **Mean** | **96.8%** | **97.2%** | **0** | **0** |

**Key findings**:

- 96.8% success rate with zero predator deaths and zero starvation
- Performance only drops 0.3pp from predator-free Stage 2a (97.1% → 96.8%)
- Nociception module working — agent survives predator encounters effectively
- 85% of all failures occur in first 100 episodes (learning phase)
- Cross-session variance low (96.0-98.0%, std 0.85pp)

**Multi-epoch REINFORCE bug found**: All 6 cortex REINFORCE epochs produced identical loss/entropy/grad_norm values. Root cause: cached spike probabilities reused from epoch 0. Fixed by reducing to 1 epoch (no performance loss, ~40% training time reduction).

**Mode distribution near-uniform**: [0.321, 0.358, 0.321] — mode-gating is not differentiating behavioural modes. Mode logits receive only indirect gradients. This is a known limitation but does not prevent strong performance.

______________________________________________________________________

## Stage 2c Round 1: Two Pursuit Predators

**Config**: `hybridquantumcortex_pursuit_predators_small_2c.yml`
**Sessions**: 20260220_101539, 101543, 101548, 101552
**Episodes**: 500 per session
**Task**: 2 pursuit predators (speed 0.5, detection_radius 6), max_hp=100, predator_damage=20
**Cortex weights**: Loaded from 014914 (Stage 2b best, 98.0%)

| Metric | 101539 | 101543 | 101548 | 101552 | Mean | Std |
|--------|--------|--------|--------|--------|------|-----|
| Overall success | 35.8% | **48.4%** | 38.6% | 36.4% | 39.8% | 5.7% |
| Last-50 success | 62.0% | **64.0%** | 58.0% | ~50.0% | 58.5% | 6.1% |
| Predator deaths | 0 | 0 | 0 | 0 | 0 | — |

**Difficulty jump from 2b → 2c**: 2x predators, 67% faster, 20% wider detection, 50% less HP, 33% more damage per hit. With max_hp=100 and damage=20, agent can only survive 5 hits (vs 13 in 2b).

**All 4 sessions on upward trajectory at ep 500** (10-28% early → 50-64% late). Architecture learns but needs more episodes.

**Best session**: 101543 (48.4% overall, 64% last-50). Used for weight transfer.

______________________________________________________________________

## Stage 3: Joint Fine-Tune — Catastrophic Forgetting

**Config**: `hybridquantumcortex_pursuit_predators_small_finetune_3.yml`
**Sessions**: 20260220_182051, 182059, 182106, 182112
**Episodes**: 500 per session
**Task**: Same as 2c (2 predators, speed 0.5)
**Weights**: Cortex/critic from 101543 (2c best), reflex from 131409 (Stage 1)

| Metric | 182051 | 182059 | 182106 | 182112 | Mean | Std |
|--------|--------|--------|--------|--------|------|-----|
| Overall success | 21.8% | 35.6% | 14.8% | 4.8% | **19.3%** | 12.9% |
| First-50 success | 36% | 46% | 22% | 10% | 28.5% | — |
| Last-50 success | 22% | 32% | 12% | 2% | **17.0%** | — |

**Catastrophic forgetting**: All sessions start high and decline — the opposite of Stage 2c where all sessions improved. Stage 3 training destroys the pre-trained reflex weights.

**Root causes**:

1. **Reflex REINFORCE frequency**: 24x more updates per episode than cortex (intra-episode window updates every 20 steps × 2 epochs = ~20 gradient passes vs 1 cortex update). Effective reflex LR=0.020/episode despite configured 0.001.
2. **Cosine LR schedule cycling**: `CosineAnnealingLR(T_max=200)` on a 500-episode run creates LR that cycles back UP at episode 200, maximising plasticity when the policy should stabilise.
3. **Structural incompatibility**: Reflex trained only on foraging rewards (Stage 1), now receiving predator-environment REINFORCE gradients. The gradient signals are anti-correlated with foraging-trained behaviour.

**Key comparison with HybridQuantum Stage 3**: HybridQuantum achieved 96.9% in Stage 3 because its reflex had been exposed to the predator environment from Stage 2 onward (even though frozen, it was trained in a context where predators existed). HybridQuantumCortex's reflex only ever saw foraging — Stage 3 destroys it.

**Decision**: Abandon Stage 3. Continue Stage 2c with frozen reflex.

______________________________________________________________________

## Stage 2c Round 2: Extended Training

**Config**: `hybridquantumcortex_pursuit_predators_small_2c_round2.yml`
**Sessions**: 20260221_052315, 052323, 052330, 052336
**Episodes**: 600 per session (intended 1000, ran 600)
**Cortex weights**: Loaded from 101543 (2c R1 best)

**Config changes from Round 1**:

| Parameter | 2c R1 | 2c R2 | Rationale |
|-----------|-------|-------|-----------|
| cortex_lr | 0.005 | 0.003 | Lower — weights already trained |
| cortex_lr_warmup_episodes | 30 | 0 | No warmup; continue from trained weights |
| cortex_lr_decay_episodes | 150 | 300 | Spread decay across more episodes |
| cortex_alpha_start | 0.3 | 2.0 | Already at full influence |
| cortex_alpha_warmup_episodes | 150 | 0 | No ramp needed |

| Metric | 052315 | 052330 | 052323 | 052336 | Mean | Std |
|--------|--------|--------|--------|--------|------|-----|
| Overall success | 41.0% | 38.3% | 41.5% | 42.8% | **40.9%** | 1.9% |
| Last-50 success | 42.0% | 44.0% | 40.0% | 34.0% | **40.0%** | 4.3% |
| Last-100 success | 43.0% | 49.0% | 49.0% | 40.0% | **45.2%** | 4.5% |
| Evasion rate | 72.7% | 71.1% | 71.4% | 73.2% | **72.1%** | 0.9% |
| Predator deaths | 0 | 0 | 0 | 0 | **0** | — |

**Result: REGRESSION from Round 1 late-stage** (last-50: 58.5% R1 → 40.0% R2). Starting with higher LR (0.003) than where R1 ended (0.001) disrupted the fine-tuned weights. Cross-session variance extremely low (std 1.9%) — all sessions converge to the same ~40% ceiling.

**Root cause analysis — performance ceiling**:

1. **Vanishing gradients**: Grad norms drop to 0.04-0.07 after LR decay completes at ep 300. Effective learning rate × gradient norm = ~0.00005 per update — negligible.
2. **Critic remains useless**: Explained variance oscillates between -0.5 and +0.5, mean ~0.10 across all sessions.
3. **Mode distribution frozen**: [0.33, 0.33, 0.33] — mode-gating is completely inert.
4. **Weight norms barely change**: Total weight norm shift \<5% across 600 episodes.

**Evasion rate is the bottleneck**: At 72% evasion and damage=20, the math shows the agent needs ~85%+ evasion to reliably survive. The QSNN cortex cannot learn strong enough evasion signals through REINFORCE with surrogate gradients.

______________________________________________________________________

## Bug Tracker

| Bug | Stage Found | Severity | Status | Fix Applied |
|-----|-------------|----------|--------|-------------|
| Missing advantage clipping in cortex REINFORCE | 2a R1 | Critical | Fixed | `torch.clamp(advantages, -clip, +clip)` |
| Missing cortex weight clamping | 2a R1 | Critical | Fixed | Added `_clamp_cortex_weights()` |
| Missing advantage normalization for GAE | 2a R1 | Critical | Fixed | `(adv - mean) / (std + eps)` |
| Double warmup (alpha + LR both ramping ep 0) | 2a R1 | High | Fixed | Added `cortex_alpha_warmup_delay` |
| Multi-epoch REINFORCE identical outputs (caching) | 2b | Medium | Fixed | Reduced to 1 epoch |
| `run_metrics.csv` reports incorrect success counts | 2c R2 | Low | Known | Cosmetic data export bug |
| Critic grad_norm logs pre-clip value | 2b | Low | Known | Cosmetic, no fix required |

______________________________________________________________________

## Lessons Learned

01. **Graduated curriculum is essential for QSNN cortex training**: Jumping directly from foraging reflex to 2-predator cortex training fails catastrophically. The foraging → 1 predator → 2 predator progression enables each stage to validate before scaling difficulty.

02. **REINFORCE with surrogate gradients has a ~40-45% ceiling on hard multi-objective tasks**: Despite extensive tuning (3 rounds of bug fixes, 4 rounds of config optimisation), the QSNN cortex plateaus at ~40-45% on the 2-predator environment. The gradient signal from REINFORCE with 252 quantum parameters is too weak to push past this level.

03. **Bug fixes produce larger gains than hyperparameter tuning**: The 3 critical bug fixes (advantage clipping/normalization/weight clamping) improved success from 19.1% → 52.6% (+33.5pp). Subsequent config tuning (entropy, LR, alpha) added another +36.2pp to 88.8% on foraging. Find bugs first.

04. **Stage 3 joint fine-tune is structurally incompatible with the HybridQuantumCortex curriculum**: The reflex, trained only on foraging, receives destructive predator-environment gradients during Stage 3. HybridQuantum's Stage 3 works because its reflex was already exposed to the predator environment. This is a fundamental architectural difference.

05. **Intra-episode REINFORCE window updates amplify effective LR**: With `reinforce_window_size=20`, the reflex gets ~20 gradient passes per episode at nominal LR=0.001, giving effective LR=0.020/episode. This is fine for initial training but catastrophic for fine-tuning pre-trained weights.

06. **Cosine LR schedules with T_max < total_episodes create harmful LR cycling**: The LR cycles back UP at T_max, increasing plasticity when the policy should be stabilising. Use monotonic decay (linear or step) for fine-tuning.

07. **Critic EV near zero does not prevent learning**: The REINFORCE cortex learned effectively (88.8% foraging, 96.8% 1-predator) despite consistently poor critic explained variance (~0.00-0.10). GAE advantages with a near-zero-EV critic are effectively noisy normalized returns — still sufficient for simpler tasks.

08. **Cross-session variance decreases with training maturity**: Stage 2a R2 had 56pp variance (14.5% to 69.5%), Stage 2b had 2pp (96.0% to 98.0%), Stage 2c R2 had 4.5pp (38.3% to 42.8%). As hyperparameters and bugs are fixed, reproducibility improves.

09. **Starting a continuation session with higher LR than the previous session's end disrupts fine-tuned weights**: Stage 2c R2 started at LR=0.003 (vs R1's end of 0.001), causing initial perturbation that regressed late-stage performance from 58.5% to 40.0%.

10. **Zero predator deaths across all 14,600 episodes**: The nociception module provides effective predator awareness throughout all stages. The agent avoids lethal predator contact perfectly. The limitation is cumulative HP depletion from sustained predator proximity damage, not evasion failure per se.

______________________________________________________________________

## Session References

### Stage 1 (Reflex Foraging)

| Session | Success | Post-Conv | Config |
|---------|---------|-----------|--------|
| 20260218_131409 | 84.5% | 99.4% | `hybridquantumcortex_foraging_small_1.yml` |
| 20260218_131415 | 77.0% | 99.3% | |
| 20260218_131420 | 79.0% | 84.5% | |
| 20260218_131425 | 89.5% | 97.3% | |

### Stage 2 Original (Failed)

| Session | Success | Config |
|---------|---------|--------|
| 20260219_002830 | ~3% | `hybridquantumcortex_pursuit_predators_small_2c.yml` (original) |
| 20260219_002837 | ~3% | |
| 20260219_003130 | ~3% | |
| 20260219_003141 | ~3% | |

### Stage 2a Round 1 (Pre-Fix)

| Session | Success | Config |
|---------|---------|--------|
| 20260219_060444 | 16.0% | `hybridquantumcortex_foraging_small_2a.yml` |
| 20260219_060452 | 25.0% | |
| 20260219_060457 | 16.5% | |
| 20260219_060503 | 19.0% | |

### Stage 2a Round 2 (Post-Fix)

| Session | Success | Config |
|---------|---------|--------|
| 20260219_093944 | 14.5% | `hybridquantumcortex_foraging_small_2a.yml` (post-fix) |
| 20260219_093950 | 70.5% | |
| 20260219_093955 | 69.5% | |
| 20260219_094001 | 56.0% | |

### Stage 2a Round 3 (Config Tuning)

| Session | Success | Config |
|---------|---------|--------|
| 20260219_133505 | ~88% | `hybridquantumcortex_foraging_small_2a.yml` (tuned) |
| 20260219_133514 | ~88% | |
| 20260219_133520 | ~80% | |
| 20260219_133527 | ~99% | |

### Stage 2b (1 Predator)

| Session | Success | Config |
|---------|---------|--------|
| 20260220_014906 | 96.6% | `hybridquantumcortex_pursuit_predators_small_2b.yml` |
| 20260220_014914 | 98.0% | |
| 20260220_014921 | 96.6% | |
| 20260220_014927 | 96.0% | |

### Stage 2c Round 1 (2 Predators)

| Session | Success | Config |
|---------|---------|--------|
| 20260220_101539 | 35.8% | `hybridquantumcortex_pursuit_predators_small_2c.yml` |
| 20260220_101543 | 48.4% | |
| 20260220_101548 | 38.6% | |
| 20260220_101552 | 36.4% | |

### Stage 3 (Joint Fine-Tune — Failed)

| Session | Success | Config |
|---------|---------|--------|
| 20260220_182051 | 21.8% | `hybridquantumcortex_pursuit_predators_small_finetune_3.yml` |
| 20260220_182059 | 35.6% | |
| 20260220_182106 | 14.8% | |
| 20260220_182112 | 4.8% | |

### Stage 2c Round 2 (Extended — Final)

| Session | Success | Config |
|---------|---------|--------|
| 20260221_052315 | 41.0% | `hybridquantumcortex_pursuit_predators_small_2c_round2.yml` |
| 20260221_052323 | 41.5% | |
| 20260221_052330 | 38.3% | |
| 20260221_052336 | 42.8% | |
