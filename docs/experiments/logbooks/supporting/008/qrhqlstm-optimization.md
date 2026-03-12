# QRH-QLSTM / CRH-QLSTM Optimization History

**Architecture**: Reservoir-LSTM composition — fixed reservoir (QRH quantum or CRH classical) as feature extractor + QLIF-LSTM temporal readout with recurrent PPO (chunk-based truncated BPTT). Tests whether composing QRH's rich reservoir features with QLIF-LSTM's temporal memory improves over either standalone architecture.

**Total**: 15 rounds, 54 sessions, ~16,500 episodes across foraging, pursuit predators, and stationary predators (classical + quantum gates, QRH + CRH reservoirs).

**Conclusion**: Reservoir-LSTM composition does not improve over simpler architectures. CRH-QLSTM excels on small pursuit predators (85.4%, best reservoir variant) but fails to scale to large grids. QRH-QLSTM fails on all multi-objective tasks. LSTM readout hurts QRH vs its simpler MLP readout.

______________________________________________________________________

## Phase 1 — Classical Gates Foraging (Stage 4a Validation)

**Purpose**: Validate reservoir+LSTM composition learns before investing in slow quantum runs.

**Config**: `qrhqlstm_foraging_small_classical.yml` / `crhqlstm_foraging_small_classical.yml` — 200 episodes, 4 sessions each, lstm_hidden_dim=64, bptt_chunk_length=32, rollout_buffer_size=256, actor_lr=0.0005, critic_lr=0.0005, entropy_coef=0.02→0.008, use_quantum_gates=false.

**Environment**: 20x20 grid, 5 foods, target 10, no predators.

### QRH-QLSTM Classical

| Session ID | Success | Post-Conv | Convergence | Time/Ep |
|------------|---------|-----------|-------------|---------|
| 20260310_122324 | 62.0% | 97.2% | Run 92 | ~1.3s |
| 20260310_122327 | 61.0% | 93.5% | Run 77 | ~1.4s |
| 20260310_122332 | 61.5% | 98.2% | Run 87 | ~1.4s |
| 20260310_122336 | 67.0% | 97.6% | Run 75 | ~1.3s |
| **Mean** | **62.9%** | **96.6%** | **82.8** | **~1.3s** |

### CRH-QLSTM Classical

| Session ID | Success | Post-Conv | Convergence | Time/Ep |
|------------|---------|-----------|-------------|---------|
| 20260310_122353 | 86.5% | 100% | Run 28 | ~0.9s |
| 20260310_122356 | 84.5% | 99.4% | Run 40 | ~0.9s |
| 20260310_122358 | 86.0% | 97.7% | Run 29 | ~0.9s |
| 20260310_122401 | 88.0% | 100% | Run 26 | ~0.9s |
| **Mean** | **86.3%** | **99.3%** | **30.8** | **~0.9s** |

**Key finding**: CRH converges 2.7x faster than QRH (31 vs 83 episodes), +23pp overall success. QRH's quantum reservoir noise delays learning.

**Decision**: **PASS** — both ≥80% post-convergence. Proceed to quantum gates.

______________________________________________________________________

## Phase 2 — Quantum Gates Foraging (Stage 4a-Q)

**Purpose**: Confirm quantum QLIF gates converge. Compare gate type impact.

**Config**: Same as Phase 1 but `use_quantum_gates: true`, shots=1024, membrane_tau=0.9.

### QRH-QLSTM Quantum

| Session ID | Success | Post-Conv | Convergence | Time/Ep |
|------------|---------|-----------|-------------|---------|
| 20260310_123914 | 60.0% | 99.1% | Run 86 | ~3.5 min |
| 20260310_123916 | 69.5% | 99.1% | Run 89 | ~3.4 min |
| 20260310_123920 | 71.0% | 95.2% | Run 76 | ~3.5 min |
| 20260310_123922 | 66.0% | 98.3% | Run 80 | ~3.3 min |
| **Mean** | **66.6%** | **97.9%** | **82.8** | **~3.4 min** |

### CRH-QLSTM Quantum

| Session ID | Success | Post-Conv | Convergence | Time/Ep |
|------------|---------|-----------|-------------|---------|
| 20260310_123944 | 78.0% | 99.4% | Run 44 | ~2.7 min |
| 20260310_123946 | 87.5% | 99.4% | Run 26 | ~2.7 min |
| 20260310_123950 | 87.5% | 100% | Run 27 | ~2.8 min |
| 20260310_123952 | 87.5% | 100% | Run 27 | ~2.2 min |
| **Mean** | **85.1%** | **99.7%** | **31.0** | **~2.6 min** |

### Classical vs Quantum Gates Comparison

| Metric | QRH Classical | QRH Quantum | CRH Classical | CRH Quantum |
|--------|--------------|-------------|---------------|-------------|
| Overall Success | 62.9% | 66.6% | 86.3% | 85.1% |
| Conv Episode | 82.8 | 82.8 | 30.8 | 31.0 |
| Post-Conv | 96.6% | 97.9% | 99.3% | 99.7% |
| Time/Episode | ~1.3s | ~3.4 min | ~0.9s | ~2.6 min |

**Key insight**: Convergence dynamics identical between classical and quantum gates. Reservoir features determine convergence speed, not gate type. Quantum gates add ~150x time overhead with negligible accuracy difference.

**Decision**: **PASS** — proceed to predators with CRH-QLSTM (QRH too slow to converge).

______________________________________________________________________

## Phase 3 — CRH-QLSTM Pursuit Predators (Quantum Gates)

**Purpose**: Test temporal memory advantage against pursuit threats.

**Config**: `crhqlstm_pursuit_predators_small.yml` — 200 episodes, 4 sessions, quantum gates.

**Environment**: 20x20 grid, 6 foods, target 10, 2 pursuit predators (speed 0.5), health system.

| Session ID | Success | Post-Conv | Convergence | Evasion | Foods |
|------------|---------|-----------|-------------|---------|-------|
| 20260311_015814 | 87.0% | 97.0% | Run 34 | 84.0% | 9.08 |
| 20260311_015819 | 83.5% | 94.8% | Run 29 | 85.4% | 8.85 |
| 20260311_015823 | 84.0% | 94.1% | Run 31 | 85.7% | 8.94 |
| 20260311_015828 | 87.0% | 97.1% | Run 27 | 85.0% | 9.06 |
| **Mean** | **85.4%** | **95.8%** | **30.3** | **85.0%** | **8.98** |

### CRH-QLSTM vs QLIF-LSTM on Pursuit Predators

| Metric | CRH-QLSTM Quantum | QLIF-LSTM Classical | QLIF-LSTM Quantum |
|--------|-------------------|---------------------|-------------------|
| Overall Success | **85.4%** | 74.7% | 70.8% |
| Convergence Ep | **30.3** | 146.3 | 158.3 |
| Post-Conv | **95.8%** | 92.4% | 90.8% |
| Evasion Rate | **85.0%** | 75.4% | 72.4% |
| Conv Variance (std) | **3.0** | 36.2 | 72.4 |

**Key finding**: CRH-QLSTM dramatically outperforms standalone QLIF-LSTM — +10.7pp success, 4.8x faster convergence, 12x lower variance. The CRH echo state network provides features that significantly help pursuit predator evasion.

**Decision**: **PASS** — proceed to advanced environments.

______________________________________________________________________

## Phase 3b — CRH-QLSTM Pursuit Predators (Classical Ablation)

**Purpose**: Measure quantum vs classical gate advantage on pursuit predators.

**Config**: `crhqlstm_pursuit_predators_small_classical.yml` — same environment, use_quantum_gates=false.

| Session ID | Success | Post-Conv | Convergence | Time/Ep |
|------------|---------|-----------|-------------|---------|
| 20260311_111030 | 79.5% | 92.8% | Run 35 | ~0.9s |
| 20260311_111033 | 75.5% | 84.8% | Run 50 | ~0.9s |
| 20260311_111035 | 85.5% | 93.8% | Run 24 | ~0.9s |
| 20260311_111038 | 88.5% | 99.4% | Run 41 | ~0.9s |
| **Mean** | **82.2%** | **92.7%** | **37.5** | **~0.9s** |

**Quantum vs classical**: +3.1pp (quantum), ~170x faster with classical gates. Small but consistent quantum advantage on multi-objective tasks.

______________________________________________________________________

## Phase 3c — QRH-QLSTM Pursuit Predators (Quantum Gates)

**Purpose**: Test QRH quantum reservoir on pursuit predators.

**Config**: `qrhqlstm_pursuit_predators_small.yml` — 200 episodes, 4 sessions, quantum gates.

| Session ID | Success | Conv | Foods | Health% |
|------------|---------|------|-------|---------|
| 20260311_031223 | 18.5% | N | 4.8 | 70.5% |
| 20260311_031227 | 18.5% | N | 5.0 | 72.5% |
| 20260311_031232 | 14.5% | N | 4.4 | 72.0% |
| 20260311_031237 | 9.5% | N | 4.1 | 68.5% |
| **Mean** | **15.2%** | **0/4** | **4.6** | **70.9%** |

**Decision**: **FAIL** — QRH-QLSTM collapses on multi-objective tasks. Quantum reservoir noise that merely slows foraging convergence (2.7x) becomes unlearnable with added predator evasion objective.

______________________________________________________________________

## Phase 4 — CRH-QLSTM Thermotaxis + Pursuit Predators Large (Classical Gates)

**Purpose**: Test CRH-QLSTM at scale with triple-objective task.

**Config**: `crhqlstm_thermotaxis_pursuit_predators_large_classical.yml` — 500 episodes, 4 sessions.

**Environment**: 100x100 grid, 45 foods, target 25, 4 pursuit predators, thermotaxis zones, health=150.

| Session ID | Success | Post-Conv | Conv | Avg Steps | Foods |
|------------|---------|-----------|------|-----------|-------|
| 20260311_112537 | 38.4% | 70.1% | 424 | 1030 | 16.6 |
| 20260311_112540 | 35.0% | 60.0% | N | 952 | 15.1 |
| 20260311_112544 | 34.6% | 60.0% | N | 973 | 15.5 |
| 20260311_112547 | 35.6% | 20.0% | N | 947 | 15.6 |
| **Mean** | **35.9%** | **52.5%** | **1/4** | **975** | **15.7** |

### Phase 4a v2: Hyperparameter-Aligned Re-run

Aligned PPO hyperparameters with QLIF-LSTM's best config. Result: 38.8% avg — near-identical, confirming gap is architectural.

| Model | SR | Steps/Food | Conv Ep |
|-------|-----|-----------|---------|
| CRH-QLSTM v1 | 35.9% | 53.1 | 424 |
| CRH-QLSTM v2 | 38.8% | 50.1 | N |
| **QLIF-LSTM** | **60.1%** | **25.3** | **198** |

**Root cause**: Path efficiency — CRH-QLSTM takes 2x more steps per food (53 vs 25). The reservoir's 75-dim feature expansion obscures gradient signals needed for long-range navigation in 100x100 grids.

______________________________________________________________________

## Phase 4b — CRH-QLSTM Thermotaxis + Stationary Predators Large (Classical Gates)

**Purpose**: Test CRH-QLSTM against fixed toxic zones.

**Config**: `crhqlstm_thermotaxis_stationary_predators_large_classical.yml` — 500 episodes, 4 sessions.

| Session ID | Success | Conv | Foods | Health% |
|------------|---------|------|-------|---------|
| 20260311_124349 | 10.4% | N | 10.1 | 81.8% |
| 20260311_124351 | 12.0% | N | 10.6 | 79.0% |
| 20260311_124355 | 15.2% | N | 11.0 | 77.4% |
| 20260311_124358 | 18.6% | N | 12.7 | 72.6% |
| **Mean** | **14.0%** | **0/4** | **11.1** | **77.7%** |

**Result**: Worse than pursuit predators. 10pp behind QLIF-LSTM (24.0%). Same path efficiency bottleneck.

______________________________________________________________________

## Stage 4d — QRH-LSTM Pursuit Predators Small (Classical Gates)

**Purpose**: Test QRH with classical LSTM readout (use_quantum_gates=false) — isolates whether LSTM temporal readout helps QRH.

**Config**: `qrhqlstm_pursuit_predators_small_classical.yml` — 200 episodes, 4 sessions.

| Session ID | Success | Conv | Foods | Health% |
|------------|---------|------|-------|---------|
| 20260311_220133 | 16.0% | N | 4.57 | 69.5% |
| 20260311_220136 | 15.5% | N | 4.65 | 69.0% |
| 20260311_220141 | 20.0% | 173 | 4.92 | 66.0% |
| 20260311_220145 | 16.5% | N | 4.71 | 66.5% |
| **Mean** | **17.0%** | **1/4** | **4.71** | **67.8%** |

**Key finding**: Classical gates provide no improvement over quantum (+1.8pp, within noise). Fixed quantum reservoir is the bottleneck.

______________________________________________________________________

## Stage 4d — QRH-LSTM Thermotaxis + Pursuit Predators Large (Classical Gates)

**Config**: `qrhqlstm_thermotaxis_pursuit_predators_large_classical.yml` — 500 episodes, 4 sessions.

| Session ID | Success | Conv | Foods | Steps |
|------------|---------|------|-------|-------|
| 20260311_221535 | 14.4% | N | 10.84 | 685 |
| 20260311_221539 | 14.6% | N | 11.05 | 687 |
| 20260311_221543 | 16.2% | N | 11.00 | 719 |
| 20260311_221547 | 20.4% | N | 12.67 | 721 |
| **Mean** | **16.4%** | **0/4** | **11.4** | **703** |

**Result**: Worst among all architectures on this environment. LSTM readout *degrades* QRH vs its MLP readout (16.4% vs 41.3%, -24.9pp).

______________________________________________________________________

## Stage 4d — QRH-LSTM Thermotaxis + Stationary Predators Large (Classical Gates)

**Purpose**: Hypothesis test — "QRH-LSTM should beat QRH-MLP on stationary predators (>=5pp improvement)."

**Config**: `qrhqlstm_thermotaxis_stationary_predators_large_classical.yml` — 500 episodes, 4 sessions.

| Session ID | Success | Conv | Foods | Steps |
|------------|---------|------|-------|-------|
| 20260311_221821 | 11.0% | N | 9.78 | 678 |
| 20260311_221825 | 15.2% | N | 11.31 | 709 |
| 20260311_221828 | 6.6% | N | 9.00 | 719 |
| 20260311_221832 | 10.2% | N | 10.60 | 719 |
| **Mean** | **10.8%** | **0/4** | **10.2** | **706** |

### Hypothesis Test Result

| Metric | QRH-MLP | QRH-LSTM | Delta |
|--------|---------|----------|-------|
| SR | 14.9% | 10.8% | **-4.2pp** |

**HYPOTHESIS REJECTED.** QRH-LSTM is 4.2pp *worse* than QRH-MLP, far from the +5pp target. LSTM temporal readout does NOT resolve QRH's multi-objective weakness.

______________________________________________________________________

## Summary: Architecture Rankings by Environment

### Small Pursuit Predators (20x20)

| Rank | Model | SR | Conv Ep |
|------|-------|-----|---------|
| 1 | CRH-QLSTM Quantum | **85.4%** | 30 |
| 2 | CRH-QLSTM Classical | 82.2% | 38 |
| 3 | QLIF-LSTM Classical | 63.3% | 132 |
| 4 | QLIF-LSTM Quantum | 57.6% | 143 |
| 5 | QRH-LSTM Classical | 17.0% | 1/4 |
| 6 | QRH-QLSTM Quantum | 15.2% | N |

### Large Thermotaxis + Pursuit Predators (100x100)

| Rank | Model | SR | Conv Ep |
|------|-------|-----|---------|
| 1 | QLIF-LSTM Classical | **60.1%** | 198 |
| 2 | QRH standalone (MLP) | 41.3% | 793 |
| 3 | CRH-QLSTM Classical v2 | 38.8% | N |
| 4 | CRH-QLSTM Classical v1 | 35.9% | 424 |
| 5 | QRH-LSTM Classical | 16.4% | N |

### Large Thermotaxis + Stationary Predators (100x100)

| Rank | Model | SR |
|------|-------|----|
| 1 | QLIF-LSTM Classical | **24.0%** |
| 2 | CRH standalone | 23.4% |
| 3 | QRH standalone (MLP) | 14.9% |
| 4 | CRH-QLSTM Classical | 14.0% |
| 5 | QRH-LSTM Classical | 10.8% |

______________________________________________________________________

## Key Conclusions

1. **CRH-QLSTM is the best reservoir-LSTM variant** — classical ESN provides stable features that help the LSTM converge 4.8x faster than standalone QLIF-LSTM on small pursuit predators.

2. **Reservoir-LSTM does NOT scale to large grids** — feature expansion (7 sensory -> 75 reservoir features) helps local reactive evasion but hurts long-range gradient-following (53 vs 25 steps/food).

3. **QRH-QLSTM fails on all multi-objective tasks** — fixed quantum reservoir features are inadequate when evasion is added. The noise that merely slows foraging convergence becomes catastrophic with added objectives.

4. **LSTM readout HURTS QRH performance** — QRH-LSTM is worse than QRH standalone MLP on every large-grid test (-24.9pp pursuit, -4.2pp stationary).

5. **Quantum QLIF gates provide no meaningful advantage** — ~3pp on CRH pursuit predators (not worth 170x speed cost), within noise on QRH.

6. **Path efficiency gap is architectural, not hyperparameter** — confirmed by v2 alignment experiment.

7. **Stage 4d hypothesis REJECTED** — LSTM temporal readout does not improve QRH on stationary predators.

______________________________________________________________________

## Session Reference

| Phase | Config | Sessions | Episodes | IDs |
|-------|--------|----------|----------|-----|
| Phase 1 | qrhqlstm_foraging_small_classical | 4 | 200 | 20260310_122324-122336 |
| Phase 1 | crhqlstm_foraging_small_classical | 4 | 200 | 20260310_122353-122401 |
| Phase 2 | qrhqlstm_foraging_small | 4 | 200 | 20260310_123914-123922 |
| Phase 2 | crhqlstm_foraging_small | 4 | 200 | 20260310_123944-123952 |
| Phase 3 | crhqlstm_pursuit_predators_small | 4 | 200 | 20260311_015814-015828 |
| Phase 3b | crhqlstm_pursuit_predators_small_classical | 4 | 200 | 20260311_111030-111038 |
| Phase 3c | qrhqlstm_pursuit_predators_small | 4 | 200 | 20260311_031223-031237 |
| Phase 4 | crhqlstm_thermotaxis_pursuit_predators_large_classical | 4 | 500 | 20260311_112537-112547 |
| Phase 4 v2 | crhqlstm_thermotaxis_pursuit_predators_large_classical | 4 | 500 | 20260311_121622-121629 |
| Phase 4b | crhqlstm_thermotaxis_stationary_predators_large_classical | 4 | 500 | 20260311_124349-124358 |
| Stage 4d | qrhqlstm_pursuit_predators_small_classical | 4 | 200 | 20260311_220133-220145 |
| Stage 4d | qrhqlstm_thermotaxis_pursuit_predators_large_classical | 4 | 500 | 20260311_221535-221547 |
| Stage 4d | qrhqlstm_thermotaxis_stationary_predators_large_classical | 4 | 500 | 20260311_221821-221832 |
| Smoke | qrhqlstm_foraging_small | 1 | 50 | 20260310_104718 |
| Smoke | crhqlstm_foraging_small | 1 | 50 | 20260310_104749 |
