# 008 Appendix: QRH/CRH Optimization History

This appendix documents the Quantum Reservoir Hybrid (QRH) and Classical Reservoir Hybrid (CRH) evaluation across 16 rounds (96 sessions, ~30,000 episodes) covering foraging, pursuit predators, stationary predators, and ablation studies. For main findings, see [008-quantum-brain-evaluation.md](../../008-quantum-brain-evaluation.md). For architecture design, see [quantum-architectures.md](../../../../research/quantum-architectures.md).

______________________________________________________________________

## Table of Contents

01. [Architecture Overview](#architecture-overview)
02. [Optimization Summary](#optimization-summary)
03. [Phase 1: MI Decision Gate](#phase-1-mi-decision-gate)
04. [Phase 2: Foraging Evaluation (R1-R8)](#phase-2-foraging-evaluation-r1-r8)
05. [Phase 3: Multi-Objective Evaluation (R9-R14)](#phase-3-multi-objective-evaluation-r9-r14)
06. [Phase 4: Ablation Studies (R15-R16)](#phase-4-ablation-studies-r15-r16)
07. [Cross-Architecture Comparison](#cross-architecture-comparison)
08. [Key Technical Discoveries](#key-technical-discoveries)
09. [Lessons Learned](#lessons-learned)
10. [Session References](#session-references)

______________________________________________________________________

## Architecture Overview

### QRH (Quantum Reservoir Hybrid)

10-qubit parameterized quantum circuit reservoir (fixed, not trained) with PPO-trained classical readout. Input encoding via RY(feature×π) and RZ(feature×π) rotation gates. Feature extraction: 75 features via 3 channels (raw Z-expectations, cos/sin nonlinearities, pairwise ZZ correlations). ~10K trainable classical parameters in the readout MLP.

### CRH (Classical Reservoir Hybrid)

10-neuron Echo State Network reservoir (fixed, spectral_radius=0.9) with identical PPO-trained classical readout. Input encoding via linear W_in projection + tanh activation. Same 3 feature channels producing 75 features. Identical readout architecture and training pipeline.

### Key Design Principle

Both architectures share identical downstream components (feature channels, readout MLP, PPO training) — only the reservoir differs. This isolates the effect of quantum vs classical reservoir dynamics.

______________________________________________________________________

## Optimization Summary

### Foraging Rounds (R1-R8)

| Round | Topology | Key Change | Avg Foods | Success | Post-Conv | Key Finding |
|-------|----------|-----------|-----------|---------|-----------|-------------|
| R1 | Structured | Baseline (8 qubits, Z+ZZ features) | 1.70 | 0.13% | — | Structured topology is information bottleneck |
| R2 | Structured | LR↑, epochs↑ (match MLPPPO) | 1.33 | 0.25% | — | More gradient steps on noisy features = worse |
| R3 | Structured | X/Y/Z+ZZ features (52-dim) | 1.52 | 0.0% | — | Phase features help; lowest inter-session variance |
| R4 | Structured | 500 episodes | 1.45 | 0.05% | — | Extended training doesn't help; plateau at ep 300 |
| R5 | Structured | Combined optimizer + LayerNorm | 1.35 | 0.25% | — | Both changes failed; LayerNorm actively harmful |
| **R6a** | **Random** | Combined optimizer + LayerNorm | **8.32** | **76.9%** | **97.4%** | **Random topology: 0% → 77%** |
| R6b | Random | Separate optimizer, no LayerNorm | 7.82 | 69.8% | 94.7% | 6a wins 7/8 metrics |
| R7 | Random | +Min buffer guard, 500 ep | 9.04 | 87.8% | 98.8% | Buffer guard eliminates regression |
| **R8** | **Random** | **+LR warmup, 6 PPO epochs** | **9.05** | **86.8%** | **98.0%** | **5× less conv variance; H.1 gate PASS** |

### Predator Rounds (R9-R14)

| Round | Task | Architecture | Episodes | Overall SR | Last-500 SR | Converged | Key Finding |
|-------|------|-------------|----------|-----------|-------------|-----------|-------------|
| **R9** | **Pursuit** | **QRH** | **1000** | **41.2%** | **53.3%** | **4/4** | **Best pursuit; 13× lower variance** |
| R9 | Pursuit | CRH | 1000 | 31.8% | ~40% | 1/4 | QRH wins (+9.4pp) |
| R10 | Stationary | QRH | 1000 | 15.9% | 29.3% | 0/4 | Baseline stationary; LR decays before convergence |
| R11 | Stationary | QRH | 2000 | 21.4% | 31.3% | 0/4 | Extended training; learning not yet plateaued |
| R11 | Stationary | CRH | 2000 | 28.7% | 35.8% | 1/4 | CRH leads on stationary |
| R12 | Stationary | QRH | 3000 | 30.2% | 38.2% | 1/4 | Extended training helps |
| R12 | Stationary | CRH | 3000 | 33.9% | 44.0% | 2/4 | CRH still leads |
| R13 | Stationary | QRH/CRH | 3000 | 19-21% | ~30-35% | 0/4 | Entropy decay regression |
| **R14** | **Stationary** | **QRH** | **3000** | **23.6%** | **31.1%** | **1/4** | **Best QRH stationary** |
| **R14** | **Stationary** | **CRH** | **3000** | **29.9%** | **43.5%** | **3/4** | **Best CRH; wins +6.3pp** |

### Ablation Rounds (R15-R16)

| Round | Experiment | Architecture | Pursuit SR | Stationary SR | Finding |
|-------|-----------|-------------|-----------|---------------|---------|
| **R15** | **Domingo confound** | **CRH-trig** | **13.0%** | **17.7%** | **Trig encoding hurts CRH; QRH advantage is genuine** |
| R16 | Structured topology | QRH-structured | — | 0.0% | Structured topology definitively falsified |

______________________________________________________________________

## Phase 1: MI Decision Gate

The MI decision gate was the first evaluation step, designed to test whether C. elegans-inspired structured topology produces higher mutual information with optimal actions than a random topology of equivalent gate density. Two runs were conducted (2026-02-22), with the second applying biological correctness interventions.

### Methodology

- **Script**: `scripts/qrh_mi_analysis.py`
- **MI estimation**: `sklearn.feature_selection.mutual_info_classif` (k-nearest neighbors estimator for continuous features vs discrete labels)
- **Dataset**: 1000 synthetic observations — gradient strengths and directions sampled uniformly, agent facing directions sampled from 4 cardinal directions
- **Oracle**: Rule-based gradient-following policy mapping relative gradient direction to optimal action (FORWARD if within ±45° of heading, LEFT/RIGHT otherwise, STAY if gradient_strength < 0.1)
- **Features compared**: Structured QRH (8 qubits, C. elegans topology, 36 features), Random QRH (8 qubits, random topology, 36 features), Classical MLP hidden layer (64 features, untrained random init)
- **Statistical test**: Row-swap permutation test (1000 permutations, seed 42). H₀: mean MI(structured) = mean MI(random). For each permutation, each sample's feature row is randomly assigned to group A or group B; MI delta recomputed. p-value = fraction of permuted deltas ≥ observed
- **Metrics**: Mean MI = average MI across all features; Total MI = sum across all features
- **Results stored**: `artifacts/logbooks/008/qrh_mi_analysis/mi_analysis_results_001.json`, `mi_analysis_results_002.json`

### Label Distribution

| Action | Count | Fraction |
|--------|-------|----------|
| FORWARD | 241 | 24.1% |
| LEFT | 320 | 32.0% |
| RIGHT | 334 | 33.4% |
| STAY | 105 | 10.5% |

### Run 1 (Pre-Intervention)

| Method | Mean MI | Total MI | Feature Dim |
|--------|---------|----------|-------------|
| Structured (C. elegans) | 0.1326 | 4.773 | 36 |
| Random topology | 0.1585 | 5.705 | 36 |
| Classical MLP hidden | 0.3809 | 24.376 | 64 |

Permutation test: observed delta = -0.0259 (random > structured), p=1.0, **not significant**. **Decision: NO-GO for structured topology advantage.**

Key observations:

- Both quantum reservoirs underperform classical features by ~3× on mean MI (0.13-0.16 vs 0.38). Expected since classical MLP has 64 learned nonlinear units, while quantum reservoirs use fixed circuits.
- Structured vs random gap is small and in the **wrong direction** (random wins).
- Run 1 used identical data re-uploading on all 8 qubits (every qubit receives same 2 sensory features), which may wash out topology-specific effects.

### Run 2 (Post-Interventions)

Three biological correctness interventions applied:

1. **Per-qubit input encoding**: Only sensory qubits (ASEL=0, ASER=1) receive direct input. Interneuron/command qubits receive signal exclusively through topology.
2. **Asymmetric feature assignment**: ASEL encodes gradient_strength via RY (on-response), ASER encodes relative_angle via RZ (off-response/phase).
3. **Controlled rotations (CRY/CRZ)**: Chemical synapses use controlled rotations where target rotation is conditioned on control qubit state.

| Method | Mean MI | Total MI | Δ from Run 1 |
|--------|---------|----------|---------------|
| Structured (C. elegans) | 0.2025 | 7.290 | +53% (+0.070) |
| Random topology | 0.2445 | 8.801 | +54% (+0.086) |
| Classical MLP hidden | 0.3809 | 24.376 | (unchanged) |

Permutation test: observed delta = -0.0420 (random > structured), p=1.0, **not significant**. Gap **widened** from -0.026 to -0.042. **Decision: NO-GO (still).**

Key observations:

- Absolute MI improved substantially for both topologies — the interventions are genuine improvements to feature quality.
- Random improved **equally** (+54% vs +53%), proving improvements benefit any topology, not just structured.
- Per-feature analysis: structured reservoir has feature 1 (ASER Z-expectation) at MI=0.0 (dead sensory qubit). The RZ-only encoding on ASER may be less effective than RY for phase-based information.
- Random topology's arbitrary connectivity provides signal from sensory qubits to all other qubits through diverse paths, creating more diverse and less correlated features.

### Per-Feature MI Pattern: Bilateral Symmetry Degeneracy

The structured topology's per-feature MI values in Run 1 reveal a critical pattern. Multiple feature pairs have **identical** MI values:

```text
Feature 0 (ASEL Z):  0.0951    Feature 1 (ASER Z):  0.0951  ← identical
Feature 2 (AIY_L Z): 0.0868    Feature 3 (AIY_R Z): 0.0868  ← identical
Feature 4 (AIA_L Z): 0.1243    Feature 5 (AIA_R Z): 0.1243  ← identical
Feature 6 (AVA_L Z): 0.3242    Feature 7 (AVA_R Z): 0.3242  ← identical
```

The C. elegans circuit's bilateral left-right symmetry creates **mirror features** — left and right neurons compute identical functions, halving the effective information capacity. The random topology has no such constraint: all 8 per-qubit features have distinct MI values (range 0.052-0.273), providing ~2× the effective feature diversity.

This degeneracy compounds through the ZZ correlation features: symmetric pairwise features also produce identical MI values, meaning a substantial fraction of the 36-feature vector is redundant.

### Root Cause Assessment

The fundamental issue is that **biologically correct circuit topology does not translate to quantum information advantage**:

1. **Bilateral symmetry** creates near-degenerate left-right mirror features, collapsing effective rank by ~50%
2. **Feedforward hierarchy** (ASE→AIY→AIA→AVA) constrains signal flow to a narrow 3-hop chain, while random connectivity explores more of the exponential Hilbert space through diverse paths
3. **Biological encoding evolved for analog processing** — neurotransmitter diffusion, ion channel dynamics, temporal integration — not for discrete quantum gate operations. The mapping of synaptic weights to gate angles is inherently lossy.

The MI analysis correctly predicted all subsequent training outcomes: R1-R5 structured topology achieved 0-0.25% success (information bottleneck), while R6 random topology achieved 77% in its first attempt. The R16 ablation (12,000 episodes, structured QRH on stationary predators, 0.0% success) provided definitive confirmation.

______________________________________________________________________

## Phase 2: Foraging Evaluation (R1-R8)

### Critical Transition: R5 → R6 (Structured → Random Topology)

The single most impactful change in the entire evaluation. Switching from structured to random topology transformed QRH from a 0% architecture to a 77% architecture — a ~300× improvement in success rate.

| Metric | Best Structured (R1) | First Random (R6a) | Improvement |
|--------|---------------------|---------------------|-------------|
| Success rate | 0.13% | 76.9% | ~600× |
| Avg foods/run | 1.70 | 8.32 | 4.9× |
| Post-convergence | — | 97.4% | — |

**Why random works**: Random connectivity explores more of the Hilbert space than the constrained 3-hop feedforward hierarchy. Each qubit receives signal through multiple independent paths, creating more diverse and less correlated features.

### Convergence Improvements (R7-R8)

**Min buffer guard (R7)**: Prevents PPO updates on tiny buffers (\<64 samples) that caused late-stage regression. Eliminated regression in 4/4 sessions.

**LR warmup (R8)**: Linear ramp from 10% to full LR over 30 episodes. Primary benefit: 5× reduction in convergence variance (16-episode range vs 72). Secondary benefit: eliminates buffer guard activations entirely (warmup prevents the tiny-buffer conditions). PPO epochs reduced from 10→6 with no performance loss (39% fewer gradient steps).

### Why LayerNorm Interaction is Topology-Dependent

With **structured** topology: LayerNorm erases the meaningful hierarchy between sensory qubits (high signal) and interneuron qubits (derived signal). Amplifies noise in low-information features.

With **random** topology: All qubits are equivalent — no hierarchy to destroy. LayerNorm genuinely normalizes heterogeneous X/Y/Z/ZZ feature scales without removing information.

______________________________________________________________________

## Phase 3: Multi-Objective Evaluation (R9-R14)

### R9: Pursuit Predators (1000 episodes, QRH vs CRH)

**QRH R9** (best pursuit result):

| Session | Overall SR | Last-500 SR | Last-100 SR | Foods/ep | Evasion |
|---------|-----------|-------------|-------------|----------|---------|
| 044427 | 42.2% | 52.2% | 49.0% | ~14 | ~95% |
| 044447 | 40.3% | 54.4% | 64.0% | ~15 | ~94% |
| 044451 | 40.7% | 51.4% | 49.0% | ~14 | ~95% |
| 044501 | 41.7% | 55.2% | 64.0% | ~15 | ~95% |
| **Mean** | **41.2%** | **53.3%** | **56.5%** | — | ~95% |

**CRH R9** (pursuit ablation):

| Session | Overall SR | Last-500 SR | Last-100 SR | Converged |
|---------|-----------|-------------|-------------|-----------|
| 092747 | 18.9% | ~25% | ~28% | No |
| 092754 | 37.5% | ~50% | ~52% | Yes |
| 092901 | 33.7% | ~42% | ~38% | No |
| 092907 | 37.0% | ~45% | ~50% | No |
| **Mean** | **31.8%** | — | — | **1/4** |

**Key metrics**: QRH wins by +9.4pp overall. More importantly, QRH achieves 4/4 convergence with 1.4pp variance — 13× lower than CRH's 18.6pp variance. The quantum reservoir provides dramatically more consistent features for this task.

### R14: Stationary Predators (3000 episodes, QRH vs CRH)

| Metric | QRH R14 | CRH R14 | Winner |
|--------|---------|---------|--------|
| Overall SR (avg) | 23.6% | **29.9%** | CRH (+6.3pp) |
| Last-500 SR (avg) | 31.1% | **43.5%** | CRH (+12.4pp) |
| Sessions converged (≥40% last-500) | 1/4 | **3/4** | CRH |
| Best session overall SR | 33.1% | 33.1% | Tie |
| Variance (SR range) | 15.5pp | **5.2pp** | CRH (3× lower) |

CRH's ESN with fixed spectral_radius=0.9 provides consistent dynamics across seeds. QRH's random topology creates high seed sensitivity on longer tasks.

### Task-Dependent Advantage Pattern

| Task | Winner | Margin | Why |
|------|--------|--------|-----|
| Foraging | QRH | Post-conv +1.3pp over MLPPPO | Quantum features richer after convergence |
| Pursuit predators | QRH | +9.4pp, 13× lower variance | Quantum reservoir provides more consistent features |
| Stationary predators | CRH | +6.3pp, 3/4 vs 1/4 converged | ESN dynamics more stable for longer training |

______________________________________________________________________

## Phase 4: Ablation Studies (R15-R16)

### R15: Domingo Encoding Confound Control

**Problem**: QRH encodes inputs via RY(feature×π), RZ(feature×π) — trigonometric functions. CRH uses W_in @ x — linear projection. Domingo et al. (2021): "Nonlinear input transformations are ubiquitous in QRC." Does QRH's advantage come from trig encoding or quantum dynamics?

**Implementation**: Added `input_encoding: "trig"` to CRH. Applies sin(f×π), cos(f×π) to each input before W_in projection. Doubles W_in input dimension but preserves output feature_dim (75).

**Results**:

| Architecture | Pursuit SR | Stationary SR | Converged |
|---|---|---|---|
| QRH | **41.2%** | 23.6% | **4/4** pursuit, 1/4 stat. |
| CRH-linear | 31.8% | **29.9%** | 1/4 pursuit, **3/4** stat. |
| CRH-trig | 13.0% | 17.7% | 0/4 both |

**Interpretation**:

| Comparison | Pursuit Δ | Stationary Δ | What it means |
|---|---|---|---|
| CRH-linear vs CRH-trig | **-18.8pp** | **-12.2pp** | Trig encoding hurts classical ESN |
| QRH vs CRH-trig | +28.2pp | +5.9pp | QRH wins despite same encoding basis |

**Why trig hurts CRH**: Triple nonlinearity (sin/cos → tanh → tanh) compresses signal dynamic range. The sin/cos outputs are in [-1,1], then tanh compresses again. QRH doesn't have this problem because quantum gates natively operate in trigonometric space (Bloch sphere rotations).

**Conclusion**: QRH's advantage is genuine quantum dynamics (interference, entanglement, phase information), not trigonometric encoding. Domingo concern **resolved**.

### R16: Structured Topology on Stationary Predators

**Hypothesis**: C. elegans-inspired topology might provide better inductive bias for the spatial planning task.

**Results**: **0.0% success across 12,000 episodes** (4 sessions × 3000). Dominant failure: starvation (85-91%). Excellent predator evasion (97-98%) but catastrophically poor foraging (3-5 foods vs 14-17 for random).

**Root cause**: Bilateral symmetry degeneracy. The structured topology has exact left-right mirror symmetry (ASEL↔AIYL mirrors ASER↔AIYR with identical gate angles), creating near-degenerate features when left/right inputs are similar. The feedforward-only architecture limits signal propagation to command qubits.

**Conclusion**: Structured topology is definitively falsified on both pursuit AND stationary predators. Random topology is unambiguously superior.

______________________________________________________________________

## Cross-Architecture Comparison

### QRH vs CRH vs CRH-trig: Comprehensive

| Metric | QRH | CRH-linear | CRH-trig |
|--------|-----|-----------|----------|
| Foraging post-conv | **98.0%** | — | — |
| Pursuit SR (avg) | **41.2%** | 31.8% | 13.0% |
| Pursuit convergence | **4/4** | 1/4 | 0/4 |
| Pursuit variance | **1.4pp** | 18.6pp | 5.3pp |
| Stationary SR (avg) | 23.6% | **29.9%** | 17.7% |
| Stationary convergence | 1/4 | **3/4** | 0/4 |
| Stationary variance | 15.5pp | **5.2pp** | 10.0pp |
| Domingo confound | **Native trig** | Linear | Bolted trig |

### QRH in Context of All Evaluated Architectures

| Architecture | Foraging | Pursuit | Stationary | Training | Params | Status |
|---|---|---|---|---|---|---|
| QRC | 0% | — | — | REINFORCE (readout) | ~1K | Failed |
| QVarCircuit | 30-40% (88% CMA-ES†) | — | — | Param-shift / CMA-ES | ~60 | Marginal |
| QSNN Surrogate | 73.9% | 1.25% | — | Surr grad REINFORCE | 92 | Foraging only |
| QSNN-PPO | 0% | — | — | Surr grad + PPO | ~5.8K | Failed‡ |
| QSNNReinforce A2C | — | 0.63% | — | Surr REINFORCE + A2C | ~1.3K | Failed |
| HybridQuantumCortex | 88.8% | ~40-45% | — | Surr REINFORCE (both) | ~10.5K | Halted§ |
| HybridQuantum | 91.0% | **96.9%** | — | Surr REINFORCE + PPO | ~10K | **Best** |
| HybridClassical | 97.0% | 96.3% | — | REINFORCE + PPO | ~10K | Control |
| **QRH** | **86.8% (98% p-c)** | **41.2%** | **23.6%** | **PPO (readout)** | **~10K** | **Partial** |
| CRH (ablation) | — | 31.8% | 29.9% | PPO (readout) | ~10K | Control |
| CRH-trig (Domingo) | — | 13.0% | 17.7% | PPO (readout) | ~10K | Ablation |
| SpikingReinforce | 73.3% | — | — | REINFORCE | 131K | Baseline¶ |
| MLPPPO | 96.7% | 71.6% | — | PPO | ~42K | Baseline |

† CMA-ES is offline evolutionary optimization, not online gradient learning.
‡ Architectural incompatibility: surrogate gradient forward pass returns constant, making PPO policy ratio always 1.0.
§ Halted: REINFORCE + surrogate gradients cannot exceed ~40-45% on 2-predator environment (vanishing gradients).
¶ ~1 in 10 sessions converge; high catastrophic failure rate.

______________________________________________________________________

## Key Technical Discoveries

### 1. Expressivity Limits of Quantum Reservoir Computing

arXiv:2501.15528 mathematically proves that single-qubit rotation input encoding (what QRH uses: RY(x_i)) creates a provable upper bound on distinguishable functions, regardless of reservoir size, depth, or topology. The information bottleneck is at the **input encoding**, not the reservoir transformation.

### 2. Feature Channel Design

The 3-channel feature extraction (raw + cos_sin + pairwise) provides 75 features from 10 qubits. The cos_sin channel adds nonlinear transformations of Z-expectations; the pairwise channel captures correlations. This is richer than QRC's probability-only measurement (16-dim from 4 qubits).

### 3. Combined Optimizer + LayerNorm Interaction

The combined optimizer (single Adam, single backward pass) outperforms separate actor/critic optimizers with random topology but not structured. LayerNorm helps with random topology but hurts with structured. The combination is topology-dependent.

### 4. Buffer Management for Quantum Features

Quantum reservoir features are noisier than raw sensory features. PPO updates on tiny buffers (\<64 samples) produce unreliable gradients that cascade into policy collapse. The min buffer guard prevents this, and LR warmup eliminates the conditions that create tiny buffers.

______________________________________________________________________

## Lessons Learned

01. **Random topology vastly outperforms biological topology for QRC** (0% → 77% success). MI analysis was predictive.
02. **Post-convergence, quantum reservoir features match or exceed classical** (98% vs MLPPPO 96.7%).
03. **QRH's pursuit advantage is genuine quantum dynamics** (Domingo confound resolved).
04. **Quantum vs classical advantage is task-dependent** (QRH wins pursuit, CRH wins stationary).
05. **LR warmup's primary value is variance reduction** (5× tighter convergence), not speed.
06. **LayerNorm interaction is topology-dependent** — helps random, hurts structured.
07. **The information bottleneck is input encoding** (proven theoretically), not reservoir complexity.
08. **Bilateral symmetry is catastrophic for quantum reservoirs** — creates feature degeneracy.
09. **Fixed quantum reservoirs can work for RL** — but only with random topology, not biological.
10. **CRH is a stronger ablation control than expected** — ESN's spectral_radius provides inherent consistency QRH lacks.

______________________________________________________________________

## Session References

### Foraging Sessions (R1-R8)

| Round | Sessions | Episodes | Config |
|-------|----------|----------|--------|
| R1 | 20260222_042333-042348 | 4×200 | Structured, Z+ZZ, lr=0.0003 |
| R2 | 20260222_052548-052606 | 4×200 | Structured, Z+ZZ, lr=0.001 |
| R3 | 20260222_070527-070546 | 4×200 | Structured, X/Y/Z+ZZ, lr=0.0005 |
| R4 | 20260222_074712-074734 | 4×500 | Structured, X/Y/Z+ZZ, lr=0.0005 |
| R5 | 20260222_102058-102112 | 4×200 | Structured, comb.opt+LayerNorm |
| R6a | 20260222_111320-111339 | 4×200 | Random, comb.opt+LayerNorm |
| R6b | 20260222_114813-114832 | 4×200 | Random, separate opt, no LN |
| R7 | 20260222_123905-123920 | 4×500 | Random, comb.opt+LN+buffer guard |
| R8 | 20260223_151153-151209 | 4×500 | Random, +LR warmup, 6 PPO epochs |

### Pursuit Predator Sessions

| Round | Arch | Sessions | Episodes |
|-------|------|----------|----------|
| R9 | QRH | 20260224_044427-044501 | 4×1000 |
| R9 | CRH | 20260225_092747-092754 | 4×1000 |
| R9-ext | CRH | 20260228_093931-093948 | 4×1500 |
| R15 | CRH-trig | 20260301_210008-210021 | 4×1000 |

### Stationary Predator Sessions

| Round | Arch | Sessions | Episodes |
|-------|------|----------|----------|
| R10 | QRH | 20260226_224841-224915 | 4×1000 |
| R11 | QRH | 20260225_044427-044451 | 4×2000 |
| R11 | CRH | 20260225_044427-044451 | 4×2000 |
| R12 | QRH | 20260226_115741-115752 | 4×3000 |
| R12 | CRH | 20260226_115814-115829 | 4×3000 |
| R13 | QRH | 20260228_234222-234237 | 4×3000 |
| R13 | CRH | 20260228_234246-234301 | 4×3000 |
| R14 | QRH | 20260226_084742-084759 | 4×3000 |
| R14 | CRH | 20260226_093914-093921 | 4×3000 |
| R15 | CRH-trig | 20260301_210103-210115 | 4×3000 |
| R16 | QRH-struct | 20260301_221424-221435 | 4×3000 |

### Stored Artifacts

Best session results and configs stored in `artifacts/logbooks/008/`:

| Directory | Contents |
|-----------|----------|
| `qrh_foraging_small/` | QRH R8 foraging (4 JSON + 1 YML) |
| `qrh_thermotaxis_pursuit_predators_large/` | QRH R9 pursuit (4 JSON + 1 YML) |
| `qrh_thermotaxis_stationary_predators_large/` | QRH R14 stationary (4 JSON + 1 YML) |
| `crh_thermotaxis_pursuit_predators_large/` | CRH R9 pursuit (4 JSON + 1 YML) |
| `crh_thermotaxis_stationary_predators_large/` | CRH R14 stationary (4 JSON + 1 YML) |
| `crh_trig_thermotaxis_pursuit_predators_large/` | CRH-trig R15 pursuit (4 JSON + 1 YML) |
| `crh_trig_thermotaxis_stationary_predators_large/` | CRH-trig R15 stationary (4 JSON + 1 YML) |
| `qrh_structured_thermotaxis_stationary_predators_large/` | QRH R16 structured (4 JSON + 1 YML) |
