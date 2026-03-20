# 008 Appendix: QEF (Quantum Entangled Features) Optimization History

This appendix documents the QEF evaluation across 24 phases (~500+ simulation runs) covering MI analysis, topology optimization, hybrid input, feature gating, classical ablation, and 12-seed statistical validation. For main findings, see [008-quantum-brain-evaluation.md](../../008-quantum-brain-evaluation.md). For architecture design, see [quantum-architectures.md](../../../../research/quantum-architectures.md).

______________________________________________________________________

## Table of Contents

01. [Architecture Overview](#architecture-overview)
02. [Optimization Summary](#optimization-summary)
03. [Phase 1-4: MI Analysis and Circuit Configuration](#phase-1-4-mi-analysis-and-circuit-configuration)
04. [Phase 5-6: Topology and Readout Optimization](#phase-5-6-topology-and-readout-optimization)
05. [Phase 7: Depth and Encoding Experiments](#phase-7-depth-and-encoding-experiments)
06. [Phase 8: Stationary Predator Evaluation](#phase-8-stationary-predator-evaluation)
07. [Phase 9: Hybrid Input](#phase-9-hybrid-input)
08. [Phase 10-11: Feature Gating and Extended Experiments](#phase-10-11-feature-gating-and-extended-experiments)
09. [Phase 12: Final Pursuit Optimization](#phase-12-final-pursuit-optimization)
10. [Phase 13-15: Classical Ablation](#phase-13-15-classical-ablation)
11. [Phase 16-19: Context and Mixed Gating](#phase-16-19-context-and-mixed-gating)
12. [Phase 20-24: Stationary Optimization Attempts](#phase-20-24-stationary-optimization-attempts)
13. [12-Seed Statistical Validation](#12-seed-statistical-validation)
14. [Key Mechanistic Findings](#key-mechanistic-findings)

______________________________________________________________________

## Architecture Overview

QEF (Quantum Entangled Features) is an 8-qubit parameterized quantum circuit with purposeful entanglement, hybrid classical-quantum input, and learnable feature gating. It extends the QRH (Quantum Reservoir Hybrid) paradigm by replacing random reservoir entanglement with configurable cross-modal entanglement topologies.

**Final architecture per task:**

- **Pursuit large**: Ring topology, CRY/CRZ gates, hybrid input (59-dim), context gating, bigbuf 1024
- **Stationary large**: Ring topology, CRY/CRZ gates, hybrid input (38-dim curated), static gating, bigbuf 1024
- **Small PP**: Ring topology, CRY/CRZ gates, hybrid input (56-dim), context gating

______________________________________________________________________

## Optimization Summary

| Phase | Key Change | Best L100 (task) | Finding |
|-------|-----------|-----------------|---------|
| 1-4 | MI analysis, CRY/CRZ gates, xyz features | — | CZ-only insufficient; CRY/CRZ + xyz required |
| 5 | LR decay + entropy annealing | 96.0% (small PP) | Eliminates catastrophic forgetting |
| 5b | Ring topology | +11.7pp overall (small PP) | Ring >> modality-paired (8 vs 4 CZ pairs) |
| 6 | Compact readout 64x2 | +5.0pp pursuit | Smaller network has higher L100 ceiling |
| 7 | Depth 3, sparse encoding tested | — | Both hurt; depth 2 + uniform confirmed optimal |
| 8 | Stationary evaluation + entropy/bigbuf tune | 75.5% stationary | Weak nociception signal is bottleneck |
| 9 | **Hybrid input** (raw + quantum) | 82.5% stat, 88.2% pursuit | Key innovation: fast convergence from raw features |
| 10 | **Static feature gating** | 90.2% stat, 90.5% pursuit | +7.2pp stationary, +2.3pp pursuit |
| 11 | Bigbuf + circuit seed sweep | 92.2% pursuit (bigbuf) | Bigbuf adds +1.7pp pursuit, halves seed variance |
| 12 | Sparse + hybrid pursuit | 94.5% pursuit | Matches uniform 2000ep in half the training |
| 13 | Classical ablation (A1-A4) | — | A3 poly 88.2% stat (buffer 512); A4 random worst |
| 14 | Gating asymmetry diagnostic | — | Gating helps quantum (+7.7pp), hurts classical (-4.0pp) on stationary |
| 15 | Cross-task gating validation | — | Gating asymmetry is stationary-specific |
| 16 | **Context gating** | 95.7% pursuit | +5.2pp pursuit over static; hurts stationary |
| 19 | **Mixed gating** | 91.0% stationary | +0.8pp stationary combining static + context |
| 21 | **Fair comparison** (matched buffer) | — | MLP PPO stationary 80.0%→90.2% with buffer 1024 |
| 23 | Cross-modal ZZ curation | 92.2% stationary | +1.2pp from dropping intra-modal ZZ + cos/sin |
| Final | 12-seed validation | 90.8% stat, 93.0% pursuit, 98.2% small | No significant quantum advantage |

______________________________________________________________________

## Phase 1-4: MI Analysis and Circuit Configuration

Mutual information analysis comparing entangled vs separable vs QRH random features. Key findings:

- CZ-only entanglement had insufficient expressivity (MI gap 9.3x vs QRH)
- Adding CRY/CRZ controlled rotations (`gate_mode: cry_crz`) closed the MI gap to 1.5x
- Switching to xyz feature mode (`feature_mode: xyz`) further improved MI
- All 3 topologies (modality-paired, ring, random) showed similar MI — topology matters less than gate/feature mode

______________________________________________________________________

## Phase 5-6: Topology and Readout Optimization

**Phase 5**: LR decay (0.0005→0.0001 over 300ep) + entropy annealing (0.02→0.005) eliminated catastrophic forgetting. All 4 seeds converged to 96% L100 on small pursuit predators.

**Phase 5b**: Ring topology (+11.7pp overall, 2x faster convergence) decisively outperformed modality-paired on pursuit predators. Ring's 8 CZ pairs create denser entanglement than modality-paired's 4 pairs.

**Phase 6**: Compact readout (64x2, ~13K params) outperformed large readout (128x3, ~80K params) by +5.0pp L100 on thermotaxis pursuit — smaller networks generalize better for quantum feature inputs.

______________________________________________________________________

## Phase 7: Depth and Encoding Experiments

Tested circuit depth 3, sparse encoding, and combinations. All performed worse than the depth-2 uniform baseline:

- Depth 3 alone: -4.1pp L100
- Sparse encoding: -9.1pp L100
- Depth 3 + bigbuf: -0.6pp L100 (nearly recovers but doesn't improve)
- Higher LR (0.001): -1.8pp L100

**Conclusion**: Depth 2, uniform encoding, LR 0.0005 confirmed as optimal circuit configuration.

______________________________________________________________________

## Phase 8: Stationary Predator Evaluation

First evaluation on thermotaxis + stationary predators (100x100 grid, 5 fixed toxic zones). QEF initially at 69.7% L100 vs MLP PPO 80.0%.

Root cause analysis revealed stationary predators have `gradient_decay_constant=5.0` (vs pursuit's 12.0), causing weak nociception signals detectable at only 3-4 cells — below the damage_radius of 4.

Entropy fix (0.005/1200ep) + big buffer (1024/4mb) improved to 75.5% L100 (+5.8pp).

______________________________________________________________________

## Phase 9: Hybrid Input

**Key innovation**: Concatenating raw sensory features (7-dim) with quantum features (52-dim) before the readout MLP. Gives the network direct access to actionable signals while benefiting from quantum correlations.

- Stationary: 70.2% → 82.5% (+12.3pp)
- Pursuit: 87.8% → 88.2% (+0.4pp)
- **Stationary surpassed MLP PPO** (82.5% vs 80.0%) — first time QEF beat classical (later corrected when buffer mismatch was discovered)

______________________________________________________________________

## Phase 10-11: Feature Gating and Extended Experiments

**Static gating** (`sigmoid(w) * quantum_features`): +7.2pp stationary, +2.3pp pursuit. The learned gate suppresses noisy ZZ correlations and amplifies useful cross-modal signals.

**Entanglement ablation**: 0% success without entanglement — confirms quantum circuit correlations are essential.

**Phase 11**: Bigbuf on pursuit adds +1.7pp and halves seed variance. Circuit seed sweep shows ±0.5pp effect — gating compensates for different entanglement angles.

______________________________________________________________________

## Phase 12: Final Pursuit Optimization

Sparse + hybrid + gating + bigbuf reached 94.5% L100 on pursuit at 1000ep — matching uniform gating+bigbuf at 2000ep in half the training time. Sparse encoding, catastrophic without hybrid, works well when raw features handle actionable signals.

______________________________________________________________________

## Phase 13-15: Classical Ablation

Four classical ablations tested on all tasks (with buffer 512 initially):

| Ablation | Input | Stationary L100 | Pursuit L100 |
|----------|-------|-----------------|-------------|
| A1: Fair MLP PPO (64x2, 7-dim) | 7 | 80.0% | 95.2% |
| A2: Capacity-matched (128x2, 7-dim) | 7 | 83.7% | 93.5% |
| A3: Polynomial (64x2, 28-dim) | 28 | 88.2% | 93.0% |
| A4: Random projection (64x2, 59-dim) | 59 | 78.5% | 95.5% |

**Phase 14 diagnostic**: Gating helps quantum features (+7.7pp) but hurts classical polynomial features (-4.0pp) on stationary — evidence that quantum features have a sparse useful structure that classical features don't.

**Phase 15**: Cross-task validation showed the gating asymmetry is stationary-specific. On pursuit/small PP, gating helps both quantum and classical features similarly.

______________________________________________________________________

## Phase 16-19: Context and Mixed Gating

**Context gating** (`sigmoid(MLP(raw_features)) * quantum_features`): +5.2pp on pursuit over static gating, but -2.5pp on stationary. Input-dependent feature selection helps when signals are strong (pursuit) but hurts when signals are weak (stationary).

**Mixed gating** (average of static + context): +0.8pp on stationary over static alone. The static component anchors the gate while context adds adaptive modulation.

Optimal gating is task-dependent: static/mixed for weak signals, context for strong signals.

______________________________________________________________________

## Phase 20-24: Stationary Optimization Attempts

Multiple attempts to close the stationary gap:

- Wider readout (128x2): -3.8pp — overfits on weak signals
- More PPO epochs (15): -4.5pp — over-training per buffer
- Higher entropy: -6.3pp — delays convergence
- Hybrid + polynomial: no improvement — redundant with quantum ZZ
- ZZZ three-body correlations: -5.5pp — adds noise, pairwise is sufficient
- **Cross-modal ZZ curation**: +1.2pp (92.2%) — best stationary result

**Phase 21 (fairness correction)**: Discovered MLP PPO and A3 poly baselines used buffer 512 while QEF used 1024. With matched buffer, MLP PPO stationary jumped from 80.0% to 90.2% and A3 poly from 88.2% to 94.2%.

______________________________________________________________________

## 12-Seed Statistical Validation

Seeds: 42, 123, 456, 789, 7, 99, 314, 555, 1001, 2024, 3141, 8888

### Stationary (1000 episodes)

| Config | Mean L100 | SE | σ | Seeds |
|--------|----------|-----|-----|-------|
| A3 Poly | 93.8% | ±1.0% | 3.4 | 96,96,95,97,96,91,95,93,97,93,85,92 |
| QEF | 90.8% | ±1.2% | 4.0 | 88,93,95,89,93,91,87,96,93,82,94,88 |
| MLP PPO | 89.6% | ±0.8% | 2.6 | 93,93,90,88,86,89,88,91,90,89,85,93 |

- QEF vs MLP PPO: +1.2pp, t(11)=0.804, **p=0.438 (ns)**
- QEF vs A3 Poly: -3.1pp, t(11)=-1.924, **p=0.081 (ns, trending)**

### Pursuit Large (1000 episodes)

| Config | Mean L100 | SE | σ | Seeds |
|--------|----------|-----|-----|-------|
| MLP PPO | 96.0% | ±0.5% | 1.6 | 96,94,94,95,98,99,96,94,96,96,97,97 |
| A3 Poly | 94.8% | ±1.1% | 4.0 | 99,97,97,95,95,88,87,100,97,95,92,95 |
| QEF | 93.0% | ±1.3% | 4.5 | 96,93,88,98,94,97,86,95,100,88,89,92 |

- QEF vs MLP PPO: -3.0pp, t(11)=-2.283, **p=0.043 (\*)**
- QEF vs A3 Poly: -1.8pp, t(11)=-1.242, **p=0.240 (ns)**

### Small Pursuit Predators (500 episodes)

Note: Seed 789 missing for MLP PPO and A3 Poly due to session collision. Paired t-tests use 11 common seeds.

| Config | Mean L100 | SE | σ | n | Seeds |
|--------|----------|-----|-----|---|-------|
| MLP PPO | 98.6% | ±0.5% | 1.7 | 11 | 99,94,100,99,100,99,99,100,99,99,97 |
| QEF | 98.2% | ±0.6% | 2.0 | 12 | 99,98,98,97,93,100,99,99,100,97,99,100 |
| A3 Poly | 97.0% | ±0.6% | 2.0 | 11 | 98,95,98,100,99,96,95,95,94,99,98 |

- QEF vs MLP PPO (11 pairs): -0.5pp, t(10)=-0.518, **p=0.616 (ns)**
- QEF vs A3 Poly (11 pairs): +1.2pp, t(10)=1.184, **p=0.264 (ns)**

______________________________________________________________________

## Key Mechanistic Findings

### 1. Entanglement is Essential

Separable ablation (no entanglement) produces 0% success. The ZZ correlations from entangled qubits carry critical cross-modal information that the readout needs.

### 2. Gating Asymmetry (Stationary-Specific)

On stationary predators, learned sigmoid gating:

- Helps quantum features: +7.7pp (82.5% → 90.2%)
- Hurts classical polynomial features: -4.0pp (88.2% → 84.2%)

This suggests quantum features have a sparse useful structure (most ZZ correlations are noise, a few encode cross-modal information) while classical polynomial features are uniformly useful. On pursuit/small PP, gating helps both equally.

### 3. Classical Polynomial Self-Silencing

Polynomial features (x_i × x_j) naturally go to zero when either input is near-zero. On stationary predators with weak nociception signals, this provides automatic noise suppression without learned gating. Quantum ZZ correlations produce non-zero values even for near-zero inputs due to the entangling structure, requiring explicit gating to achieve similar noise suppression.

### 4. Trainability-Advantage Dilemma

The 8-qubit depth-2 circuit is classically simulatable. ZZ correlations are expressible as classical functions of the input rotation angles. The "quantum" aspect is the entanglement structure selecting which correlations to compute, but polynomial expansion achieves a similar effect. Genuine quantum advantage may require deeper circuits, larger qubit counts, or tasks with inherently quantum structure.
