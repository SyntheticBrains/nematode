# 008: Quantum Brain Architecture Evaluation

**Status**: `in_progress`

**Branch**: `feature/add-qrc-brain`

**Date Started**: 2026-02-05

## Objective

Evaluate and benchmark novel quantum brain architectures against classical baselines. This experiment covers:

- **QRCBrain** (Quantum Reservoir Computing): Fixed quantum reservoir + trainable classical readout
- **QSNN** (Quantum Spiking Neural Network): Planned - QLIF neurons with local learning rules
- **QVarCircuitBrain**: Existing - Variational quantum circuit with trainable parameters

Goal: Identify which quantum architectures are viable for nematode navigation tasks.

## Background

Classical baselines are well-established (see Logbook 004, 007):

- **MLPReinforceBrain**: ~70-85% success on foraging
- **MLPPPOBrain**: 84-98% post-convergence across all thermotaxis configs

Quantum approaches face unique challenges:

- Barren plateaus in variational circuits
- Measurement noise
- Limited qubit counts
- Gradient estimation overhead

______________________________________________________________________

## QRCBrain Evaluation

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM RESERVOIR COMPUTING (QRC)                        │
│                  Fixed Reservoir + Trainable Readout                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS                  QUANTUM RESERVOIR (FIXED)                          │
│  ┌────────────────┐      ┌───────────────────────────────────────────┐      │
│  │ gradient_str   │──┐   │  ┌───────────────────────────────────┐    │      │
│  │ [0.0 - 1.0]    │  ├──▶│  │ Layer × 3 (depth):                │    │      │
│  ├────────────────┤  │   │  │  H(all) → RY(input) → Rrand →CZ   │    │      │
│  │ relative_angle │──┘   │  └───────────────────────────────────┘    │      │
│  │ [-1.0, 1.0]    │      │  Random angles seeded (reproducible)      │      │
│  └────────────────┘      │  4 qubits → 16-dim measurement vector     │      │
│                          └────────────┬──────────────────────────────┘      │
│                                       ▼                                     │
│                           ┌──────────────────────────────────────────┐      │
│                           │  CLASSICAL READOUT (TRAINABLE)           │      │
│                           │  Linear(16→64) + ReLU → Linear(64→4)     │      │
│                           │  REINFORCE policy gradient               │      │
│                           └────────────┬─────────────────────────────┘      │
│                                        ▼                                    │
│                           ┌──────────────────────────────────────────┐      │
│                           │  Softmax → Action: FWD/LEFT/RIGHT/STAY   │      │
│                           └──────────────────────────────────────────┘      │
│                                                                             │
│  PROBLEM: Fixed random rotations → non-discriminative representations       │
│           Different inputs → similar measurement distributions              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key design choices:

- **Fixed reservoir**: Random rotation angles seeded for reproducibility
- **Data re-uploading**: Input encoded before each reservoir layer
- **REINFORCE learning**: Only readout network trains

### Configuration (Final Tuned)

```yaml
brain:
  name: qrc
  config:
    num_reservoir_qubits: 4      # Reduced from 8 (16-dim vs 256-dim output)
    reservoir_depth: 3
    readout_type: mlp
    readout_hidden: 64
    learning_rate: 0.01          # 10x baseline (weak gradients)
    entropy_coef: 0.005
    shots: 1024
```

### Results Summary

| Task | Runs | Success | Chemotaxis | Status |
|------|------|---------|------------|--------|
| Foraging (2 inputs) | 1600+ | 0% | -0.13 to -0.23 | ❌ Failed |
| Predators (4 inputs) | 25 | 0% | -0.131 | ❌ Failed |

### Key Experiments

| Session | Config Changes | Success | CI | Finding |
|---------|----------------|---------|----|---------|
| 20260204_122807-122818 | Baseline (8 qubits) | 0% | -0.14 | Input encoding sparse |
| 20260204_131441-131456 | Dense encoding + entropy | 0% | -0.10 | Marginal improvement |
| 20260204_135543-135557 | 4 qubits, linear readout | 0% | -0.10 | Simpler, same result |
| 20260204_220604 | LR=0.01, MLP readout | 0% | -0.15 | Learning signal stronger |
| 20260204_222450 | Data re-uploading | 0% | -0.21 | Worsened |
| 20260204_231515 | Multi-sensory (predators) | 0% | -0.13 | Slightly better CI |

### Root Cause Analysis

Debug logging revealed:

1. **Learning IS happening**: Policy updates execute, entropy decreases (1.386 → 1.360)
2. **Gradients are weak**: Norm ~0.02-0.14, need many episodes to converge
3. **Reservoir outputs are non-discriminative**: High entropy (~2.4/2.77), similar for different inputs
4. **Policy converges to wrong behavior**: Negative chemotaxis = moving AWAY from food

### Conclusion

**QRCBrain is not viable for chemotaxis/foraging tasks.**

The fixed random reservoir doesn't create representations that distinguish "toward food" from "away from food". The architecture may suit:

- Time-series prediction (memory effects)
- Tasks with richer input signals
- Pre-trained readout networks

______________________________________________________________________

## QSNN Evaluation

**Status**: Foraging baseline established (73.9%), predator evaluation Round P0 complete (14.5% avg)

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                  QUANTUM SPIKING NEURAL NETWORK (QSNN)                      │
│            QLIF Neurons with Hybrid Quantum-Classical Training              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS (2-4 features)           SENSORY LAYER (6 neurons)                  │
│  ┌────────────────┐              ┌──────────────────────────┐               │
│  │ gradient_str   │──sigmoid──▶ │  S₁  S₂  S₃  S₄  S₅  S₆   │               │
│  │ relative_angle │──sigmoid──▶ │  (spike probabilities)    │               │
│  └────────────────┘              └────────────┬─────────────┘               │
│                                               │ W_sh (6×8 weights)          │
│                                               ▼                             │
│                                  ┌──────────────────────────┐               │
│                                  │    HIDDEN LAYER (8 QLIF) │               │
│  ┌─QLIF NEURON DETAIL───────┐    │  H₁ H₂ H₃ H₄ H₅ H₆ H₇ H₈││               │
│  │                          │    │  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐│               │
│  │  |0⟩ ─ RY(θ+tanh(wx)π)   │    │  │Q││Q││Q││Q││Q││Q││Q││Q││               │
│  │         │                │    │  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘│               │
│  │       RX(θ_leak)         │    │  (8 quantum circuits)    │               │
│  │         │                │    └────────────┬─────────────┘               │
│  │      Measure             │                 │ W_hm (8×4 weights)          │
│  │   |1⟩=spike |0⟩=quiet    │                 ▼                             │
│  └──────────────────────────┘    ┌──────────────────────────┐               │
│                                  │    MOTOR LAYER (4 QLIF)  │               │
│  × 10 TIMESTEPS                  │     M₁   M₂   M₃   M₄    │               │
│  (averaged for noise             │     ┌─┐  ┌─┐  ┌─┐  ┌─┐   │               │
│   reduction)                     │     │Q│  │Q│  │Q│  │Q│   │               │
│                                  │     └─┘  └─┘  └─┘  └─┘   │               │
│                                  └────────────┬─────────────┘               │
│                                               ▼                             │
│                                  ┌──────────────────────────┐               │
│                                  │  Logit Scaling → Softmax │               │
│                                  │  + ε-greedy exploration  │               │
│                                  │  → FWD / LEFT / RIGHT /  │               │
│                                  │    STAY                  │               │
│                                  └──────────────────────────┘               │
│                                                                             │
│  PARAMETERS: ~92 total                                                      │
│    W_sh: 6×8=48 │ W_hm: 8×4=32 │ θ_hidden: 8 │ θ_motor: 4                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key design choices (based on Brand & Petruccione 2024):

- **QLIF neurons**: Quantum Leaky Integrate-and-Fire with minimal 2-gate circuit
- **Network topology**: 6 sensory → 8 hidden → 4 motor neurons (~92 trainable parameters)
- **Dual learning modes**: Surrogate gradient REINFORCE (primary) or 3-factor Hebbian (legacy)
- **Trainable parameters**: Weight matrices (W_sh, W_hm), membrane biases (θ_hidden, θ_motor)
- **Multi-timestep integration**: 10 QLIF timesteps averaged per decision to reduce quantum shot noise
- **Adaptive entropy regulation**: Two-sided (floor boost + ceiling suppression) prevents both policy collapse and entropy explosion

### QLIF Neuron Circuit

```python
# Minimal circuit per Brand & Petruccione (2024)
|0⟩ → RY(θ_membrane + tanh(w·x) × π) → RX(θ_leak) → Measure

# θ_membrane: trainable membrane potential bias
# w·x: weighted input from presynaptic layer
# tanh bounds input to [-1, 1] before scaling by π
# θ_leak: leak rate = (1 - membrane_tau) × π
```

### Surrogate Gradient Learning

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│              HYBRID QUANTUM-CLASSICAL TRAINING (QLIFSurrogateSpike)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORWARD PASS (quantum)                 BACKWARD PASS (classical)           │
│  ┌────────────────────────────┐         ┌───────────────────────────────┐   │
│  │  ry_angle = θ + tanh(w·x)π │         │  Sigmoid surrogate gradient   │   │
│  │         │                  │         │  centered at π/2:             │   │
│  │  |0⟩ → RY(ry_angle)        │         │                               │   │
│  │         │                  │         │  d/d(angle) ≈ α·σ(α·(angle    │   │
│  │       RX(leak)             │         │    - π/2))·(1 - σ(...))       │   │
│  │         │                  │         │                               │   │
│  │      Measure × 1024 shots  │         │  α = 1.0 (smooth, reduces     │   │
│  │         │                  │         │  quantum noise amplification) │   │
│  │  spike_prob = P(|1⟩)       │         │                               │   │
│  └────────────┬───────────────┘         └──────────────┬────────────────┘   │
│               │                                        │                    │
│               │     ┌───────────────────────────┐      │                    │
│               └────▶│  torch.autograd.Function  │◀─────┘                    │
│                     │  forward: quantum result  │                           │
│                     │  backward: surrogate grad │                           │
│                     └──────────────┬────────────┘                           │
│                                    │                                        │
│                                    ▼                                        │
│               ┌─────────────────────────────────────┐                       │
│               │  REINFORCE POLICY GRADIENT          │                       │
│               │                                     │                       │
│               │  loss = -(log_probs × advantages)   │                       │
│               │       - entropy_coef × entropy      │                       │
│               │                                     │                       │
│               │  Adam optimizer (LR 0.01→0.001)     │                       │
│               │  Gradient clipping (norm 1.0)       │                       │
│               │  Weight clamping ([-3, 3])          │                       │
│               └─────────────────────────────────────┘                       │
│                                                                             │
│  KEY INSIGHT: Quantum circuits preserve QLIF dynamics in the forward pass.  │
│  Classical surrogate avoids parameter-shift rule cost (no extra circuits)   │
│  and barren plateaus (gradients computed classically, not through quantum). │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration (Final Tuned)

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
    use_local_learning: false    # Surrogate gradient mode
    shots: 1024
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.02
    weight_clip: 3.0
    update_interval: 20
    num_integration_steps: 10
```

### Results Summary

| Phase | Success | CI | Key Finding |
|-------|---------|-----|-------------|
| Initial (Hebbian learning) | 0% | -0.154 | Local learning too weak for this task |
| After 17 rounds of tuning | **73.9%** | **+0.41** | **Matches SpikingReinforceBrain (73.3%)** |

**Best round (R12o) — 4 sessions, 200 episodes each:**

| Session | Success | Convergence | Post-Conv | Late-50 |
|---------|---------|-------------|-----------|---------|
| 234508 | 79.5% | Ep 56 | 96.7% | 97.5% |
| 234514 | 75.0% | Ep 45 | 95.5% | 97.5% |
| 234519 | 68.0% | Ep 63 | 91.2% | 92.5% |
| 234524 | 73.0% | Ep 52 | 96.0% | 100% |
| **Average** | **73.9%** | **100% convergence** | **94.9%** | **96.9%** |

### Key Breakthroughs

1. **Surrogate gradients replaced Hebbian learning**: Local Hebbian learning (Rounds 0-11, 12 iterations) never exceeded 0% success. Switching to REINFORCE with `QLIFSurrogateSpike` (sigmoid surrogate on the RY angle) provided the dense gradient signal needed. Quantum circuits still run in the forward pass; only the backward pass uses the classical surrogate.

2. **Multi-timestep integration**: Averaging spike probabilities across 10 QLIF timesteps per decision reduces quantum shot noise variance by 10x (effective samples = 10,240). This was essential for stable REINFORCE training — 5 timesteps showed 52.6% success with 50% convergence vs 73.9% and 100% convergence with 10.

3. **Adaptive entropy regulation**: Two-sided system prevents both entropy collapse (20x boost when entropy < 0.5 nats) and entropy explosion (suppression when entropy > 95% of max). This solved the premature policy commitment failure mode that caused 50% session failure rate.

4. **Warm-start initialization**: θ_hidden = π/4 provides initial spike probability ~0.15 with surrogate gradient at ~60% of peak, giving meaningful gradient sensitivity from step 1. Weight scale 0.15 breaks symmetry without destabilizing.

### Comparison with Classical Spiking

| Metric | QSNN | SpikingReinforce |
|--------|------|------------------|
| Success Rate | 73.9% | 73.3% |
| Session Reliability | 4/4 converged (100%) | ~1 in 10 sessions converges |
| Architecture | 6→8→4 (92 params) | 16→256→256→4 (131K params) |
| Convergence | Ep 45-63 | Ep 22 (when it converges) |
| Learning | Quantum forward + surrogate backward | Fully classical surrogate gradient |
| Timesteps/decision | 10 (quantum) | 100 (classical) |

QSNN achieves equivalent success rate with **1,400x fewer parameters** and **far more reliable training**. SpikingReinforce's 73.3% headline number came from its best session; in practice, roughly 9 out of 10 sessions suffer catastrophic entropy collapse or policy divergence, producing near-0% success. QSNN's adaptive entropy regulation and multi-timestep integration provide much more robust convergence — all 4 sessions in the best round converged successfully. QSNN does converge more slowly and has higher computational cost per timestep due to quantum circuit simulation.

### Root Cause Analysis: Why Hebbian Failed

After 12 rounds of tuning the 3-factor Hebbian learning rule (Rounds 0-11), the fundamental tension was identified:

- Updating all W_hm columns causes correlated collapse (all columns converge to same direction)
- Updating only the chosen action's column causes starvation collapse (unchosen columns atrophy)

The local learning signal is simply too weak for this task. Dense gradient information via surrogate gradients was required. See [008-appendix-qsnn-optimization.md](008-appendix-qsnn-optimization.md) for the full optimization history.

### File Locations

- QSNN implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnn.py`
- QSNN tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnn.py`
- QSNN configs: `configs/examples/qsnn_*.yml`

______________________________________________________________________

## QSNN Predator Evasion Evaluation

## QVarCircuitBrain Comparison

**Status**: Existing baseline

QVarCircuit uses trainable rotation angles in a variational ansatz. Prior results show it achieves ~30-40% success on foraging with gradient-based learning, and 88% with CMA-ES evolutionary optimization.

| Metric | QRC | QSNN | QVarCircuit | SpikingReinforce |
|--------|-----|------|-------------|------------------|
| Success Rate | 0% | **73.9%** | 30-40% (88% CMA-ES) | 73.3%\* |
| Session Reliability | N/A | **100%** | ~50% (gradient) | ~10% |
| Chemotaxis | -0.13 to -0.23 | **+0.41** | ~0.1-0.3 | ~0.3 |
| Trainable Params | Readout only | 92 | Quantum + Readout | 131K |
| Gradient Issue | Weak signal | Solved (surrogate) | Barren plateaus | High variance\* |

\* SpikingReinforce's 73.3% is its best session; ~9/10 sessions fail catastrophically.

______________________________________________________________________

## Analysis

### Quantum Architecture Comparison

| Architecture | Trainable | Gradient Approach | Best Success | Viable? |
|--------------|-----------|-------------------|--------------|---------|
| QRC | Readout only | REINFORCE on readout | 0% | No |
| QSNN (Hebbian) | Weights + θ | 3-factor local Hebbian | 0% | No |
| **QSNN (Surrogate)** | **Weights + θ** | **Quantum forward + sigmoid surrogate backward** | **73.9%** | **Yes** |
| QVarCircuit (gradient) | Full circuit | Parameter-shift rule | ~40% | Marginal |
| QVarCircuit (CMA-ES) | Full circuit | Evolutionary | 88% | Yes (but not gradient-based) |

```text
QUANTUM ARCHITECTURE SUCCESS RATES (Foraging, Gradient-Based Learning)
═══════════════════════════════════════════════════════════════════════════

Architecture                  Success Rate
────────────────────────────────────────────────────────────────────────────
QRC (fixed reservoir)         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%   ❌
QSNN Hebbian                  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%   ❌
QVarCircuit (param-shift)     ████████████████░░░░░░░░░░░░░░░░░░░░░░░  ~40% ⚠️
QSNN Surrogate                ██████████████████████████████░░░░░░░░░  73.9% ✓
QVarCircuit (CMA-ES)          ███████████████████████████████████░░░░  88%   ✓*
────────────────────────────────────────────────────────────────────────────
SpikingReinforce (classical)  █████████████████████████████░░░░░░░░░░  73.3%† (ref)

* CMA-ES is evolutionary, not gradient-based
† SpikingReinforce best session only; ~9/10 sessions fail (~10% reliability)
  QSNN achieves 73.9% with 100% session reliability (4/4 converge)

KEY INSIGHT: QSNN Surrogate is the first quantum architecture to match a
classical baseline using gradient-based learning (no evolution needed).
It also surpasses SpikingReinforce in training reliability.
═══════════════════════════════════════════════════════════════════════════
```

```text
PARAMETER EFFICIENCY AND RELIABILITY: QSNN vs CLASSICAL SPIKING
═══════════════════════════════════════════════════════════════════════════

                   Parameters (log scale)          Success   Reliability
───────────────────────────────────────────────────────────────────────────
QSNN               ██  92                          73.9%     4/4 (100%)
SpikingReinforce   ██████████████████████████████  131,000   73.3%*   ~1/10

* SpikingReinforce's 73.3% is from its best session. Most sessions (~9/10)
  suffer catastrophic failure (entropy collapse or policy divergence).
  QSNN achieves equal peak success with 1,400x fewer parameters and
  dramatically more reliable convergence across seeds.
═══════════════════════════════════════════════════════════════════════════
```

```text
QSNN OPTIMIZATION JOURNEY (17 Rounds)
═══════════════════════════════════════════════════════════════════════════

Success
  80% ┤                                                         ●── 73.9%
  70% ┤                                                    ●───●
  60% ┤
  50% ┤                                              ●────●
  40% ┤                                         ●───●
  30% ┤
  20% ┤
  10% ┤                                    ●
   0% ┼●●●●●●●●●●●●●●●●●●●●●●●●●────●───●
      R0  R2  R4  R6  R8  R10  R11   R12e R12f R12h R12l R12n R12o
      └──── Hebbian (0%) ─────┘ └── Surrogate Gradient (0→73.9%) ──┘

      Phase 1: Hebbian       │  Phase 2: Surrogate    │  Phase 3: Tuning
      12 rounds, 0% success  │  Foundation (R12-12e)  │  Multi-timestep,
      Local learning too     │  First success at 9%   │  adaptive entropy,
      weak for RL tasks      │  with LR=0.01, α=1.0   │  exploration decay
═══════════════════════════════════════════════════════════════════════════
```

### Key Learnings

1. **Fixed reservoirs don't work**: Random quantum circuits don't preserve input structure (QRC: 0%)
2. **Local Hebbian learning is insufficient**: Despite 12 rounds of tuning, the learning signal is too weak for RL tasks
3. **Surrogate gradients unlock quantum SNNs**: Using classical surrogate backward pass with quantum forward pass achieves classical parity
4. **Multi-timestep integration is essential**: Averaging across timesteps reduces quantum shot noise enough for stable REINFORCE training
5. **Adaptive entropy regulation prevents failure modes**: Two-sided regulation (floor + ceiling) eliminates both entropy collapse and explosion
6. **Parameter efficiency**: QSNN achieves 73.9% with 92 parameters vs SpikingReinforce's 131K (1,400x fewer)
7. **Training reliability matters**: SpikingReinforce's 73.3% headline number is misleading — only ~1 in 10 sessions converges, with the rest collapsing catastrophically. QSNN converges in 4/4 sessions (100%), making it a more practical architecture despite similar peak performance

______________________________________________________________________

## Next Steps

- [x] Implement QSNNBrain with QLIF neurons
- [x] Run QSNN benchmark (200 episodes on foraging) - 0% success (Hebbian)
- [x] Tune QSNN hyperparameters (17 rounds of optimization)
- [x] Add surrogate gradient learning mode
- [x] Achieve classical SNN parity on foraging (73.9% vs 73.3%)
- [x] Evaluate QSNN on predator evasion with multi-sensory config — Round P0: 14.5% avg (0–44%), 1/4 converge
- [x] Round P1: Sensory fix + entropy + reward rebalancing — REGRESSION to 1.25% (proximity penalty too aggressive)
- [x] Round P2a: Fix proximity penalty, match gradient decay, LR schedule — 0% success, config tuning exhausted
- [x] Round P2b: Add intra-episode REINFORCE — 3.0% avg success, improved over P2a but no convergence. Weight explosion → entropy collapse is new bottleneck
- [x] Round P2c: Weight clip 3.0→2.0 + 300 episodes — **22.3% avg, 74.3% post-convergence (best), 2-3/4 converge. Surpasses SpikingReinforce post-convergence (62.8%) with 1,400x fewer params**
- [ ] Round P2d: Evasion shaping (proximity 0.05→0.15) + entropy stabilization (0.05→0.08) — targeting evasion ceiling and 0% failure sessions
- [ ] Compare QSNN vs SpikingReinforceBrain on predator tasks (comprehensive comparison)
- [ ] Evaluate QVarCircuit with actor-critic (lower variance)

______________________________________________________________________

## Data References

### QRC Sessions

| Config | Sessions | Notes |
|--------|----------|-------|
| Foraging baseline | 20260204_122807-122818 | 4x200 runs, 0% success |
| Dense encoding | 20260204_131441-131456 | 4x200 runs, 0% success |
| Reduced qubits | 20260204_135543-135557 | 4x200 runs, 0% success |
| High LR + MLP | 20260204_220604-222819 | Various fixes, 0% success |
| Multi-sensory | 20260204_231515 | 25 runs with predators, 0% success |

### QSNN Best Sessions (R12o Foraging Baseline)

| Session | Success | Convergence | Post-Conv | Notes |
|---------|---------|-------------|-----------|-------|
| 20260208_234508 | 79.5% | Ep 56 | 96.7% | Best session |
| 20260208_234514 | 75.0% | Ep 45 | 95.5% | Fastest convergence |
| 20260208_234519 | 68.0% | Ep 63 | 91.2% | Slowest convergence |
| 20260208_234524 | 73.0% | Ep 52 | 96.0% | 100% late mastery |

Full session results and config: `artifacts/logbooks/008/qsnn_foraging_small/`

### QSNN Predator Sessions (Round P0)

| Session | Success | Post-Conv | Notes |
|---------|---------|-----------|-------|
| 20260209_101857 | 0% | N/A | Starvation-dominated, food-seeking improved but insufficient |
| 20260209_101904 | 44% | 72.1% | Best session, converged ep 140, outperforms SpikingReinforce post-conv |
| 20260209_101910 | 14% | ~50% (last 20) | Late learner, still improving at ep 200 |
| 20260209_101915 | 0% | N/A | Entropy collapse, negative CI (-0.287) |

### Appendix

For the full QSNN optimization history (17 rounds, key decisions, failure analysis), see:
[008-appendix-qsnn-optimization.md](008-appendix-qsnn-optimization.md)

### File Locations

- QRC implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrc.py`
- QRC configs: `configs/examples/qrc_*.yml`
