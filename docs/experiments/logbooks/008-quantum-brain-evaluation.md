# 008: Quantum Brain Architecture Evaluation

**Status**: `in_progress` — HybridQuantum brain achieves 96.9% post-convergence on pursuit predators (+25.3 pts over MLP PPO baseline) with 4.3x fewer parameters. HybridClassical ablation (96.3%) confirms architecture + curriculum drive performance, not QSNN. HybridQuantumCortex (QSNN cortex, ~11% quantum fraction) achieved 96.8% on 1-predator but plateaued at ~40-45% on 2-predator environment — halted. Next: evaluate next-generation quantum architectures (H.1 QRH, H.4 QKAN-QLIF).

**Branch**: `feature/add-qsnn-brain`

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

**Status**: Foraging baseline 73.9% (matches classical SNN). Predator evaluation complete: 25.1% best avg on random predators (16 rounds, 64 sessions). Approach halted — transitioning to QSNN-PPO hybrid.

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

The local learning signal is simply too weak for this task. Dense gradient information via surrogate gradients was required. See [qsnn-foraging-optimization.md](supporting/008/qsnn-foraging-optimization.md) for the full optimization history.

### File Locations

- QSNN implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnnreinforce.py`
- QSNN tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnnreinforce.py`
- QSNN configs: `configs/examples/qsnnreinforce_*.yml`

______________________________________________________________________

## QSNN Predator Evasion Evaluation

**Status**: Complete — 16 rounds, 64 sessions. Best: 25.1% avg / 74.3% post-convergence on random predators, 1.25% on pursuit predators. Approach halted.

### Task

Multi-objective RL: collect 10 foods while surviving predators on 20x20 grid. Two phases tested:

- **Random predators** (10 rounds, 40 sessions): speed 1.0, detection_radius 8, 2 predators
- **Pursuit predators** (6 rounds, 24 sessions): speed 0.5, detection_radius 6, movement_pattern pursuit, health system enabled

Uses separated sensory modules (`food_chemotaxis` + `nociception`) for gradient input.

### Results Summary

#### Random Predators — Top Results

| Round | Key Change | Success | Convergence | Key Finding |
|-------|-----------|---------|-------------|-------------|
| P2c | Weight clip 3.0→2.0 + 300 episodes | **22.3%** | 2-3/4 | **Best avg.** 74.3% post-convergence (best session) |
| P2d | entropy_coef 0.05→0.08 | **25.1%** | **3-4/4** | **Best reliability.** Proximity penalty didn't help evasion |
| P3a | PPO clipping + logit_scale 20→5 | 11.6% | 1/4 | Fixed action death and stay-lock |

#### Pursuit Predators — Top Results

| Round | Key Change | Success | Convergence | Key Finding |
|-------|-----------|---------|-------------|-------------|
| PP8 | Holistic param overhaul (16 hidden) | **1.25%** | 0/4 | **First-ever pursuit success** (1 session, 5%) |
| PP9 | Stabilise learning (adv clip, degen skip) | 0% | 0/4 | PP8 was lucky seed, not reproducible |

#### Best Session Details

| Session | Task | Success | Post-Conv | Avg Foods | Evasion | CI |
|---------|------|---------|-----------|-----------|---------|-----|
| 20260210_064304 | Random | 48.3% | **74.3%** | 6.84 | 86.5% | +0.410 |
| 20260213_150444 | Pursuit | 5.0% | — | 2.14 | 47.8% | — |

### Comparison with Classical Baselines

| Metric | QSNN Best (Random) | QSNN Avg (Random) | SpikingReinforce\* | MLP PPO (Pursuit) |
|--------|---------------------|---------------------|--------------------|-------------------|
| Success Rate | 48.3% | 25.1% (P2d) | 61% | 93.5% |
| Post-Conv Success | 74.3% | — | 62.8% | — |
| Evasion Rate | 86.5% | ~88% | ~95% | — |
| Parameters | 94 | 94 | 131K | 34,949 |
| Session Reliability | 2-3/4 (P2c) | 3-4/4 (P2d) | ~1/10 | ~4/4 |

\* SpikingReinforce best session only; ~9/10 sessions fail catastrophically.

QSNN's best post-convergence (74.3%) **exceeds** SpikingReinforce's (62.8%) with 1,400x fewer parameters. But average success and pursuit predator performance remain far below classical baselines.

### Key Findings

1. **Per-encounter evasion never improved through training** across all 64 sessions (P0–PP9). The ~88% random / ~35% pursuit evasion rates are essentially innate from the nociception module, not learned behavior. This is the fundamental limitation.

2. **Cumulative predator risk caps success**: At 88% per-encounter evasion and ~7 encounters/episode: P(survive all) ≈ 0.88^7 ≈ 40%. Even a perfectly food-efficient agent faces this ceiling.

3. **Weight explosion → entropy collapse** was the dominant failure pattern. Solved by weight clipping (P2c), which made collapse recoverable rather than terminal.

4. **Intra-episode REINFORCE** (P2b) was the most impactful code change. Episode-end updates dilute the death signal to noise (25x weaker); 20-step windows keep it actionable.

5. **Fan-in-aware tanh scaling** (PP7) fixed the true root cause of gradient death in wider networks: `tanh(w*x)` saturates when `fan_in * avg_spike * |w| > ~2`. The fix `tanh(w*x / sqrt(fan_in))` keeps gradients alive regardless of layer width.

6. **Pursuit predators exposed fundamental capacity limits**: 94-param actor with single-pass REINFORCE vs MLP PPO's 34,949 params and 40 gradient passes — a combined ~15,000x gap. Classical critic attempts (PP4/PP5) failed completely.

### Architecture Limitations Identified

After 16 rounds across both predator types, the evidence points to structural limitations:

1. **No value function**: Vanilla REINFORCE with 20-step windows has enormous variance. The critic approach failed due to insufficient capacity.
2. **Separated gradient inputs**: The QSNN never learned to use nociception for directional evasion. SpikingReinforce also 0% with separated gradients (logbook 003).
3. **Network capacity**: 94-212 params for dual-objective task vs MLP PPO's 34,949.
4. **Learning efficiency**: 1-3 gradient passes per window vs MLP PPO's 40.

### Conclusion

The standalone QSNN approach has been exhaustively explored. The architecture achieves 73.9% on foraging (matching classical SNN) but cannot reliably solve predator evasion. The next step is the **QSNN-PPO hybrid** — keeping the QSNN actor (which works for foraging) and adding a proper classical critic with GAE advantages and full PPO training loop. See [quantum-architectures.md](../../research/quantum-architectures.md) for the hybrid architecture design.

For the full round-by-round optimization history, see [qsnn-predator-optimization.md](supporting/008/qsnn-predator-optimization.md).

______________________________________________________________________

## QSNN-PPO Hybrid Evaluation

**Status**: Complete — 4 rounds, 16 sessions. Halted: PPO fundamentally incompatible with surrogate gradient spiking networks.

### Architecture

QSNN actor (212 quantum params, QLIF circuits with surrogate gradients) + classical MLP critic (5,569 params) + PPO clipped surrogate with quantum caching. Total ~5.8K params.

```text
Sensory Input (8 features) → QSNN Actor (8→16→4 QLIF, surrogate gradient)
                            → Classical Critic (24-dim → 64 → 64 → 1)
Training: PPO with rollout buffer, GAE advantages, multi-epoch quantum caching
```

### Task

Same pursuit predator environment as QSNN standalone: 2 predators (speed 0.5, detection_radius 6), health system (max_hp 100, predator_damage 20), 20x20 grid, food_chemotaxis + nociception sensory modules.

### Results Summary

| Round | Key Changes | Success | Avg Food | Avg Steps | Evasion | Key Finding |
|-------|-----------|---------|----------|-----------|---------|-------------|
| PPO-0 | Initial architecture | 0% | 0.52 | 111.3 | 43.5% | Buffer never fills; entropy collapse cycles; theta_hidden collapse |
| PPO-1 | Cross-ep buffer, entropy=0.08, LR=0.003 | 0% | 0.56 | 99.5 | 35.1% | **100% policy_loss=0** — PPO completely inert |
| PPO-2 | logit_scale=20, entropy decay, theta_motor init | 0% | 0.48 | 87.5 | 25.1% | Motor spike probs at 0.02 (not 0.5) — wrong hypothesis corrected |
| PPO-3 | theta_hidden=pi/2, theta_motor=linspace(pi/4,3pi/4) | 0% | 0.77 | 105.8 | 42.5% | Motor probs fixed (0.15-0.91) but **policy_loss still 0** |

**Best single episode**: Session 085951, Episode 50 — 7 food, 500 steps (max), reward +31.31. Driven by entropy gradient, not policy learning.

### Root Cause: Architectural Incompatibility

PPO requires the forward pass to produce parameter-dependent action probabilities for importance sampling (`ratio = exp(new_log_prob - old_log_prob)`). The QLIF surrogate gradient approach produces:

- **Forward pass**: Returns quantum-measured spike probability — a **constant** independent of current weights
- **Backward pass**: Computes gradient via classical sigmoid surrogate — parameter-dependent

During PPO re-evaluation, `QLIFSurrogateSpike.forward()` returns the cached quantum measurement, so `pi_new(a|s) == pi_old(a|s)` always, `ratio == 1.0` always, `policy_loss == 0` always. This is **irreconcilable** without replacing the quantum forward pass with a classical analytical approximation, which would strip the architecture of its quantum character.

REINFORCE only needs `d(log_prob)/d(theta)` (the backward pass gradient), which the surrogate provides correctly. This is why QSNNReinforce works but QSNN-PPO cannot.

### Key Discoveries

1. **Motor spike probability suppression** (PPO-2): QLIF motor neurons were barely firing (spike prob ~0.02) due to small theta_motor init. Fixed in PPO-3 with `theta_motor = linspace(pi/4, 3*pi/4)`.

2. **theta_hidden init matters**: `pi/2` places hidden neurons at maximum sensitivity point (`sin²(pi/4) = 0.5`). Prior init at `pi/4` produced spike probs ~0.15.

3. **Entropy gradient provides weak learning**: Even with zero policy_loss, the entropy bonus is differentiable through surrogate gradients. This drove the late-episode food improvements in PPO-3 but is far too slow for convergence.

4. **PPO infrastructure is reusable**: The critic MLP, GAE computation, and training loop code informed the QSNNReinforce A2C implementation.

### Conclusion

**PPO is fundamentally incompatible with surrogate gradient spiking networks.** After 4 rounds (16 sessions, 1,000 episodes), every session produced 0% success with 100% of PPO updates having policy_loss=0. The correct path is actor-critic variance reduction on top of REINFORCE (A2C), which preserves the quantum circuit in both passes. Development shifted to QSNNReinforce A2C.

For the full round-by-round optimization history, see [qsnnppo-optimization.md](supporting/008/qsnnppo-optimization.md).

______________________________________________________________________

## QSNNReinforce A2C Evaluation

**Status**: Complete — 4 rounds, 16 sessions, 3,200 episodes. Halted: A2C critic cannot learn V(s) in this environment. All actor improvement driven by REINFORCE backbone.

### Motivation

After QSNN-PPO failed (PPO incompatible with surrogate gradients), A2C was the natural pivot: add a classical critic for GAE advantage estimation while preserving the REINFORCE backbone that works. A2C only needs backward-pass gradients for the actor (which the surrogate provides) and trains the critic separately — no importance sampling required.

### Architecture

```text
QSNN Actor (212 params, unchanged)     Classical MLP Critic (353–5,569 params)
8 sensory → 16 hidden → 4 motor QLIF   Input: sensory features ± hidden spike rates
Surrogate gradient REINFORCE backbone  Huber loss, separate optimizer

Training: REINFORCE with GAE advantages from critic
1. Collect 20-step windows (intra-episode)
2. Compute GAE advantages using critic V(s) estimates
3. REINFORCE policy gradient with GAE advantages (2 actor epochs)
4. Train critic on same window (5 gradient steps)
```

### Task

Same pursuit predator environment: 2 predators (speed 0.5, detection_radius 6), health system (max_hp 100, predator_damage 20), 20x20 grid, food_chemotaxis + nociception sensory modules.

### Results Summary

| Round | Key Changes | Success | Avg Food | Q4 Food | EV (Q4) | Key Finding |
|-------|-----------|---------|----------|---------|---------|-------------|
| A2C-0 | Initial A2C (50 eps) | 0% | 0.62 | — | ~0 | Critic not learning; EV oscillates near zero |
| A2C-1 | 200 eps, smaller critic, lower LR | 0.13% | 1.52 | 2.05 | -0.008 | Actor improves (REINFORCE); critic still fails; found 4 bugs |
| A2C-2 | Bug fixes (multi-step, bootstrap, grad clip) | 0.63% | 1.32 | 2.28 | **-0.295** | Bugs fixed but EV worse; critic overfits 20-step windows |
| A2C-3 | Sensory-only critic input | 0.50% | 1.50 | 2.05 | **-0.620** | Non-stationarity hypothesis disproved; A2C abandoned |

**Best single episode**: Session 135025, Episode 167 — 10 food, 265 steps, reward +31.65.

### Systematic Hypothesis Elimination

The A2C investigation followed a rigorous hypothesis-testing methodology:

| Round | Hypothesis | Intervention | Result |
|-------|-----------|-------------|--------|
| A2C-0→1 | Insufficient data, too much capacity | 4x episodes (50→200), smaller critic (5.4K→1.1K params) | Critic still fails (EV≈0) |
| A2C-1→2 | Implementation bugs | Fixed 4 bugs: multi-step training, bootstrap ordering, grad clip, EV timing | Critic **worse** (EV -0.295) |
| A2C-2→3 | Non-stationary hidden spike features | Removed hidden spikes from critic input (stationary 8-dim) | Critic **even worse** (EV -0.620) |

After systematically eliminating every hypothesized cause, the root causes are fundamental to the environment:

1. **Partial observability**: The critic sees local gradient features, not global state (position, HP, food count, predator positions). Return prediction is ill-posed.
2. **Policy non-stationarity**: V(s) changes every time the actor updates. With 2 actor epochs per 20-step window, the critic can never converge.
3. **High return variance**: Returns span [-20, +30] depending on stochastic predator encounters.
4. **Short training windows**: 20-step windows with gamma=0.99 poorly approximate true discounted returns over 50-500 step episodes.

### Actor Learning (Independent of Critic)

The REINFORCE actor showed steady improvement across all rounds, entirely independent of the (non-functional) critic:

| Quartile | A2C-1 Foods | A2C-2 Foods | A2C-3 Foods |
|----------|-----------|-----------|-----------|
| Q1 (ep 1-50) | 0.79 | 0.60 | 0.64 |
| Q2 (ep 51-100) | 1.48 | 0.90 | 1.36 |
| Q3 (ep 101-150) | 1.78 | 1.48 | 1.93 |
| Q4 (ep 151-200) | 2.05 | 2.28 | 2.05 |

Food collection improves 2-4x from Q1 to Q4 in every round. This plateau (~2.0 Q4 foods) represents the ceiling of what REINFORCE alone achieves in 200 episodes on pursuit predators.

### Critic Harm (New Finding)

A2C-3 revealed a new failure mode: **the critic actively degrades the actor** when EV is deeply negative. 2 of 4 sessions showed Q4 regression (performance peaks in Q3, declines in Q4). This pattern was not observed in vanilla REINFORCE runs, confirming that the critic's noisy GAE advantages can corrupt the policy gradient.

### Conclusion

**A2C is not viable for QSNNReinforce on pursuit predators.** After 4 rounds, the critic never achieved the 0.2 EV target and progressively worsened (0 → -0.008 → -0.295 → -0.620). All task performance improvements came from the REINFORCE backbone alone. The critic is at best deadweight and at worst harmful.

For the full round-by-round optimization history, see [qsnnreinforce-a2c-optimization.md](supporting/008/qsnnreinforce-a2c-optimization.md).

______________________________________________________________________

## HybridQuantum Brain Evaluation

**Status**: Complete — 4 rounds, 16 sessions, 4,200 episodes across all 3 training stages. **96.9% post-convergence on pursuit predators**, beating the MLP PPO unified baseline by +25.3 points with 4.3x fewer parameters.

### Architecture

```text
Sensory Input
       |
       ├──────────────────────────┐
       v                          v
QSNN Reflex Layer          Classical Cortex (MLP)
6→8→4 QLIF neurons         6 → 64 → 64 → 7
~92 quantum params         (4 action biases + 3 mode logits)
Surrogate grad REINFORCE   PPO training, ~5K actor params
       |                          |
       v                          v
  ┌───────────────────────────────────┐
  │ Fusion: reflex * trust + biases   │
  └────────────────┬──────────────────┘
                   v
         Action Selection (4 actions)

Classical Critic: 6 → 64 → 64 → 1 = V(s), ~5K params
Total: ~10K params (92 quantum + ~10K classical)
```

Key design choices:

- **QSNN reflex** for reactive foraging (proven 73.9% on foraging alone)
- **Classical cortex** for strategic multi-objective behaviour (PPO with GAE)
- **Mode-gated fusion**: cortex learns WHEN to trust QSNN (forage/evade/explore modes)
- **Sensory-only critic** (6-dim from sensory modules, no hidden spikes — lesson from A2C failure)
- **Three-stage curriculum**: isolates proven components, enables incremental validation

### Three-Stage Curriculum

| Stage | What Trains | What's Frozen | Task | Purpose |
|-------|------------|---------------|------|---------|
| 1 | QSNN (REINFORCE) | Cortex unused | Foraging only | Validate quantum reflex |
| 2 | Cortex (PPO) | QSNN frozen | Pursuit predators | Learn strategic behaviour |
| 3 | Both (REINFORCE + PPO) | Nothing | Pursuit predators | Joint fine-tune |

### Results Summary

| Round | Stage | Sessions | Success | Post-Conv | Key Finding |
|-------|-------|----------|---------|-----------|-------------|
| 1 | 1 (QSNN only) | 4 × 200 eps | 91.0%\* | 99.3%\* | QSNN reflex validated for foraging |
| 2 | 2 (cortex PPO) | 4 × 200 eps | 66.4% | 81.9% | Cortex learns foraging + evasion |
| 3 | 2 (tuned) | 4 × 500 eps | 84.3% | 91.7% | LR schedule + 12 PPO epochs + mechanosensation |
| 4 | 3 (joint) | 4 × 500 eps | **96.9%** | **96.9%** | **Best result; immediate convergence** |

\*Stage 1 best 3 of 4 sessions (1 outlier excluded).

### Stage 3 Highlights (Round 4)

| Session | Success | Post-Conv | Evasion | HP Deaths | Dist Eff | Composite |
|---------|---------|-----------|---------|-----------|----------|-----------|
| 061309 | 96.6% | 96.6% | 89.9% | 17 | 0.510 | 0.854 |
| 061317 | 97.2% | 97.2% | 90.0% | 14 | 0.554 | 0.871 |
| 061323 | 96.4% | 96.4% | 92.1% | 17 | 0.528 | 0.863 |
| 061329 | 97.2% | 97.2% | 91.6% | 14 | 0.546 | 0.870 |
| **Avg** | **96.9%** | **96.9%** | **90.9%** | **15.5** | **0.534** | **0.864** |

- **Immediate convergence**: All sessions at 90-100% from episode 1 (pre-trained weights)
- **Only 3.1% failure rate** (all health depletion from cumulative predator damage)
- **0.8 pt session variance** vs 8.8 pts in Stage 2 (pre-trained init eliminates convergence lottery)

### Comparison with Classical Baselines

| Metric | Hybrid Stage 3 | MLP PPO Unified | MLP PPO Legacy | Gap (vs Unified) |
|--------|---------------|-----------------|----------------|------------------|
| Post-conv success | **96.9%** | 71.6% | 94.5% | **+25.3 pts** |
| HP death rate | **3.1%** | 44.8% | 6.8% | **-41.7 pts** |
| Evasion rate | **90.9%** | ~82% | — | **+8.9 pts** |
| Trainable params | **9,828** | ~42K | ~42K | **4.3x fewer** |

The hybrid brain beats the apples-to-apples MLP PPO baseline by +25.3 points and even surpasses the "cheating" legacy MLP PPO (which uses pre-computed combined gradient) by +2.4 points.

### Key Findings

1. **Three-stage curriculum works**: Isolating QSNN, cortex, then joint fine-tune prevented interference and enabled incremental validation. Each stage improved on the previous.

2. **QSNN reflex provides qualitative but not quantitative value**: The classical ablation (HybridClassical) achieves equivalent task performance (96.3% vs 96.9% mean post-conv), proving the QSNN is not the key ingredient. However, the QSNN earns ~1.5x more trust from the cortex (0.55 vs 0.37), achieves higher chemotaxis indices, and enables a collaborative strategy rather than the cortex-dominant strategy the classical reflex produces.

3. **Mode gating acts as static trust**: The 3-mode design works as a learned mixing parameter rather than a dynamic per-step mode switch. HybridQuantum converges to QSNN-collaborative (trust ~0.55, forage mode dominant), while HybridClassical converges to cortex-dominant (trust ~0.37, evade mode dominant) — fundamentally different strategies yielding equivalent performance.

4. **W_hm is the "plastic" weight in joint fine-tune**: Hidden→motor weights grew +27.7% while sensory→hidden weights barely moved (+2.7%), indicating the QSNN's output mapping adapted while input encoding was preserved from Stage 1.

5. **Pre-trained initialisation eliminates convergence variance**: Stage 3 session variance was 0.8 pts (vs Stage 2's 8.8 pts).

### File Locations

- Implementation: `packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py`
- Tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridquantum.py`
- Stage 1 config: `configs/examples/hybridquantum_foraging_small.yml`
- Stage 2 config: `configs/examples/hybridquantum_pursuit_predators_small.yml`
- Stage 3 config: `configs/examples/hybridquantum_pursuit_predators_small_finetune.yml`

Full optimization history (4 rounds, 16 sessions): [hybridquantum-optimization.md](supporting/008/hybridquantum-optimization.md)

______________________________________________________________________

## HybridClassical Ablation Study

**Status**: Complete — 12 sessions across 3 stages (4,200 episodes). Classical ablation confirms QSNN quantum reflex is not the key performance driver.

### Purpose

Isolate the QSNN quantum reflex contribution in the HybridQuantum brain. The HybridQuantum achieved 96.9% post-convergence, but we cannot determine whether that advantage comes from the QSNN itself, the three-stage curriculum, or the mode-gated fusion architecture. **HybridClassical** replaces the QSNN reflex (92 quantum params) with a small classical MLP reflex (~116 classical params), keeping everything else identical.

### Architecture

```text
HybridQuantum (control)             HybridClassical (ablation)
========================             ==========================
QSNN Reflex (92 params)             Classical MLP Reflex (~116 params)
  6→8→4 QLIF neurons                  Linear(2→16) + ReLU + Linear(16→4)
  Quantum circuits + surrogates       + sigmoid output scaling
        |                                   |
        v                                   v
  Mode-Gated Fusion (same)           Mode-Gated Fusion (same)
        ^                                   ^
        |                                   |
  Classical Cortex (same)             Classical Cortex (same)
  PPO training                        PPO training
```

### Results Summary

| Stage | Sessions | Episodes | Success | Post-Conv | Key Finding |
|-------|----------|----------|---------|-----------|-------------|
| 1 (reflex only) | 4 × 200 | 800 | 97.0% | 99.6% | Faster convergence than QSNN (6.75 vs 18 eps) |
| 2 (cortex PPO) | 4 × 500 | 2,000 | 92.0% | 94.5% | Competitive with quantum Stage 2 (94.5% vs 91.7%) |
| 3 (joint fine-tune) | 4 × 500 | 2,000 | **96.1%** | **96.3%** | **Best session 97.8% exceeds quantum best 97.2%** |

### Stage 3 Highlights

| Session | Success | Post-Conv | Composite | Evasion | Notes |
|---------|---------|-----------|-----------|---------|-------|
| 000530 | **97.8%** | **97.8%** | **0.892** | 90.4% | Best overall — exceeds quantum best |
| 000537 | 95.0% | 95.5% | 0.861 | 89.5% | |
| 000543 | 96.2% | 96.5% | 0.871 | 90.4% | 164-ep success streak |
| 000549 | 95.2% | 95.2% | 0.863 | 90.3% | |
| **Mean** | **96.1%** | **96.3%** | **0.872** | **90.2%** | |

### Final Ablation Comparison

| Metric | HybridQuantum (QSNN) | HybridClassical (MLP) | Verdict |
|--------|----------------------|----------------------|---------|
| Stage 3 best post-conv | 97.2% | **97.8%** | Classical +0.6 pp |
| Stage 3 mean post-conv | **96.9%** | 96.3% | Within noise |
| Stage 3 best composite | 0.871 | **0.892** | Classical better |
| Stage 1 chemotaxis index | higher | 0.411 (sub-biological) | Quantum more biological |
| Reflex params | 92 | 116 | Quantum more compact |

### Fusion Trust Analysis

The two architectures adopt **fundamentally different strategies** that achieve equivalent task performance:

| Metric | HybridClassical | HybridQuantum |
|--------|----------------|---------------|
| Late reflex trust | **0.37** (cortex-dominant) | **0.55** (collaborative) |
| Dominant mode | **Evade** (~0.48) | **Forage** (~0.55) |

The QSNN earns ~1.5x more trust than the classical MLP, yet task performance is equivalent — the classical cortex compensates by doing more itself.

### Ablation Conclusion

**The QSNN quantum reflex is not the key performance ingredient.** Performance is statistically indistinguishable between quantum and classical reflexes. What drives performance:

1. The **three-stage curriculum** (pre-train reflex → cortex PPO → joint fine-tune)
2. The **mode-gated fusion architecture** (reflex + cortex specialization)
3. The **cortex PPO network** (handles multi-objective behaviour)

**Where QSNN retains value**: biological fidelity (higher chemotaxis indices), parameter efficiency (92 vs 116 params), and scientific interest as a biologically plausible neural computation model.

### File Locations

- Implementation: `packages/quantum-nematode/quantumnematode/brain/arch/hybridclassical.py`
- Tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridclassical.py`
- Stage 1 config: `configs/examples/hybridclassical_foraging_small.yml`
- Stage 2 config: `configs/examples/hybridclassical_pursuit_predators_small.yml`
- Stage 3 config: `configs/examples/hybridclassical_pursuit_predators_small_finetune.yml`

Full ablation experiment details (12 sessions, trust analysis): [hybridclassical-ablation.md](supporting/008/hybridclassical-ablation.md)

______________________________________________________________________

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

| Architecture | Trainable | Gradient Approach | Best Success (Foraging) | Best Success (Pursuit) | Viable? |
|--------------|-----------|-------------------|------------------------|------------------------|---------|
| QRC | Readout only | REINFORCE on readout | 0% | 0% | No |
| QSNN (Hebbian) | Weights + θ | 3-factor local Hebbian | 0% | N/A | No |
| QSNN-PPO | QSNN actor + MLP critic | Surrogate + PPO (incompatible) | N/A | 0% | **No** |
| QSNNReinforce A2C | QSNN actor + MLP critic | Surrogate + A2C critic | N/A | 0.63% | **No** |
| QSNN (Surrogate) | Weights + θ | Quantum forward + surrogate backward | **73.9%** | 1.25% | Partial |
| QVarCircuit (gradient) | Full circuit | Parameter-shift rule | ~40% | N/A | Marginal |
| QVarCircuit (CMA-ES) | Full circuit | Evolutionary | 88% | 76.1%\* | Yes (not online) |
| **HybridQuantum** | **QSNN + cortex MLP** | **Surrogate REINFORCE + PPO** | **91.0%** | **96.9%** | **Yes** |
| HybridClassical (ablation) | MLP reflex + cortex MLP | Backprop REINFORCE + PPO | 97.0% | 96.3% | Yes (control) |

\*CMA-ES is evolutionary, not gradient-based.

```text
QUANTUM ARCHITECTURE SUCCESS RATES (Best Post-Convergence, Gradient-Based)
═══════════════════════════════════════════════════════════════════════════

Architecture                  Foraging    Pursuit Pred    Params
────────────────────────────────────────────────────────────────────────────
QRC (fixed reservoir)          0%          0%              ~1K       ❌
QSNN Hebbian                   0%          N/A             92        ❌
QSNN-PPO Hybrid                N/A         0%              5.8K      ❌†
QSNNReinforce A2C              N/A         0.6%            ~1.3K     ❌‡
QVarCircuit (param-shift)      ~40%        N/A             ~60       ⚠️
QSNN Surrogate                 73.9%       1.25%           92        ✓ (forage)
QVarCircuit (CMA-ES)           88%         76.1%           ~60       ✓*
HybridQuantum                  91.0%       96.9%           ~10K      ✓✓
HybridClassical (ablation)     97.0%       96.3%           ~10K      ✓ (control)
────────────────────────────────────────────────────────────────────────────
MLPPPOBrain (classical)        96.7%       71.6%†† / 94.5% ~42K      (ref)

† QSNN-PPO: PPO incompatible with surrogate gradients (policy_loss=0)
‡ QSNNReinforce A2C: critic never learned (EV -0.620)
* CMA-ES is evolutionary, not gradient-based
†† Unified sensory modules (apples-to-apples comparison)

KEY INSIGHT: HybridQuantum is the first quantum architecture to SURPASS
a classical baseline on a multi-objective task using gradient-based
online learning. It beats MLP PPO unified by +25.3 points on pursuit
predators with 4.3x fewer parameters. However, classical ablation
(HybridClassical) shows equivalent performance — the three-stage
curriculum and mode-gated fusion drive the result, not the QSNN.
═══════════════════════════════════════════════════════════════════════════
```

```text
HYBRID QUANTUM BRAIN: THREE-STAGE CURRICULUM PROGRESSION
═══════════════════════════════════════════════════════════════════════════

Post-Convergence Success Rate (Pursuit Predators)
 100% ┤                                              ●●●●── 96.9%
  95% ┤                                        ●────●
  90% ┤                                  ●────●
  85% ┤                            ●────●
  80% ┤                      ●────●
  75% ┤                ●────●                         ·····  71.6% MLP PPO
  70% ┤          ●────●                                      (unified baseline)
  65% ┤
  60% ┤
       Stage 1   Stage 2    Stage 2    Stage 3
       (QSNN)    Round 2    Round 3   (Joint FT)
       forage     81.9%      91.7%      96.9%
       only

  Stage 1: QSNN reflex on foraging (REINFORCE)
  Stage 2: Cortex PPO with frozen QSNN (pursuit predators)
  Stage 3: Joint fine-tune (both trainable, pre-trained weights)
═══════════════════════════════════════════════════════════════════════════
```

```text
PARAMETER EFFICIENCY: HYBRID QUANTUM vs CLASSICAL BASELINES
═══════════════════════════════════════════════════════════════════════════

                     Parameters      Pursuit Post-Conv   Reliability
───────────────────────────────────────────────────────────────────────────
QSNN (standalone)    ██  92           1.25%              0/24 converge
HybridQuantum        ████  ~10K      96.9%              4/4 (100%)
MLP PPO (unified)    ████████  ~42K  71.6%              ~4/4
MLP PPO (legacy)     ████████  ~42K  94.5%              ~4/4

HybridQuantum achieves SOTA with 4.3x fewer parameters than MLP PPO.
The QSNN component provides 92 quantum parameters; the cortex adds ~10K
classical parameters for strategic multi-objective learning.
═══════════════════════════════════════════════════════════════════════════
```

### Key Learnings

01. **Fixed reservoirs don't work**: Random quantum circuits don't preserve input structure (QRC: 0%)
02. **Local Hebbian learning is insufficient**: Despite 12 rounds of tuning, the learning signal is too weak for RL tasks
03. **Surrogate gradients unlock quantum SNNs**: Using classical surrogate backward pass with quantum forward pass achieves classical parity on foraging
04. **Multi-timestep integration is essential**: Averaging across timesteps reduces quantum shot noise enough for stable REINFORCE training
05. **Adaptive entropy regulation prevents failure modes**: Two-sided regulation (floor + ceiling) eliminates both entropy collapse and explosion
06. **Parameter efficiency**: QSNN achieves 73.9% with 92 parameters vs SpikingReinforce's 131K (1,400x fewer)
07. **Training reliability matters**: SpikingReinforce's 73.3% headline number is misleading — only ~1 in 10 sessions converges, with the rest collapsing catastrophically. QSNN converges in 4/4 sessions (100%), making it a more practical architecture despite similar peak performance
08. **Standalone QSNN cannot solve multi-objective tasks**: Despite 16 rounds and 64 sessions of predator optimization, per-encounter evasion never improved through training. The architecture learns foraging but not evasion — a hybrid approach is needed
09. **Fan-in-aware scaling is critical for wider networks**: `tanh(w*x / sqrt(fan_in))` prevents gradient death that otherwise occurs when layer width exceeds ~10 neurons
10. **PPO is incompatible with surrogate gradient spiking networks**: The QLIFSurrogateSpike forward pass returns a constant (quantum measurement), making PPO's importance sampling ratio always 1.0. REINFORCE-based methods (which only need backward-pass gradients) are the correct algorithm family for QSNN
11. **theta_motor init near pi/2 is critical**: Motor neurons initialised with small theta (~0) produce spike probs ~0.02, effectively dead. Initialising in `linspace(pi/4, 3*pi/4)` places neurons in the responsive range (spike probs 0.15-0.85)
12. **A2C critic cannot learn V(s) with partial observations**: After 4 rounds (16 sessions), the classical critic never achieved meaningful explained variance on pursuit predators. Root causes: partial observability (critic sees local gradients, not global state), policy non-stationarity, high return variance, and short 20-step GAE windows. Systematically eliminated data quantity, capacity, bugs, and feature non-stationarity as causes.
13. **The REINFORCE actor learns independently of the critic**: All food collection improvements (0.6→2.0 Q4 foods) across 4 A2C rounds were driven by the REINFORCE backbone, not critic-provided advantages. The critic was confirmed as non-functional deadweight.
14. **A non-functional critic can actively harm learning**: When explained variance is deeply negative, GAE advantages inject noise into the policy gradient that is worse than normalized-returns REINFORCE. A2C-3 showed Q4 regression in 2/4 sessions — a pattern absent in vanilla REINFORCE runs.
15. **Hierarchical hybrid architecture solves multi-objective**: Combining QSNN reflex (REINFORCE) with classical cortex (PPO) via mode-gated fusion achieves 96.9% on pursuit predators — surpassing both the unified MLP PPO baseline (+25.3 pts) and the legacy "cheating" baseline (+2.4 pts) with 4.3x fewer parameters.
16. **Three-stage curriculum prevents interference**: Training QSNN alone (stage 1), cortex alone (stage 2), then jointly (stage 3) enables incremental validation. Each stage improved over the previous.
17. **Mode gating acts as static trust, not dynamic switching**: The cortex learns a stable mixing parameter between QSNN reflex and its own action biases. In Stage 3, all sessions converge to QSNN-collaborative (trust ~0.5), not cortex-dominant.
18. **Pre-trained initialisation eliminates convergence variance**: Stage 3 session variance was 0.8 pts vs Stage 2's 8.8 pts, because pre-trained weights remove seed-dependent convergence lottery.
19. **Classical PPO critic works when given its own sensory input**: Unlike A2C where the critic shared the QSNN's features (and failed), the hybrid cortex critic receives independent sensory module features. Explained variance reached +0.29 (vs A2C's -0.620).
20. **LR scheduling is essential for cortex PPO**: Warmup + cosine decay produced +9.8 pts improvement over flat LR in Stage 2.
21. **QSNN quantum reflex is not the key performance driver**: HybridClassical ablation (classical MLP reflex, ~116 params) achieves 96.3% mean / 97.8% best post-convergence — statistically indistinguishable from HybridQuantum's 96.9%. The three-stage curriculum and mode-gated fusion architecture are what matter. The QSNN retains value for biological fidelity (higher chemotaxis indices) and parameter efficiency (92 vs 116 params).
22. **Different reflexes produce different strategies at equivalent performance**: HybridQuantum's cortex trusts the QSNN (trust ~0.55, forage-mode dominant), while HybridClassical's cortex partially gates out the MLP reflex (trust ~0.37, evade-mode dominant). The cortex adapts its strategy to the quality of the reflex signal — compensating when the reflex is weaker.
23. **QSNN cortex (REINFORCE) hits ~40-45% ceiling on hard multi-objective tasks**: HybridQuantumCortex achieved 96.8% on 1-predator and 88.8% on foraging, but plateaued at ~40-45% on 2-predator environment despite 9 rounds of optimisation. Vanishing gradients (norms drop to 0.04-0.07 after LR decay), ineffective critic (EV ~0.10), and frozen mode distributions indicate the REINFORCE+surrogate gradient combination lacks the gradient signal strength for complex multi-objective tasks.
24. **Graduated curriculum is essential for QSNN cortex training**: Jumping directly from foraging reflex to 2-predator cortex fails catastrophically (3.1%). The foraging → 1 predator → 2 predator progression (Stages 2a → 2b → 2c) enables incremental validation.
25. **Joint fine-tuning can cause catastrophic forgetting when reflex and cortex were trained on different tasks**: HybridQuantumCortex Stage 3 destroyed the foraging-tuned reflex with predator-environment REINFORCE gradients (19.3%, declining trajectory). HybridQuantum's Stage 3 worked because its reflex was already exposed to predators.
26. **Bug fixes produce larger gains than hyperparameter tuning**: Three critical HybridQuantumCortex bugs (missing advantage clipping/normalization/weight clamping) improved cortex foraging from 19.1% → 52.6% (+33.5pp). Subsequent config tuning added another +36.2pp to 88.8%.

______________________________________________________________________

## Next Steps

- [x] Implement QSNNBrain with QLIF neurons
- [x] Run QSNN benchmark (200 episodes on foraging) — 0% success (Hebbian)
- [x] Tune QSNN hyperparameters (17 rounds of optimization)
- [x] Add surrogate gradient learning mode
- [x] Achieve classical SNN parity on foraging (73.9% vs 73.3%)
- [x] Evaluate QSNN on random predator evasion (10 rounds, 40 sessions) — best: 25.1% avg (P2d), 74.3% post-convergence best session (P2c)
- [x] Evaluate QSNN on pursuit predator evasion (6 rounds, 24 sessions) — best: 1.25% avg (PP8), approach exhausted
- [x] Halt standalone QSNN predator approach — architecture limitation confirmed
- [x] Implement QSNN-PPO hybrid (QSNN actor + classical critic with GAE + PPO training loop) — 4 rounds, 16 sessions
- [x] Halt QSNN-PPO — PPO incompatible with surrogate gradients (policy_loss=0 in 100% of updates)
- [x] Implement QSNNReinforce A2C (actor-critic variance reduction on REINFORCE backbone)
- [x] Evaluate QSNNReinforce A2C on pursuit predators — 4 rounds, 16 sessions, 3,200 episodes. Critic never learned (EV: 0 → -0.620). Approach halted.
- [x] Implement HybridQuantum brain (QSNN reflex + classical cortex MLP + mode-gated fusion)
- [x] Stage 1: Validate QSNN reflex in hybrid wrapper — 4 sessions, 91.0% (best 3/4)
- [x] Stage 2: Train cortex PPO with frozen QSNN — 8 sessions across 2 rounds, 81.9% → 91.7% post-convergence
- [x] Stage 3: Joint fine-tune — 4 sessions, **96.9% post-convergence**, +25.3 pts over MLP PPO baseline
- [x] Three-stage curriculum validated end-to-end — quantum multi-objective learning achieved
- [x] Implement HybridClassical brain (classical ablation — MLP reflex replacing QSNN)
- [x] Run HybridClassical 3-stage ablation — 12 sessions, 4,200 episodes
- [x] Ablation conclusion: QSNN not key ingredient; curriculum + fusion architecture drive performance
- [x] Fusion trust analysis: quantum trusted 1.5x more but performance equivalent
- [x] Implement HybridQuantumCortex brain (QSNN cortex replacing classical MLP cortex, ~11% quantum fraction)
- [x] Stage 1: Validate QSNN reflex in HybridQuantumCortex wrapper — 4 sessions, 82.5% foraging
- [x] Stage 2a: Graduated curriculum for cortex training — 12 sessions across 3 rounds, 19.1% → 52.6% → 88.8%
- [x] Stage 2b: 1 pursuit predator — 4 sessions, 96.8% success, zero deaths
- [x] Stage 2c: 2 pursuit predators — 8 sessions across 2 rounds, plateaued at ~40-45%
- [x] Stage 3: Joint fine-tune — 4 sessions, catastrophic forgetting (19.3%), abandoned
- [x] HybridQuantumCortex halted — REINFORCE with surrogate gradients cannot push past ~40-45% on 2-predator environment

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

### QSNN Best Predator Sessions

| Session | Task | Success | Post-Conv | Key Finding |
|---------|------|---------|-----------|-------------|
| 20260210_064304 | Random (P2c) | **48.3%** | **74.3%** | Best overall result, 1,400x fewer params than SpikingReinforce |
| 20260210_111144 | Random (P2d) | 35.3% | — | Best reliability round (3-4/4 converge) |
| 20260213_150444 | Pursuit (PP8) | 5.0% | — | First and only pursuit predator success |

Full predator session references (64 sessions across 16 rounds): [qsnn-predator-optimization.md](supporting/008/qsnn-predator-optimization.md)

### QSNN-PPO Sessions

| Round | Sessions | Result |
|-------|----------|--------|
| PPO-0 | 20260215_040128-040155 | 0%, buffer never fills, entropy collapse cycles |
| PPO-1 | 20260215_063301-063319 | 0%, 100% policy_loss=0 (PPO completely inert) |
| PPO-2 | 20260215_082646-082702 | 0%, motor spike probs at 0.02 (wrong hypothesis corrected) |
| PPO-3 | 20260215_085929-085951 | 0%, motor probs fixed but policy_loss still 0 (root cause identified) |

Full QSNN-PPO optimization history (4 rounds, 16 sessions): [qsnnppo-optimization.md](supporting/008/qsnnppo-optimization.md)

### QSNNReinforce A2C Sessions

| Round | Sessions | Episodes | Result |
|-------|----------|----------|--------|
| A2C-0 | 20260215_103816-103835 | 50 | 0%, critic EV ≈ 0, worse than vanilla REINFORCE |
| A2C-1 | 20260215_121727-121748 | 200 | 0.13%, actor improves via REINFORCE, 4 critic bugs found |
| A2C-2 | 20260215_135006-135025 | 200 | 0.63%, bugs fixed but EV worse (-0.295) |
| A2C-3 | 20260215_221154-221213 | 200 | 0.50%, sensory-only critic, EV worst yet (-0.620) |

Full QSNNReinforce A2C optimization history (4 rounds, 16 sessions): [qsnnreinforce-a2c-optimization.md](supporting/008/qsnnreinforce-a2c-optimization.md)

### HybridQuantum Sessions

| Round | Stage | Sessions | Episodes | Result |
|-------|-------|----------|----------|--------|
| 1 | 1 | 20260216_100503, 100507, 100512, 100516 | 200 | 91.0% foraging (best 3/4); QSNN reflex validated |
| 2 | 2 | 20260216_132604, 132609, 132614, 132619 | 200 | 81.9% post-conv; beats MLP PPO unified +10.3 pts |
| 3 | 2 | 20260216_213406, 20260217_012722, 012729, 012735 | 500 | 91.7% post-conv; beats MLP PPO unified +20.1 pts |
| 4 | 3 | 20260217_061309, 061317, 061323, 061329 | 500 | **96.9% post-conv**; beats MLP PPO unified +25.3 pts |

Full optimization history (4 rounds, 16 sessions): [hybridquantum-optimization.md](supporting/008/hybridquantum-optimization.md)

Experiment results: `artifacts/logbooks/008/hybridquantum_foraging_small/`, `artifacts/logbooks/008/hybridquantum_pursuit_predators_small/`

### HybridClassical Ablation Sessions

| Stage | Sessions | Episodes | Config | Key Result |
|-------|----------|----------|--------|------------|
| 1 | 20260217_214132, 214138, 214143, 214148 | 200 | `hybridclassical_foraging_small.yml` | 97.0% foraging, 99.6% post-conv, all 4 sessions reliable |
| 2 | 20260217_223325, 223331, 223336, 223340 | 500 | `hybridclassical_pursuit_predators_small.yml` | 94.5% post-conv, competitive with quantum |
| 3 | 20260218_000530, 000537, 000543, 000549 | 500 | `hybridclassical_pursuit_predators_small_finetune.yml` | **96.3% post-conv mean, 97.8% best** — matches quantum |

Best weights:

| Component | Session | Path |
|-----------|---------|------|
| Reflex (Stage 1) | 214143 | `artifacts/models/20260217_214143/reflex_weights.pt` |
| Cortex (Stage 2) | 223336 | `artifacts/models/20260217_223336/cortex_weights.pt` |
| Both (Stage 3) | 000530 | `artifacts/models/20260218_000530/reflex_weights.pt` + `cortex_weights.pt` |

Experiment results: `artifacts/logbooks/008/hybridclassical_foraging_small/`, `artifacts/logbooks/008/hybridclassical_pursuit_predators_small/`

Full ablation details (12 sessions, trust analysis): [hybridclassical-ablation.md](supporting/008/hybridclassical-ablation.md)

### HybridQuantumCortex Sessions

The HybridQuantumCortex replaces the classical cortex MLP (~5K params) with a QSNN-based cortex (~252 quantum params) using grouped QLIF neurons per sensory modality, raising the quantum fraction from ~1% to ~11%. Trained via surrogate gradient REINFORCE with critic-provided GAE advantages (not PPO). Graduated curriculum: Stage 1 (reflex foraging) → 2a (cortex foraging) → 2b (1 predator) → 2c (2 predators).

| Round | Stage | Sessions | Episodes | Result |
|-------|-------|----------|----------|--------|
| 1 | 1 | 20260218_131409-131425 | 4×200 | 82.5% foraging; QSNN reflex validated |
| 2 | 2 (orig) | 20260219_002830-003141 | 4×500 | 3.1% catastrophic failure; curriculum needed |
| 3 | 2a R1 | 20260219_060444-060503 | 4×200 | 19.1%; 3 critical bugs found |
| 4 | 2a R2 | 20260219_093944-094001 | 4×200 | 52.6%; bug fixes +33.5pp |
| 5 | 2a R3 | 20260219_133505-133527 | 4×200 | **88.8%** foraging; exceeds Stage 1 baseline |
| 6 | 2b | 20260220_014906-014927 | 4×500 | **96.8%** 1-predator; zero deaths |
| 7 | 2c R1 | 20260220_101539-101552 | 4×500 | 39.8%; still improving at ep 500 |
| 8 | 3 | 20260220_182051-182112 | 4×500 | 19.3% catastrophic forgetting; abandoned |
| 9 | 2c R2 | 20260221_052315-052336 | 4×600 | 40.9%; plateau confirmed; halted |

**Best results**: 96.8% with 1 predator (Stage 2b), 88.8% on foraging (Stage 2a R3). Architecture hits ~40-45% ceiling on 2-predator environment under REINFORCE with surrogate gradients.

**Outcome**: Architecture halted. The QSNN cortex under REINFORCE with surrogate gradients cannot push past ~40-45% on the 2-predator environment. The limitation is the training method (vanishing gradients, ineffective critic) not the architecture itself. Stage 2a-2b results prove the cortex architecture works on simpler tasks.

Full optimization history (9 rounds, 32 sessions): [hybridquantumcortex-optimization.md](supporting/008/hybridquantumcortex-optimization.md)

Experiment results: `artifacts/logbooks/008/hybridquantumcortex_foraging_small/`, `artifacts/logbooks/008/hybridquantumcortex_pursuit_predators_small/`

### Appendices

- QSNN foraging optimization history (17 rounds): [qsnn-foraging-optimization.md](supporting/008/qsnn-foraging-optimization.md)
- QSNN predator optimization history (16 rounds, 64 sessions): [qsnn-predator-optimization.md](supporting/008/qsnn-predator-optimization.md)
- QSNN-PPO optimization history (4 rounds, 16 sessions): [qsnnppo-optimization.md](supporting/008/qsnnppo-optimization.md)
- QSNNReinforce A2C optimization history (4 rounds, 16 sessions): [qsnnreinforce-a2c-optimization.md](supporting/008/qsnnreinforce-a2c-optimization.md)
- HybridQuantum optimization history (4 rounds, 16 sessions): [hybridquantum-optimization.md](supporting/008/hybridquantum-optimization.md)
- HybridClassical ablation (12 sessions, trust analysis): [hybridclassical-ablation.md](supporting/008/hybridclassical-ablation.md)
- HybridQuantumCortex optimization history (9 rounds, 32 sessions): [hybridquantumcortex-optimization.md](supporting/008/hybridquantumcortex-optimization.md)

### File Locations

- QRC implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrc.py`
- QRC configs: `configs/examples/qrc_*.yml`
- QSNN implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnnreinforce.py`
- QSNN tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnnreinforce.py`
- QSNN configs: `configs/examples/qsnnreinforce_*.yml`
- QSNN-PPO implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnnppo.py`
- QSNN-PPO tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnnppo.py`
- QSNN-PPO configs: `configs/examples/qsnnppo_*.yml`
- HybridQuantum implementation: `packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py`
- HybridQuantum tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridquantum.py`
- HybridQuantum configs: `configs/examples/hybridquantum_*.yml`
- HybridClassical implementation: `packages/quantum-nematode/quantumnematode/brain/arch/hybridclassical.py`
- HybridClassical tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridclassical.py`
- HybridClassical configs: `configs/examples/hybridclassical_*.yml`
- HybridQuantumCortex implementation: `packages/quantum-nematode/quantumnematode/brain/arch/hybridquantumcortex.py`
- HybridQuantumCortex tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridquantumcortex.py`
- HybridQuantumCortex configs: `configs/examples/hybridquantumcortex_*.yml`
