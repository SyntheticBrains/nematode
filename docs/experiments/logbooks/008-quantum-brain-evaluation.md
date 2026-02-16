# 008: Quantum Brain Architecture Evaluation

**Status**: `in_progress` — QSNN-PPO halted; QSNNReinforce A2C halted (critic cannot learn V(s) in pursuit predator environment). Evaluating next approach.

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

The local learning signal is simply too weak for this task. Dense gradient information via surrogate gradients was required. See [008-appendix-qsnn-foraging-optimization.md](008-appendix-qsnn-foraging-optimization.md) for the full optimization history.

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

For the full round-by-round optimization history, see [008-appendix-qsnn-predator-optimization.md](008-appendix-qsnn-predator-optimization.md).

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

For the full round-by-round optimization history, see [008-appendix-qsnnppo-optimization.md](008-appendix-qsnnppo-optimization.md).

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

For the full round-by-round optimization history, see [008-appendix-qsnnreinforce-a2c-optimization.md](008-appendix-qsnnreinforce-a2c-optimization.md).

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

| Architecture | Trainable | Gradient Approach | Best Success | Viable? |
|--------------|-----------|-------------------|--------------|---------|
| QRC | Readout only | REINFORCE on readout | 0% | No |
| QSNN (Hebbian) | Weights + θ | 3-factor local Hebbian | 0% | No |
| QSNN-PPO | QSNN actor + MLP critic | Surrogate + PPO (incompatible) | 0% | **No** |
| QSNNReinforce A2C | QSNN actor + MLP critic | Surrogate + A2C critic | 0.63% (pursuit) | **No** |
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
QSNN-PPO Hybrid               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%   ❌†
QSNNReinforce A2C             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.6% ❌‡‡
QVarCircuit (param-shift)     ████████████████░░░░░░░░░░░░░░░░░░░░░░░  ~40% ⚠️
QSNN Surrogate                ██████████████████████████████░░░░░░░░░  73.9% ✓
QVarCircuit (CMA-ES)          ███████████████████████████████████░░░░  88%   ✓*
────────────────────────────────────────────────────────────────────────────
SpikingReinforce (classical)  █████████████████████████████░░░░░░░░░░  73.3%‡ (ref)

* CMA-ES is evolutionary, not gradient-based
† QSNN-PPO: PPO incompatible with surrogate gradients (policy_loss=0 always)
‡‡ QSNNReinforce A2C: pursuit predator only; critic never learned (EV -0.620)
‡ SpikingReinforce best session only; ~9/10 sessions fail (~10% reliability)
  QSNN achieves 73.9% with 100% session reliability (4/4 converge)

KEY INSIGHT: QSNN Surrogate is the first quantum architecture to match a
classical baseline using gradient-based learning (no evolution needed).
Neither PPO nor A2C can be combined with QSNN for multi-objective tasks —
PPO fails (importance sampling), A2C fails (critic can't learn V(s)).
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
- [ ] Determine next approach for quantum multi-objective learning

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

Full predator session references (64 sessions across 16 rounds): [008-appendix-qsnn-predator-optimization.md](008-appendix-qsnn-predator-optimization.md)

### QSNN-PPO Sessions

| Round | Sessions | Result |
|-------|----------|--------|
| PPO-0 | 20260215_040128-040155 | 0%, buffer never fills, entropy collapse cycles |
| PPO-1 | 20260215_063301-063319 | 0%, 100% policy_loss=0 (PPO completely inert) |
| PPO-2 | 20260215_082646-082702 | 0%, motor spike probs at 0.02 (wrong hypothesis corrected) |
| PPO-3 | 20260215_085929-085951 | 0%, motor probs fixed but policy_loss still 0 (root cause identified) |

Full QSNN-PPO optimization history (4 rounds, 16 sessions): [008-appendix-qsnnppo-optimization.md](008-appendix-qsnnppo-optimization.md)

### QSNNReinforce A2C Sessions

| Round | Sessions | Episodes | Result |
|-------|----------|----------|--------|
| A2C-0 | 20260215_103816-103835 | 50 | 0%, critic EV ≈ 0, worse than vanilla REINFORCE |
| A2C-1 | 20260215_121727-121748 | 200 | 0.13%, actor improves via REINFORCE, 4 critic bugs found |
| A2C-2 | 20260215_135006-135025 | 200 | 0.63%, bugs fixed but EV worse (-0.295) |
| A2C-3 | 20260215_221154-221213 | 200 | 0.50%, sensory-only critic, EV worst yet (-0.620) |

Full QSNNReinforce A2C optimization history (4 rounds, 16 sessions): [008-appendix-qsnnreinforce-a2c-optimization.md](008-appendix-qsnnreinforce-a2c-optimization.md)

### Appendices

- QSNN foraging optimization history (17 rounds): [008-appendix-qsnn-foraging-optimization.md](008-appendix-qsnn-foraging-optimization.md)
- QSNN predator optimization history (16 rounds, 64 sessions): [008-appendix-qsnn-predator-optimization.md](008-appendix-qsnn-predator-optimization.md)
- QSNN-PPO optimization history (4 rounds, 16 sessions): [008-appendix-qsnnppo-optimization.md](008-appendix-qsnnppo-optimization.md)
- QSNNReinforce A2C optimization history (4 rounds, 16 sessions): [008-appendix-qsnnreinforce-a2c-optimization.md](008-appendix-qsnnreinforce-a2c-optimization.md)

### File Locations

- QRC implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrc.py`
- QRC configs: `configs/examples/qrc_*.yml`
- QSNN implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnnreinforce.py`
- QSNN tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnnreinforce.py`
- QSNN configs: `configs/examples/qsnnreinforce_*.yml`
- QSNN-PPO implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnnppo.py`
- QSNN-PPO tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnnppo.py`
- QSNN-PPO configs: `configs/examples/qsnnppo_*.yml`
