# 008: Quantum Brain Architecture Evaluation

**Status**: `complete` — All quantum architecture evaluations complete (QA-1 through QA-7). QA-7 (Quantum Plasticity) was the final experiment: classical baselines show zero backward forgetting (11/12 seeds BF=0.0, 1/12 BF=0.02), making the quantum anti-forgetting hypothesis untestable at current environment complexity. Quantum runs halted. Campaign concludes with 300+ sessions across 11+ architectures. Pivot to environment enrichment (Phases 1-3) confirmed. See [quantum-architectures.md Strategic Assessment](../../research/quantum-architectures.md#strategic-assessment-environment-complexity--quantum-advantage).

**Branch**: `feature/add-qsnn-brain`, `feature/add-quantum-reservoir-hybrid-brain`, `feat/add-qliflstm-brain`, `feat/add-qrh-qlstm-variant`, `feat/add-qef-brain`, `feat/add-qef-brain-eval`

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
| **QRH** | **Readout only** | **PPO on readout** | **86.8% (98% post-conv)** | **41.2%** | **Partial** |
| CRH (ablation) | Readout only | PPO on readout | N/A | 31.8% / 29.9%† | Partial (control) |
| **QLIF-LSTM Classical** | **LSTM cell + readout** | **Recurrent PPO (BPTT)** | **86.25%** | **74.7% (98% last-100)** | **Yes** |
| QLIF-LSTM Quantum | LSTM cell + readout | Recurrent PPO (BPTT) | 85.63% | 70.8% (94% last-100) | Yes (no Q advantage) |

\*CMA-ES is evolutionary, not gradient-based.
†CRH: 31.8% pursuit / 29.9% stationary (outperforms QRH on stationary).

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
QRH (quantum reservoir)        86.8%§      41.2%           ~10K      ✓ (forage+pursuit)
CRH (classical reservoir)      N/A         31.8% / 29.9%‖  ~10K      ✓ (control)
HybridQuantum                  91.0%       96.9%           ~10K      ✓✓
HybridClassical (ablation)     97.0%       96.3%           ~10K      ✓ (control)
QLIF-LSTM Classical            86.25%      74.7% (98%¶)    ~11K      ✓ (temporal)
QLIF-LSTM Quantum              85.63%      70.8% (94%¶)    ~11K      ✓ (no Q adv)
────────────────────────────────────────────────────────────────────────────
MLPPPOBrain (classical)        96.7%       71.6%†† / 94.5% ~42K      (ref)

† QSNN-PPO: PPO incompatible with surrogate gradients (policy_loss=0)
‡ QSNNReinforce A2C: critic never learned (EV -0.620)
* CMA-ES is evolutionary, not gradient-based
§ QRH: 98.0% post-convergence (exceeds MLPPPO); Domingo confound resolved
‖ CRH: pursuit 31.8% / stationary 29.9% (outperforms QRH on stationary)
¶ QLIF-LSTM: last-100 SR on 500-ep sessions; first temporal architecture
†† Unified sensory modules (apples-to-apples comparison)

KEY INSIGHT: HybridQuantum is the first quantum architecture to SURPASS
a classical baseline on a multi-objective task using gradient-based
online learning. It beats MLP PPO unified by +25.3 points on pursuit
predators with 4.3x fewer parameters. However, classical ablation
(HybridClassical) shows equivalent performance — the three-stage
curriculum and mode-gated fusion drive the result, not the QSNN.

QRH validates fixed-reservoir quantum computing for RL — the approach
that QRC failed to deliver. Random topology (not biological) is key.
Post-convergence QRH (98%) exceeds MLPPPO (96.7%). Domingo encoding
confound control confirms QRH's pursuit advantage is genuine quantum
dynamics, not trigonometric input encoding.

QLIF-LSTM is the first temporal (recurrent) architecture evaluated.
The classical ablation achieves the highest last-100 SR on pursuit
predators (98%) across all architectures, demonstrating genuine value
of within-episode memory. However, quantum QLIF gates provide no
measurable advantage over classical sigmoid — consistent across all
tasks and training durations. Stationary predators remain a weakness
(37% classical ceiling vs MLP PPO's 96.5%) due to the architecture's
limited spatial memory capacity.
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
27. **Random topology vastly outperforms biologically-inspired topology for quantum reservoirs**: QRH with C. elegans-inspired structured topology achieved 0-0.25% across 5 rounds (4,400 episodes). Random topology achieved 77% in the first round. MI analysis predicted this (random MI > structured, p=1.0). Biological circuits evolved for analog, continuous-time processing — not discrete quantum gates.
28. **Fixed quantum reservoirs can achieve competitive foraging**: QRH with random topology achieves 98% post-convergence on foraging (exceeding MLPPPO's 96.7%), validating the fixed-reservoir approach that QRC failed to deliver. The key differences: 10 qubits (vs QRC's 4), richer feature extraction (75 features via 3 channels), PPO training (vs REINFORCE), and random topology (vs random angles).
29. **QRH demonstrates genuine quantum advantage on pursuit predators**: Domingo encoding confound control (CRH-trig) confirmed that trigonometric input encoding hurts classical ESN (-18.8pp). QRH's +9.4pp pursuit advantage over CRH comes from quantum dynamics (interference, entanglement, phase information), not encoding.
30. **LR warmup reduces convergence variance without improving mean speed**: 5× improvement in convergence range (16-episode range vs 72) by preventing both destructive early updates and lucky fast convergences. An unexpected bonus: eliminates buffer guard activations entirely.
31. **Quantum vs classical reservoir advantage is task-dependent**: QRH excels on pursuit predators (4/4 converged, 13× lower variance) while CRH excels on stationary predators (3/4 converged, +6.3pp). The quantum reservoir provides better variance reduction for simpler tasks; the classical ESN provides more consistent dynamics for harder tasks requiring longer training.
32. **LayerNorm interaction depends on feature structure**: LayerNorm hurts structured topology (erases meaningful hierarchy between sensory and interneuron qubits) but helps random topology (genuinely normalizes heterogeneous feature scales without removing information).
33. **QLIF-LSTM temporal memory provides genuine value for pursuit predators**: The first temporal architecture achieves 98% classical last-100 SR on pursuit predators — the highest late-session performance across all architectures. Within-episode memory (h_t, c_t) enables temporal predator tracking that memoryless architectures cannot achieve.
34. **Quantum QLIF gates provide no measurable advantage over classical sigmoid**: Across all QLIF-LSTM tasks and training durations (foraging, pursuit, stationary, large environments), classical sigmoid gates match or exceed quantum QLIF gates. The hypothesis that quantum measurement noise provides beneficial exploration stochasticity is not supported.
35. **Temporal memory has task-specific limitations**: QLIF-LSTM excels at temporal evasion (pursuit predators: 98%) but struggles with spatial memory (stationary predators: 37% ceiling vs MLP PPO's 96.5%). The LSTM's 48-dim hidden state cannot implicitly encode 5 zone locations across a 100×100 grid from an 11×11 viewport.
36. **Entropy floor prevents late-session quantum destabilisation**: `entropy_coef_end=0.015` eliminates entropy rebound observed with 0.005. Validated across pursuit and large environment configs.
37. **Actor bottleneck: [features, h_t] outperforms h_t-only**: The actor needs direct access to current sensory signals alongside temporal context. Pure h_t bottleneck costs ~4pp on hard tasks.
38. **Apparent quantum regularisation disappears with sufficient training**: In 200-episode sessions, quantum shows lower variance than classical. With 500+ episodes, classical has time to fully converge and this advantage disappears.

______________________________________________________________________

## QRH Brain Evaluation (Quantum Reservoir Hybrid)

**Status**: Complete — 16 rounds, 96 sessions, ~30,000 episodes across foraging, pursuit predators, and stationary predators. 86.8% foraging success (98.0% post-convergence, Best: R8), 41.2% pursuit predator success (4/4 converged), 23.6% stationary predator success. Classical ablation (CRH) and Domingo encoding confound control completed.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│               QUANTUM RESERVOIR HYBRID (QRH)                                │
│         Fixed 10-Qubit Reservoir + PPO-Trained Classical Readout            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sensory Input (4-8 features)                                               │
│  ┌────────────────────────────┐                                             │
│  │ food_chemotaxis (2)        │                                             │
│  │ nociception (2)            │──┐                                          │
│  │ thermotaxis (1)            │  │  Input Encoding: RY(feature × π)         │
│  │ mechanosensation (2)       │  │                  RZ(feature × π)         │
│  └────────────────────────────┘  │  on each of 10 qubits                    │
│                                  │                                          │
│  ┌───────────────────────────────┴──────────────────────────────────┐       │
│  │  QUANTUM RESERVOIR (FIXED — not trained)                         │       │
│  │                                                                  │       │
│  │  10 qubits, random topology (CZ entangling + RY/RZ rotations)    │       │
│  │  3 reservoir layers, data re-uploading per layer                 │       │
│  │  Statevector simulation (no shot noise)                          │       │
│  │                                                                  │       │
│  │  Feature channels:                                               │       │
│  │    raw:      10 per-qubit Z expectations                         │       │
│  │    cos_sin:  20 (cos/sin of Z expectations)                      │       │
│  │    pairwise: 45 (C(10,2) ZZ correlations)                        │       │
│  │  Total: 75 features                                              │       │
│  └──────────────────────┬───────────────────────────────────────────┘       │
│                         │                                                   │
│  ┌──────────────────────┴───────────────────────────────────────────┐       │
│  │  CLASSICAL READOUT (PPO-TRAINED)                                 │       │
│  │                                                                  │       │
│  │  LayerNorm(75) → Actor: MLP(75→64→64→4)                          │       │
│  │                → Critic: MLP(75→64→64→1)                         │       │
│  │  Combined optimizer, PPO with GAE                                │       │
│  │  ~10K classical params total                                     │       │
│  └──────────────────────┬───────────────────────────────────────────┘       │
│                         │                                                   │
│                         ▼                                                   │
│              Softmax → Action: FWD / LEFT / RIGHT / STAY                    │
│                                                                             │
│  KEY DIFFERENCE FROM FAILED QRC:                                            │
│  - 10 qubits (vs QRC's 4) with richer feature extraction                   │
│  - 75 features via 3 channels (raw + cos_sin + pairwise ZZ)                │
│  - PPO training with full classical ML stack (LayerNorm, LR warmup,        │
│    min buffer guard, combined optimizer)                                    │
│  - Random topology (not structured C. elegans circuit)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MI Decision Gate (Pre-Training Evaluation)

Before any PPO training, a mutual information analysis compared structured (C. elegans) vs random reservoir topology. The hypothesis was that biologically-inspired structured topology would produce features with higher MI with optimal actions.

**Methodology**: 1000 synthetic observations with rule-based gradient-following oracle labels. MI estimated via `sklearn.feature_selection.mutual_info_classif` (k-nearest neighbors). Significance tested by row-swap permutation test (1000 permutations, p < 0.01). Script: `scripts/qrh_mi_analysis.py`. Full results: `artifacts/logbooks/008/qrh_mi_analysis/`.

| Run | Method | Mean MI | Total MI | Δ |
|-----|--------|---------|----------|---|
| 1 (baseline) | Structured (C. elegans) | 0.1326 | 4.773 | — |
| 1 (baseline) | Random topology | 0.1585 | 5.705 | — |
| 1 (baseline) | Classical MLP (64 hidden) | 0.3809 | 24.376 | — |
| 2 (per-qubit encoding, CRY/CRZ) | Structured | 0.2025 | 7.290 | +53% |
| 2 (per-qubit encoding, CRY/CRZ) | Random | 0.2445 | 8.801 | +54% |

**Decision: NO-GO for structured topology** (both runs). Random > structured with p=1.0 (wrong direction). Biological correctness interventions improved both topologies equally — random's advantage is fundamental, not fixable by encoding improvements.

**Root cause — bilateral symmetry degeneracy**: The structured topology's left-right mirror connectivity creates feature pairs with identical MI values (ASEL=ASER, AIY_L=AIY_R, etc.), collapsing ~50% of effective feature diversity. Random topology produces all-distinct MI values per feature. This MI result correctly predicted all subsequent training outcomes: R1-R5 structured achieved 0-0.25%, R6 random achieved 77% on first attempt, R16 structured achieved 0.0% across 12,000 episodes.

For detailed per-feature analysis and methodology, see [qrh-optimization.md Phase 1](supporting/008/qrh-optimization.md#phase-1-mi-decision-gate).

### Configuration (Best — Foraging R8)

```yaml
brain:
  name: qrh
  config:
    num_reservoir_qubits: 10
    reservoir_depth: 3
    use_random_topology: true          # Critical: random >> structured
    feature_channels: [raw, cos_sin, pairwise]  # 75 features
    actor_lr: 0.0005
    critic_lr: 0.0005
    ppo_epochs: 6                      # Reduced from 10 (less overfitting)
    ppo_buffer_size: 256
    ppo_minibatches: 2
    entropy_coeff: 0.02
    max_grad_norm: 0.5
    gae_lambda: 0.95
    lr_warmup_episodes: 30             # Ramp from 10% to full LR
    lr_warmup_start: 0.00005
```

### Results Summary

#### Foraging (Best: R8, 2000 episodes, 4 sessions)

| Session | Success | Post-Conv | Conv. Episode | Dist. Efficiency |
|---------|---------|-----------|---------------|------------------|
| 151153 | 88.2% | 96.8% | 70 | 0.459 |
| 151158 | 87.2% | 98.8% | 77 | 0.462 |
| 151204 | 83.8% | 97.4% | 76 | 0.450 |
| 151209 | 88.0% | 99.1% | 61 | 0.481 |
| **Mean** | **86.8%** | **98.0%** | **71** | **0.463** |

Post-convergence QRH (98.0%) **exceeds** MLPPPO baseline (96.7%).

#### Pursuit Predators (Best: R9, 1000 episodes, 4 sessions)

| Session | Overall SR | Last-500 SR | Last-100 SR | Converged |
|---------|-----------|-------------|-------------|-----------|
| 044427 | 42.2% | 52.2% | 49.0% | Yes |
| 044447 | 40.3% | 54.4% | 64.0% | Yes |
| 044451 | 40.7% | 51.4% | 49.0% | Yes |
| 044501 | 41.7% | 55.2% | 64.0% | Yes |
| **Mean** | **41.2%** | **53.3%** | **56.5%** | **4/4** |

QRH achieves **4/4 convergence** on pursuit predators with 13× lower variance than CRH.

#### Stationary Predators (Best: R14, 3000 episodes, 4 sessions)

| Session | Overall SR | Last-500 SR | Last-100 SR | Converged |
|---------|-----------|-------------|-------------|-----------|
| 084742 | 17.6% | 20.0% | 20.0% | No |
| 084751 | 23.7% | 34.2% | 36.0% | No |
| 084755 | 20.0% | 20.6% | 24.0% | No |
| 084759 | 33.1% | 49.2% | 47.0% | Yes |
| **Mean** | **23.6%** | **31.1%** | **31.8%** | **1/4** |

High seed sensitivity on stationary predators. CRH outperforms QRH here (+6.3pp, 3/4 converged).

### Key Experiments (Optimization Journey)

| Round | Task | Key Change | Success | Key Finding |
|-------|------|-----------|---------|-------------|
| R1-R5 | Foraging | Structured topology, various PPO tuning | 0-0.25% | Structured topology is an information bottleneck |
| **R6** | Foraging | **Random topology** | **77%** | Random topology transforms QRH from 0% to 77% |
| R7 | Foraging | +Min buffer guard, 500ep | 87.8% | Buffer guard eliminates late regression; 98.8% post-conv |
| **R8** | **Foraging** | **+LR warmup, PPO epochs 6** | **86.8%** | **5× less convergence variance (16ep range vs 72)** |
| R9 | Pursuit | Best pursuit config | **41.2%** | 4/4 converged, 13× lower variance than CRH |
| R10-R14 | Stationary | Extended training, LR/entropy schedules | 19-30% | CRH beats QRH on stationary; seed sensitivity remains |
| **R15** | **Ablation** | **Domingo encoding confound (CRH-trig)** | **13%** | **Trig encoding hurts CRH; QRH advantage is genuine** |
| R16 | Ablation | Structured topology on stationary | **0.0%** | Structured topology definitively falsified |

### Key Findings

1. **Random topology is essential**: Structured C. elegans-inspired topology achieved 0-0.25% across 5 rounds (4,400 episodes). Random topology achieved 77% in the first round. MI analysis predicted this: random MI > structured MI (p=1.0 wrong direction). The biological circuit was evolved for analog, continuous-time signal processing — not discrete quantum gates.

2. **Post-convergence, QRH matches or exceeds MLPPPO**: 98.0-98.8% on foraging (vs MLPPPO's 96.7%). The quantum reservoir features provide equal or better information for action selection once the readout is trained. The pre-convergence gap (~70 episodes of low performance) is the only weakness.

3. **QRH's pursuit advantage is genuine quantum dynamics**: The Domingo encoding confound control (CRH-trig) showed that trigonometric encoding *hurts* classical ESN (-18.8pp on pursuit). QRH's +9.4pp advantage over CRH on pursuit comes from quantum interference and entanglement, not encoding.

4. **QRH excels at variance reduction on pursuit**: 4/4 convergence with 1.4pp variance (vs CRH's 1/4 convergence, 18.6pp variance). The quantum reservoir provides a more consistent feature space for pursuit predator learning.

5. **CRH wins on stationary predators**: +6.3pp overall, 3/4 converged (vs QRH's 1/4). The ESN's fixed spectral_radius provides consistent dynamics across seeds, while QRH's random topology creates high seed sensitivity on harder tasks.

6. **LR warmup dramatically reduces convergence variance**: 5× improvement (16-episode range vs 72-episode range) without changing mean convergence speed. Prevents both "lucky fast" and "unlucky slow" convergence patterns.

### Conclusion

QRH is the **second quantum architecture to achieve competitive performance** with classical baselines. It validates the fixed-reservoir approach that QRC failed to deliver, with the key insight that **random topology vastly outperforms biologically-inspired topology**. The architecture demonstrates genuine quantum advantage on pursuit predators (confirmed by Domingo ablation) but not on stationary predators where the classical reservoir is more consistent.

**Where QRH excels**: foraging (98% post-conv), pursuit predator convergence reliability (4/4, lowest variance), computational simplicity (no quantum gradient training).

**Where QRH falls short**: stationary predator seed sensitivity, pre-convergence exploration period (~70 episodes), task generalization across predator types.

### File Locations

- QRH implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrh.py`
- QRH tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qrh.py`
- QRH configs: `configs/examples/qrh_*.yml`
- CRH implementation: `packages/quantum-nematode/quantumnematode/brain/arch/crh.py`
- CRH tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_crh.py`
- CRH configs: `configs/examples/crh_*.yml`

Full optimization history (16 rounds, 96 sessions): [qrh-optimization.md](supporting/008/qrh-optimization.md)

______________________________________________________________________

## CRH Brain Evaluation (Classical Reservoir Hybrid — Ablation Control)

**Status**: Complete — classical ablation of QRH. 10-neuron Echo State Network reservoir with same PPO readout architecture. Outperforms QRH on stationary predators, underperforms on pursuit.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│              CLASSICAL RESERVOIR HYBRID (CRH)                               │
│       10-Neuron ESN Reservoir + PPO-Trained Classical Readout               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sensory Input (4-8 features)                                               │
│       │                                                                     │
│  ┌────┴─────────────────────────────────────────────────────────────┐       │
│  │  ECHO STATE NETWORK RESERVOIR (FIXED — not trained)              │       │
│  │                                                                  │       │
│  │  W_in: random input weights (10 × input_dim)                     │       │
│  │  W_res: random recurrent weights, spectral_radius = 0.9          │       │
│  │  Activation: tanh                                                │       │
│  │  reservoir_depth: 3 (layers of state update)                     │       │
│  │                                                                  │       │
│  │  Feature channels (same as QRH):                                 │       │
│  │    raw:      10 neuron activations                               │       │
│  │    cos_sin:  20 (cos/sin of activations)                         │       │
│  │    pairwise: 45 (C(10,2) pairwise products)                      │       │
│  │  Total: 75 features                                              │       │
│  └──────────────────────┬───────────────────────────────────────────┘       │
│                         │                                                   │
│  ┌──────────────────────┴───────────────────────────────────────────┐       │
│  │  CLASSICAL READOUT (PPO-TRAINED) — identical to QRH              │       │
│  │  LayerNorm(75) → Actor + Critic MLP, ~10K params                 │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Results Summary

#### Pursuit Predators (R9 config, 1000 episodes, 4 sessions)

| Metric | QRH (R9) | CRH (R9) | Winner |
|--------|----------|----------|--------|
| Overall SR (avg) | **41.2%** | 31.8% | **QRH (+9.4pp)** |
| Sessions converged | **4/4** | 1/4 | **QRH** |
| Variance (SR range) | **1.4pp** | 18.6pp | **QRH (13× lower)** |
| Last-100 SR (avg) | **56.5%** | ~42% | **QRH** |

#### Stationary Predators (R14 config, 3000 episodes, 4 sessions)

| Metric | QRH (R14) | CRH (R14) | Winner |
|--------|-----------|-----------|--------|
| Overall SR (avg) | 23.6% | **29.9%** | **CRH (+6.3pp)** |
| Sessions converged (≥40% last-500) | 1/4 | **3/4** | **CRH** |
| Variance (SR range) | 15.5pp | **5.2pp** | **CRH (3× lower)** |
| Last-500 SR (avg) | 31.1% | **43.5%** | **CRH (+12.4pp)** |

### Domingo Encoding Confound Control (R15)

Added `input_encoding: "trig"` to CRH — applies sin(f×π), cos(f×π) to inputs before W_in projection, matching QRH's trigonometric gate encoding.

| Architecture | Pursuit SR | Stationary SR | Converged |
|---|---|---|---|
| **QRH** | **41.2%** | 23.6% | **4/4** pursuit, 1/4 stationary |
| CRH-linear | 31.8% | **29.9%** | 1/4 pursuit, **3/4** stationary |
| CRH-trig | 13.0% | 17.7% | 0/4 both |

**Trig encoding hurts CRH** (-18.8pp pursuit, -12.2pp stationary). The triple nonlinearity (sin/cos → tanh → tanh) compresses dynamic range. QRH's trig encoding is native to quantum gate operations (Bloch sphere rotations) — no double nonlinearity. **Domingo confound resolved: QRH's advantage is genuine quantum dynamics, not encoding.**

### Key Findings

1. **CRH validates the QRH architecture**: A classical ESN with identical feature channels and PPO readout achieves competitive performance, proving the reservoir+readout approach works regardless of reservoir type.

2. **Task-dependent quantum vs classical advantage**: QRH wins on pursuit (where variance reduction matters) while CRH wins on stationary (where consistency matters). Neither dominates across all tasks.

3. **ESN consistency advantage**: CRH's fixed spectral_radius=0.9 provides consistent reservoir dynamics regardless of seed. QRH's random quantum topology creates high seed sensitivity, which hurts on harder tasks requiring longer training.

### File Locations

- CRH implementation: `packages/quantum-nematode/quantumnematode/brain/arch/crh.py`
- CRH tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_crh.py`
- CRH configs: `configs/examples/crh_*.yml`

______________________________________________________________________

## QLIF-LSTM Brain Evaluation (H.4 — Quantum LIF Long Short-Term Memory)

**Status**: Complete — 12 rounds, ~66 sessions, ~36,000 episodes across foraging, pursuit predators, and stationary predators (classical + quantum). First temporal architecture in the codebase. Classical last-100 SR: 98% pursuit, 82% large pursuit, 37% stationary. Quantum gates provide no measurable advantage.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│               QLIF-LSTM (H.4) — Quantum Temporal Brain                      │
│      Custom LSTM Cell with QLIF Quantum Gates + Recurrent PPO               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sensory Input (2-9 features)                                               │
│  ┌────────────────────────────┐                                             │
│  │ food_chemotaxis (2)        │                                             │
│  │ nociception (2)            │  Via extract_classical_features()           │
│  │ mechanosensation (2)       │  or preprocess() for legacy 2-input mode    │
│  │ thermotaxis (3)            │                                             │
│  └────────────┬───────────────┘                                             │
│               │                                                             │
│  ┌────────────┴───────────────────────────────────────────────────┐         │
│  │  QLIF-LSTM CELL (core innovation)                              │         │
│  │                                                                │         │
│  │  z = [x_t, h_{t-1}]  (concatenation)                           │         │
│  │                                                                │         │
│  │  Forget gate: f_t = QLIF(W_f·z / √fan_in)  ← quantum P(|1⟩)    │         │
│  │  Input gate:  i_t = QLIF(W_i·z / √fan_in)  ← quantum P(|1⟩)    │         │
│  │  Cell cand:   ĉ_t = tanh(W_c·z)            ← classical         │         │
│  │  Output gate: o_t = σ(W_o·z)               ← classical         │         │
│  │                                                                │         │
│  │  c_t = f_t * c_{t-1} + i_t * ĉ_t                               │         │
│  │  h_t = o_t * tanh(c_t)                                         │         │
│  │                                                                │         │
│  │  QLIF circuit: |0⟩ → RY(θ + tanh(scaled_input)·π) → RX(leak)   │         │
│  │  Surrogate gradient: sigmoid on RY angle for backward pass     │         │
│  │  Batched execution: 1 circuit per neuron, single backend.run() │         │
│  └────────────┬───────────────────────────────────────────────────┘         │
│               │                                                             │
│  ┌────────────┴───────────────────────────────────────────────────┐         │
│  │  Actor: Linear([features, h_t] → 4 actions) + Categorical      │         │
│  │  Critic: MLP([features, h_t.detach()] → V(s))                  │         │
│  │  Training: Recurrent PPO with chunk-based truncated BPTT       │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  KEY INNOVATION: LSTM gates driven by quantum measurement P(|1⟩)            │
│  instead of classical sigmoid. Temporal memory via h_t/c_t persists         │
│  within episodes, resets between episodes.                                  │
│                                                                             │
│  Classical ablation: use_quantum_gates=false → sigmoid for gates            │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key design choices:

- **QLIF quantum gates**: Forget and input gates use quantum measurement P(|1⟩) from single-qubit QLIF circuits, with surrogate gradients for backpropagation
- **Fan-in scaling**: Linear projections scaled by 1/√fan_in to keep tanh in responsive regime
- **Recurrent PPO**: Chunk-based truncated BPTT with sequential chunks from rollout buffer
- **Episode-scoped memory**: h_t/c_t persist within episodes, reset via `prepare_episode()`
- **Classical ablation**: `use_quantum_gates: false` replaces QLIF with torch.sigmoid()

### Configuration (Best — Pursuit Predators R4 + R4b entropy floor)

```yaml
brain:
  name: qliflstm
  config:
    lstm_hidden_dim: 32
    shots: 1024
    membrane_tau: 0.9
    gamma: 0.99
    gae_lambda: 0.98
    clip_epsilon: 0.2
    entropy_coef: 0.05
    entropy_coef_end: 0.015        # Higher floor prevents late-session destabilisation
    entropy_decay_episodes: 400
    value_loss_coef: 0.5
    num_epochs: 2
    rollout_buffer_size: 512
    max_grad_norm: 0.5
    actor_lr: 0.003
    critic_lr: 0.001
    bptt_chunk_length: 16
    use_quantum_gates: true
    sensory_modules: [food_chemotaxis, nociception]
```

### Results Summary

#### Foraging (Best: R1 classical / R2 quantum, 200 episodes, 4+4 sessions)

| Metric | Classical (R1) | Quantum (R2) | Winner |
|--------|---------------|-------------|--------|
| Success rate (avg) | **86.25%** | 85.63% | Equivalent |
| Post-convergence | 99.85% | 99.85% | Tie |
| Convergence run | 31.25 | 31.0 | Tie |
| Distance efficiency | **0.403** | 0.372 | Classical |

Both variants achieve near-perfect post-convergence performance. Quantum maintains higher entropy (~0.94-1.07 vs ~0.65-0.87 final) due to measurement noise. Task too simple to differentiate.

#### Pursuit Predators (Best: R4, 500 episodes, 4+4 sessions)

| Metric | Classical (R4) | Quantum (R4) | Winner |
|--------|---------------|-------------|--------|
| Success rate (avg) | **74.70%** | 70.80% | Classical (+3.9pp) |
| Post-convergence | 92.40% | 90.78% | Classical |
| **Last 100 SR** | **98%** | **94%** | **Classical (+4pp)** |
| Per-encounter evasion | 83.45% | 84.00% | Equivalent |
| SR std | **2.33%** | 4.70% | Classical |

Strong performance from both variants. The QLIF-LSTM architecture demonstrably learns temporal predator evasion — 98% last-100 classical is the best result for this task among temporal architectures.

#### Thermotaxis + Pursuit Predators Large (Best: R5, 500 episodes, 4+4 sessions)

| Metric | Classical (R5) | Quantum (R5) | Winner |
|--------|---------------|-------------|--------|
| Success rate (avg) | **60.10%** | 45.35% | Classical (+14.8pp) |
| Last 100 SR | 82.00% | 81.50% | Equivalent |
| Convergence run | **182** | 256 | Classical (74 runs faster) |
| SR std | 8.26% | **2.30%** | Quantum (3.6× lower) |
| Evasion rate | **88.60%** | 84.68% | Classical |

Classical converges faster, but late-session performance is equivalent. Quantum shows remarkably consistent convergence (2.30% std across 4 sessions). All 8 sessions learn the forage > evade > thermoregulate priority hierarchy.

#### Stationary Predators Large (Best: R10 classical + quantum comparison, 500-1000 eps)

| Metric | Classical (R10, 1000ep) | Quantum (500ep) | MLP PPO ref |
|--------|------------------------|-----------------|-------------|
| Last-100 SR | 36.8% | 30.5% | **96.5%** |
| Peak R50 | 46.5% | 44.0% | — |
| Health deaths | 70.8% | 77.6% | — |

Both variants struggle on stationary predators. The LSTM temporal memory doesn't help with spatial zone avoidance — the 9-dim sensory input and 11×11 viewport are insufficient for encoding 5 zone locations across a 100×100 grid. MLP PPO's reactive gradient sensing is far more effective.

### Key Findings

1. **First temporal architecture in the codebase** — introduces within-episode memory (h_t, c_t), enabling temporal predator evasion that memoryless architectures cannot achieve.

2. **Quantum gates provide no measurable advantage**: Across all tasks — foraging (≈0), pursuit small (-4pp), pursuit large (-15pp overall but ≈0 late), stationary (-2pp) — classical sigmoid gates match or exceed quantum QLIF gates.

3. **Task-specific strengths and limitations**: Strong on pursuit predators (temporal evasion, 98% classical late), weak on stationary predators (spatial memory, 37% ceiling). The architecture is well-suited for temporal tasks but not spatial tasks.

4. **Entropy floor is critical for quantum stability**: `entropy_coef_end=0.015` prevents late-session entropy rebound and policy destabilisation. Validated in R4b.

5. **Actor [features, h_t] fix**: The actor needs direct sensory access alongside LSTM context. Pure h_t bottleneck limits performance by ~4pp on hard tasks.

6. **Quantum variance reversal**: In short sessions (200ep), quantum shows lower variance than classical (regularisation benefit). This advantage disappears with sufficient training (500+ep) as classical converges fully.

### Comparison with Other Architectures (Pursuit Predators)

| Architecture | Best Post-Conv | Params | Training | Temporal? |
|--------------|---------------|--------|----------|-----------|
| **HybridQuantum** | **96.9%** | ~10K | 3-stage curriculum | No |
| HybridClassical | 96.3% | ~10K | 3-stage curriculum | No |
| **QLIF-LSTM Classical** | **98% last-100** | ~11K | Recurrent PPO | **Yes** |
| QLIF-LSTM Quantum | 94% last-100 | ~11K | Recurrent PPO | **Yes** |
| MLP PPO (unified) | 71.6% | ~42K | PPO | No |
| QRH | 56.5% last-100 | ~10K | PPO | No |

QLIF-LSTM classical achieves the highest late-session performance on pursuit predators across all architectures, demonstrating the value of temporal memory for predator evasion. However, HybridQuantum achieves higher overall success rate (96.9% vs 74.7%) due to much faster convergence from its pre-trained curriculum approach.

### File Locations

- QLIF-LSTM implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qliflstm.py`
- QLIF-LSTM tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qliflstm.py`
- QLIF-LSTM configs: `configs/examples/qliflstm_*.yml`

Full optimization history (12 rounds, ~66 sessions): [qliflstm-optimization.md](supporting/008/qliflstm-optimization.md)

______________________________________________________________________

## QRH-QLSTM / CRH-QLSTM Brain Evaluation

**Date**: 2026-03-10 – 2026-03-12

**Scope**: 15 rounds, 54 sessions, 16,500 episodes

**Goal**: Test whether composing reservoir features (QRH quantum or CRH classical) with QLIF-LSTM temporal readout improves over either standalone architecture (Stage 4d).

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│              RESERVOIR-LSTM COMPOSITION (QRH-QLSTM / CRH-QLSTM)             │
│       Fixed Reservoir Feature Extractor + QLIF-LSTM Temporal Readout        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sensory Input (4-8 features)                                               │
│       │                                                                     │
│  ┌────┴─────────────────────────────────────────────────────────────┐       │
│  │  RESERVOIR (FIXED — not trained)                                 │       │
│  │                                                                  │       │
│  │  QRH variant: N-qubit quantum reservoir (random topology)         │       │
│  │    Features: 3N + N(N-1)/2  (N=8 → 52-D, N=10 → 75-D)           │       │
│  │  CRH variant: 10-neuron classical ESN (spectral_radius=0.9)      │       │
│  │    Features: raw(10) + cos_sin(20) + pairwise(45) = 75-D         │       │
│  └──────────────────────┬───────────────────────────────────────────┘       │
│                         │                                                   │
│  ┌──────────────────────┴───────────────────────────────────────────┐       │
│  │  QLIF-LSTM TEMPORAL READOUT (PPO-TRAINED)                        │       │
│  │                                                                  │       │
│  │  z = [reservoir_features, h_{t-1}]                               │       │
│  │  Forget gate: f_t = QLIF(W_f·z) or sigmoid(W_f·z)                │       │
│  │  Input gate:  i_t = QLIF(W_i·z) or sigmoid(W_i·z)                │       │
│  │  Cell cand:   ĉ_t = tanh(W_c·z)                                  │       │
│  │  Output gate: o_t = σ(W_o·z)                                     │       │
│  │  c_t = f_t * c_{t-1} + i_t * ĉ_t                                 │       │
│  │  h_t = o_t * tanh(c_t)                                           │       │
│  │                                                                  │       │
│  │  Actor: Linear([features, h_t] → 4 actions) + Categorical        │       │
│  │  Critic: MLP([features, h_t.detach()] → V(s))                    │       │
│  │  Training: Recurrent PPO with chunk-based truncated BPTT         │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  VARIANTS:                                                                  │
│    QRH-QLSTM: quantum reservoir + quantum QLIF gates (full quantum)         │
│    CRH-QLSTM: classical ESN reservoir + quantum QLIF gates                  │
│    QRH-LSTM:  quantum reservoir + classical sigmoid gates (ablation)        │
│    CRH-LSTM:  classical reservoir + classical sigmoid gates (ablation)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key design choices:

- **Reservoir-LSTM composition**: Fixed reservoir provides rich features (QRH: 3N+N(N-1)/2, e.g. 52-D for N=8; CRH: 75-D for N=10); QLIF-LSTM adds within-episode temporal memory
- **Shared base class**: `ReservoirLSTMBase` abstracts the LSTM readout + recurrent PPO; subclasses provide only reservoir creation and feature dim computation
- **Classical ablation**: `use_quantum_gates: false` replaces QLIF circuits with torch.sigmoid() for both forget and input gates

### Configuration (Best — CRH-QLSTM Pursuit Predators)

```yaml
brain:
  name: crhqlstm
  config:
    num_reservoir_neurons: 10
    reservoir_depth: 3
    spectral_radius: 0.9
    feature_channels: [raw, cos_sin, pairwise]
    lstm_hidden_dim: 64
    shots: 1024
    membrane_tau: 0.9
    gamma: 0.99
    gae_lambda: 0.98
    clip_epsilon: 0.2
    entropy_coef: 0.02
    entropy_coef_end: 0.008
    entropy_decay_episodes: 100
    value_loss_coef: 0.5
    num_epochs: 2
    rollout_buffer_size: 1024
    max_grad_norm: 0.5
    actor_lr: 0.0005
    critic_lr: 0.0005
    bptt_chunk_length: 32
    use_quantum_gates: true
    sensory_modules: [food_chemotaxis, nociception]
```

### Results Summary

#### Foraging (Phase 1-2, 200 episodes, 4 sessions each variant)

| Metric | QRH-QLSTM Classical | QRH-QLSTM Quantum | CRH-QLSTM Classical | CRH-QLSTM Quantum |
|--------|---------------------|-------------------|---------------------|-------------------|
| Success rate | 62.9% | 66.6% | **86.3%** | 85.1% |
| Post-convergence | 96.6% | 97.9% | **99.3%** | 99.7% |
| Convergence ep | 82.8 | 82.8 | **30.8** | 31.0 |

CRH converges 2.7x faster than QRH. Quantum vs classical gates make negligible difference. Both pass Stage 4a.

#### Pursuit Predators Small (Phase 3, 200 episodes, 4 sessions each)

| Metric | CRH-QLSTM Quantum | CRH-QLSTM Classical | QRH-QLSTM Quantum | QLIF-LSTM Classical |
|--------|-------------------|---------------------|-------------------|---------------------|
| Success rate | **85.4%** | 82.2% | 15.2% | 74.7% |
| Post-convergence | **95.8%** | 92.7% | — | 92.4% |
| Convergence ep | **30.3** | 37.5 | — | 146.3 |
| Evasion rate | **85.0%** | — | — | 75.4% |

CRH-QLSTM is the best reservoir-LSTM variant: +10.7pp over standalone QLIF-LSTM, 4.8x faster convergence. QRH-QLSTM fails completely on multi-objective tasks.

#### Thermotaxis + Pursuit Predators Large (Phase 4, 500 episodes, 4 sessions)

| Model | SR | Conv | Steps/Food |
|-------|----|------|------------|
| QLIF-LSTM Classical | **60.1%** | **198** | **25.3** |
| QRH standalone (MLP) | 41.3% | 793 | — |
| CRH-QLSTM Classical v2 | 38.8% | — | 50.1 |
| QRH-LSTM Classical | 16.4% | — | — |

CRH-QLSTM does not scale to large grids — CRH-QLSTM Classical v2 has 2x worse path efficiency than QLIF-LSTM Classical (50.1 vs 25.3 steps/food). Reservoir feature expansion obscures gradient signals needed for long-range navigation.

#### Thermotaxis + Stationary Predators Large (Phase 4b + Stage 4d, 500 episodes, 4 sessions)

| Model | SR |
|-------|----|
| QLIF-LSTM Classical | **24.0%** |
| CRH standalone | 23.4% |
| QRH standalone (MLP) | 14.9% |
| CRH-QLSTM Classical | 14.0% |
| QRH-LSTM Classical | **10.8%** |

Stage 4d hypothesis **REJECTED**: QRH-LSTM is -4.1pp worse than QRH-MLP on stationary predators. LSTM temporal readout does NOT resolve QRH's multi-objective weakness.

### Key Findings

1. **CRH-QLSTM excels on small pursuit predators** (85.4%) but fails to scale to large grids
2. **QRH-QLSTM fails on all multi-objective tasks** — quantum reservoir noise becomes catastrophic with added objectives
3. **LSTM readout HURTS QRH performance** — worse than QRH standalone MLP on every large-grid test (-24.9pp pursuit, -4.1pp stationary)
4. **Quantum QLIF gates provide ~3pp advantage** on CRH pursuit predators, but not worth 170x speed cost
5. **Path efficiency gap is architectural, not hyperparameter** — confirmed by hyperparameter alignment experiment
6. **Reservoir-LSTM composition does not improve over simpler architectures** at scale

### Cross-Architecture Comparison (All Architectures, Pursuit Predators)

| Model | Small (20x20) | Large (100x100) | Architecture |
|-------|--------------|-----------------|--------------|
| CRH-QLSTM Quantum | **85.4%** | — | Reservoir-LSTM |
| CRH-QLSTM Classical | 82.2% | 38.8% | Reservoir-LSTM |
| HybridQuantum Stage 3 | 96.9% | — | Curriculum fusion |
| QLIF-LSTM Classical | 74.7% | **60.1%** | Standalone LSTM |
| QRH standalone (MLP) | — | 41.3% | Reservoir-MLP |
| CRH standalone | — | — | Reservoir-MLP |
| QRH-LSTM Classical | 17.0% | 16.4% | Reservoir-LSTM |
| QRH-QLSTM Quantum | 15.2% | — | Reservoir-LSTM |

### File Locations

- QRH-QLSTM implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrhqlstm.py`
- CRH-QLSTM implementation: `packages/quantum-nematode/quantumnematode/brain/arch/crhqlstm.py`
- Reservoir-LSTM base: `packages/quantum-nematode/quantumnematode/brain/arch/_reservoir_lstm_base.py`
- Tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qrhqlstm.py`, `test_crhqlstm.py`
- Configs: `configs/examples/qrhqlstm_*.yml`, `configs/examples/crhqlstm_*.yml`

Full optimization history (15 rounds, 54 sessions): [qrhqlstm-optimization.md](supporting/008/qrhqlstm-optimization.md)

______________________________________________________________________

## QEF Brain Evaluation (Quantum Entangled Features)

### Architecture

```text
Sensory Input (7 features)
    │
    ├─── Raw Features (7-dim) ────────────────────────────┐
    │                                                     │
    ▼                                                     │
┌──────────────────────────────────────────────┐          │
│  8-Qubit PQC (depth 2, ring entanglement)    │          │
│  H → [RY encoding + CRY/CRZ ring]^2 → |ψ⟩    │          │
│  → Z + ZZ correlations (31-52 features)      │          │
└──────────────────────────────────────────────┘          │
    │                                                     │
    ▼                                                     │
┌──────────────────────────────────────────────┐          │
│  Feature Gating (static/context/mixed)       │          │
│  sigmoid(w) * quantum_features               │          │
└──────────────────────────────────────────────┘          │
    │                                                     │
    ├─── Gated Quantum Features ──────────────────────────┤
    │                                                     │
    ▼                                                     ▼
┌──────────────────────────────────────────────────────────┐
│  PPO Actor-Critic Readout (64×2 MLP)                     │
│  [raw | quantum | poly] → LayerNorm → Actor/Critic       │
└──────────────────────────────────────────────────────────┘
```

QEF extends the QRH paradigm by replacing the random reservoir with purposeful cross-modal entanglement. Key innovations developed during evaluation: hybrid input (raw + quantum features), learnable feature gating, and cross-modal ZZ curation.

### Configuration (Task-Specific)

**Pursuit predators (large)**: hybrid + context gating, 59-dim input (7 raw + 52 quantum), bigbuf 1024
**Stationary predators (large)**: hybrid + curated cross-modal ZZ + static gating, 38-dim input (7 raw + 31 quantum), bigbuf 1024
**Pursuit predators (small)**: hybrid + context gating, 56-dim input (4 raw + 52 quantum)

### Results Summary (12-Seed Validation)

| Task | QEF L100 | MLP PPO L100 | A3 Poly L100 | QEF vs MLP | QEF vs Poly | Status |
|------|---------|-------------|-------------|-----------|------------|--------|
| Stationary (1000ep) | 90.8% ± 1.2% | 89.6% ± 0.8% | **93.8% ± 1.0%** | +1.2pp (ns) | -3.1pp (ns) | Competitive |
| Pursuit large (1000ep) | 93.0% ± 1.3% | **96.0% ± 0.5%** | 94.8% ± 1.1% | -3.0pp (\*) | -1.8pp (ns) | Behind MLP |
| Small PP (500ep) | 98.2% ± 0.6% | **98.6% ± 0.5%** | 97.0% ± 0.6% | -0.5pp (ns) | +1.2pp (ns) | Competitive |

Paired t-tests with 12 seeds. (ns) = not significant at p\<0.05, (\*) = significant at p\<0.05.
Additional: A3 Poly vs MLP PPO on stationary: +4.2pp, p=0.001 (\*\*). Full per-seed data in [appendix](supporting/008/qef-optimization.md#12-seed-statistical-validation).

### Key Findings

1. **No statistically significant quantum advantage on any task.** QEF does not significantly outperform either classical baseline at p\<0.05.
2. **QEF significantly trails MLP PPO on pursuit** (-3.0pp, p=0.043). MLP PPO's simple 7-feature input with low variance (σ=1.6) outperforms QEF's 59-dim hybrid input with higher variance (σ=4.5).
3. **A3 polynomial features are strongest on stationary** (93.8%), significantly beating MLP PPO (p=0.001). Classical polynomial features are most effective on the weak-signal task.
4. **Entanglement is essential** — 0% success without it (separable ablation).
5. **Gating interacts differently with quantum vs classical features**: On stationary, gating helps quantum (+7.7pp) but hurts classical (-4.0pp). On pursuit/small PP, gating helps both equally.
6. **Feature curation helps on weak-signal tasks**: Cross-modal ZZ only (dropping intra-modal pairs and cos/sin) improved stationary from 91.0% to 92.2%.

### Optimization Journey (24 Phases)

| Phase | Key Change | Stationary L100 | Pursuit L100 | Finding |
|-------|-----------|-----------------|-------------|---------|
| 1-4 | MI analysis, topology/gate/feature modes | — | — | CRY/CRZ gates essential; xyz features better |
| 5 | LR decay + entropy annealing | — | 96.0% (small PP) | Eliminates catastrophic forgetting |
| 5b | Ring topology (vs modality-paired) | — | +12.5pp | Ring wins decisively |
| 6 | Compact readout 64×2 | — | +5.0pp | Smaller network → higher ceiling |
| 8 | Stationary evaluation | 69.7% | — | Weak nociception signal is the bottleneck |
| 8c | Entropy fix + bigbuf | 75.5% | — | +5.8pp on stationary |
| 9 | Hybrid input (raw + quantum) | 82.5% | 88.2% | Key innovation — fast convergence |
| 10 | Feature gating (static) | 90.2% | 90.5% | +7.2pp stationary, +2.3pp pursuit |
| 13 | Classical ablation | — | — | A3 poly = 88.2% stationary (buffer 512) |
| 16 | Context gating | — | 95.7% | +5.2pp pursuit over static gating |
| 19 | Mixed gating | 91.0% | — | +0.8pp stationary over static |
| 21 | Fair comparison (matched buffer) | — | — | MLP PPO stationary jumps to 90.2% |
| 23 | Curated cross-modal ZZ | 92.2% | — | +1.2pp from feature curation |
| Final | 12-seed validation | 90.8% | 93.0% | No significant quantum advantage |

### Conclusion

QEF is **quantum-competitive but not quantum-advantageous**. It matches classical approaches within ~1-3pp across all tested environments using quantum-derived features but does not provide a statistically significant performance improvement. The architecture's value lies in demonstrating that entangled PQC features can serve as viable alternatives to classical features, and in the mechanistic finding that learned gating interacts fundamentally differently with quantum vs classical feature structures.

The -3.1pp gap on stationary (vs A3 polynomial, p=0.081) appears structural — classical polynomial features have a self-silencing property on weak signals (products go to zero when either input is zero) that quantum ZZ correlations lack. The 8-qubit depth-2 circuit is also classically simulatable, limiting the theoretical basis for quantum advantage in this regime.

**Recommended future directions**: QA-6 (weak-measurement feedback for temporal memory), higher-dimensional observation tasks, or multi-agent interaction environments where entanglement has stronger theoretical motivation.

### File Locations

- **Implementation**: `packages/quantum-nematode/quantumnematode/brain/arch/qef.py`
- **Tests**: `tests/quantumnematode_tests/brain/arch/test_qef.py` (79 tests)
- **MLP PPO ablation**: `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py` (feature expansion + gating)
- **MLP PPO tests**: `tests/quantumnematode_tests/brain/arch/test_mlpppo.py` (58 tests)
- **Configs**: `configs/examples/qef_*.yml`, `configs/examples/mlpppo_*_fair.yml`
- **Artifacts**: `artifacts/logbooks/008/qef_*`, `artifacts/logbooks/008/mlpppo_*_fair`
- **Appendix**: [QEF optimization history](supporting/008/qef-optimization.md)

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
- [x] Implement QRH brain (Quantum Reservoir Hybrid — 10-qubit fixed reservoir + PPO readout)
- [x] Implement CRH brain (Classical Reservoir Hybrid — 10-neuron ESN ablation control)
- [x] MI decision gate: structured topology does NOT outperform random (p=1.0, wrong direction)
- [x] QRH foraging R1-R5 (structured topology): 0-0.25% — information bottleneck identified
- [x] QRH foraging R6-R8 (random topology): 77% → 87.8% — random topology transforms QRH
- [x] QRH pursuit predators R9: 41.2%, 4/4 converged, 13× lower variance than CRH
- [x] QRH stationary predators R10-R14: 23.6%, 1/4 converged — high seed sensitivity
- [x] CRH pursuit predators R9: 31.8%, 1/4 converged
- [x] CRH stationary predators R14: 29.9%, 3/4 converged — CRH beats QRH on stationary
- [x] Domingo encoding confound control (CRH-trig R15): trig encoding hurts CRH (-18.8pp pursuit); QRH advantage is genuine quantum dynamics
- [x] Structured topology R16: 0.0% across 12,000 episodes — definitively falsified
- [x] Implement QLIF-LSTM brain (H.4 — Quantum LIF LSTM with recurrent PPO)
- [x] QLIF-LSTM Stage 4a: classical ablation foraging — 86.25%, 4/4 converged, PASS
- [x] QLIF-LSTM Stage 4a-Q: quantum foraging — 85.63%, equivalent to classical, PASS
- [x] QLIF-LSTM Stage 4b: pursuit predators 200ep — partial PASS, insufficient training
- [x] QLIF-LSTM Stage 4b: pursuit predators 500ep — 74.7% classical, 70.8% quantum, **98% last-100 classical**, PASS
- [x] QLIF-LSTM entropy floor validation — entropy_coef_end=0.015 prevents late destabilisation
- [x] QLIF-LSTM Stage 4c: thermotaxis pursuit predators large — 60.1% classical (82% last-100), 45.4% quantum (82% last-100), PASS
- [x] QLIF-LSTM Stage 4c: stationary predators — 37% classical ceiling, 31% quantum. 6 rounds tuning, actor [features, h_t] fix. FAIL vs MLP PPO (96.5%)
- [x] QLIF-LSTM quantum comparison complete — quantum QLIF gates provide no measurable advantage on any task
- [x] QRH-QLSTM composition (Stage 4d) — 15 rounds, 54 sessions: CRH-QLSTM 85.4% small pursuit (best reservoir), but -21pp vs QLIF-LSTM at scale. LSTM hurts QRH (-4.1pp). Hypothesis REJECTED
- [x] Implement QEF brain (QA-5 — Quantum Entangled Features, 8-qubit PQC + PPO readout)
- [x] QEF evaluation: 24 phases, ~500+ runs across 3 tasks (stationary, pursuit, small PP)
- [x] Hybrid input + feature gating + curated features — systematic optimization
- [x] Classical ablation: capacity-matched, polynomial, random projection, polynomial + gating
- [x] 12-seed validation: QEF competitive but no significant quantum advantage (p>0.05 on all tasks except pursuit where QEF trails MLP PPO at p=0.04)
- [x] QEF evaluation halted — architecture has reached ceiling in current simulation regime

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

### QRH Best Sessions

| Round | Task | Sessions | Episodes | Result |
|-------|------|----------|----------|--------|
| R8 | Foraging | 20260223_151153-151209 | 4×500 | **86.8% avg, 98.0% post-conv**; 5× less convergence variance |
| R9 | Pursuit | 20260224_044427-044501 | 4×1000 | **41.2% avg**, 4/4 converged, 13× lower variance than CRH |
| R14 | Stationary | 20260226_084742-084759 | 4×3000 | 23.6% avg, 1/4 converged; CRH outperforms |

Experiment results: `artifacts/logbooks/008/qrh_foraging_small/`, `artifacts/logbooks/008/qrh_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/qrh_thermotaxis_stationary_predators_large/`

### CRH Ablation Sessions

| Round | Task | Sessions | Episodes | Result |
|-------|------|----------|----------|--------|
| R9 | Pursuit | 20260225_092747-092754 | 4×1000 | 31.8% avg, 1/4 converged; QRH wins (+9.4pp) |
| R14 | Stationary | 20260226_093914-093921 | 4×3000 | **29.9% avg**, 3/4 converged; CRH wins (+6.3pp) |
| R15 (Domingo) | Pursuit (trig) | 20260301_210008-210021 | 4×1000 | 13.0% avg; trig encoding hurts CRH |
| R15 (Domingo) | Stationary (trig) | 20260301_210103-210115 | 4×3000 | 17.7% avg; Domingo confound resolved |
| R16 | Stationary (struct) | 20260301_221424-221435 | 4×3000 | **0.0%** across 12K episodes; structured falsified |

Experiment results: `artifacts/logbooks/008/crh_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/crh_thermotaxis_stationary_predators_large/`, `artifacts/logbooks/008/crh_trig_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/crh_trig_thermotaxis_stationary_predators_large/`, `artifacts/logbooks/008/qrh_structured_thermotaxis_stationary_predators_large/`

### QLIF-LSTM Best Sessions

| Round | Task | Sessions | Episodes | Result |
|-------|------|----------|----------|--------|
| R1 | Foraging (classical) | 20260305_140313-140321 | 4×200 | **86.25% avg**, 4/4 converged |
| R2 | Foraging (quantum) | 20260305_141819-141831 | 4×200 | **85.63% avg**, equivalent to classical |
| R4 | Pursuit (classical) | 20260305_232259-232309 | 4×500 | **74.7% avg, 98% last-100** |
| R4 | Pursuit (quantum) | 20260305_232312-233816 | 4×500 | **70.8% avg, 93.5% last-100** |
| R4b | Pursuit validation | 20260306_045940-045946 | 2×500 | Entropy floor (0.015) validated |
| R5 | Pursuit large (classical) | 20260306_081836-081846 | 4×500 | **60.1% avg, 82% last-100** |
| R5 | Pursuit large (quantum) | 20260306_112902-112910 | 4×500 | **45.4% avg, 82% last-100** |
| R10 | Stationary (classical) | 20260307_123010-123019 | 4×1000 | **28.8% overall, 36.8% last-100** (ceiling) |
| R10-Q | Stationary (quantum) | 20260307_132444-132457 | 4×500 | **21.1% overall, 30.5% last-100** |

Full optimization history (12 rounds, ~66 sessions): [qliflstm-optimization.md](supporting/008/qliflstm-optimization.md)

Experiment results: `artifacts/logbooks/008/qliflstm_foraging_small/`, `artifacts/logbooks/008/qliflstm_pursuit_predators_small/`, `artifacts/logbooks/008/qliflstm_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/qliflstm_thermotaxis_stationary_predators_large/`

### QRH-QLSTM / CRH-QLSTM Best Sessions

| Phase | Task | Variant | Sessions | Episodes | Result |
|-------|------|---------|----------|----------|--------|
| Phase 1 | Foraging (classical) | QRH-QLSTM | 20260310_122324-122336 | 4×200 | 62.9% avg, 96.6% post-conv |
| Phase 1 | Foraging (classical) | CRH-QLSTM | 20260310_122353-122401 | 4×200 | **86.3% avg**, 99.3% post-conv |
| Phase 2 | Foraging (quantum) | QRH-QLSTM | 20260310_123914-123922 | 4×200 | 66.6% avg, 97.9% post-conv |
| Phase 2 | Foraging (quantum) | CRH-QLSTM | 20260310_123944-123952 | 4×200 | 85.1% avg, 99.7% post-conv |
| Phase 3 | Pursuit (quantum) | CRH-QLSTM | 20260311_015814-015828 | 4×200 | **85.4% avg**, 95.8% post-conv |
| Phase 3b | Pursuit (classical) | CRH-QLSTM | 20260311_111030-111038 | 4×200 | 82.2% avg, 92.7% post-conv |
| Phase 3c | Pursuit (quantum) | QRH-QLSTM | 20260311_031223-031237 | 4×200 | 15.2% avg, FAIL |
| Phase 4 | Pursuit large (classical) | CRH-QLSTM | 20260311_112537-112547 | 4×500 | 35.9% avg |
| Phase 4 v2 | Pursuit large (classical) | CRH-QLSTM | 20260311_121622-121629 | 4×500 | **38.8% avg** (hyperparams aligned) |
| Phase 4b | Stationary large (classical) | CRH-QLSTM | 20260311_124349-124358 | 4×500 | 14.0% avg |
| Stage 4d | Pursuit small (classical) | QRH-LSTM | 20260311_220133-220145 | 4×200 | 17.0% avg |
| Stage 4d | Pursuit large (classical) | QRH-LSTM | 20260311_221535-221547 | 4×500 | 16.4% avg |
| Stage 4d | Stationary large (classical) | QRH-LSTM | 20260311_221821-221832 | 4×500 | 10.8% avg, hypothesis REJECTED |

Full optimization history (15 rounds, 54 sessions): [qrhqlstm-optimization.md](supporting/008/qrhqlstm-optimization.md)

Experiment results: `artifacts/logbooks/008/qrhqlstm_foraging_small/`, `artifacts/logbooks/008/qrhqlstm_pursuit_predators_small/`, `artifacts/logbooks/008/qrhqlstm_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/qrhqlstm_thermotaxis_stationary_predators_large/`, `artifacts/logbooks/008/crhqlstm_foraging_small/`, `artifacts/logbooks/008/crhqlstm_pursuit_predators_small/`, `artifacts/logbooks/008/crhqlstm_thermotaxis_pursuit_predators_large/`, `artifacts/logbooks/008/crhqlstm_thermotaxis_stationary_predators_large/`

______________________________________________________________________

## Strategic Pivot: Environment Enrichment Before Further Quantum Evaluation

**Date**: 2026-03-20
**Decision**: Defer QA-6 (QRH+ weak-measurement feedback). Promote QA-7 (Quantum Plasticity) as final quantum experiment at current environment complexity. After QA-7, pivot to environment enrichment (roadmap Phases 1-3).

### Rationale

After 300+ experiment sessions across 11+ quantum architectures (QRC, QSNN, QVarCircuit, QSNN-PPO, QSNNReinforce A2C, HybridQuantum, HybridClassical, HybridQuantumCortex, QRH, CRH, QLIF-LSTM, QRH-QLSTM, CRH-QLSTM, QRH-LSTM, QEF), the theoretical and empirical evidence converges: our current environment complexity is below the threshold for quantum advantage.

**Key factors**:

- **Observation space (2-9D)**: Dequantization results prove classical algorithms match quantum at this scale
- **Action space (4 discrete)**: Quantum search advantages require >10^20 actions
- **State space (~10K)**: Polynomial, not the exponential spaces where quantum excels
- **Classical performance (94-98%)**: No headroom for quantum to demonstrate advantage
- **Empirical**: Every trainable quantum component either fails or matches its classical ablation

**QA-7 is the exception** because it tests a fundamentally different hypothesis — optimisation landscape properties (anti-forgetting via PQC unitarity), not computational performance. This is testable at current complexity since multi-objective sequential training is already our core challenge.

### What Changes

- **QA-6**: DEFERRED — even +5pp target yields ~28% absolute vs classical 90%+
- **QA-3**: DEFERRED — QA-5 showed entangled features competitive but not advantageous; trainable entangled circuits unlikely to help
- **QA-7**: COMPLETED — classical baselines show zero forgetting; quantum runs halted (see below)
- **Post QA-7**: Pivot to environment enrichment confirmed. Return to quantum evaluation when environments reach complexity thresholds (>30 input features, multi-agent, long non-Markovian horizons)

Full analysis: [quantum-architectures.md Strategic Assessment](../../research/quantum-architectures.md#strategic-assessment-environment-complexity--quantum-advantage)

______________________________________________________________________

## QA-7 Quantum Plasticity Test Results

**Date**: 2026-03-21
**Status**: COMPLETED — classical baselines show zero backward forgetting; quantum runs halted.

### Protocol

Sequential multi-objective training: Foraging (A) → Pursuit Predators (B) → Thermotaxis+Pursuit (C) → Foraging Return (A'). All phases on 100×100 grid with 200 training episodes per phase. 50-episode evaluation blocks at each of 5 transition points (9 eval blocks total per seed). Brain state snapshot/restore ensures eval blocks leave no trace on training.

Convergence threshold for Plasticity Retention: 60% success rate in trailing-20 window.

### Classical Baseline Results (4 seeds × 3 architectures = 12 sessions)

| Architecture | BF (mean ± std) | FT (mean ± std) | PR (mean ± std) |
|---|---|---|---|
| **MLP PPO** | 0.000 ± 0.000 | 0.205 ± 0.067 | 1.000 ± 0.000 |
| **CRH** | 0.000 ± 0.000 | -0.085 ± 0.104 | 1.000 ± 0.000 |
| **HybridClassical** | 0.005 ± 0.009 | 0.130 ± 0.022 | 1.000 ± 0.000 |

**Per-seed Backward Forgetting values:**

- MLP PPO: [0.0, 0.0, 0.0, 0.0]
- CRH: [0.0, 0.0, 0.0, 0.0]
- HybridClassical: [0.0, 0.0, 0.0, 0.02]

**11/12 seeds show exactly zero backward forgetting.** The one non-zero value (HybridClassical seed 512, BF=0.02) is noise — a single eval episode failure out of 50.

### Decision: Halt Quantum Runs

The quantum plasticity hypothesis (arXiv:2511.17228) posits that PQC unitarity prevents catastrophic forgetting compared to classical networks. Classical baselines show **zero forgetting** on these tasks. There is no classical forgetting for quantum to improve upon.

Running QRH and HybridQuantum would at best produce "quantum = classical = no forgetting" — an uninformative comparison with no statistical power to detect a difference.

### Key Findings

1. **No catastrophic forgetting at this environment complexity.** All classical architectures maintain 100% foraging success throughout training on pursuit predators and thermotaxis+pursuit. The network has enough capacity to retain all skills simultaneously.

2. **Forward transfer is architecture-dependent.** MLP PPO shows strong positive transfer (FT=0.21 — foraging training helps pursuit). CRH shows slight negative transfer (FT=-0.09 — ESN reservoir features learned for foraging slightly hurt pursuit). HybridClassical is intermediate (FT=0.13).

3. **Perfect plasticity retention.** PR=1.0 across all architectures and seeds — relearning speed is identical to original learning speed. No plasticity loss.

4. **The continual learning challenge doesn't exist at this scale.** The tasks (foraging, pursuit, thermotaxis) use different reward signals and environment dynamics, but the network's gradient-based learning easily accommodates all simultaneously. Catastrophic forgetting requires tasks that share and compete for the same representational capacity — which doesn't happen here.

### Implications for QA-7 Hypothesis

The arXiv:2511.17228 result (PQC unitarity prevents forgetting) may be valid for tasks where classical networks *do* show forgetting. Our environments are below that threshold. This further validates the strategic pivot: advance to richer environments (Phases 1-3) where classical approaches encounter actual learning conflicts before revisiting quantum plasticity.

### Experiment Artifacts

- Script: `scripts/run_plasticity_test.py`
- Configs: `configs/studies/plasticity/campaign/*.yml`
- Logs: `build/studies/plasticity_test/logs/*.log`
- Scratchpad: `build/studies/plasticity_test/plasticity_test_scratchpad.md`

### Appendices

- QSNN foraging optimization history (17 rounds): [qsnn-foraging-optimization.md](supporting/008/qsnn-foraging-optimization.md)
- QSNN predator optimization history (16 rounds, 64 sessions): [qsnn-predator-optimization.md](supporting/008/qsnn-predator-optimization.md)
- QSNN-PPO optimization history (4 rounds, 16 sessions): [qsnnppo-optimization.md](supporting/008/qsnnppo-optimization.md)
- QSNNReinforce A2C optimization history (4 rounds, 16 sessions): [qsnnreinforce-a2c-optimization.md](supporting/008/qsnnreinforce-a2c-optimization.md)
- HybridQuantum optimization history (4 rounds, 16 sessions): [hybridquantum-optimization.md](supporting/008/hybridquantum-optimization.md)
- HybridClassical ablation (12 sessions, trust analysis): [hybridclassical-ablation.md](supporting/008/hybridclassical-ablation.md)
- HybridQuantumCortex optimization history (9 rounds, 32 sessions): [hybridquantumcortex-optimization.md](supporting/008/hybridquantumcortex-optimization.md)
- QRH/CRH optimization history (16 rounds, 96 sessions): [qrh-optimization.md](supporting/008/qrh-optimization.md)
- QLIF-LSTM optimization history (12 rounds, ~66 sessions): [qliflstm-optimization.md](supporting/008/qliflstm-optimization.md)

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
- QRH implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrh.py`
- QRH tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qrh.py`
- QRH configs: `configs/examples/qrh_*.yml`
- CRH implementation: `packages/quantum-nematode/quantumnematode/brain/arch/crh.py`
- CRH tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_crh.py`
- CRH configs: `configs/examples/crh_*.yml`
- QLIF-LSTM implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qliflstm.py`
- QLIF-LSTM tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qliflstm.py`
- QLIF-LSTM configs: `configs/examples/qliflstm_*.yml`
