# Advanced Quantum Brain Architectures - Implementation Notes

**Purpose**: Detailed specifications for novel quantum brain implementations beyond QVarCircuitBrain
**Status**: Research & Planning
**Last Updated**: 2026-02-20

______________________________________________________________________

## Table of Contents

01. [Executive Summary](#executive-summary)
02. [Current Results & Lessons Learned](#current-results--lessons-learned)
03. [QSNN Brain Specification](#qsnn-brain-specification)
04. [QRC Brain Specification](#qrc-brain-specification)
05. [QSNN-PPO Hybrid](#qsnn-ppo-hybrid)
06. [PPO-Q Style PQC Actor](#ppo-q-style-pqc-actor)
07. [HybridQuantum Brain Architecture](#hybridquantum-brain-architecture)
08. [Data Re-Uploading Enhancement](#data-re-uploading-enhancement)
09. [Optimizer Variants](#optimizer-variants)
10. [Barren Plateau Mitigation](#barren-plateau-mitigation)
11. [External Research Survey (2024-2026)](#external-research-survey-2024-2026)
12. [Next-Generation Architecture Proposals (H.1-H.6)](#next-generation-architecture-proposals)
13. [Benchmarking Plan](#benchmarking-plan)
14. [Research References](#research-references)
15. [Open Questions](#open-questions)

______________________________________________________________________

## Executive Summary

### Goal

Find a quantum or hybrid quantum architecture that can match or surpass classical architectures for multi-objective learning in the nematode simulation: foraging, predator evasion, thermotaxis navigation, and future advanced environments. The architecture must support **gradient-based online learning** (not just evolutionary/offline optimization).

### Why New Architectures?

The current QVarCircuitBrain achieves 88% success with CMA-ES evolutionary optimization but suffers from:

- Parameter-shift gradients showing barren-plateau-like behavior
- High sensitivity to initial parameters
- Gradient-based learning only achieving ~40% success
- Only evaluated up to random predators (not pursuit predators)
- CMA-ES does not support online/live learning

### Current Landscape

```text
GRADIENT-BASED ONLINE LEARNING EFFECTIVENESS
═══════════════════════════════════════════════════════════════════════════

Architecture                      Foraging   Pursuit Pred   Training Method         Viable?
──────────────────────────────────────────────────────────────────────────────────────────────
QRC                               0%         0%             REINFORCE (readout)     NO
QSNN (Hebbian)                    0%         N/A            3-factor local          NO
QSNN-PPO Hybrid                   N/A        0% (16 sess)   Surr grad + PPO         NO†
QVarCircuit (parameter-shift)     ~40%       Not tested     Parameter-shift SGD     MARGINAL
──────────────────────────────────────────────────────────────────────────────────────────────
QSNN (Surrogate gradient)         73.9%      0% (60 sess)   Quantum fwd + surr bwd  PARTIAL
QSNNReinforce A2C                 N/A        0.5% (16 sess) Surr grad + A2C critic  NO‡
QVarCircuit (CMA-ES)              99.8%      76.1%**        Evolutionary            NOT ONLINE
──────────────────────────────────────────────────────────────────────────────────────────────
HybridQuantum                     91.0%      96.9%          Surr REINFORCE + PPO    YES (BEST)
HybridClassical (ablation)        97.0%      96.3%          Backprop + PPO          YES (CONTROL)
──────────────────────────────────────────────────────────────────────────────────────────────
SpikingReinforceBrain             73.3%§     ~61%§          Surrogate grad (class)  UNRELIABLE
MLPReinforceBrain                 95.1%      73.4%          REINFORCE (classical)   YES
MLPPPOBrain                       96.7%      71.6%††/94.5%  PPO (classical)         YES
──────────────────────────────────────────────────────────────────────────────────────────────

** QVarCircuit predator result is on random predators only, with CMA-ES (not gradient-based)
† QSNN-PPO: PPO incompatible with surrogate gradients — policy_loss=0 in 100% of updates
‡ QSNNReinforce A2C: critic never learned V(s) (EV -0.620); all improvement from REINFORCE backbone
§ SpikingReinforce numbers from best session only; ~90% of sessions fail catastrophically
†† MLP PPO unified sensory modules (apples-to-apples comparison); 94.5% uses pre-computed gradient
```

### Key Finding: Architecture + Curriculum Drive Performance, Not QSNN

The **HybridQuantum brain** is the first quantum architecture to surpass a classical baseline on a multi-objective RL task using gradient-based online learning. It combines the QSNN reflex layer (proven 73.9% foraging) with a classical cortex MLP (PPO) via mode-gated fusion, achieving **96.9% post-convergence on pursuit predators** — beating the apples-to-apples MLP PPO baseline by **+25.3 points** with **4.3x fewer parameters**.

However, **classical ablation (HybridClassical)** — replacing the QSNN reflex with a small classical MLP reflex (~116 params) while keeping everything else identical — achieves **96.3% mean / 97.8% best post-convergence**, proving the QSNN quantum reflex is **not the key performance driver**. The three-stage curriculum and mode-gated fusion architecture are what matter.

The architecture was validated through a three-stage curriculum:

1. **Stage 1**: QSNN reflex on foraging (REINFORCE) — 91.0% success, 4 sessions
2. **Stage 2**: Cortex PPO with frozen QSNN (pursuit predators) — 91.7% post-convergence, 8 sessions across 2 rounds
3. **Stage 3**: Joint fine-tune (both trainable) — 96.9% post-convergence, 4 sessions, immediate convergence

**Ablation insight**: The cortex adapts its strategy to the reflex quality. With the QSNN (trust ~0.55), the cortex delegates foraging to the reflex. With the classical MLP (trust ~0.37), the cortex handles more itself. Both strategies achieve equivalent performance.

**Where QSNN retains value**: biological fidelity (higher chemotaxis indices closer to real C. elegans), parameter efficiency (92 vs 116 params), and as a scientifically interesting model of quantum neural computation.

**QSNN's surrogate gradient approach** (quantum forward, classical backward) remains the core proven quantum technique. It sidesteps barren plateaus while providing dense gradient signals. However, standalone QSNN cannot solve multi-objective tasks (0% across 60 sessions on pursuit predators). The hybrid architecture resolves this by delegating strategic behaviour to the classical cortex while preserving the quantum reflex.

### Architecture Evaluation History

```text
COMPLETED:
  HybridQuantum — QSNN reflex + classical cortex MLP + mode-gated fusion.
    4 rounds, 16 sessions, 4,200 episodes. 96.9% post-convergence.
    Beats MLP PPO unified by +25.3 pts. Three-stage curriculum validated.
    STATUS: SUCCESS — best quantum architecture for multi-objective tasks.

  HybridClassical — Classical ablation of HybridQuantum.
    MLP reflex (116 params) replacing QSNN reflex (92 params).
    12 sessions, 4,200 episodes across 3 stages. 96.3% mean / 97.8% best.
    Proves QSNN is NOT the key ingredient — curriculum + fusion drive perf.
    STATUS: ABLATION COMPLETE — QSNN value is biological, not task perf.

  QSNNReinforce A2C — A2C critic cannot learn V(s) in pursuit predator env.
    4 rounds, 16 sessions. Critic EV: 0 → -0.008 → -0.295 → -0.620.
    All actor improvement from REINFORCE backbone, not critic.
    STATUS: HALTED — critic fails under partial observability.

  QSNN-PPO — PPO incompatible with surrogate gradients (policy_loss=0 always).
    4 rounds, 16 sessions. Fundamental: forward pass returns constant.
    STATUS: HALTED — architectural incompatibility.

NOT EVALUATED:
  PPO-Q Style PQC Actor — PQC wrapped in classical pre/post-processing.
    Uses parameter-shift, not surrogate gradients. Not pursued given
    HybridQuantum's success.
```

______________________________________________________________________

## Current Results & Lessons Learned

### Architecture Evaluation Summary

#### QVarCircuit (Modular Variational Quantum Circuit)

- **Training**: Parameter-shift rule gradients with momentum SGD; also supports CMA-ES
- **Foraging**: 99.8% (CMA-ES), ~40% (gradient-based)
- **Predators**: 76.1% on random predators (CMA-ES only), not tested on pursuit predators
- **Verdict**: Strong with evolutionary optimization but gradient-based learning is limited by barren-plateau-like behavior. Parameter-shift rule requires 2N circuit evaluations per gradient step, which is computationally expensive and produces weak gradients.

#### QSNN (Quantum Spiking Neural Network)

- **Training**: Surrogate gradient REINFORCE (quantum forward, classical backward)
- **Foraging**: 73.9% success, **100% session reliability** (4/4), 92 params
- **Predators (random)**: 22.3% avg, best session 48.3% post-convergence
- **Predators (pursuit)**: 0% across 60 sessions (24 on pursuit specifically)
- **Strengths**: Avoids barren plateaus via surrogate gradients; 1,400x parameter efficiency vs classical spiking; provably reliable session convergence on single-objective tasks
- **Weaknesses**: No critic (high REINFORCE variance); 92-212 params insufficient for multi-objective; per-encounter evasion rate never improved through training; unbounded weight growth

**Key technical findings from 13 rounds of QSNN predator experiments:**

1. **Hebbian local learning failed** (0% success) — learning signal magnitude ~0.001 per update, insufficient for RL tasks
2. **Surrogate gradients work** — sigmoid surrogate at RY transition point (pi/2) provides ~1,000x stronger gradients than Hebbian, enabling 73.9% foraging
3. **Fan-in-aware tanh scaling essential** — `tanh(w*x / sqrt(fan_in))` prevents gradient death when hidden layer width increases. Without it, gradients die within 10-20 episodes
4. **Multi-timestep integration critical** — averaging 10 quantum measurements per decision reduces shot noise 10x (5 timesteps: 52.6% → 10 timesteps: 73.9% on foraging)
5. **Adaptive entropy regulation** — two-sided system (floor boost + ceiling suppress) prevents both entropy collapse and entropy lock at maximum
6. **Weight growth is unbounded** — W_sh and W_hm norms grow 3-6x over 200 episodes with no plateau, progressively re-saturating tanh despite fan-in scaling
7. **Entropy ceiling suppresses early exploration** — `entropy_ceiling_fraction=0.95` zeroes entropy_coef in the first ~30 episodes, enabling premature direction-lock

#### QRC (Quantum Reservoir Computing)

- **Training**: REINFORCE on classical readout only (fixed quantum reservoir)
- **Foraging**: 0% across 1,600+ runs
- **Root cause**: Non-discriminative representations — fixed random reservoir angles produce similar measurement distributions for different sensory inputs. The readout network learns the WRONG behavior (negative chemotaxis = away from food).
- **Verdict**: Not viable for spatial navigation tasks. May have potential for time-series forecasting where temporal structure provides richer signal.

#### Classical Baselines — Why They Succeed

**MLP PPO** (83.3% on pursuit predators, ~17K params):

- Actor-critic with GAE provides dense per-step advantage estimates
- PPO clipping prevents catastrophic policy updates
- 20 epochs × 2 minibatches = 40 gradient passes per 512-step buffer
- Orthogonal weight initialization for stable learning
- Classical gradients do not suffer from barren plateaus

**SpikingReinforce** (61% best session, 131K params):

- Temporal dynamics via 100 LIF timesteps per decision
- Intra-episode updates every 20 steps (denser credit assignment than QSNN)
- High variance: ~90% of sessions fail catastrophically (vs QSNN's 0% failure on foraging)

### The Fundamental Gaps

To match classical baselines on multi-objective predator tasks, a quantum architecture needs:

| Requirement | MLP PPO (classical) | QSNN (current) | Gap | Attempted Fix | Result |
|---|---|---|---|---|---|
| **Parameters** | ~17K | 212 | 80x | — | — |
| **Gradient passes per buffer** | 40 | 2 | 20x | QSNN-PPO (4 epochs) | PPO incompatible |
| **Value function** | Classical critic with GAE | None (REINFORCE only) | Missing | A2C critic (4 rounds) | Critic can't learn V(s) |
| **Temporal credit assignment** | Per-step via TD(λ) | Per-window (uniform) | Missing | A2C GAE | Fails with critic |
| **Policy stability** | PPO clipping (ε=0.2) | Partial (advantage_clip) | Incomplete | QSNN-PPO | PPO incompatible |
| **Weight regularization** | Implicit (optimizer) | Per-element clip only | Weak | — | — |
| **Full state access** | Full observation | Local viewport gradients | Critical | — | Root cause of critic failure |

______________________________________________________________________

## QSNN Brain Specification

### A.1 Overview

Quantum Spiking Neural Networks combine quantum computing with neuromorphic principles for reflex-like sensorimotor control.

**Key Advantage**: Surrogate gradient learning (quantum forward pass, classical backward pass) avoids barren plateaus while maintaining dense gradient signals. Experimentally verified: 73.9% foraging success with 92 parameters and 100% session reliability.

**Key Limitation**: The architecture lacks a value function, making it unsuitable for high-variance multi-objective tasks when trained with vanilla REINFORCE.

### A.2 Architecture

#### Quantum Leaky Integrate-and-Fire (QLIF) Neuron

Based on Brand & Petruccione (2024), a minimal QLIF neuron requires:

- **Only 2 rotation gates** (RY for membrane potential, RX for leak)
- **No CNOT gates** needed for single neuron
- High fidelity with minimal circuit depth

```text
Circuit per QLIF neuron:
|0⟩ → RY(θ_membrane + input) → RX(θ_leak) → Measure

Where:
- θ_membrane: trainable membrane potential parameter
- θ_leak: leak rate parameter (1 - decay_rate)
- input: weighted sum of incoming spikes
```

#### Actual Implementation (as built)

The implemented QSNN uses a 3-layer feedforward spiking architecture with fan-in-aware scaling:

```text
Sensory (8 QLIF) → Hidden (16 QLIF) → Motor (4 QLIF)
  W_sh: 8×16          W_hm: 16×4
  (128 params)        (64 params)
  θ_h: 16 params      θ_m: 4 params
  Total: 212 params

ry_angle = θ + tanh(w·x / sqrt(fan_in)) × π    # Fan-in scaling prevents gradient death
spike_prob = P(measure |1⟩)                      # Quantum measurement
surrogate_grad = sigmoid(α × (ry_angle - π/2))  # Classical backward pass
```

Each decision requires 10 integration steps (multi-timestep averaging), each running the quantum circuit for all neurons. With 1,024 shots per circuit, this provides 10,240 effective samples per decision for noise reduction.

#### Network Structure

```python
class QSNNBrainConfig(BrainConfig):
    name: Literal["qsnn"] = "qsnn"
    num_sensory_neurons: int = 6      # One per sensory module
    num_hidden_neurons: int = 4       # Interneuron layer
    num_motor_neurons: int = 5        # One per action
    membrane_tau: float = 0.9         # Leak time constant
    threshold: float = 0.5            # Firing threshold
    refractory_period: int = 2        # Timesteps after firing
    use_local_learning: bool = True   # Use local vs backprop
```

### A.3 Sensory Spike Encoding

Convert continuous sensor values to spike trains:

```python
def encode_sensory_spikes(brain_params: BrainParams) -> dict[str, float]:
    """Encode sensory inputs as spike probabilities."""
    encodings = {}

    # Chemotaxis: gradient magnitude → spike rate
    grad_mag = np.linalg.norm([brain_params.gradient_x, brain_params.gradient_y])
    encodings['chemotaxis'] = sigmoid(grad_mag * 5.0)  # Scale to [0,1]

    # Thermotaxis: temperature deviation from preference
    temp_diff = abs(brain_params.temperature - brain_params.preferred_temp)
    encodings['thermotaxis'] = sigmoid(-temp_diff)  # Higher when at preferred

    # Nociception: predator proximity → high spike rate
    encodings['nociception'] = sigmoid(brain_params.predator_threat * 10.0)

    # Mechanosensation: wall/obstacle contact
    encodings['mechanosensation'] = 1.0 if brain_params.touching_wall else 0.0

    # Proprioception: current velocity/heading
    encodings['proprioception'] = normalize(brain_params.velocity)

    # Food chemotaxis: food gradient
    food_grad = np.linalg.norm([brain_params.food_gradient_x, brain_params.food_gradient_y])
    encodings['food_chemotaxis'] = sigmoid(food_grad * 5.0)

    return encodings
```

### A.4 Timestep Update Rule

```python
def qsnn_timestep(self, sensory_spikes: dict[str, float]) -> np.ndarray:
    """Single timestep of QSNN dynamics."""

    # 1. Encode sensory spikes into qubit rotations
    for i, (module, spike_prob) in enumerate(sensory_spikes.items()):
        # RY rotation proportional to spike probability
        self.circuit.ry(spike_prob * np.pi, self.sensory_qubits[i])

    # 2. Synaptic integration (weighted connections)
    for h in range(self.num_hidden):
        # Sum weighted inputs from sensory layer
        weighted_input = sum(
            self.weights_sh[s, h] * sensory_spikes[s]
            for s in sensory_spikes
        )
        # Apply to hidden neuron membrane
        self.circuit.ry(weighted_input * np.pi, self.hidden_qubits[h])
        # Apply leak
        self.circuit.rx(self.leak_angle, self.hidden_qubits[h])

    # 3. Hidden → Motor connections
    for m in range(self.num_motor):
        weighted_input = sum(
            self.weights_hm[h, m] * self.hidden_activations[h]
            for h in range(self.num_hidden)
        )
        self.circuit.ry(weighted_input * np.pi, self.motor_qubits[m])
        self.circuit.rx(self.leak_angle, self.motor_qubits[m])

    # 4. Measure motor neurons for firing probabilities
    motor_probs = self.measure_motor_neurons()

    # 5. Convert to action logits
    action_logits = self.firing_to_action(motor_probs)

    return action_logits
```

### A.5 Firing Probability via Measurement

```python
def measure_motor_neurons(self) -> np.ndarray:
    """Measure motor qubits to get firing probabilities."""
    # Execute circuit
    result = self.backend.run(self.circuit, shots=self.shots).result()
    counts = result.get_counts()

    # Extract per-motor-neuron firing probability
    firing_probs = np.zeros(self.num_motor)
    total = sum(counts.values())

    for bitstring, count in counts.items():
        for m in range(self.num_motor):
            if bitstring[m] == '1':
                firing_probs[m] += count / total

    return firing_probs
```

### A.6 Motor Primitive Output

```python
def firing_to_action(self, motor_probs: np.ndarray) -> np.ndarray:
    """Convert motor neuron firing to action logits."""
    # Motor neurons map to actions:
    # 0: forward, 1: backward, 2: left, 3: right, 4: stay

    # Option 1: Direct mapping (motor_probs ARE the logits)
    action_logits = motor_probs

    # Option 2: Learned linear transform
    # action_logits = self.readout_weights @ motor_probs + self.readout_bias

    return action_logits
```

### A.7 Learning Rules

#### Option A: Local Learning (tested, failed)

Three-factor Hebbian learning was tested extensively (17 rounds) and produced 0% success. The learning signal magnitude (~0.001 per update) is insufficient for RL tasks. This approach is not recommended.

```python
def local_learning_update(self, pre_spike, post_spike, reward):
    """Three-factor local learning rule."""
    # Eligibility trace: correlation of pre and post activity
    eligibility = pre_spike * post_spike

    # Weight update: reward-modulated Hebbian
    delta_w = self.lr * eligibility * reward

    # Apply update
    self.weights += delta_w
```

#### Option B: Surrogate Gradient (proven, recommended)

The surrogate gradient approach is the proven winner. The quantum circuit runs the forward pass, but the backward pass uses a classical sigmoid surrogate centered at the RY quantum transition point (pi/2):

```python
class QLIFSurrogateSpike(torch.autograd.Function):
    """Custom autograd for quantum-classical hybrid gradient."""

    @staticmethod
    def forward(ctx, ry_angle, quantum_spike_prob, alpha):
        ctx.save_for_backward(ry_angle)
        ctx.alpha = alpha
        return quantum_spike_prob  # Actual quantum measurement

    @staticmethod
    def backward(ctx, grad_output):
        ry_angle, = ctx.saved_tensors
        alpha = ctx.alpha
        # Sigmoid surrogate centered at pi/2 (quantum transition point)
        sig = torch.sigmoid(alpha * (ry_angle - math.pi / 2))
        surrogate_grad = alpha * sig * (1 - sig)
        return grad_output * surrogate_grad, None, None
```

This provides ~1,000x stronger gradient signal than Hebbian learning while avoiding the barren plateau issues of parameter-shift gradients.

______________________________________________________________________

## QRC Brain Specification

### B.1 Overview

Quantum Reservoir Computing uses a fixed (random) quantum reservoir with only a classical readout layer trained. This **inherently avoids barren plateaus** since reservoir parameters are never optimized.

**Experimental Result**: QRC was tested across 1,600+ runs on foraging and achieved 0% success. The fixed random reservoir produces non-discriminative representations — different sensory inputs yield similar measurement distributions. The readout network consistently learned the wrong behavior (negative chemotaxis = moving away from food). **Not recommended for spatial navigation tasks.**

### B.2 Architecture

```python
class QRCBrainConfig(BrainConfig):
    name: Literal["qrc"] = "qrc"
    num_reservoir_qubits: int = 8     # Size of quantum reservoir
    reservoir_depth: int = 3          # Circuit depth (entanglement layers)
    reservoir_seed: int = 42          # For reproducibility
    readout_hidden: int = 32          # Classical readout hidden size
    readout_type: Literal["linear", "mlp"] = "mlp"
```

### B.3 Fixed Quantum Reservoir

```python
def build_reservoir_circuit(self) -> QuantumCircuit:
    """Build fixed random quantum reservoir."""
    qc = QuantumCircuit(self.num_qubits)
    rng = np.random.default_rng(self.seed)

    # Initial superposition
    for q in range(self.num_qubits):
        qc.h(q)

    # Random entangling layers
    for layer in range(self.depth):
        # Random single-qubit rotations
        for q in range(self.num_qubits):
            theta = rng.uniform(0, 2*np.pi, 3)
            qc.rx(theta[0], q)
            qc.ry(theta[1], q)
            qc.rz(theta[2], q)

        # Entangling layer (circular CZ)
        for q in range(self.num_qubits - 1):
            qc.cz(q, q + 1)
        qc.cz(self.num_qubits - 1, 0)  # Wrap around

    return qc
```

### B.4 Input Encoding

```python
def encode_input(self, brain_params: BrainParams) -> QuantumCircuit:
    """Encode sensory input into reservoir."""
    qc = self.reservoir_circuit.copy()

    # Encode each sensory feature as rotation on corresponding qubits
    features = self.brain_params_to_features(brain_params)

    for i, feat in enumerate(features):
        qubit = i % self.num_qubits
        qc.ry(feat * np.pi, qubit)  # Input rotation

    return qc
```

### B.5 Reservoir State Extraction

```python
def get_reservoir_state(self, circuit: QuantumCircuit) -> np.ndarray:
    """Execute circuit and extract reservoir state."""
    result = self.backend.run(circuit, shots=self.shots).result()
    counts = result.get_counts()

    # Convert to probability distribution
    probs = np.zeros(2 ** self.num_qubits)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count / total

    return probs  # Shape: (2^num_qubits,)
```

### B.6 Classical Readout (ONLY trainable part)

```python
class QRCReadout(nn.Module):
    """Classical readout network - ONLY this is trained."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, reservoir_state: torch.Tensor) -> torch.Tensor:
        return self.net(reservoir_state)
```

### B.7 Training

Only train the classical readout:

```python
def train_step(self, brain_params, action, reward):
    """Train only the readout network."""
    # Get reservoir state (no gradient through quantum)
    reservoir_state = self.get_reservoir_state(self.encode_input(brain_params))
    reservoir_tensor = torch.tensor(reservoir_state, dtype=torch.float32)

    # Forward through readout
    action_logits = self.readout(reservoir_tensor)

    # REINFORCE loss
    log_prob = F.log_softmax(action_logits, dim=-1)[action]
    loss = -log_prob * reward

    # Backprop through readout only
    self.readout_optimizer.zero_grad()
    loss.backward()
    self.readout_optimizer.step()
```

### B.8 Why QRC Failed — Lessons Learned

Across 17 experimental sessions, QRC consistently failed due to:

1. **Non-discriminative representations**: Reservoir output entropy stayed at ~2.4/2.77 bits maximum — the reservoir does not produce meaningfully different outputs for different inputs
2. **Wrong behavior learned**: Readout network entropy dropped to 1.36 (indicating learning DID occur), but the policy learned negative chemotaxis (moving AWAY from food)
3. **Weak gradients**: Gradient norms of 0.02-0.14 — too small for effective REINFORCE learning
4. **Root cause**: Random reservoir angles have no task-relevant structure. The reservoir must be either (a) trained (defeating the barren-plateau-avoidance purpose) or (b) structurally designed for the task

**Potential revival path**: Structured reservoir (angles derived from task structure, not random), or use as temporal feature extractor in a hybrid architecture where spatial features come from a different source.

______________________________________________________________________

## QSNN-PPO Hybrid

**Status**: Implemented and evaluated across 4 rounds (16 sessions, 1,000 episodes). **HALTED** — PPO is fundamentally incompatible with surrogate gradient spiking networks.

### Overview

Paired the QSNN actor (proven surrogate gradients, barren-plateau-free) with a classical critic for value estimation and the PPO training algorithm. Designed to address the three diagnosed root causes of QSNN's predator failure:

1. **No critic** → Add classical MLP critic with GAE advantages
2. **High REINFORCE variance** → PPO clipped surrogate objective
3. **Insufficient gradient passes** → Multi-epoch updates with quantum caching

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       QSNN-PPO Hybrid                       │
│                                                             │
│  Sensory Input (food_chemotaxis + nociception = 8 features) │
│       │                                                     │
│       ├────────────────────────────┐                        │
│       ▼                            ▼                        │
│  ┌────────────────────────┐  ┌───────────────────────┐      │
│  │  QSNN Actor (existing) │  │  Classical Critic     │      │
│  │  8 sensory → 16 hidden │  │  (NEW)                │      │
│  │  → 4 motor QLIF        │  │                       │      │
│  │                        │  │  Input: raw sensory   │      │
│  │  Surrogate gradient    │  │  (8-dim) + QSNN       │      │
│  │  REINFORCE backbone    │  │  hidden spike rates   │      │
│  │                        │  │  (16-dim) = 24-dim    │      │
│  │  ~212 quantum params   │  │                       │      │
│  └──────────┬─────────────┘  │  Linear(24, 64) +     │      │
│             │                │  ReLU → Linear(64,    │      │
│             ▼                │  64) + ReLU →         │      │
│      Action selection        │  Linear(64, 1) = V(s) │      │
│      (Categorical)           │                       │      │
│                              │  ~5K classical params │      │
│                              └──────────┬────────────┘      │
│                                         │                   │
│                                         ▼                   │
│                                    GAE Advantages           │
│                                         │                   │
│  Training Loop:                         │                   │
│  ┌──────────────────────────────────────┘                   │
│  │                                                          │
│  │  1. Collect rollout buffer (512 steps)                   │
│  │  2. Compute GAE advantages using critic V(s)             │
│  │  3. For each epoch (4 epochs):                           │
│  │     a. Recompute QSNN forward (quantum on epoch 0,       │
│  │        cached spike probs on epochs 1+)                  │
│  │     b. PPO clipped surrogate loss on actor               │
│  │     c. MSE/Huber loss on critic                          │
│  │     d. Step both optimizers                              │
│  │  4. Clear buffer, repeat                                 │
│  │                                                          │
│  │  Total: ~5.2K params (212 quantum + 5K classical)        │
│  │  Gradient passes: 4 per buffer (vs QSNN's 1)             │
│  └──────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### Why This Could Work

1. **Proven QSNN foundation**: Surrogate gradients avoid barren plateaus. 100% session reliability on foraging. The quantum forward / classical backward approach is the architecture's core strength.

2. **Classical critic fills the gap**: The critic provides per-step V(s) estimates for GAE advantages — the single most important missing component for multi-objective learning. MLP PPO's success (83.3% on pursuit predators) is primarily due to its critic, not its actor complexity.

3. **Hidden spike rates as critic input**: Feeding the critic the QSNN's 16-dim hidden spike rates (after quantum measurement) gives it access to the QSNN's learned internal representation. This is a 24-dim input (16 hidden + 8 raw sensory) — 6x richer than the previous failed critic attempt which only saw 4 raw features.

4. **Multi-epoch infrastructure exists**: QSNN already supports `num_reinforce_epochs` with quantum output caching. Adapting this for PPO epochs is straightforward.

5. **PPO clipping partially exists**: The `clip_epsilon` config field and advantage clipping are already in the QSNN codebase.

### Why Previous Critic Attempts Failed

The QSNN-AC approach was tried and failed (0% across 8 sessions). The failures had specific, fixable causes:

- **Bug: gradient death from tanh saturation** — fan-in scaling had not been implemented yet, so weight gradients died within 10-20 episodes. The critic was training against a brain that couldn't learn.
- **Bug: weight clamping inside epoch loop** — weights were clamped 3x per window instead of once, destroying gradient information.
- **Bug: reward normalization active during AC mode** — made critic targets non-stationary.
- **Design flaw: critic only saw 4-dim raw input** — the critic had no access to the QSNN's internal state, making it impossible to track the value landscape.

All of these bugs have since been fixed. A properly designed QSNN-PPO with hidden spike rate inputs to the critic is an untested configuration.

### Key Design Decisions

1. **Critic input**: 24-dim (8 raw sensory + 16 hidden spike rates). The hidden spike rates are detached from the autograd graph (no gradient flow from critic through quantum circuit).

2. **Rollout buffer**: 512 steps (matching MLP PPO's predator config). Stores states, actions, log_probs, rewards, values, hidden_spike_rates.

3. **GAE computation**: Standard GAE with λ=0.95, γ=0.99. Advantages normalized per-buffer.

4. **Actor loss**: PPO clipped surrogate using QSNN's existing surrogate gradient backward pass. Ratio = exp(new_log_prob - old_log_prob), clipped to [1-ε, 1+ε].

5. **Critic loss**: Huber loss (robust to extreme death penalties of -10.0).

6. **Weight regularization**: Add L2 weight decay (λ=0.001) to address the unbounded weight growth problem observed in all QSNN predator experiments.

### Experimental Results (4 Rounds, 16 Sessions)

| Round | Key Changes | Success | Avg Food | Evasion | Key Finding |
|-------|-----------|---------|----------|---------|-------------|
| PPO-0 | Initial architecture | 0% | 0.52 | 43.5% | Buffer never fills; entropy collapse; theta_h collapse |
| PPO-1 | Cross-ep buffer, entropy=0.08, LR=0.003 | 0% | 0.56 | 35.1% | 100% policy_loss=0 — PPO completely inert |
| PPO-2 | logit_scale=20, entropy decay, motor init | 0% | 0.48 | 25.1% | Motor spike probs at 0.02 not 0.5 |
| PPO-3 | theta_h=pi/2, theta_m=linspace(pi/4,3pi/4) | 0% | 0.77 | 42.5% | Motor probs fixed but policy_loss still 0 |

### Why QSNN-PPO Failed: Architectural Incompatibility

PPO requires the forward pass to produce **parameter-dependent** action probabilities for importance sampling (`ratio = exp(log_pi_new - log_pi_old)`). The QLIF surrogate gradient approach produces:

- **Forward pass**: Returns quantum-measured spike probability — a **constant** independent of current weights
- **Backward pass**: Computes gradient via classical sigmoid surrogate — parameter-dependent

During PPO re-evaluation, `QLIFSurrogateSpike.forward()` returns the cached quantum measurement, so `pi_new(a|s) == pi_old(a|s)` always, `ratio == 1.0` always, `policy_loss == 0` always. This is irreconcilable without replacing the quantum forward pass with a classical approximation.

**REINFORCE works** because it only needs `d(log_prob)/d(theta)` (the backward pass gradient), not a re-evaluated forward pass.

### Actual Impact vs Expected

| Metric | Expected | Actual |
|---|---|---|
| Pursuit predator success | 20-50% | **0%** (16 sessions) |
| Session reliability | 2-3/4 | **0/16** |
| Policy learning | Non-zero policy gradient | **policy_loss=0 in 100% of 600+ updates** |

### Key Technical Discoveries

1. **Motor spike probability suppression**: QLIF motor neurons with small theta_motor produce spike probs ~0.02 (barely firing). Init near `pi/2` is critical — `linspace(pi/4, 3*pi/4)` places neurons in responsive range (0.15-0.85).
2. **theta_hidden init matters**: `pi/2` gives maximum sensitivity (`sin²(pi/4) = 0.5`). Prior init at `pi/4` produced probs ~0.15.
3. **Entropy gradient provides weak learning**: Even with zero policy_loss, entropy regularisation is differentiable through surrogate gradients, creating slow exploration-driven improvement.

### Pivot: QSNNReinforce A2C (Also Halted)

A2C was implemented to preserve REINFORCE's compatibility with surrogate gradients while adding a classical critic for GAE advantage estimation. After 4 rounds (16 sessions, 3,200 episodes), the critic never learned:

- **Explained variance** progressively worsened: 0 → -0.008 → -0.295 → -0.620
- All actor improvement driven by REINFORCE backbone, not critic
- Root causes: partial observability, policy non-stationarity, high return variance, short 20-step GAE windows
- The critic actively **harmed** the actor in late training (Q4 regression in 2/4 A2C-3 sessions)

**Conclusion**: Neither PPO nor A2C is viable with QSNN's surrogate gradient architecture for multi-objective pursuit predator tasks. See logbook 008 and [008-appendix-qsnnreinforce-a2c-optimization.md](../../experiments/logbooks/008-appendix-qsnnreinforce-a2c-optimization.md) for full results.

______________________________________________________________________

## PPO-Q Style PQC Actor

### Overview

Based on the PPO-Q paper (arXiv:2501.07085, Jan 2025), which demonstrated competitive results on 8 RL environments including BipedalWalker using only 2-4 qubits. This replaces the QSNN actor entirely with a parameterized quantum circuit (PQC) wrapped in classical pre/post-processing networks.

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    PPO-Q Style Actor                        │
│                                                             │
│  Sensory Input (8 features)                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────┐                           │
│  │ Classical Pre-Encoder         │                          │
│  │ Linear(8 → 6), Tanh rescale  │  ~54 params               │
│  └──────────┬───────────────────┘                           │
│             ▼                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PQC Actor (6 qubits, 3 data-re-uploading layers)    │   │
│  │                                                      │   │
│  │  For each re-upload layer:                           │   │
│  │    RY(data_i) on each qubit    (data encoding)       │   │
│  │    RX(θ), RY(θ), RZ(θ) on each qubit (trainable)     │   │
│  │    CZ entanglement (nearest-neighbor)                │   │
│  │                                                      │   │
│  │  Measure all 6 qubits → 6 expectation values         │   │
│  │                                                      │   │
│  │  3 layers × 6 qubits × 3 gates = 54 quantum params   │   │
│  │  + 6 final measurement scaling params                │   │
│  │  Total: ~60 quantum params                           │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             ▼                                               │
│  ┌──────────────────────────────┐                           │
│  │ Classical Post-Processor     │                           │
│  │ Linear(6 → 4), action logits │  ~28 params               │
│  └──────────┬───────────────────┘                           │
│             ▼                                               │
│      Softmax → Action (Categorical)                         │
│                                                             │
│  Classical Critic: same as QSNN-PPO (MLP, ~5K params)       │
│  Training: Standard PPO with parameter-shift or adjoint     │
│  Total: ~142 quantum + ~5K classical ≈ 5.1K params          │
└─────────────────────────────────────────────────────────────┘
```

### Why This Could Work

- **Data re-uploading** keeps the circuit shallow (3 layers) while providing universal approximation capability. Coelho et al. (2024) showed re-uploading improves both performance and trainability of VQC-based RL agents.
- **6 qubits with nearest-neighbor CZ** is shallow enough to likely avoid barren plateaus (the safe zone in the literature is O(log n) depth).
- **Classical pre/post-processing** handles dimensionality adaptation efficiently. The quantum circuit focuses on feature transformation, not input/output formatting.
- **PPO-Q demonstrated real results** on BipedalWalker, which is more complex than CartPole-level benchmarks.

### Key Risks

1. **Barren plateaus remain possible** — even with shallow circuits, the parameter-shift rule produces weak gradients. The Cerezo et al. (2025) Nature Comms result suggests shallow circuits avoiding BPs may be classically simulable.
2. **No existing codebase** — this would be built from scratch, unlike QSNN-PPO which builds on existing infrastructure.
3. **No surrogate gradient shortcut** — unlike QSNN, this architecture must use parameter-shift or adjoint differentiation through the quantum circuit, which is computationally expensive (2N circuit evaluations per gradient step for N parameters).

### Comparison with QSNN-PPO

| Aspect | QSNN-PPO | PPO-Q Style |
|---|---|---|
| Quantum core | QLIF spiking neurons | PQC with re-uploading |
| Gradient method | Surrogate (cheap) | Parameter-shift (expensive) |
| Barren plateau risk | None (proven) | Low but possible |
| Existing code | ~80% exists | From scratch |
| Biological analogy | Spiking neural network | Variational circuit |
| Temporal dynamics | Native (spiking) | None (feedforward) |

______________________________________________________________________

## HybridQuantum Brain Architecture

### C.1 Overview

**Status**: Implemented and validated. 96.9% post-convergence on pursuit predators across 4 rounds, 16 sessions. Best quantum architecture for multi-objective tasks. **Classical ablation (HybridClassical)** confirms the architecture and curriculum drive performance, not the QSNN component specifically — a classical MLP reflex achieves 96.3% mean / 97.8% best post-conv.

Combines QSNN (fast reflexes) with a classical cortex MLP (slow strategic decisions) in a hierarchical architecture mimicking biological spinal cord / cortex separation. **Variation B (Pragmatic)** was implemented and succeeded.

Two variations were considered:

- **Variation A (Original)**: Fully quantum — QSNN reflex + VQC/QRC cortex (not pursued)
- **Variation B (Pragmatic, Implemented)**: Quantum reflex + classical cortex — leverages QSNN's proven strengths while using classical MLP for the strategic layer where gradient-based learning is most critical

### C.2a Architecture Diagram (Variation A — Fully Quantum)

```text
                          ┌─────────────────────────────────────┐
                          │         HybridQuantumBrain          │
                          └─────────────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
       ┌─────────────┐            ┌─────────────────┐          ┌──────────────┐
       │   Sensors   │            │  Cortex Module  │          │ Fusion Layer │
       └─────────────┘            │  (VQC or QRC)   │          └──────────────┘
              │                   │  runs every N   │                  │
              │                   │    timesteps    │                  │
              │                   └─────────────────┘                  │
              │                            │                           │
              │                   Options / Gating                     │
              │                            │                           │
              ▼                            ▼                           ▼
       ┌─────────────┐            ┌─────────────────┐          ┌──────────────┐
       │ QSNN Actor  │◄───────────│   Conditioning  │──────────│   Actions    │
       │  (reflexes) │            │   or Gating     │          └──────────────┘
       └─────────────┘            └─────────────────┘
```

### C.2b Architecture Diagram (Variation B — Quantum Reflex + Classical Cortex)

```text
┌─────────────────────────────────────────────────────────────┐
│           Hierarchical Hybrid Brain (Variation B)           │
│                                                             │
│  Sensory Input                                              │
│       │                                                     │
│       ├──────────────────────────┐                          │
│       ▼                          ▼                          │
│  ┌────────────────────┐   ┌──────────────────────────┐      │
│  │ QSNN Reflex Layer  │   │ Classical Cortex (MLP)   │      │
│  │ (runs every step)  │   │ (runs every N steps)     │      │
│  │                    │   │                          │      │
│  │ 8→16→4 QLIF        │   │ MLP: 8→64→64             │      │
│  │ Surrogate gradient │   │ Outputs:                 │      │
│  │ 10 integration     │   │ - 4 action biases        │      │
│  │ steps              │   │ - mode signal            │      │
│  │                    │   │   (forage/evade/explore) │      │
│  │ ~212 quantum params│   │ ~5K classical params     │      │
│  └──────────┬─────────┘   └───────────┬──────────────┘      │
│             │                         │                     │
│             ▼                         ▼                     │
│  ┌───────────────────────────────────────────────┐          │
│  │  Fusion: QSNN_logits + cortex_bias            │          │
│  │  mode=forage: boost food-directed actions     │          │
│  │  mode=evade: boost threat-away actions        │          │
│  │  mode=explore: boost entropy                  │          │
│  └────────────────────┬──────────────────────────┘          │
│                       ▼                                     │
│                Final action selection                       │
│                                                             │
│  Training Curriculum:                                       │
│  Phase 1: Train QSNN on foraging only (proven: 73.9%)       │
│  Phase 2: Freeze QSNN, train cortex with PPO                │
│  Phase 3: Fine-tune jointly (low LR)                        │
│                                                             │
│  Total: ~5.2K params (212 quantum + 5K classical)           │
└─────────────────────────────────────────────────────────────┘
```

**Variation B rationale**: The original design used a VQC/QRC cortex, but experimental results show that gradient-based learning through variational quantum circuits is limited by barren plateaus. The cortex needs to learn strategic switching between objectives — this is exactly where classical MLPs excel. Using a classical cortex where it matters (strategic decisions) and a quantum reflex layer where QSNN is proven (fast sensorimotor responses) plays to each paradigm's strengths.

### C.3 Configuration

```python
class HybridQuantumBrainConfig(BrainConfig):
    name: Literal["hybrid_quantum"] = "hybrid_quantum"

    # Reflex layer (QSNN)
    reflex_type: Literal["qsnn", "qrc"] = "qsnn"
    reflex_config: QSNNBrainConfig | QRCBrainConfig

    # Cortex layer (VQC, QRC, or classical MLP)
    cortex_type: Literal["vqc", "qrc", "mlp"] = "mlp"
    cortex_config: QVarCircuitBrainConfig | QRCBrainConfig | dict

    # Integration
    cortex_update_frequency: int = 10        # Run cortex every N steps
    fusion_strategy: Literal["option", "gating"] = "option"
    num_options: int = 4                      # For option conditioning
```

### C.4 Fusion Strategies

#### Strategy 1: Option Conditioning

Cortex outputs discrete "option" that conditions QSNN behavior:

```python
def option_conditioning(self, cortex_output: np.ndarray) -> int:
    """Cortex selects high-level option."""
    # Cortex outputs option probabilities
    option = np.random.choice(self.num_options, p=softmax(cortex_output))
    return option

def qsnn_with_option(self, brain_params: BrainParams, option: int) -> np.ndarray:
    """QSNN conditioned on cortex option."""
    # Option modifies QSNN parameters
    # e.g., option 0 = "explore", option 1 = "exploit food"
    #       option 2 = "evade predator", option 3 = "follow temperature"

    option_embedding = self.option_embeddings[option]  # Learned embeddings

    # Modify QSNN circuit based on option
    self.qsnn.set_option_bias(option_embedding)

    return self.qsnn.forward(brain_params)
```

#### Strategy 2: Logit Gating

Cortex outputs bias/gain that modulates QSNN action logits:

```python
def logit_gating(self, qsnn_logits: np.ndarray, cortex_output: np.ndarray) -> np.ndarray:
    """Cortex modulates QSNN output via gating."""
    # Cortex outputs gain (multiplicative) and bias (additive)
    gain = cortex_output[:self.num_actions]   # Per-action gain
    bias = cortex_output[self.num_actions:]   # Per-action bias

    # Modulate QSNN logits
    gated_logits = qsnn_logits * sigmoid(gain) + bias

    return gated_logits
```

### C.5 Forward Pass

```python
def forward(self, brain_params: BrainParams) -> np.ndarray:
    """Hierarchical forward pass."""

    # 1. Get QSNN reflex output (every timestep)
    qsnn_logits = self.reflex_module.forward(brain_params)

    # 2. Update cortex periodically
    if self.timestep % self.cortex_update_frequency == 0:
        self.cortex_output = self.cortex_module.forward(brain_params)

    # 3. Fuse based on strategy
    if self.fusion_strategy == "option":
        option = self.option_conditioning(self.cortex_output)
        final_logits = self.qsnn_with_option(brain_params, option)
    else:  # "gating"
        final_logits = self.logit_gating(qsnn_logits, self.cortex_output)

    self.timestep += 1
    return final_logits
```

### C.6 Training Curriculum (Validated)

Three-stage training pipeline — all stages validated experimentally (4 rounds, 16 sessions):

```text
Stage 1: Train QSNN reflexes alone (VALIDATED: 91.0% foraging, 4 sessions)
  - Dense reward shaping, legacy 2-feature mode
  - Surrogate gradient REINFORCE, 2 epochs, 20-step window
  - Tasks: foraging (91.0% success, 99.3% post-convergence best 3/4)
  - QSNN weights saved for stage 2

Stage 2: Freeze QSNN, train cortex (VALIDATED: 91.7% pursuit, 8 sessions)
  - QSNN weights frozen, forward pass provides reflex logits
  - Train cortex with PPO (12 epochs, 512-step buffer, LR schedule)
  - Tasks: pursuit predators (91.7% post-convergence, beats MLP PPO +20.1 pts)
  - Key hyperparameters: LR warmup/decay, mechanosensation module, 12 PPO epochs
  - Cortex weights saved for stage 3

Stage 3: Joint fine-tuning (VALIDATED: 96.9% pursuit, 4 sessions)
  - QSNN unfrozen with 10x lower LR (0.001 vs 0.01)
  - Cortex LR halved from stage 2 (0.0005 vs 0.001)
  - Both REINFORCE + PPO run simultaneously
  - Immediate convergence from pre-trained weights
  - 96.9% post-convergence, +25.3 pts over MLP PPO baseline
```

**Key experimental insights**:

- Stage 2 required LR scheduling (warmup + decay) and 12 PPO epochs to reach ≥90%
- Stage 3 was unambiguously beneficial (+5.2 pts post-conv, -12.6 pts HP death rate)
- Pre-trained initialisation reduced session variance from 8.8 pts to 0.8 pts
- QSNN W_hm grew +27.7% during stage 3 (output mapping adapted) while W_sh stayed stable (+2.7%, input encoding preserved)

______________________________________________________________________

## Data Re-Uploading Enhancement

### D.1 Concept

Data re-uploading encodes classical inputs multiple times into the quantum circuit, interleaved with trainable gates. This dramatically increases expressivity without adding qubits.

Recent empirical evidence (Coelho et al., 2024) shows that data re-uploading improves both **performance** and **trainability** of VQC-based RL agents, and that gradient magnitudes remain substantial throughout training when combined with Deep Q-Learning's moving targets.

### D.2 Implementation

Add to existing QVarCircuitBrain:

```python
class QVarCircuitBrainConfig(BrainConfig):
    # ... existing fields ...
    use_data_reuploading: bool = False
    reupload_layers: int = 3          # Number of re-upload repetitions
```

```python
def build_circuit_with_reuploading(self, features: dict[str, np.ndarray]) -> QuantumCircuit:
    """Build circuit with data re-uploading."""
    qc = QuantumCircuit(self.num_qubits)

    for layer in range(self.reupload_layers):
        # Re-upload data
        for module, qubit_indices in self.module_mapping.items():
            if module in features:
                for i, q in enumerate(qubit_indices):
                    feat = features[module][i] if i < len(features[module]) else 0.0
                    qc.ry(feat * np.pi / 2, q)  # Data encoding

        # Trainable layer
        for q in range(self.num_qubits):
            qc.rx(self.params[f"rx_{layer}_{q}"], q)
            qc.ry(self.params[f"ry_{layer}_{q}"], q)
            qc.rz(self.params[f"rz_{layer}_{q}"], q)

        # Entangling layer
        for q in range(self.num_qubits - 1):
            qc.cz(q, q + 1)

    return qc
```

______________________________________________________________________

## Optimizer Variants

### E.1 Quantum Natural Gradient (QNG)

QNG uses the Fisher information matrix to precondition gradients:

```python
class QNGOptimizer:
    """Quantum Natural Gradient with momentum."""

    def __init__(self, params: np.ndarray, lr: float = 0.01, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = np.zeros_like(params)

    def step(self, circuit_fn, loss_fn):
        """One QNG update step."""
        # Compute Fisher information matrix
        F = self.compute_fisher_matrix(circuit_fn)

        # Compute gradient
        grad = self.compute_gradient(loss_fn)

        # Natural gradient: F^{-1} @ grad
        F_inv = np.linalg.pinv(F + 1e-6 * np.eye(len(self.params)))
        nat_grad = F_inv @ grad

        # Momentum update
        self.velocity = self.momentum * self.velocity + nat_grad
        self.params -= self.lr * self.velocity

    def compute_fisher_matrix(self, circuit_fn) -> np.ndarray:
        """Compute quantum Fisher information matrix."""
        n = len(self.params)
        F = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                # Use parameter shift for Fisher matrix elements
                F[i, j] = self.fisher_element(circuit_fn, i, j)
                F[j, i] = F[i, j]

        return F
```

### E.2 Modified Conjugate QNG (CQNG)

```python
class CQNGOptimizer(QNGOptimizer):
    """Conjugate QNG with dynamic hyperparameter adjustment."""

    def step(self, circuit_fn, loss_fn):
        # Compute natural gradient
        nat_grad = self.compute_natural_gradient(circuit_fn, loss_fn)

        # Conjugate direction
        if self.prev_grad is not None:
            beta = self.compute_beta(nat_grad, self.prev_grad)  # Fletcher-Reeves or Polak-Ribière
            self.direction = nat_grad + beta * self.prev_direction
        else:
            self.direction = nat_grad

        # Line search for optimal step size
        alpha = self.line_search(circuit_fn, loss_fn, self.direction)

        # Update
        self.params -= alpha * self.direction

        self.prev_grad = nat_grad
        self.prev_direction = self.direction
```

______________________________________________________________________

## Barren Plateau Mitigation

### F.0 The Trainability-Advantage Dilemma

A fundamental theoretical tension exists between trainability and quantum advantage (Cerezo et al., Nature Communications, 2025):

> Many commonly used models whose loss landscapes avoid barren plateaus can also be classically simulated. This means trainability and quantum advantage may be fundamentally in tension for variational quantum circuits.

**Practical implications for this project:**

- Shallow circuits (O(log n) depth) with local measurements avoid barren plateaus but may not provide genuine quantum advantage over classical networks of equivalent size
- Deep, expressive circuits that cannot be classically simulated likely hit barren plateaus, preventing gradient-based training
- **QSNN's surrogate gradient approach may sidestep this dilemma** — the quantum circuit provides the forward dynamics (genuine quantum measurement), while the backward pass is classical (no barren plateau). Whether this constitutes "quantum advantage" depends on whether the quantum measurement distribution provides information that a classical sigmoid cannot

### F.1 Neural Network-Generated Initialization

Use a classical neural network to generate good initial parameters:

```python
class ParameterGenerator(nn.Module):
    """Generate quantum circuit parameters to avoid barren plateaus."""

    def __init__(self, input_dim: int, param_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim),
            nn.Tanh()  # Output in [-1, 1], scale to [-π, π]
        )

    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """Generate parameters conditioned on task."""
        return self.net(task_embedding) * np.pi
```

### F.2 Beta Distribution Initialization

```python
def beta_initialization(num_params: int, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
    """Initialize parameters from beta distribution to avoid saddle points."""
    # Beta(0.5, 0.5) concentrates mass near 0 and 1, avoiding saddle points
    samples = np.random.beta(alpha, beta, num_params)
    # Scale to [-π, π]
    return (samples - 0.5) * 2 * np.pi
```

### F.3 Structured Initialization (Lie-algebraic)

```python
def lie_algebraic_initialization(circuit_structure: dict) -> np.ndarray:
    """Initialize based on Lie algebra decomposition."""
    params = []
    for gate_type, count in circuit_structure.items():
        if gate_type in ['rx', 'ry', 'rz']:
            # Small rotations for single-qubit gates
            params.extend(np.random.normal(0, 0.1, count))
        elif gate_type in ['crx', 'cry', 'crz']:
            # Larger rotations for entangling gates
            params.extend(np.random.normal(0, 0.5, count))
    return np.array(params)
```

### F.4 Strategies Proven Effective in This Project

Based on QSNN experimental results:

1. **Surrogate gradient bypass**: Use quantum circuits for forward pass only; compute backward pass with classical surrogates. Avoids parameter-shift rule entirely. Gradient magnitude ~1,000x larger than Hebbian, ~100x larger than parameter-shift on equivalent circuits.

2. **Fan-in-aware scaling**: `tanh(w*x / sqrt(fan_in))` keeps tanh in responsive regime regardless of layer width. Without this, gradient death occurs within 10-20 episodes for hidden layers wider than ~10 neurons.

3. **Multi-timestep averaging**: Running 10 quantum integration steps per decision reduces shot noise by ~sqrt(10). Critical for stable learning — 5 timesteps: 52.6% vs 10 timesteps: 73.9% on foraging.

4. **Adaptive entropy regulation**: Two-sided system with floor boost and ceiling suppress prevents both entropy collapse and entropy lock. Required for policy crystallization on harder tasks.

______________________________________________________________________

## External Research Survey (2024-2026)

### State of Quantum RL with Gradient-Based Online Learning

No published work as of February 2026 demonstrates quantum RL with gradient-based online learning matching classical baselines on tasks approaching the complexity of multi-objective predator pursuit in a grid world. The most relevant findings:

### PPO-Q (arXiv:2501.07085, Jan 2025)

Hybrid quantum-classical PPO with pre-encoding NN + PQC + post-processing NN. Used 2-4 qubits, 1-3 variational layers, 32-152 total actor parameters.

- **Results**: "Comparable performance" to classical PPO on 8 environments including BipedalWalker
- **Tested on real hardware**: Dongling superconducting chip (105 qubits)
- **Limitation**: All tested environments are single-objective
- **Relevance**: Architecture template for the PPO-Q Style PQC Actor proposal above

### hDQNN-TD3 (arXiv:2503.09119, Mar 2025)

10-qubit, 10-layer PQC embedded in classical TD3. Uses a classical "tangential DNN" (qtDNN) to locally approximate the PQC for backpropagation, avoiding gradient flow through the quantum circuit entirely.

- **Results**: 6,011 return on Humanoid-v4 (best seed), vs 5,306 classical TD3
- **Catch**: High seed-to-seed variance; qtDNN requires 2^(N+1) neurons, scaling exponentially with qubit count
- **Relevance**: Demonstrates that avoiding quantum backpropagation (similar to QSNN's surrogate approach) enables competitive results

### Quantum Advantage Actor-Critic (arXiv:2401.07043, Jan 2024)

Pure quantum actor-critic with VQC for both actor and critic.

- **Results**: Pure quantum A2C **failed completely** on CartPole — could not learn across any runs. Average gradients: approximately -0.000056 (vanishing). Hybrid HA2C (VQC + classical post-processing) succeeded.
- **Relevance**: Strong evidence that pure quantum actor-critic is not viable. Classical scaffolding is essential.

### VQC-Based RL with Data Re-uploading (Springer, 2024)

Data re-uploading in VQC-based Deep Q-Learning.

- **Results**: Re-uploading improves both performance and trainability. Gradient magnitudes remain substantial due to DQL's moving targets. Increasing qubit count does NOT lead to exponential gradient vanishing.
- **Relevance**: Strongest evidence that data re-uploading + DQL can maintain gradient health. May not transfer to policy gradient methods.

### Dissecting QRL (arXiv:2511.17112, Nov 2025)

Systematic evaluation of QRL components using SimplyQRL framework.

- **Key findings**:
  - Data re-uploading helps trainability but effectiveness depends on embedding style
  - Entanglement can **hurt** — stronger entanglement "can degrade optimization" and cause complete learning failure
  - Output reuse consistently improves hybrid agents
  - "Trainability in hybrid QRL arises from a delicate balance of interdependent components"

### Benchmarking QRL (ICML 2025, arXiv:2502.04909)

Most rigorous QRL evaluation to date, tested on gridworld environments.

- **Conclusion**: Previous claims of QRL superiority are "based on insufficient statistical evaluation." "It is still uncertain if QRL can show any advantage over classical RL beyond artificial problem formulations."
- **Relevance**: Directly relevant — tested on gridworld environments similar to our nematode simulation.

### Barren Plateau Trainability Dilemma (Nature Communications, 2025)

Cerezo et al. establish that many models whose loss landscapes avoid barren plateaus can also be classically simulated.

- **Implication**: There may be a fundamental trade-off between trainability and quantum advantage for variational quantum circuits
- **Relevance**: Reinforces the value of QSNN's surrogate gradient approach, which sidesteps this dilemma by using quantum measurement for dynamics and classical computation for gradients

### Hybrid QRL in Latent Spaces (Springer, 2025)

Classical autoencoder compresses high-dimensional observations to a low-dimensional latent space tailored for a quantum agent.

- **Relevance**: Addresses the input dimensionality problem — quantum circuits work with few inputs, but multi-objective tasks require rich state representations. A classical encoder could compress grid world state into quantum-friendly representation.

### SQDR-CNN: Spiking-Quantum Data Re-uploading CNN (arXiv:2512.03895, Dec 2025)

Parameter-efficient hybrid spiking-quantum CNN with surrogate gradients and quantum data re-uploading. Proves that surrogate gradients work end-to-end with multi-layer data re-uploading circuits, not just single-layer QLIF.

- **Results**: Achieves 86% of top SNN baseline accuracy on MNIST-family benchmarks using 4-8 qubits with 3-6 re-uploading layers, with only 0.5% of the smallest SNN baseline's parameter count
- **Key finding**: Surrogate gradient backward pass scales to deeper circuits without gradient degradation when combined with data re-uploading
- **Relevance**: Validates our surrogate gradient approach for deeper quantum circuits. Could enable data re-uploading QLIF with more expressive multi-layer circuits

### QKAN-LSTM: Quantum-Inspired Activations in Temporal Networks (arXiv:2512.05049, Dec 2025)

Replaces classical activation functions in LSTM gates with quantum-inspired Kolmogorov-Arnold Network (KAN) activations (DARUAN modules — single-qubit data re-uploading circuits executable on classical hardware without entanglement). Achieves 79% parameter reduction for equivalent performance on time-series tasks.

- **Results**: Achieves 79% parameter reduction vs classical LSTM on physics simulation (Damped SHM, Bessel Function) and urban telecom forecasting benchmarks
- **Key innovation**: Quantum-inspired variational activations (classically simulated single-qubit data re-uploading circuits) replace tanh/sigmoid in forget and input gates, providing richer Fourier spectral representation
- **Relevance**: Partially applicable — demonstrates the LSTM gate activation replacement pattern and parameter efficiency gains. **Note**: QKAN-LSTM uses classical simulation of quantum-inspired circuits; deploying actual QLIF quantum circuits per gate activation would have substantially higher per-step cost (one circuit execution per hidden unit per gate per timestep). Quantum advantage over classical simulation is not established by this paper.

### Structured Quantum Reservoirs (Multiple Sources, 2024-2025)

Multiple recent studies on quantum reservoir computing show that reservoir topology significantly affects performance, with structured (non-random) topologies outperforming random circuits.

- **Key findings**: Homogeneous 1D Bose-Hubbard chains with open boundaries outperform periodic and all-to-all topologies — open boundaries break translational symmetry, generating diverse features without requiring disorder (Llodrà et al., "QRC in atomic lattices", *Chaos, Solitons & Fractals*, 2025). Intermediate-regularity graphs outperform both fully random and fully connected networks (Ivaki et al., "QRC on random regular graphs", *Phys. Rev. A*, 2025). Two-qubit correlations (ZZ observables) as readout features enhance performance over single-qubit measurements by accessing higher-dimensional Hilbert space structure (Martínez-Peña et al., "Role of coherence in many-body QRC", *Commun. Phys.*, 2024).
- **Design principle**: Reservoir topology should encode domain-specific inductive biases (e.g., symmetries, locality). The optimal topology is task-dependent — no universal best structure exists.
- **Relevance**: Explains our QRC failure — random reservoir angles produced non-discriminative representations. Structured reservoirs with C. elegans-inspired connectivity could revive the QRC approach. Quantitative advantage of structured vs random topology for RL feature extraction is an open question to be validated in H.1.

______________________________________________________________________

## Next-Generation Architecture Proposals

Based on 200+ experiment sessions across 62 days of evaluation, combined with the latest external research (2025-2026), four next-generation architectures are proposed. Each is designed to address the barren plateau-advantage dilemma through a different strategy, and each includes falsification criteria for rapid go/no-go decisions.

### H.1 Quantum Reservoir Hybrid (QRH) — Priority 1

**Strategy**: Don't train the quantum part
**Risk**: Medium | **Estimated effort**: 2-3 weeks

#### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│               Quantum Reservoir Hybrid (QRH)                │
│                                                             │
│  Sensory Input (8 features)                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Structured Quantum Reservoir (FIXED, not trained)   │   │
│  │                                                      │   │
│  │  8-12 qubits with C. elegans-inspired topology:      │   │
│  │  - Gap junctions → CZ connectivity (symmetric)       │   │
│  │  - Chemical synapses → fixed RY/RZ rotations         │   │
│  │  - Hub-and-spoke matching interneuron connectivity   │   │
│  │                                                      │   │
│  │  Input: RY(feature_i * π) on sensory qubits          │   │
│  │  3 entanglement layers with structured angles        │   │
│  │                                                      │   │
│  │  Output: 2^N measurement probabilities               │   │
│  │  (or subset: per-qubit expectations + correlations)  │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Classical Readout + Cortex (TRAINED with PPO)       │   │
│  │                                                      │   │
│  │  Reservoir features → MLP(64, 64) → action logits    │   │
│  │  + Value head for critic                             │   │
│  │  Standard PPO training (no quantum gradients needed) │   │
│  │                                                      │   │
│  │  ~5K classical params                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Key difference from failed QRC:                            │
│  - STRUCTURED topology (not random angles)                  │
│  - Biologically-inspired connectivity                       │
│  - PPO training on readout (not REINFORCE)                  │
│  - Richer feature extraction (correlations, not just probs) │
└─────────────────────────────────────────────────────────────┘
```

#### Why This Could Work

- **Avoids barren plateaus entirely** — no quantum gradient training required
- Our QRC failed because of *random* circuits; recent QRC research (Llodrà et al., 2025; Ivaki et al., 2025) shows structured topologies consistently outperform random circuits across benchmarks
- **Reservoir computing has biological plausibility** — cortical microcircuits in real brains function as reservoirs
- Classical readout handles optimization with standard PPO, quantum reservoir provides the feature space
- **C. elegans connectivity data is available** — the 302-neuron connectome is fully mapped, providing biologically grounded circuit topology

#### Key Design Decisions

1. **Reservoir circuit topology**: Use C. elegans sensory-interneuron connectivity as template. Gap junctions → CZ gates (symmetric bidirectional coupling, matching gap junction symmetry), chemical synapses → parameterized RY/RZ with angles derived from synapse strength ratios
2. **Feature extraction**: Per-qubit Z-expectations + pairwise ZZ-correlations (richer than full probability distribution, scales as O(N²) not O(2^N))
3. **Readout**: MLP (2 hidden layers, 64 units) trained with PPO — uses proven classical training pipeline
4. **Reservoir size**: 8-12 qubits matching sensory neuron count in the simulation

#### Decision Gate (Week 1)

- **Reservoir feature analysis**: Do structured reservoir outputs show higher mutual information with optimal actions than random reservoir outputs?
- If MI(structured) ≤ MI(random): Architecture is not viable, stop
- If MI(structured) > MI(random) but < MI(classical_MLP_features): Proceed with caution, may need topology refinement
- If MI(structured) > MI(classical_MLP_features): Strong go-ahead — proceed with full PPO readout training

#### Falsification Criteria

- Must achieve ≥80% foraging success (QSNN surrogate baseline: 73.9%; HybridQuantum Stage 1 baseline: 91.0%)
- Reservoir features must show statistically significant difference from random features (p < 0.01 on permutation test)
- Must demonstrate that structured topology outperforms random topology on feature quality metric

### H.2 SQS-QLIF Hybrid — Priority 2

**Strategy**: Local learning rules (quantum STDP)
**Risk**: High | **Estimated effort**: 4-6 weeks

#### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     SQS-QLIF Hybrid                         │
│                                                             │
│  Sensory Input (8 features)                                 │
│       │                                                     │
│       ├──────────────────────────┐                          │
│       ▼                          ▼                          │
│  ┌────────────────────┐   ┌──────────────────────────┐      │
│  │ SQS Reflex Layer   │   │ Classical Cortex (MLP)   │      │
│  │ (replaces QLIF)    │   │ (PPO-trained)            │      │
│  │                    │   │                          │      │
│  │ Multi-qubit SQS    │   │ MLP: 8→64→64             │      │
│  │ neurons:           │   │ Mode-gated fusion        │      │
│  │ 2-3 qubits/neuron  │   │ ~5K classical params     │      │
│  │ Quantum memory     │   │                          │      │
│  │ (entanglement      │   └───────────┬──────────────┘      │
│  │  across timesteps) │               │                     │
│  │                    │               │                     │
│  │ Local learning:    │               │                     │
│  │ Quantum STDP +     │               │                     │
│  │ eligibility traces │               │                     │
│  │                    │               │                     │
│  │ ~300-500 quantum   │               │                     │
│  │ params             │               │                     │
│  └──────────┬─────────┘               │                     │
│             │                         │                     │
│             ▼                         ▼                     │
│  ┌───────────────────────────────────────────────┐          │
│  │  Mode-gated fusion (same as HybridQuantum)    │          │
│  └───────────────────────────────────────────────┘          │
│                                                             │
│  Key innovation:                                            │
│  - Multi-qubit neurons with quantum memory                  │
│  - Local learning avoids barren plateaus by construction    │
│  - Richer representations than single-qubit QLIF            │
└─────────────────────────────────────────────────────────────┘
```

#### SQS Neuron Circuit (per neuron)

```text
Per SQS neuron (3 qubits: q0=membrane, q1=memory, q_anc=readout ancilla):
  Timestep t:
  |ψ_t⟩ = U_readout · U_memory · U_input · |ψ_{t-1}⟩

  U_input:   RY(θ + f(w·x)) on q0           (membrane potential)
  U_memory:  CZ(q0, q1), RZ(φ) on q1        (memory coupling — CZ is symmetric)
  U_readout: CNOT(q0 → q_anc)               (copy spike to fresh ancilla)
  Measurement: Measure q_anc (spike output), reset q0 for next timestep
               q1 remains coherent (not entangled with measured qubit)

  Note: Measuring q0 directly would collapse the entangled q0-q1 state,
  destroying quantum memory. The ancilla-based readout decouples the spike
  signal from the memory register, preserving q1's quantum coherence across
  timesteps. Verify this matches arXiv:2506.21324's measurement scheme
  during implementation.

  Learning: Quantum STDP
  Δw ∝ ⟨pre_spike · post_spike⟩ × eligibility_trace × reward_signal
  Local rule — no global gradient computation needed
```

#### Why This Could Work

- **SQS neurons explicitly model quantum memory** (entanglement between timesteps), addressing our finding that multi-timestep integration is critical (5 steps: 52.6% → 10 steps: 73.9%)
- **Local learning rules avoid barren plateaus by construction** — no global loss landscape to get stuck in
- **Multi-qubit neurons provide richer representations** than single-qubit QLIF — each neuron has 2-3 qubits with internal entanglement
- **Biological plausibility**: STDP is the dominant learning rule in real C. elegans neurons
- Reuses the proven HybridQuantum architecture and three-stage curriculum

#### Key Design Decisions

1. **SQS neuron implementation**: Follow arXiv:2506.21324 with adaptations for Qiskit Aer. Each neuron: 3 qubits (membrane + memory + readout ancilla). Ancilla-based measurement decouples spike readout from memory qubit to preserve quantum coherence across timesteps. Verify paper's measurement scheme during implementation — if the paper uses a different approach (e.g., weak measurement, mid-circuit reset), adapt accordingly
2. **Learning rule**: Quantum STDP with eligibility traces, compatible with REINFORCE outer loop for the cortex
3. **Network topology**: Small-world network matching C. elegans connectivity statistics (clustering coefficient ~0.28, path length ~2.65)
4. **Integration**: Replace QLIF layer in `_qlif_layers.py` with SQS layer; keep cortex and fusion infrastructure unchanged

#### Decision Gate (Week 2)

- **Single SQS neuron characterization**: Does a single SQS neuron show richer input-output mapping than QLIF?
- Measure information capacity: bits of mutual information between input and output per neuron
- If capacity(SQS) ≤ capacity(QLIF): Reconsider neuron design
- If quantum STDP learning rule doesn't converge on XOR classification: Architecture is not viable

#### Falsification Criteria

- Must demonstrate quantum memory effect: performance with entanglement preservation > without (controlled ablation)
- Must achieve ≥75% foraging success within 500 episodes
- Training time must be < 10x classical equivalent (SQS overhead from multi-qubit circuits)

### H.3 Entangled QLIF with qtDNN Surrogate — Priority 3

**Strategy**: Classical gradient surrogates for entangled circuits
**Risk**: Medium-High | **Estimated effort**: 3-4 weeks

#### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│            Entangled QLIF + qtDNN Surrogate                 │
│                                                             │
│  Sensory Input (8 features)                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Entangled QLIF Layer                                │   │
│  │                                                      │   │
│  │  8-16 QLIF neurons WITH entanglement:                │   │
│  │  RY(θ + f(w·x)) → CNOT(pairs) → RX(leak) → Measure   │   │
│  │                                                      │   │
│  │  Key change: CNOT/CZ gates between neuron qubits     │   │
│  │  (current QLIF has ZERO entanglement — all single-   │   │
│  │   qubit circuits)                                    │   │
│  │                                                      │   │
│  │  Forward: Quantum measurement (same as current)      │   │
│  │  Backward: qtDNN surrogate (replaces sigmoid surr.)  │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                               │
│             │ spike probabilities                           │
│             ▼                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  qtDNN (Classical Tangential Network)                │   │
│  │                                                      │   │
│  │  Learns to approximate: θ_quantum → ∂L/∂θ_quantum    │   │
│  │  Architecture: MLP(N_params → 64 → 64 → N_params)    │   │
│  │                                                      │   │
│  │  Trained online: minimize ‖qtDNN(θ) - PSR_grad(θ)‖   │   │
│  │  (periodically calibrated against parameter-shift)   │   │
│  │                                                      │   │
│  │  Once trained, provides gradients ~100x faster       │   │
│  │  than parameter-shift rule                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Training loop:                                             │
│  1. Forward: entangled quantum circuit → measurements       │
│  2. Backward: qtDNN provides approximate gradients          │
│  3. Every K steps: calibrate qtDNN against true PSR grads   │
│  4. REINFORCE outer loop with qtDNN-provided inner grads    │
│                                                             │
│  Cortex: Same classical PPO cortex as HybridQuantum         │
└─────────────────────────────────────────────────────────────┘
```

#### Why This Could Work

- **Entanglement is the key quantum resource we're not using** — current QLIF has zero entanglement (all single-qubit circuits). Adding CNOT/CZ gates between neurons creates correlated spike patterns that cannot be reproduced classically
- **qtDNN surrogate** (arXiv:2503.09119) enables training circuits that would otherwise have barren plateaus, by approximating quantum gradients with a classical network
- **Separates advantage source from training mechanism** — entangled circuit provides the quantum dynamics, classical qtDNN handles the optimization
- Our existing surrogate gradient infrastructure (`QLIFSurrogateSpike`) provides a natural starting point — qtDNN replaces the fixed sigmoid surrogate with a learned approximation

#### Key Design Decisions

1. **Entanglement topology**: Start with linear nearest-neighbor CNOT chain (hardware-friendly). Compare against all-to-all and C. elegans-inspired connectivity
2. **qtDNN architecture**: MLP (N_params → 64 → 64 → N_params) trained to predict parameter-shift gradients. Calibrated every 50 episodes against true PSR gradients. **Note**: The original hDQNN-TD3 paper uses 2^(N+1) neurons (exponential in qubit count) for high-fidelity approximation. Our fixed-width architecture is a deliberate cost reduction — the decision gate's 0.5 correlation threshold (see below) serves as the empirical validation for whether this simplification retains sufficient gradient quality
3. **Circuit depth**: Start shallow — 2 layers of `[RY(data) → CNOT(pairs) → RX(leak) → RZ(phase)]`. Data re-uploading between layers
4. **Measurement strategy**: Measure all qubits (spike probabilities). Correlations between measured qubits encode entanglement effects

#### Decision Gate (Week 1)

- **qtDNN gradient quality**: Does qtDNN gradient correlation with true parameter-shift gradient exceed 0.5?
- If correlation < 0.3: qtDNN approach not viable for our entangled circuits
- If entangled circuit expressibility (measured by distribution uniformity) ≤ product-state circuit: No advantage from entanglement, stop

#### Falsification Criteria

- Entangled version must outperform non-entangled QLIF by ≥5% on at least one environment (foraging or pursuit predators)
- qtDNN training overhead must be < 3x forward pass cost (amortized over calibration interval)
- Must not exhibit barren plateau symptoms: gradient variance must not decay exponentially with qubit count (test at 4, 8, 12, 16 qubits)

### H.4 QKAN-QLIF Temporal Brain — Priority 4

**Strategy**: Quantum activations in classical temporal architecture
**Risk**: Low-Medium | **Estimated effort**: 2-3 weeks

#### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│               QKAN-QLIF Temporal Brain                      │
│                                                             │
│  Sensory Input (8 features)                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  QLIF-LSTM Layer                                     │   │
│  │                                                      │   │
│  │  Standard LSTM structure with quantum activations:   │   │
│  │                                                      │   │
│  │  Forget gate:  f_t = QLIF(W_f · [h_{t-1}, x_t])      │   │
│  │  Input gate:   i_t = QLIF(W_i · [h_{t-1}, x_t])      │   │
│  │  Cell update:  c̃_t = tanh(W_c · [h_{t-1}, x_t])      │   │
│  │  Output gate:  o_t = σ(W_o · [h_{t-1}, x_t])         │   │
│  │                                                      │   │
│  │  Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       │   │
│  │  Hidden state: h_t = o_t ⊙ tanh(c_t)                 │   │
│  │                                                      │   │
│  │  QLIF activation replaces sigmoid in f_t and i_t:    │   │
│  │  RY(θ + tanh(w·x/√fan_in)·π) → RX(leak) → Measure    │   │
│  │  Forward: quantum measurement probability            │   │
│  │  Backward: sigmoid surrogate (proven approach)       │   │
│  │                                                      │   │
│  │  Hidden dim: 16-32                                   │   │
│  │  ~200-400 quantum params + ~1K classical params      │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Action Head + Value Head                            │   │
│  │  Linear(hidden_dim → 4) for actions                  │   │
│  │  Linear(hidden_dim → 1) for critic                   │   │
│  │  PPO training                                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Key advantage:                                             │
│  - Adds temporal memory (roadmap Phase 3 requirement)       │
│  - QKAN-LSTM showed 79% param reduction with quantum-       │
│    inspired activations; actual QLIF may differ             │
│  - Builds on proven QLIF + surrogate gradient pipeline      │
│  - Low risk: falls back to classical LSTM if quantum fails  │
│                                                             │
│  Key cost consideration:                                    │
│  - One quantum circuit per hidden unit per gate per step    │
│  - Hidden dim 16, 2 quantum gates → 32 circuits/timestep   │
│  - Mitigated by batched Aer execution + small circuits      │
└─────────────────────────────────────────────────────────────┘
```

#### Why This Could Work

- **Architectural pattern validated**: QKAN-LSTM (arXiv:2512.05049) demonstrated 79% parameter reduction by replacing LSTM gate activations with quantum-inspired (classically simulated) variational circuits. Our H.4 extends this to *actual* QLIF quantum circuits — the parameter efficiency may differ, but the gate replacement pattern is proven
- **Minimal quantum circuit** (single-qubit per activation), avoiding barren plateaus by design
- **Adds temporal memory** — addresses roadmap Phase 3 requirement for short-term/intermediate-term activity memory (STAM/ITAM)
- **Builds directly on proven infrastructure** — uses the same QLIF neuron and surrogate gradient backward pass that achieved 73.9% foraging success
- **Low risk**: if quantum activations don't help, the architecture gracefully degrades to a standard LSTM with slightly different activations
- **Computational cost caveat**: Unlike QKAN-LSTM's classical simulation, actual QLIF circuits require one quantum circuit execution per hidden unit per quantum gate per timestep (hidden_dim=16 × 2 gates = 32 circuits per step). Mitigated by batched Aer execution and minimal circuit depth (2 gates per QLIF)

#### Key Design Decisions

1. **Which gates get quantum activations**: Start with forget gate (f_t) and input gate (i_t) — most critical for memory gating. Keep cell update (tanh) and output gate (sigmoid) classical initially
2. **QLIF integration**: Use existing `QLIFSurrogateSpike` autograd function. Each gate activation = one QLIF neuron per hidden unit
3. **Memory horizon**: Target 10-50 timestep memory. The nematode simulation runs at ~1 step/decision, so this covers 10-50 prior decisions
4. **Training**: Standard BPTT through LSTM structure, with QLIF surrogate gradients in the backward pass for quantum gate activations. PPO for the overall policy

#### Decision Gate (Week 1)

- Does QLIF activation in LSTM gate produce smoother/more expressive gating than classical sigmoid?
- Measure gate activation distributions: QLIF should show richer structure (bimodal, shot-noise-induced exploration)
- If parameter count reduction < 30% for equivalent performance: Limited practical benefit
- If training is unstable (loss divergence within 100 episodes): Need gradient clipping or different gate selection

#### Falsification Criteria

- Must match classical LSTM performance on foraging (≥80% success)
- Must show parameter reduction ≥30% for equivalent performance
- Must demonstrate meaningful temporal memory: performance on tasks requiring recall of past observations > memoryless baseline (e.g., remembering predator location after it leaves viewport, or leveraging food gradient history for more efficient search). Note: a dedicated sequential task environment may need to be added to properly evaluate this criterion

### H.5 Multi-Objective & Sensory Extensibility

All four proposals are designed for multi-objective learning (foraging + predator evasion + thermotaxis) and extensible to future sensory modalities:

| Architecture | Multi-Objective Mechanism | Sensory Extensibility |
|---|---|---|
| H.1 QRH | Classical PPO readout learns objective trade-offs (same as proven HybridQuantum cortex) | Add qubits or multiplex encoding; readout MLP scales naturally |
| H.2 SQS-QLIF | Mode-gated fusion delegates objectives to reflex (fast) vs cortex (strategic) | Add SQS neurons to small-world network for new modalities |
| H.3 Entangled QLIF + qtDNN | Entangled spike correlations encode active objective; cortex PPO handles switching | Add QLIF neurons + entanglement edges for new inputs |
| H.4 QKAN-QLIF | LSTM memory tracks objective context over time; PPO trains end-to-end | Widen LSTM input dimension (standard approach) |

**Current sensory modules**: food_chemotaxis, nociception, thermotaxis, mechanosensation, proprioception

**Planned future modalities** (roadmap Phase 4+): oxygen gradients (aerotaxis), osmolarity sensing, pheromone/social signalling, noxious chemical avoidance. These require only adding sensory modules in config and increasing the input feature dimension — no architectural changes to the quantum components.

### H.6 Implementation Roadmap

```text
Note: Scheduling follows risk level (lowest-risk architectures first to
maximize early learnings), not priority order. Priority reflects expected
scientific value if all risks were equal.

Week 1-2:  QRH (H.1) — Structured reservoir experiments
           - Build C. elegans-inspired reservoir circuit
           - Mutual information analysis vs random reservoir
           - Decision gate at end of Week 1

Week 3:    QRH results analysis + QKAN-QLIF (H.4) prototype
           - H.1 go/no-go decision
           - Begin QLIF-LSTM implementation

Week 4-5:  Entangled QLIF + qtDNN (H.3) — regardless of H.1 outcome
           (accelerate if H.1 failed; run in parallel if H.1 proceeding)
           - QKAN-QLIF (H.4) full evaluation
           - Entangled circuit design + qtDNN training

Week 6-8:  SQS-QLIF (H.2) if earlier architectures show promise
           - Cross-architecture comparison
           - Final architecture selection

Decision Gates:
  After Week 2: Is QRH viable?
    → If yes: proceed with QRH as primary, H.3/H.4 as secondary
    → If no: accelerate H.3 and H.4

  After Week 4: Do any architectures show quantum advantage?
    → If yes: focus resources on best candidate
    → If no: revisit problem formulation — consider quantum-native
      environments or redefine "advantage" (parameter efficiency,
      biological fidelity, hardware readiness)

  After Week 6: Final architecture selection for Phase 2 deliverable
```

______________________________________________________________________

## Benchmarking Plan

### G.1 Current Benchmark Results

| Architecture | Foraging | Predator (random) | Predator (pursuit) | Training Method |
|---|---|---|---|---|
| QVarCircuit (CMA-ES) | 99.8% ± 0.6% | 76.1% ± 2.1% | Not tested | Evolutionary |
| QVarCircuit (gradient) | ~40% | Not tested | Not tested | Parameter-shift |
| QSNN (surrogate) | 73.9% | 22.3% avg | 0% (60 sessions) | Surrogate gradient |
| QRC | 0% | 0% | Not tested | REINFORCE (readout) |
| **HybridQuantum** | **91.0%** | Not tested | **96.9%** | **Surrogate + PPO** |
| HybridClassical (ablation) | 97.0% | Not tested | 96.3% (mean) / 97.8% (best) | Backprop + PPO |
| MLPReinforce | 95.1% ± 1.9% | 73.4% ± 10.9% | Not tested | REINFORCE |
| MLPPPOBrain | 96.7% ± 1.3% | — | 71.6% (unified) / 94.5% (legacy) | PPO |
| SpikingReinforce | 73.3%\* | ~61%\* | Not tested | Surrogate gradient |

\*Best session only; ~90% of sessions fail

### G.2 New Benchmark Categories

Add to existing benchmark system:

```yaml
# benchmarks/categories.yml additions

qsnn_foraging_small:
  brain_type: qsnn
  environment: foraging
  grid_size: [10, 20]
  metrics: [success_rate, steps_to_food, convergence_episode]

qsnn_predator_small:
  brain_type: qsnn
  environment: predator
  grid_size: [10, 20]
  metrics: [survival_rate, evasion_latency, food_collected]

qrc_foraging_small:
  brain_type: qrc
  environment: foraging
  grid_size: [10, 20]
  metrics: [success_rate, steps_to_food, convergence_episode]

hybrid_quantum_predator_small:
  brain_type: hybrid_quantum
  environment: predator
  grid_size: [10, 20]
  metrics: [survival_rate, strategic_score, reflex_accuracy]
```

### G.3 Evaluation Metrics

**Reflex Performance** (QSNN focus):

- Evasion latency: timesteps from predator detection to evasion action
- Collision rate: wall/obstacle collisions per episode
- Thermotaxis slope following: accuracy of following temperature gradient

**Strategic Performance** (VQC/QRC cortex focus):

- Food intake vs risk tradeoff: Pareto frontier analysis
- Long-horizon reward: discounted cumulative reward over 500+ steps
- Option consistency: how often cortex maintains coherent strategy

**Hybrid Performance**:

- Fusion efficiency: improvement over best individual component
- Adaptation speed: episodes to adapt cortex to new environment

### G.4 Quantum Advantage Metrics (Next-Generation Proposals)

**Quantum Feature Quality** (QRH focus):

- Mutual information: MI(reservoir_features, optimal_action) — must exceed MI(random_features, optimal_action)
- Feature discriminability: KL divergence between feature distributions for different sensory contexts
- Classical simulability test: Can a classical network of equivalent parameter count match reservoir feature quality?

**Quantum Memory Effect** (SQS-QLIF focus):

- Memory ablation: Performance delta with/without entanglement preservation across timesteps
- Temporal correlation: Autocorrelation of spike patterns — quantum memory should show longer-range correlations
- Classical memory comparison: SQS temporal features vs classical RNN hidden states of equivalent dimension

**Gradient Quality** (Entangled QLIF + qtDNN focus):

- qtDNN fidelity: Pearson correlation between qtDNN-predicted gradient and true parameter-shift gradient
- Gradient variance scaling: Measure gradient variance vs qubit count — must not decay exponentially (barren plateau test)
- Training efficiency: Episodes to convergence with qtDNN vs parameter-shift vs sigmoid surrogate

**Parameter Efficiency** (QKAN-QLIF focus):

- Performance per parameter: Success rate / total parameter count, compared to classical LSTM equivalent
- Gate activation richness: Entropy of gate activation distributions — quantum gates should show richer structure

**Temporal Memory Capacity** (SQS-QLIF and QKAN-QLIF shared):

- Memory capacity: Maximum sequence length that can be reliably recalled
- Temporal credit assignment: Performance on tasks requiring decisions based on observations N steps ago

**Quantum Advantage Metric** (all architectures):

- Controlled ablation: For each quantum component, replace with classical equivalent of equal parameter count and measure performance delta
- Task-performance advantage: ΔSuccess = Success(quantum) - Success(classical_equivalent)
- Biological fidelity advantage: Chemotaxis index, evasion trajectory similarity to real C. elegans
- Resource advantage: Training time × parameter count for equivalent performance level

______________________________________________________________________

## Research References

### Primary Papers

01. **QLIF Neurons**

    - Brand, D., & Petruccione, F. (2024). "A quantum leaky integrate-and-fire spiking neuron and network." *npj Quantum Information*.
    - [Paper Link](https://www.nature.com/articles/s41534-024-00921-x)

02. **Stochastic QSNN with Memory**

    - arXiv:2506.21324 (2025). "Stochastic Quantum Spiking Neural Networks with Quantum Memory and Local Learning."

03. **QRC Fundamentals**

    - arXiv:2602.03522 (2025). "QRC-Lab: An Educational Toolbox for Quantum Reservoir Computing."
    - [QuEra Tutorials](https://github.com/QuEraComputing/QRC-tutorials/)

04. **Data Re-uploading**

    - Pérez-Salinas, A., et al. "Data re-uploading for a universal quantum classifier."
    - Coelho, R., et al. (2024). "VQC-based reinforcement learning with data re-uploading: performance and trainability." *Quantum Machine Intelligence*.

05. **Barren Plateau Mitigation**

    - arXiv:2411.08238v3 (2025). "Neural-network Generated Quantum State Mitigates Barren Plateau."
    - arXiv:2407.17706 (2024). "Investigating and Mitigating Barren Plateaus in Variational Quantum Circuits: A Survey."
    - Cerezo, M., et al. (2025). "Does provable absence of barren plateaus imply classical simulability?" *Nature Communications*.
    - Larocca, M., et al. (2024). "A Lie algebraic theory of barren plateaus." *Nature Communications*.

06. **QNG Variants**

    - arXiv:2501.05847 (2025). "Modified conjugate quantum natural gradient."
    - arXiv:2409.03638 (2024). "Quantum Natural Gradient with Geodesic Corrections."

07. **Hybrid Architectures**

    - arXiv:2408.03884v2 (2024). "Quantum Computing and Neuromorphic Computing for Multi-Agent RL."

08. **Quantum RL with PPO**

    - arXiv:2501.07085 (2025). "PPO-Q: Proximal Policy Optimization with Parametrized Quantum Policies or Values."
    - [GitHub: BAQIS-Quantum/PPO-Q](https://github.com/BAQIS-Quantum/PPO-Q)

09. **Hybrid Deep Quantum RL**

    - arXiv:2503.09119 (2025). "Training Hybrid Deep Quantum Neural Network for Efficient Reinforcement Learning."

10. **Quantum Actor-Critic**

    - arXiv:2401.07043 (2024). "Quantum Advantage Actor-Critic for Reinforcement Learning."

11. **QRL Benchmarking**

    - arXiv:2502.04909 (2025). "Benchmarking Quantum Reinforcement Learning." Accepted at ICML 2025.
    - arXiv:2511.17112 (2025). "Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components."

12. **QRL in Latent Spaces**

    - "Hybrid quantum-classical reinforcement learning in latent observation spaces." *Quantum Machine Intelligence* (2025).

13. **SQDR-CNN (Surrogate Gradient + Data Re-uploading)**

    - arXiv:2512.03895 (2025). "Parameter efficient hybrid spiking-quantum convolutional neural network with surrogate gradient and quantum data-reupload."

14. **QKAN-LSTM (Quantum-Inspired Activations in Temporal Networks)**

    - arXiv:2512.05049 (2025). "QKAN-LSTM: Quantum-inspired Kolmogorov-Arnold Long Short-term Memory."

15. **Structured Quantum Reservoirs**

    - Llodrà, G., Mujal, P., Zambrini, R., & Giorgi, G. L. (2025). "Quantum reservoir computing in atomic lattices." *Chaos, Solitons & Fractals*, 195, 116289. arXiv:2411.13401.
    - Ivaki, M. N., Lazarides, A., & Ala-Nissila, T. (2025). "Quantum reservoir computing on random regular graphs." *Physical Review A*, 112, 012622. arXiv:2409.03665.
    - Zhu, S., et al. (2025). "Minimalistic and scalable quantum reservoir computing enhanced with feedback." *npj Quantum Information*. DOI: 10.1038/s41534-025-01144-4.
    - Martínez-Peña, R., et al. (2024). "Role of coherence in many-body quantum reservoir computing." *Communications Physics*, 7, 369. DOI: 10.1038/s42005-024-01859-4.

### Frameworks

- **PennyLane**: [pennylane.ai](https://pennylane.ai) - Differentiable quantum programming
- **Qiskit Machine Learning**: [qiskit.org/machine-learning](https://qiskit.org/machine-learning)
- **BrainCog**: [brain-cog.network](https://www.brain-cog.network) - QSSNN tutorials
- **SimplyQRL**: Modular benchmarking library for hybrid QRL (2026)
- **CleanQRL**: Lightweight single-file QRL implementations (2025)

______________________________________________________________________

## Open Questions

01. ~~**QSNN-PPO viability**~~: **ANSWERED — NO.** PPO is fundamentally incompatible with surrogate gradient spiking networks. The forward pass returns a constant (quantum measurement), making PPO's importance sampling ratio always 1.0. policy_loss=0 in 100% of 600+ updates across 16 sessions. Only REINFORCE-family methods work with surrogate gradients.
02. ~~**QSNNReinforce A2C viability**~~: **ANSWERED — NO.** After 4 rounds (16 sessions, 3,200 episodes), the classical critic never learned V(s). Explained variance progressively worsened: 0 → -0.008 → -0.295 → -0.620. Systematically eliminated data quantity, capacity, implementation bugs, and feature non-stationarity as causes. Root causes are fundamental: partial observability, policy non-stationarity, high return variance, short GAE windows. The critic is at best deadweight and at worst actively harmful (Q4 regression in 2/4 A2C-3 sessions).
03. ~~**Surrogate gradient algorithm compatibility**~~: **ANSWERED.** Surrogate gradients are backward-only. REINFORCE works directly; PPO/TRPO do not. A2C fails empirically. **However**, the HybridQuantum architecture resolves this by using REINFORCE for the QSNN component and PPO for a separate classical cortex — each algorithm applied to the component it's compatible with.
04. **Surrogate gradient vs parameter-shift**: QSNN's surrogate gradient provides ~1,000x stronger signals than parameter-shift. Is this approach transferable to other quantum circuit architectures, or is it specific to the QLIF neuron structure?
05. ~~**Weight regularization**~~: **PARTIALLY ANSWERED.** HybridQuantum Stage 3 showed W_hm growth of +27.7% over 500 episodes without performance degradation. The `weight_clip=3.0` per-element cap prevents individual weights from exploding. However, monotonic norm growth could be problematic for longer runs (>2,000 episodes). L2 regularization was not tested.
06. ~~**Trainability-advantage trade-off**~~: **PARTIALLY ANSWERED.** The HybridClassical ablation shows that a classical MLP reflex achieves equivalent task performance to the QSNN reflex (96.3% vs 96.9%). This suggests the quantum measurement may not provide meaningful task-performance advantage over a classical sigmoid in this architecture. However, the QSNN achieves higher biological fidelity (chemotaxis indices) and earns 1.5x more trust from the cortex, indicating it produces a qualitatively different signal. Whether this constitutes "quantum advantage" remains open — it may be advantage in biological plausibility rather than computational performance.
07. **Hardware deployment**: Which architecture maps best to real QPU for eventual hardware validation? The HybridQuantum brain's QSNN component uses simple 2-gate QLIF circuits that should map well to near-term hardware.
08. ~~**Multi-objective scaling**~~: **ANSWERED — YES.** HybridQuantum achieves 96.9% on a dual-objective task (forage + evade). The architecture handles foraging via QSNN reflex and evasion via cortex PPO. Whether it scales to 3+ objectives (adding thermotaxis, mechanosensation as objectives rather than just sensory inputs) is untested but architecturally straightforward — additional modes could be added to the mode gate.
09. **Harder environments**: The small pursuit predator environment (20x20, 2 predators) is largely solved at 96.9%. Does the HybridQuantum architecture generalise to larger grids, more predators, or faster predators?
10. **Mode gating dynamics**: Mode gating converges to a static trust parameter in all sessions, but the **trust level depends on reflex quality**: HybridQuantum → 0.55 trust (collaborative, forage-dominant), HybridClassical → 0.37 trust (cortex-dominant, evade-dominant). The cortex adapts its strategy to the reflex signal strength. Would a more structured mode switching mechanism improve performance on harder environments?
11. **Structured reservoir viability**: Can structured quantum reservoirs with C. elegans-inspired topology provide richer feature spaces than classical random features? If so, does the quantum feature space encode information that a classical feature extractor cannot?
12. **Multi-qubit quantum memory**: Do multi-qubit SQS neurons with quantum memory (entanglement preserved across timesteps) provide measurably different computational capabilities than classical spiking neurons with standard recurrence?
13. **qtDNN gradient approximation**: Can a classical tangential DNN (qtDNN) approximate entangled quantum circuit gradients accurately enough (correlation > 0.5 with true parameter-shift) to enable training circuits that would otherwise exhibit barren plateaus?
14. **Trainability-advantage sweet spot**: Is the barren plateau-advantage dilemma a fundamental barrier, or can biological network topology constraints (sparse connectivity, small-world structure, modular organization) provide a "sweet spot" of trainability + advantage that random architectures cannot access?

### Proposed Next Steps: Bridging the Trainability-Advantage Gap

The central challenge for quantum brain architectures is the **Barren Plateau-Advantage Dilemma** (Cerezo et al., Nature Communications, 2025): provably trainable quantum circuits are classically simulable, while circuits offering genuine quantum advantage suffer barren plateaus. Our experimental data confirms this — HybridClassical (96.3%) matches HybridQuantum (96.9%), showing the trainable single-qubit QLIF provides no measurable task-performance quantum advantage.

Three strategies can bridge this gap:

1. **Don't train the quantum part** — Use fixed structured quantum reservoirs as feature extractors with classical readout networks. Avoids barren plateaus entirely while potentially leveraging quantum dynamics for richer feature spaces.
2. **Use local learning rules** — Replace global gradient-based training with local quantum learning rules (e.g., quantum STDP). Local rules avoid barren plateaus by construction and have biological plausibility.
3. **Use classical gradient surrogates** — Train entangled circuits (which could provide genuine quantum advantage) using a classical tangential DNN (qtDNN) that approximates quantum gradients, separating the advantage source from the training mechanism.

Four concrete architectures have been proposed (see [Next-Generation Architecture Proposals](#next-generation-architecture-proposals)):

| Priority | Architecture | Strategy | Risk | Key Question |
|----------|-------------|----------|------|-------------|
| 1 | Quantum Reservoir Hybrid (QRH) | Don't train quantum | Medium | Can structured reservoirs outperform random? |
| 2 | SQS-QLIF Hybrid | Local learning rules | High | Can multi-qubit quantum memory neurons learn? |
| 3 | Entangled QLIF + qtDNN | Classical surrogates | Medium-High | Can qtDNN approximate entangled circuit gradients? |
| 4 | QKAN-QLIF Temporal Brain | Actual quantum activations (inspired by QKAN-LSTM's classical pattern) | Low-Medium | Can actual QLIF circuits replace LSTM activations efficiently? |

Each proposal includes falsification criteria and decision gates to enable rapid go/no-go decisions.
