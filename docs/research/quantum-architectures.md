# Advanced Quantum Brain Architectures - Implementation Notes

**Purpose**: Detailed specifications for novel quantum brain implementations beyond QVarCircuitBrain
**Status**: Research & Planning
**Last Updated**: 2026-02-17

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
12. [Benchmarking Plan](#benchmarking-plan)
13. [Research References](#research-references)

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

### Key Finding: HybridQuantum Achieves SOTA on Multi-Objective Tasks

The **HybridQuantum brain** is the first quantum architecture to surpass a classical baseline on a multi-objective RL task using gradient-based online learning. It combines the QSNN reflex layer (proven 73.9% foraging) with a classical cortex MLP (PPO) via mode-gated fusion, achieving **96.9% post-convergence on pursuit predators** — beating the apples-to-apples MLP PPO baseline by **+25.3 points** with **4.3x fewer parameters**.

The architecture was validated through a three-stage curriculum:

1. **Stage 1**: QSNN reflex on foraging (REINFORCE) — 91.0% success, 4 sessions
2. **Stage 2**: Cortex PPO with frozen QSNN (pursuit predators) — 91.7% post-convergence, 8 sessions across 2 rounds
3. **Stage 3**: Joint fine-tune (both trainable) — 96.9% post-convergence, 4 sessions, immediate convergence

**QSNN's surrogate gradient approach** (quantum forward, classical backward) remains the core proven quantum technique. It sidesteps barren plateaus while providing dense gradient signals. However, standalone QSNN cannot solve multi-objective tasks (0% across 60 sessions on pursuit predators). The hybrid architecture resolves this by delegating strategic behaviour to the classical cortex while preserving the quantum reflex.

### Architecture Evaluation History

```text
COMPLETED:
  HybridQuantum — QSNN reflex + classical cortex MLP + mode-gated fusion.
    4 rounds, 16 sessions, 4,200 episodes. 96.9% post-convergence.
    Beats MLP PPO unified by +25.3 pts. Three-stage curriculum validated.
    STATUS: SUCCESS — best quantum architecture for multi-objective tasks.

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

**Status**: Implemented and validated. 96.9% post-convergence on pursuit predators across 4 rounds, 16 sessions. Best quantum architecture for multi-objective tasks.

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
06. **Trainability-advantage trade-off**: Given Cerezo et al. (2025), does QSNN's surrogate gradient approach constitute genuine quantum advantage, or is the quantum measurement functionally equivalent to a classical sigmoid?
07. **Hardware deployment**: Which architecture maps best to real QPU for eventual hardware validation? The HybridQuantum brain's QSNN component uses simple 2-gate QLIF circuits that should map well to near-term hardware.
08. ~~**Multi-objective scaling**~~: **ANSWERED — YES.** HybridQuantum achieves 96.9% on a dual-objective task (forage + evade). The architecture handles foraging via QSNN reflex and evasion via cortex PPO. Whether it scales to 3+ objectives (adding thermotaxis, mechanosensation as objectives rather than just sensory inputs) is untested but architecturally straightforward — additional modes could be added to the mode gate.
09. **Harder environments**: The small pursuit predator environment (20x20, 2 predators) is largely solved at 96.9%. Does the HybridQuantum architecture generalise to larger grids, more predators, or faster predators?
10. **Mode gating dynamics**: In all 12 Stage 2/3 sessions, mode gating converged to a static trust parameter rather than dynamic per-step switching. Would a more structured mode switching mechanism (e.g., threshold-based, or conditioned on specific sensory signals) improve performance on harder environments?
