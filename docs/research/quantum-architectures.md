# Advanced Quantum Brain Architectures - Implementation Notes

**Purpose**: Detailed specifications for novel quantum brain implementations beyond QVarCircuitBrain
**Status**: Research & Planning

______________________________________________________________________

## Table of Contents

01. [Executive Summary](#executive-summary)
02. [QSNN Brain Specification](#qsnn-brain-specification)
03. [QRC Brain Specification](#qrc-brain-specification)
04. [HybridQuantum Brain Architecture](#hybridquantum-brain-architecture)
05. [Data Re-Uploading Enhancement](#data-re-uploading-enhancement)
06. [Training Curriculum](#c6-training-curriculum)
07. [Optimizer Variants](#optimizer-variants)
08. [Barren Plateau Mitigation](#barren-plateau-mitigation)
09. [Benchmarking Plan](#benchmarking-plan)
10. [Research References](#research-references)

______________________________________________________________________

## Executive Summary

### Why New Architectures?

The current QVarCircuitBrain achieves 88% success with CMA-ES evolutionary optimization but suffers from:

- Parameter-shift gradients showing barren-plateau-like behavior
- High sensitivity to initial parameters
- Gradient-based learning only achieving 22.5% success

### Proposed Solution: Hierarchical Quantum Nervous System

```text
Sensors → QSNN Actor (fast reflexes) → Action logits
                    ↑
          VQC/QRC Cortex (slow planning, every N steps)
                    ↓
             Options / Gating / Priors
```

**QSNN = spinal cord** (predator evasion, thermotaxis following, collision response)
**VQC/QRC = cortex** (high-level options, strategic gating signals)

______________________________________________________________________

## QSNN Brain Specification

### A.1 Overview

Quantum Spiking Neural Networks combine quantum computing with neuromorphic principles for reflex-like sensorimotor control.

**Key Advantage**: Local learning rules avoid global backpropagation, potentially circumventing barren plateaus.

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

#### Option A: Local Learning (Preferred for avoiding barren plateaus)

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

#### Option B: Surrogate Gradient (for comparison)

Same as SpikingReinforceBrain - use smooth surrogate for spike derivative:

```python
def surrogate_gradient(membrane_potential, threshold):
    """Smooth approximation of spike gradient."""
    return 1.0 / (1.0 + np.abs(membrane_potential - threshold) * 10) ** 2
```

______________________________________________________________________

## QRC Brain Specification

### B.1 Overview

Quantum Reservoir Computing uses a fixed (random) quantum reservoir with only a classical readout layer trained. This **inherently avoids barren plateaus** since reservoir parameters are never optimized.

**Best for**: Fast sensorimotor reflexes, pattern recognition, temporal dynamics

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

______________________________________________________________________

## HybridQuantum Brain Architecture

### C.1 Overview

Combines QSNN (fast reflexes) with VQC/QRC (slow planning) in a hierarchical architecture mimicking biological spinal cord / cortex separation.

### C.2 Architecture Diagram

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

### C.3 Configuration

```python
class HybridQuantumBrainConfig(BrainConfig):
    name: Literal["hybrid_quantum"] = "hybrid_quantum"

    # Reflex layer (QSNN or QRC)
    reflex_type: Literal["qsnn", "qrc"] = "qsnn"
    reflex_config: QSNNBrainConfig | QRCBrainConfig

    # Cortex layer (VQC or QRC)
    cortex_type: Literal["vqc", "qrc"] = "vqc"
    cortex_config: QVarCircuitBrainConfig | QRCBrainConfig

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

### C.6 Training Curriculum

Three-stage training pipeline:

```text
Stage 1: Train QSNN reflexes alone
  - Dense reward shaping
  - PPO / REINFORCE / Local learning
  - Tasks: predator evasion, collision response, thermotaxis following
  - Exit criteria: >80% survival rate on predator tasks

Stage 2: Freeze QSNN, train cortex planner
  - QSNN weights frozen
  - Train cortex with CMA-ES (proven effective)
  - Tasks: foraging with strategic tradeoffs
  - Exit criteria: >70% on multi-objective benchmarks

Stage 3: Optional fine-tuning
  - Unfreeze both modules
  - Low learning rate joint training
  - QNG or policy gradients
  - Exit criteria: improvement over Stage 2
```

______________________________________________________________________

## Data Re-Uploading Enhancement

### D.1 Concept

Data re-uploading encodes classical inputs multiple times into the quantum circuit, interleaved with trainable gates. This dramatically increases expressivity without adding qubits.

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

______________________________________________________________________

## Benchmarking Plan

### G.1 New Benchmark Categories

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

### G.2 Comparison Matrix

| Architecture | Foraging | Predator | Thermotaxis | Multi-Objective |
|--------------|----------|----------|-------------|-----------------|
| QVarCircuit (CMA-ES) | 0.762 | 0.675 | TBD | TBD |
| QSNN (local) | TBD | TBD | TBD | TBD |
| QRC | TBD | TBD | TBD | TBD |
| HybridQuantum | TBD | TBD | TBD | TBD |
| SpikingReinforce | 0.733 | 0.556 | TBD | TBD |
| MLPReinforce | 0.822 | 0.740 | TBD | TBD |

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

1. **QLIF Neurons**

   - Brand, D., & Petruccione, F. (2024). "A quantum leaky integrate-and-fire spiking neuron and network." *npj Quantum Information*.
   - [Paper Link](https://www.nature.com/articles/s41534-024-00921-x)

2. **Stochastic QSNN with Memory**

   - arXiv:2506.21324 (2025). "Stochastic Quantum Spiking Neural Networks with Quantum Memory and Local Learning."

3. **QRC Fundamentals**

   - arXiv:2602.03522 (2025). "QRC-Lab: An Educational Toolbox for Quantum Reservoir Computing."
   - [QuEra Tutorials](https://github.com/QuEraComputing/QRC-tutorials/)

4. **Data Re-uploading**

   - Pérez-Salinas, A., et al. "Data re-uploading for a universal quantum classifier."

5. **Barren Plateau Mitigation**

   - arXiv:2411.08238v3 (2025). "Neural-network Generated Quantum State Mitigates Barren Plateau."
   - arXiv:2407.17706 (2024). "Investigating and Mitigating Barren Plateaus in Variational Quantum Circuits: A Survey."

6. **QNG Variants**

   - arXiv:2501.05847 (2025). "Modified conjugate quantum natural gradient."
   - arXiv:2409.03638 (2024). "Quantum Natural Gradient with Geodesic Corrections."

7. **Hybrid Architectures**

   - arXiv:2408.03884v2 (2024). "Quantum Computing and Neuromorphic Computing for Multi-Agent RL."

### Frameworks

- **PennyLane**: [pennylane.ai](https://pennylane.ai) - Differentiable quantum programming
- **Qiskit Machine Learning**: [qiskit.org/machine-learning](https://qiskit.org/machine-learning)
- **BrainCog**: [brain-cog.network](https://www.brain-cog.network) - QSSNN tutorials

______________________________________________________________________

## Implementation Timeline (Suggested)

### Phase 2 Integration (Q2-Q3 2026)

**Sprint 1** (2 weeks): QRC Brain

- Implement fixed reservoir circuit
- Classical readout training
- Basic benchmarks

**Sprint 2** (2 weeks): QSNN Brain

- QLIF neuron implementation
- Local learning rules
- Reflex task benchmarks

**Sprint 3** (2 weeks): Data Re-uploading

- Add to QVarCircuitBrain
- Compare expressivity vs depth
- Ablation studies

**Sprint 4** (2 weeks): HybridQuantum Brain

- Integrate QSNN + VQC/QRC
- Implement fusion strategies
- Training curriculum

**Sprint 5** (2 weeks): Comprehensive Benchmarking

- All architectures on all tasks
- Statistical analysis
- Documentation and comparison paper

______________________________________________________________________

## Open Questions

1. **QSNN vs QRC for reflexes**: Which is faster/more accurate for predator evasion?
2. **Optimal cortex update frequency**: Every 5, 10, or 20 timesteps?
3. **Fusion strategy effectiveness**: Option conditioning vs logit gating?
4. **Hardware deployment**: Which architecture maps best to real QPU?
5. **Transfer learning**: Can trained cortex transfer to new environments?
