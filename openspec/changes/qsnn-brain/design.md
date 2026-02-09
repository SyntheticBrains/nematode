## Context

QRCBrain failed (0% success) because fixed random reservoirs don't create discriminative representations. QVarCircuitBrain achieves 88% with CMA-ES but only 22.5% with gradient-based learning due to barren plateaus. We need a quantum architecture that:

1. Has trainable quantum parameters (unlike QRC)
2. Avoids global backpropagation through quantum circuits (unlike QVarCircuit)

QSNN uses QLIF neurons with a hybrid quantum-classical training approach: quantum circuits run in the forward pass, and classical surrogate gradients provide the backward pass. Reference: Brand & Petruccione (2024) "A quantum leaky integrate-and-fire spiking neuron and network."

## Goals / Non-Goals

**Goals:**

- Implement QSNNBrain with QLIF neurons (2-gate circuit per neuron)
- Implement hybrid training: quantum forward + surrogate gradient backward
- Achieve classical SNN parity on foraging (target: 73.3% SpikingReinforceBrain baseline)
- Support dual learning modes: surrogate gradient (primary) and Hebbian (legacy)
- Integrate with existing brain factory and CLI

**Non-Goals:**

- Hardware QPU optimization (future phase)
- Hybrid QSNN+VQC architecture (future HybridQuantumBrain)
- STDP-style timing-dependent learning
- Multi-hidden-layer networks (start simple: sensory->hidden->motor)

## Decisions

### Decision 1: QLIF Circuit Structure

**Choice:** Minimal 2-gate circuit: `|0> -> RY(theta + tanh(w*x)*pi) -> RX(theta_leak) -> Measure`

**Rationale:**

- Brand & Petruccione (2024) shows this captures LIF dynamics with minimal circuit depth
- No CNOT gates needed for single neurons -> lower noise on NISQ devices
- RY encodes membrane potential + input; RX implements leak
- `tanh` bounds weighted input to [-1, 1] before scaling by pi, preventing angle wrapping
- Measurement outcome `|1>` = spike, `|0>` = no spike

### Decision 2: Network Architecture

**Choice:** Three-layer feedforward: sensory (6) -> hidden (8) -> motor (4)

**Rationale:**

- Matches C. elegans sensorimotor pathway abstraction
- 6 sensory = sufficient for food chemotaxis (2) + nociception (2) + thermotaxis (2)
- 8 hidden = provides adequate representational capacity while keeping circuit count tractable
- 4 motor = forward, backward, left, right
- ~92 total trainable parameters (vs SpikingReinforceBrain's 131K)

### Decision 3: Hybrid Quantum-Classical Training

**Choice:** Quantum forward pass + classical surrogate gradient backward pass (QLIFSurrogateSpike)

**Rationale:**

- Forward pass preserves quantum dynamics (QLIF circuits execute on quantum simulator)
- Backward pass uses sigmoid surrogate centered at pi/2 (RY gate transition point)
- Avoids parameter-shift rule cost (no additional circuit evaluations for gradients)
- Avoids barren plateaus (gradients computed classically, not through deep quantum circuits)
- REINFORCE policy gradient with advantage normalization provides dense credit assignment

**Alternatives Considered:**

- 3-factor Hebbian learning: Implemented as legacy mode, but 12 rounds of tuning proved local learning too weak for RL (0% success). Fundamental tension: updating all columns causes correlated collapse, updating only chosen column causes starvation.
- Parameter-shift rule: Expensive (2 circuit evaluations per parameter per gradient) and susceptible to barren plateaus in variational circuits
- CMA-ES: Works but defeats purpose of gradient-based quantum learning

### Decision 4: Multi-Timestep Integration

**Choice:** Average spike probabilities across 10 QLIF timesteps per decision

**Rationale:**

- Reduces quantum shot noise variance by 10x (effective samples = 10 * 1024 = 10,240)
- Essential for stable REINFORCE training: 5 timesteps showed 52.6% success vs 73.9% with 10
- Classical SpikingReinforceBrain uses 100 timesteps; QSNN uses fewer because each quantum timestep already has 1024 measurement samples

### Decision 5: Adaptive Entropy Regulation

**Choice:** Two-sided entropy regulation (floor boost + ceiling suppression)

**Rationale:**

- Floor: When entropy < 0.5 nats, scale entropy_coef up to 20x to prevent policy collapse
- Ceiling: When entropy > 95% of max, suppress entropy bonus to let policy gradient sharpen
- Prevents both failure modes: deterministic policy collapse and drift to uniform random
- 20x boost produces effective_coef = 0.40, competitive with REINFORCE gradient force (~1.0)

### Decision 6: Weight Storage and Optimizer

**Choice:** PyTorch tensors with Adam optimizer and cosine annealing LR decay

**Rationale:**

- Synaptic weights are classical (connection strengths between quantum neurons)
- Adam provides adaptive per-parameter learning rates for stable training
- Cosine annealing (0.01 -> 0.001 over 200 episodes) prevents late-episode catastrophic forgetting
- Weight clamping ([-3, 3]) provides hard stability bounds

### Decision 7: Implement ClassicalBrain Protocol

**Choice:** Inherit from ClassicalBrain (like QRCBrain), not QuantumBrain

**Rationale:**

- QSNNBrain trains classical weights; quantum is only for neuron forward dynamics
- Matches QRCBrain pattern (quantum feature extraction + classical learning)
- No parameter-shift gradient infrastructure needed

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Quantum shot noise overwhelms gradient signal | Multi-timestep integration (10 steps = 10x noise reduction) |
| Entropy collapse during training | Adaptive entropy floor (20x boost when entropy < 0.5 nats) |
| Entropy explosion to uniform random | Adaptive entropy ceiling (suppression when > 95% max) |
| Late-episode catastrophic forgetting | Cosine annealing LR decay (0.01 -> 0.001 over 200 eps) |
| Premature policy commitment | Extended exploration decay (80 episodes) |
| Weight instability | Gradient clipping (norm 1.0), advantage clipping ([-2, 2]), weight clamping ([-3, 3]) |
