## Context

QRCBrain failed (0% success) because fixed random reservoirs don't create discriminative representations. QVarCircuitBrain achieves 88% with CMA-ES but only 22.5% with gradient-based learning due to barren plateaus. We need a quantum architecture that:

1. Has trainable quantum parameters (unlike QRC)
2. Avoids global backpropagation (unlike QVarCircuit)

QSNN uses QLIF neurons with local learning rules, combining the best of both approaches. Reference: Brand & Petruccione (2024) "A quantum leaky integrate-and-fire spiking neuron and network."

## Goals / Non-Goals

**Goals:**

- Implement QSNNBrain with QLIF neurons (2-gate circuit per neuron)
- Implement 3-factor local learning (reward-modulated Hebbian)
- Integrate with existing brain factory and CLI
- Achieve >0% success rate on foraging (baseline improvement over QRC)
- Provide comparison baseline vs SpikingReinforceBrain (73.3%)

**Non-Goals:**

- Hardware QPU optimization (future phase)
- Hybrid QSNN+VQC architecture (future HybridQuantumBrain)
- STDP-style timing-dependent learning (local Hebbian is simpler, proven)
- Multi-hidden-layer networks (start simple: sensory→hidden→motor)

## Decisions

### Decision 1: QLIF Circuit Structure

**Choice:** Minimal 2-gate circuit: `|0⟩ → RY(θ_membrane + input) → RX(θ_leak) → Measure`

**Rationale:**

- Brand & Petruccione (2024) shows this captures LIF dynamics with minimal circuit depth
- No CNOT gates needed for single neurons → lower noise on NISQ devices
- RY encodes membrane potential + input; RX implements leak
- Measurement outcome `|1⟩` = spike, `|0⟩` = no spike

**Alternatives Considered:**

- Full LIF ODE simulation: Too complex, not quantum-native
- Multi-qubit entangled neurons: Higher depth, more noise, barren plateau risk
- Parametric controlled gates: Unnecessary complexity for basic dynamics

### Decision 2: Network Architecture

**Choice:** Three-layer feedforward: sensory (6) → hidden (4) → motor (4-5)

**Rationale:**

- Matches C. elegans sensorimotor pathway abstraction
- 6 sensory = food chemotaxis (2), nociception (2), thermotaxis (2)
- 4 hidden = minimal interneuron processing
- 4-5 motor = forward, backward, left, right, (optional stay)
- Small network keeps circuit execution tractable

**Alternatives Considered:**

- Recurrent connections: Adds complexity, unclear benefit for reactive tasks
- Deeper networks: More parameters, harder to train with local rules
- Direct sensory→motor: Too simple, no representation learning

### Decision 3: Local Learning Rule

**Choice:** 3-factor Hebbian: `Δw = lr × pre_spike × post_spike × reward`

**Rationale:**

- Avoids barren plateaus (no global gradient computation)
- Biologically plausible (reward-modulated Hebbian)
- Computationally cheap (O(synapses) per update)
- Eligibility trace maintains spike correlations for delayed reward

**Alternatives Considered:**

- REINFORCE with surrogate gradients: Falls back to global backprop, may hit barren plateaus
- STDP (timing-dependent): More complex, requires spike timing precision
- CMA-ES: Works but defeats purpose of gradient-free quantum learning

### Decision 4: Weight Storage

**Choice:** PyTorch tensors for weight matrices, not quantum parameters

**Rationale:**

- Synaptic weights are classical (connection strengths)
- Only membrane dynamics are quantum (QLIF circuit)
- PyTorch enables easy gradient fallback if needed
- Matches existing brain architecture patterns

**Alternatives Considered:**

- Pure NumPy: Less flexible for gradient fallback
- Quantum parameter storage: Unnecessary, weights don't need quantum representation

### Decision 5: Implement ClassicalBrain Protocol

**Choice:** Inherit from ClassicalBrain (like QRCBrain), not QuantumBrain

**Rationale:**

- QSNNBrain trains classical weights, quantum is only for neuron dynamics
- Matches QRCBrain pattern (quantum feature extraction + classical learning)
- Simpler integration with existing infrastructure
- No parameter-shift gradient needed (local learning)

**Alternatives Considered:**

- QuantumBrain: Would require gradient infrastructure we don't need
- Custom protocol: Unnecessary, ClassicalBrain fits perfectly

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Local learning may converge slowly | Add configurable learning rate, monitor eligibility traces |
| Shot noise may overwhelm weak signals | Start with high shots (2048), tune down if stable |
| Refractory period may cause dead neurons | Make refractory_period configurable, default 2 is conservative |
| Network too small for complex tasks | Parameters are configurable; can scale up if baseline works |
| Eligibility trace decay may lose signal | Implement trace decay factor (optional), accumulate per episode |

## Open Questions

1. **Eligibility trace accumulation**: Should traces decay within episode or accumulate? Start with accumulation (simpler), add decay if learning is unstable.

2. **Weight initialization**: Xavier/Glorot vs small uniform? Follow SpikingReinforceBrain pattern initially.

3. **Sensory module mapping**: Fixed 6-neuron mapping or dynamic based on config? Start fixed, generalize if needed.
