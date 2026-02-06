## Why

QRCBrain achieved 0% success rate on foraging tasks because its fixed random reservoir cannot create discriminative representations - different inputs produce similar high-entropy outputs. We need a quantum architecture with trainable quantum parameters that avoids the barren plateau problem affecting QVarCircuitBrain's gradient-based learning (22.5% vs 88% with CMA-ES).

QSNN (Quantum Spiking Neural Network) addresses both problems: trainable quantum parameters solve the representation issue, while local learning rules (reward-modulated Hebbian) avoid global backpropagation and barren plateaus.

## What Changes

- Add new `QSNNBrain` architecture implementing Quantum Leaky Integrate-and-Fire (QLIF) neurons
- Implement minimal 2-gate QLIF circuit per neuron: `|0⟩ → RY(θ_membrane + input) → RX(θ_leak) → Measure`
- Add 3-factor local learning rule: `Δw = lr × pre_spike × post_spike × reward`
- Support network topology: sensory → hidden → motor layers (6 → 4 → 4-5 neurons)
- Integrate with existing brain factory, config system, and CLI
- Add example configs for foraging and predator evasion tasks

## Capabilities

### New Capabilities

- `qsnn-brain`: Quantum Spiking Neural Network brain architecture with QLIF neurons and local learning rules

### Modified Capabilities

<!-- None - this is a new architecture that doesn't change existing spec requirements -->

## Impact

**Code Changes:**

- New: `quantumnematode/brain/arch/qsnn.py` (QSNNBrain, QSNNBrainConfig)
- Modify: `brain/arch/dtypes.py` (add QSNN to BrainType enum)
- Modify: `brain/arch/__init__.py` (export new classes)
- Modify: `utils/config_loader.py` (add to BRAIN_CONFIG_MAP)
- Modify: `utils/brain_factory.py` (add instantiation case)
- New: `configs/examples/qsnn_foraging_small.yml`
- New: `configs/examples/qsnn_predators_small.yml`

**Dependencies:**

- Qiskit (existing) - quantum circuit construction and measurement
- PyTorch (existing) - weight management for classical synaptic connections

**Testing:**

- Unit tests for QLIF dynamics, spike encoding, local learning
- Benchmark against SpikingReinforceBrain (73.3% baseline) and QRC (0%)

**Documentation:**

- Update `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN results
