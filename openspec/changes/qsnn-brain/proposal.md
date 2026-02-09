## Why

QRCBrain achieved 0% success rate on foraging tasks because its fixed random reservoir cannot create discriminative representations - different inputs produce similar high-entropy outputs. QVarCircuitBrain achieves 88% with CMA-ES but only 22.5% with gradient-based learning due to barren plateaus. We need a quantum architecture with trainable quantum parameters that avoids these issues.

QSNN (Quantum Spiking Neural Network) addresses both problems: trainable quantum parameters solve the representation issue, while a hybrid quantum-classical training approach (quantum forward pass + classical surrogate gradient backward pass) sidesteps both parameter-shift rule costs and barren plateaus.

## What Changes

- Add new `QSNNBrain` architecture implementing Quantum Leaky Integrate-and-Fire (QLIF) neurons
- Implement minimal 2-gate QLIF circuit per neuron: `|0> -> RY(theta + tanh(w*x)*pi) -> RX(theta_leak) -> Measure`
- Implement `QLIFSurrogateSpike` autograd function for hybrid quantum-classical training
- Add surrogate gradient REINFORCE learning (primary) and 3-factor Hebbian (legacy)
- Support network topology: sensory -> hidden -> motor layers (6 -> 8 -> 4 neurons, ~92 params)
- Add multi-timestep integration (10 QLIF timesteps per decision) for quantum shot noise reduction
- Add adaptive entropy regulation (two-sided: floor boost + ceiling suppression)
- Integrate with existing brain factory, config system, and CLI
- Add example configs for foraging and predator evasion tasks

## Capabilities

### New Capabilities

- `qsnn-brain`: Quantum Spiking Neural Network brain architecture with QLIF neurons, hybrid quantum-classical surrogate gradient learning, and adaptive entropy regulation

### Modified Capabilities

<!-- None - this is a new architecture that doesn't change existing spec requirements -->

## Impact

**Code Changes:**

- New: `quantumnematode/brain/arch/qsnn.py` (QSNNBrain, QSNNBrainConfig, QLIFSurrogateSpike)
- Modify: `brain/arch/dtypes.py` (add QSNN to BrainType enum)
- Modify: `brain/arch/__init__.py` (export new classes)
- Modify: `utils/config_loader.py` (add to BRAIN_CONFIG_MAP)
- Modify: `utils/brain_factory.py` (add instantiation case)
- New: `configs/examples/qsnn_foraging_small.yml`
- New: `configs/examples/qsnn_predators_small.yml`

**Dependencies:**

- Qiskit + Qiskit-Aer (existing) - quantum circuit construction and simulation
- PyTorch (existing) - weight management, autograd for surrogate gradients, Adam optimizer

**Testing:**

- 100 unit tests covering configuration, QLIF dynamics, surrogate gradients, multi-timestep integration, adaptive entropy, weight initialization, and learning
- Benchmark: 73.9% success on foraging (matches SpikingReinforceBrain's 73.3%)

**Documentation:**

- Updated `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN results
- Appendix with full optimization history (17 rounds)
