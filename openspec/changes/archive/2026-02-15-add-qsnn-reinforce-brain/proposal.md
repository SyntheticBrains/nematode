## Why

QRCBrain achieved 0% success rate on foraging tasks because its fixed random reservoir cannot create discriminative representations - different inputs produce similar high-entropy outputs. QVarCircuitBrain achieves 88% with CMA-ES but only 22.5% with gradient-based learning due to barren plateaus. We need a quantum architecture with trainable quantum parameters that avoids these issues.

QSNN (Quantum Spiking Neural Network) addresses both problems: trainable quantum parameters solve the representation issue, while a hybrid quantum-classical training approach (quantum forward pass + classical surrogate gradient backward pass) sidesteps both parameter-shift rule costs and barren plateaus.

## What Changes

- Add `QSNNReinforceBrain` architecture implementing Quantum Leaky Integrate-and-Fire (QLIF) neurons
- Implement minimal 2-gate QLIF circuit per neuron: `|0> -> RY(theta + tanh(w*x/sqrt(fan_in))*pi) -> RX(theta_leak) -> Measure`
- Extract shared QLIF components (`QLIFSurrogateSpike`, circuit execution, sensory encoding) into `_qlif_layers.py`
- Add surrogate gradient REINFORCE learning (primary) and 3-factor Hebbian (legacy)
- Support network topology: sensory -> hidden -> motor layers (default 8 -> 16 -> 4 neurons, ~212 params)
- Add multi-timestep integration (10 QLIF timesteps per decision) for quantum shot noise reduction
- Add multi-epoch REINFORCE with quantum output caching for increased gradient passes
- Add adaptive entropy regulation (two-sided: floor boost + ceiling suppression)
- Add reward normalization (EMA), theta motor norm clamping, degenerate batch skipping
- Integrate with existing brain factory, config system, and CLI
- Add example configs for foraging, predator evasion, and pursuit predator tasks

## Capabilities

### New Capabilities

- `qsnn-reinforce-brain`: Quantum Spiking Neural Network brain architecture with QLIF neurons, hybrid quantum-classical surrogate gradient learning, and adaptive entropy regulation

### Modified Capabilities

<!-- None - this is a new architecture that doesn't change existing spec requirements -->

## Impact

**Code Changes:**

- New: `quantumnematode/brain/arch/qsnnreinforce.py` (QSNNReinforceBrain, QSNNReinforceBrainConfig)
- New: `quantumnematode/brain/arch/_qlif_layers.py` (QLIFSurrogateSpike, shared QLIF circuit execution)
- Modify: `brain/arch/dtypes.py` (add QSNN_REINFORCE to BrainType enum)
- Modify: `brain/arch/__init__.py` (export new classes)
- Modify: `utils/config_loader.py` (add to BRAIN_CONFIG_MAP)
- Modify: `utils/brain_factory.py` (add instantiation case)
- New: `configs/examples/qsnnreinforce_foraging_small.yml`
- New: `configs/examples/qsnnreinforce_predators_small.yml`
- New: `configs/examples/qsnnreinforce_pursuit_predators_small.yml`

**Dependencies:**

- Qiskit + Qiskit-Aer (existing) - quantum circuit construction and simulation
- PyTorch (existing) - weight management, autograd for surrogate gradients, Adam optimizer

**Testing:**

- 175 unit tests covering configuration, QLIF dynamics, surrogate gradients, multi-timestep integration, adaptive entropy, weight initialization, learning, and multi-epoch caching
- Benchmark: 67% success on foraging (4x200 episodes)
- Predator evaluation: per-encounter evasion did not improve through training; standalone QSNN Reinforce halted on predator tasks (see logbook 008)

**Documentation:**

- Updated `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN results
- Appendix with full optimization history (17 rounds foraging + 16 rounds predator)
