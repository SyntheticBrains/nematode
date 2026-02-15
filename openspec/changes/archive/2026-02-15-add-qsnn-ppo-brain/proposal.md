## Why

QSNNReinforceBrain achieved 67% success on foraging but failed on predator evasion — 16 rounds of optimization showed per-encounter evasion never improved through training. Three root causes were diagnosed: (1) no critic to provide value-based feedback, (2) high REINFORCE variance from single-sample policy gradient, and (3) insufficient gradient passes per data collection. QSNN-PPO addresses all three by pairing the QSNN actor with a classical MLP critic and training via PPO with quantum output caching for multi-epoch updates.

## What Changes

- Add `QSNNPPOBrain` architecture pairing a QSNN actor (QLIF circuits with surrogate gradients) with a classical MLP critic
- Implement PPO clipped surrogate objective with GAE advantages
- Implement `QSNNRolloutBuffer` storing quantum spike caches and hidden spike rates alongside standard PPO fields
- Implement `QSNNPPOCritic` MLP with orthogonal initialization, accepting raw sensory features + hidden spike rates as input
- Implement multi-epoch PPO with quantum caching (epoch 0 runs circuits, epochs 1+ reuse cached spike probs)
- Use separate optimizers for actor (raw tensors + L2 weight decay) and critic (nn.Module)
- Reuse shared QLIF components from `_qlif_layers.py` (established by QSNNReinforce change)
- Register in brain factory, config system, dtypes, and module exports
- Add example config for pursuit predator task

## Capabilities

### New Capabilities

- `qsnn-ppo-brain`: Quantum Spiking Neural Network with PPO training — QSNN actor with classical MLP critic, rollout buffer with quantum spike caching, GAE advantages, and clipped surrogate objective

### Modified Capabilities

<!-- None - this is a new brain architecture that doesn't change existing spec requirements -->

## Impact

**Code Changes:**

- New: `quantumnematode/brain/arch/qsnnppo.py` (QSNNPPOBrain, QSNNPPOBrainConfig, QSNNRolloutBuffer, QSNNPPOCritic)
- Modify: `brain/arch/dtypes.py` (add QSNN_PPO to BrainType enum, QUANTUM_BRAIN_TYPES, BRAIN_TYPES)
- Modify: `brain/arch/__init__.py` (export new classes)
- Modify: `utils/config_loader.py` (add to BRAIN_CONFIG_MAP)
- Modify: `utils/brain_factory.py` (add instantiation case)
- New: `configs/examples/qsnnppo_pursuit_predators_small.yml`

**Dependencies:**

- Qiskit + Qiskit-Aer (existing) — quantum circuit simulation
- PyTorch (existing) — actor weight management, critic MLP, surrogate gradients, Adam optimizers
- Shared: `_qlif_layers.py` from QSNNReinforce change (QLIF circuit execution, surrogate spike, sensory encoding)

**Testing:**

- 65 unit tests across 10 test classes covering config validation, rollout buffer, critic, initialization, preprocessing, forward pass, PPO learning, episode lifecycle, reproducibility, integration, error handling, and registration
- Benchmark: pending initial experiment results on pursuit predator task

**Documentation:**

- Updated `docs/research/quantum-architectures.md` with QSNN-PPO architecture design
