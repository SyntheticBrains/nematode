## Why

After 64+ experiment sessions across 5 quantum/hybrid architectures (QRC, QVarCircuit, QSNN-REINFORCE, QSNN-PPO, QSNNReinforce-A2C), we've established that QSNN-REINFORCE produces strong reactive policies (73.9% foraging success) but cannot learn multi-objective behaviours (predator evasion never improved across 16 optimization rounds). QSNN-PPO is fundamentally incompatible with surrogate gradients (importance ratio always 1.0), and the A2C critic fails on quantum features (explained variance stayed at ~0 across 4 rounds). Meanwhile, classical PPO achieves 83-98% on the same tasks. A hierarchical hybrid architecture that preserves the proven quantum reflex layer while adding a classical cortex for strategic modulation is the logical next step, supported by external research showing hybrid approaches (quantum for ONE component) succeed where pure quantum fails (QA2C arXiv:2401.07043, PPO-Q arXiv:2501.07085).

## What Changes

- Add a new `HybridQuantumBrain` that combines a QSNN reflex layer (QLIFNetwork, ~212 quantum params, REINFORCE) with a classical cortex MLP (~5K actor params, PPO) for strategic multi-objective behaviour
- Implement a fusion mechanism where the cortex learns mode gating (forage/evade/explore) to modulate trust in the QSNN reflex, plus additive action biases
- Add a classical-only critic (8-dim sensory input, ~5K params) trained alongside the cortex via PPO — using sensory-only input based on lessons from A2C failure
- Support three-stage curriculum training: (1) QSNN reflex on foraging, (2) freeze QSNN + train cortex PPO on multi-objective, (3) optional joint fine-tune
- Register the new brain type in the existing plugin system (dtypes, factory, config loader)
- Add example config files for foraging (stage 1) and pursuit-predator (stage 2/3) environments

## Capabilities

### New Capabilities

- `hybrid-quantum-brain`: Hierarchical hybrid quantum brain architecture combining QSNN reflex layer with classical cortex MLP, including fusion mechanism, stage-aware training, PPO rollout buffer, and mode gating

### Modified Capabilities

- None. This is a new brain type that reuses existing infrastructure (`_qlif_layers.py`, `QuantumBrain` base class) without modifying their requirements.

## Impact

- **New files**: `hybridquantum.py` (~800-1000 lines), `test_hybridquantum.py` (~400-500 lines), 2 example config YAMLs
- **Modified files**: `dtypes.py` (new enum), `__init__.py` (new exports), `brain_factory.py` (new elif), `config_loader.py` (new config map entry)
- **Dependencies**: No new external dependencies. Reuses `_qlif_layers.py` (QLIFNetwork), `torch.nn` (cortex MLP), existing PPO patterns from `mlpppo.py`
- **Documentation**: `quantum-architectures.md` to be updated with new architecture section
