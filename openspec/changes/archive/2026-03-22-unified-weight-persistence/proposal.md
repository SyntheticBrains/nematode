## Why

MLP PPO — our primary classical baseline — has no weight save/load support, blocking curriculum learning (train on easy config → save → load into harder config → continue). The hybrid brains have weight persistence but use ad-hoc private methods with no shared interface, making it impossible to write generic tooling. There is no CLI support for loading or saving weights at the entrypoint level. This change introduces a unified, component-based weight persistence system that works across brain architectures. Linked to GitHub issue #7.

## What Changes

- **New `WeightPersistence` protocol** — A separate `@runtime_checkable` protocol (not added to `Brain`) with component-based save/load. Brains opt in by implementing `get_weight_components()` and `load_weight_components()`. Brains that don't implement it are treated as no-ops with warnings (no generic fallback — explicit implementation required).
- **New `brain/weights.py` module** — Core types (`WeightComponent` dataclass), the protocol, and `save_weights()`/`load_weights()` free functions that handle dispatch, metadata, and component filtering.
- **MLP PPO weight persistence** — `MLPPPOBrain` implements `WeightPersistence` with components: policy, value, optimizer (single joint Adam), training state. Supports `weights_path` config field for loading pre-trained weights.
- **Hybrid brain wrappers** — `HybridQuantumBrain`, `HybridClassicalBrain`, and `HybridQuantumCortexBrain` gain thin `WeightPersistence` wrappers delegating to existing private `_save/_load` methods. Existing `qsnn_weights_path`/`cortex_weights_path`/`critic_weights_path` config fields continue working.
- **CLI flags** — `--load-weights PATH` and `--save-weights PATH` on `run_simulation.py`. CLI overrides config `weights_path` when both specified.
- **Auto-save final weights** — Brains implementing `WeightPersistence` auto-save `final.pt` to `exports/{session_id}/weights/` at training end (no flag needed). Non-implementing brains skip auto-save silently.
- **`weights_path` on `BrainConfig`** — Generic config field for any brain to specify pre-trained weights to load at initialization.
- **Saved file metadata** — Each `.pt` file includes a `_metadata` key with brain type, timestamp, component names, tensor shapes, and episode count for diagnostic validation.

## Capabilities

### New Capabilities

- `weight-persistence`: Core weight persistence protocol, types, save/load functions, component filtering, metadata validation, and implementations for MLP PPO and HybridClassical brains.

### Modified Capabilities

- `cli-interface`: Add `--load-weights` and `--save-weights` arguments to `run_simulation.py`, plus auto-save of final weights to session export directory.
- `configuration-system`: Add `weights_path` field to `BrainConfig` base class for config-based weight loading.
- `hybrid-quantum-brain`: Add `WeightPersistence` protocol wrapper around existing private save/load methods. No behavioral changes to existing config-based loading.
- `hybrid-quantum-cortex-brain`: Add `WeightPersistence` protocol wrapper around existing private save/load methods (reflex, cortex, critic components).

## Impact

- **Code**: New module `brain/weights.py`. Edits to `brain/arch/dtypes.py` (BrainConfig), `brain/arch/mlpppo.py`, `brain/arch/hybridquantum.py`, `brain/arch/hybridclassical.py`, `brain/arch/hybridquantumcortex.py`, `scripts/run_simulation.py`.
- **Config**: All brain YAML configs gain optional `weights_path` field (backward compatible, defaults to null).
- **Dependencies**: No new dependencies — uses existing PyTorch `torch.save`/`torch.load`.
- **File format**: Single `.pt` file per save point containing component-keyed state dicts plus `_metadata`. Coexists with hybrid brains' existing separate-file format.
- **Existing behavior**: No breaking changes. Hybrid config fields (`qsnn_weights_path`, `cortex_weights_path`) continue working. The auto-save adds a `weights/` subdirectory to session exports.
