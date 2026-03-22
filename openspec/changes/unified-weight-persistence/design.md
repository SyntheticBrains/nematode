## Context

The project has 18 brain architectures but no unified weight persistence. The current state:

- **MLP PPO** (`mlpppo.py`): No save/load at all. Training results are lost when the process exits.
- **Hybrid brains** (`hybridquantum.py`, `hybridclassical.py`): Ad-hoc private methods (`_save_qsnn_weights`, `_save_cortex_weights`) that auto-save every episode to `exports/{session_id}/`. Weight paths are config fields (`qsnn_weights_path`, `cortex_weights_path`). Shared helpers in `_hybrid_common.py`.
- **Plasticity system** (`plasticity/snapshot.py`): Generic in-memory snapshot/restore via `_get_torch_modules()` discovery. Also has `save_checkpoint()` for disk persistence. Brain-agnostic but doesn't support component filtering or metadata.
- **CLI** (`run_simulation.py`): No weight-related arguments. Brains receive session IDs via `set_session_id()` (duck-typed, not on protocol).

The primary use case driving this work is **curriculum learning** for MLP PPO: train on an easy environment config, save weights, load into a harder config, continue training. Secondary use cases: standardizing hybrid weight persistence and enabling generic tooling.

## Goals / Non-Goals

**Goals:**

- Unified `save_weights()` / `load_weights()` API that works for any brain
- Component-based architecture supporting partial save/load (hybrid curriculum stages)
- MLP PPO full weight persistence (policy, value, optimizers, training state)
- CLI flags (`--load-weights`, `--save-weights`) on `run_simulation.py`
- Auto-save `final.pt` to session export directory after training
- Config-based weight loading via `weights_path` on `BrainConfig`
- Metadata in saved files for diagnostics (brain type, shapes, episode count)
- Hybrid brain wrappers (thin, delegating to existing private methods)

**Non-Goals:**

- Periodic checkpointing (`--checkpoint-interval`) — deferred
- Best-model tracking (requires metric definition) — deferred
- Quantum parameter persistence (QVarCircuit `dict[str, float]`) — out of scope
- Full hybrid method consolidation (replacing private methods) — deferred
- Model-level persistence (saving architecture definition, not just weights) — out of scope

## Decisions

### 1. Separate protocol instead of extending `Brain`

**Decision**: Create a new `WeightPersistence` protocol in `brain/weights.py` rather than adding methods to the `Brain` protocol.

**Rationale**: `Brain` is a `@runtime_checkable Protocol` — adding methods would break structural subtyping for every brain that doesn't implement them (all 18 today). A separate protocol enables opt-in. The free functions `save_weights()` / `load_weights()` check for `WeightPersistence` first, then fall back to the generic `plasticity/snapshot.py` system.

**Alternative considered**: Default no-op methods on `Brain`. Rejected because protocols don't support default implementations (they're structural, not inherited), and adding concrete methods would require converting to an ABC, which is a much larger change.

### 2. Component-based architecture

**Decision**: Weight save/load uses named components. Each brain declares components via `get_weight_components()` returning `dict[str, WeightComponent]`. Load accepts an optional component filter.

**Rationale**: Hybrid brains need partial loading — stage 1 saves QSNN only, stage 2 loads QSNN and saves cortex. A flat save/load cannot express this. Components also enable meaningful metadata per component.

**Component naming convention**: Brain-defined names following a convention — `policy`/`value` for actor/critic networks, `policy_optimizer`/`value_optimizer` for optimizer state, `training_state` for episode count and similar. Hybrid brains use prefixed names: `qsnn`, `cortex.policy`, `cortex.value`. Not enforced by protocol — documented convention.

### 3. Single `.pt` file format with component keys

**Decision**: All components go into one file, keyed by component name, plus a `_metadata` key.

```python
{
    "policy": {"0.weight": tensor, "0.bias": tensor, ...},
    "value": {"0.weight": tensor, "0.bias": tensor, ...},
    "policy_optimizer": {optimizer state_dict},
    "value_optimizer": {optimizer state_dict},
    "training_state": {"episode_count": int, ...},
    "_metadata": {
        "brain_type": "MLPPPOBrain",
        "saved_at": "2026-03-22T14:30:00Z",
        "components": ["policy", "value", ...],
        "shapes": {"policy.0.weight": [64, 12], ...},
        "episode_count": 500,
    },
}
```

**Rationale**: Atomic writes (no partial file states), single artifact to share, component filtering at load time provides the same flexibility as separate files. Coexists with hybrid brains' existing separate-file format.

**Alternative considered**: Directory with one file per component. Rejected — non-atomic, more files to manage, and the hybrid wrapper needs to support both old and new formats anyway.

### 4. Fallback to plasticity/snapshot.py for non-implementing brains

**Decision**: `save_weights()` and `load_weights()` check `isinstance(brain, WeightPersistence)`. If false, fall back to `snapshot_brain_state()` / `restore_brain_state()` from `plasticity/snapshot.py`.

**Rationale**: This means every brain with PyTorch modules gets basic save/load for free, without implementing `WeightPersistence`. The snapshot system already discovers modules via `_get_torch_modules()`. The fallback doesn't support component filtering or brain-specific validation, but it works for auto-save and simple round-trips.

### 5. CLI overrides config

**Decision**: `--load-weights` CLI flag overrides `config.weights_path`. Both are optional; at most one source is used for loading.

**Resolution order**:

1. If `--load-weights` provided → use that path
2. Else if `config.weights_path` is set → use that path
3. Else → no weight loading

For saving: `--save-weights` provides an explicit additional output path. Auto-save to `exports/{session_id}/weights/final.pt` always happens regardless.

### 6. Auto-save placement in training loop

**Decision**: Auto-save happens after the training loop completes (after line ~785 in `run_simulation.py`, before plots/exports), not inside `post_process_episode()`.

**Rationale**: Placing it in the training loop keeps the brain classes clean — brains shouldn't know about session export structure. The hybrid brains' existing per-episode auto-save in `post_process_episode()` stays untouched (wrapper approach). The new auto-save is additive.

### 7. `weights_only=True` for `torch.load()`

**Decision**: Always use `weights_only=True` when loading weight files.

**Rationale**: `weights_only=False` allows arbitrary code execution via pickle deserialization — a security risk. The original brief (`save_load_weights.md`) proposed `weights_only=False`, but our weight files contain only state dicts (tensors and primitives), which are safe with `weights_only=True`. The `_metadata` dict contains only strings, lists, and ints — all supported by `weights_only=True`.

**Implication for training_state**: The `training_state` component stores `episode_count` as a plain int inside a dict, which `weights_only=True` supports. Optimizer state dicts (containing tensors and scalars) are also supported.

### 8. Hybrid wrapper scope

**Decision**: Thin wrappers only. `HybridQuantumBrain.get_weight_components()` calls `_save_qsnn_weights`-style logic to build component dicts. `load_weight_components()` delegates to `_load_qsnn_weights` / `_load_cortex_weights`. Existing config fields and auto-save behavior unchanged.

**What this means concretely**:

- `save_weights(hybrid_brain, path)` produces a single `.pt` with `qsnn`, `cortex.policy`, `cortex.value` keys
- `load_weights(hybrid_brain, path, components={"qsnn"})` loads only the QSNN component
- `config.qsnn_weights_path` continues to work via existing `__init__` logic (separate from `WeightPersistence`)
- Per-episode auto-save in `post_process_episode()` continues using the old separate-file format

## Risks / Trade-offs

**[Two weight formats coexist for hybrid brains]** → Accepted trade-off of the wrapper approach. The old format (separate `qsnn_weights.pt`, `cortex_weights.pt`) continues for config-based loading and per-episode auto-save. The new format (single file with component keys) is used by `save_weights()` / `load_weights()` and CLI flags. A future change can consolidate. Documented in code comments.

**\[`_metadata` brain_type is class name, not BrainType enum\]** → Using the class name (e.g. `"MLPPPOBrain"`) rather than the enum value (e.g. `"mlpppo"`) because the free functions receive a `Brain` instance, not a `BrainType`. The class name is more specific and works for brains not in the enum. Mismatch produces a warning, not an error — allows loading weights from compatible but renamed brains.

**[Auto-save adds files to every session]** → One extra `.pt` file per session in `exports/{session_id}/weights/`. For MLP PPO this is ~1-5MB. The `exports/` directory is already gitignored and intended for ephemeral session outputs. Acceptable.

**\[`weights_only=True` may reject future complex state\]** → If a brain needs to save non-tensor, non-primitive state (e.g. custom Python objects), `weights_only=True` will reject it. Mitigation: `WeightComponent.state` is typed as `dict[str, Any]` but in practice must contain only torch-safe types. Document this constraint.

**[Component name collisions across brains]** → If two different brain types use the same component name (e.g. `"policy"`) with different semantics, loading weights from brain A into brain B would succeed but produce garbage. Mitigation: the `_metadata.brain_type` check warns on mismatch. Brain-specific shape validation in `load_state_dict(strict=True)` catches dimension mismatches.
