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

**Rationale**: `Brain` is a `@runtime_checkable Protocol` — adding methods would break structural subtyping for every brain that doesn't implement them (all 18 today). A separate protocol enables opt-in. The free functions `save_weights()` / `load_weights()` check for `WeightPersistence` and no-op or error for non-implementing brains.

**Alternative considered**: Default no-op methods on `Brain`. Rejected because protocols don't support default implementations (they're structural, not inherited), and adding concrete methods would require converting to an ABC, which is a much larger change.

### 2. Component-based architecture

**Decision**: Weight save/load uses named components. Each brain declares components via `get_weight_components()` returning `dict[str, WeightComponent]`. Load accepts an optional component filter.

**Rationale**: Hybrid brains need partial loading — stage 1 saves QSNN only, stage 2 loads QSNN and saves cortex. A flat save/load cannot express this. Components also enable meaningful metadata per component.

**Component naming convention**: Brain-defined names following a convention — `policy`/`value` for actor/critic networks, `optimizer` for optimizer state (single or joint), `training_state` for episode count and similar. Hybrid brains use prefixed names: `qsnn`, `cortex.policy`, `cortex.value`. Not enforced by protocol — documented convention.

### 3. Single `.pt` file format with component keys

**Decision**: All components go into one file, keyed by component name, plus a `_metadata` key.

```python
{
    "policy": {"0.weight": tensor, "0.bias": tensor, ...},
    "value": {"0.weight": tensor, "0.bias": tensor, ...},
    "optimizer": {optimizer state_dict},
    "training_state": {"episode_count": int, ...},
    "_metadata": {
        "brain_type": "MLPPPOBrain",
        "saved_at": "2026-03-22T14:30:00Z",
        "components": ["policy", "value", "optimizer", "training_state"],
        "shapes": {"policy.0.weight": [64, 2], ...},
        "episode_count": 500,
    },
}
```

**Rationale**: Atomic writes (no partial file states), single artifact to share, component filtering at load time provides the same flexibility as separate files. Coexists with hybrid brains' existing separate-file format.

**Alternative considered**: Directory with one file per component. Rejected — non-atomic, more files to manage, and the hybrid wrapper needs to support both old and new formats anyway.

### 4. No fallback — explicit implementation required

**Decision**: `save_weights()` and `load_weights()` check `isinstance(brain, WeightPersistence)`. If false, they no-op with a warning (for auto-save) or raise an error (for explicit CLI flags).

**Rationale**: The `plasticity/snapshot.py` system was designed for in-memory plasticity evaluation, not persistent weight files. Repurposing it as a fallback would create a fragile format bridge — snapshot uses `{module_attr_name: state_dict}` while the new system uses `{component_name: state_dict, _metadata: {...}}`. A no-op is honest: if a brain doesn't implement `WeightPersistence`, the user learns immediately rather than getting a file that may not restore correctly. Any brain that needs persistence can implement the two-method protocol.

**Strictness levels**:

- Auto-save in training loop: skip silently (debug log) if brain doesn't implement `WeightPersistence`
- `--load-weights` / `--save-weights` CLI flags: raise an error — the user explicitly asked for weight persistence and the brain can't provide it

**Alternative considered**: Fallback to `snapshot_brain_state()` / `restore_brain_state()`. Rejected — format mismatch between snapshot dict and component-keyed dict would require a translation layer, and snapshot doesn't support component filtering, metadata, or buffer reset for the `WeightPersistence` path.

### 5. CLI overrides config

**Decision**: `--load-weights` CLI flag overrides `config.weights_path`. Both are optional; at most one source is used for loading.

**Resolution order**:

1. If `--load-weights` provided → use that path
2. Else if `config.weights_path` is set → use that path
3. Else → no weight loading

For saving: `--save-weights` provides an explicit additional output path. Auto-save to `exports/{session_id}/weights/final.pt` always happens regardless.

### 6. Auto-save placement in training loop

**Decision**: Auto-save happens after the training loop completes (after line ~785 in `run_simulation.py`, before plots/exports), not inside `post_process_episode()`. Auto-save also happens on `KeyboardInterrupt` to preserve partial training progress.

**Rationale**: Placing it in the training loop keeps the brain classes clean — brains shouldn't know about session export structure. The hybrid brains' existing per-episode auto-save in `post_process_episode()` stays untouched (wrapper approach). The new auto-save is additive.

**Known gap**: The `manyworlds_mode` code path in `run_simulation.py` (line ~478) returns early and bypasses the normal training loop. Auto-save does not apply to manyworlds mode.

### 7. `weights_only=True` for `torch.load()`

**Decision**: Always use `weights_only=True` when loading weight files.

**Rationale**: `weights_only=False` allows arbitrary code execution via pickle deserialization — a security risk. The original brief (`save_load_weights.md`) proposed `weights_only=False`, but our weight files contain only state dicts (tensors and primitives), which are safe with `weights_only=True`. The `_metadata` dict contains only strings, lists, and ints — all supported by `weights_only=True`.

**Implication for training_state**: The `training_state` component stores `episode_count` as a plain int inside a dict, which `weights_only=True` supports. Optimizer state dicts (containing tensors and scalars) are also supported.

### 8. Hybrid wrapper scope

**Decision**: Thin wrappers only for all three hybrid brains. `get_weight_components()` builds component dicts from existing internal state. `load_weight_components()` delegates to existing load logic. Existing config fields and auto-save behavior unchanged.

**Covered brains**:

- `HybridQuantumBrain`: components `qsnn`, `cortex.policy`, `cortex.value`
- `HybridClassicalBrain`: components `reflex`, `cortex.policy`, `cortex.value`
- `HybridQuantumCortexBrain`: components `reflex`, `cortex`, `critic` (three-component system with 4 training stages)

**What this means concretely**:

- `save_weights(hybrid_brain, path)` produces a single `.pt` with all component keys
- `load_weights(hybrid_brain, path, components={"qsnn"})` loads only the QSNN component
- `config.qsnn_weights_path` continues to work via existing `__init__` logic (separate from `WeightPersistence`)
- Per-episode auto-save in `post_process_episode()` continues using the old separate-file format

### 9. Weight loading is caller-controlled, not in `__init__`

**Decision**: `MLPPPOBrain.__init__` does NOT load weights from `config.weights_path`. Instead, the training loop (`run_simulation.py`) resolves the weight path (CLI overrides config) and calls `load_weights()` once after brain construction.

**Rationale**: If both `config.weights_path` and `--load-weights` are set, loading in `__init__` wastes work (loads config path, then CLI path overwrites it). Keeping resolution in one place (the training script) avoids double-load and matches the existing pattern where hybrid brains handle their own specific config fields, while the new generic `weights_path` is consumed by the caller.

### 10. PPO buffer reset on weight load

**Decision**: `load_weight_components()` implementations SHALL reset the PPO rollout buffer after loading weights.

**Rationale**: Stale experience in the buffer was collected under different weights. If not cleared, the first PPO update after loading would use off-policy data, potentially corrupting the loaded policy. The existing `restore_brain_state()` in `plasticity/snapshot.py` already does this — our implementations must match.

## Risks / Trade-offs

**[Two weight formats coexist for hybrid brains]** → Accepted trade-off of the wrapper approach. The old format (separate `qsnn_weights.pt`, `cortex_weights.pt`) continues for config-based loading and per-episode auto-save. The new format (single file with component keys) is used by `save_weights()` / `load_weights()` and CLI flags. A future change can consolidate. Documented in code comments.

**\[`_metadata` brain_type is class name, not BrainType enum\]** → Using the class name (e.g. `"MLPPPOBrain"`) rather than the enum value (e.g. `"mlpppo"`) because the free functions receive a `Brain` instance, not a `BrainType`. The class name is more specific and works for brains not in the enum. Mismatch produces a warning, not an error — allows loading weights from compatible but renamed brains.

**[Auto-save adds files to every session]** → One extra `.pt` file per session in `exports/{session_id}/weights/`. For MLP PPO this is ~1-5MB. The `exports/` directory is already gitignored and intended for ephemeral session outputs. Acceptable.

**\[`weights_only=True` may reject future complex state\]** → If a brain needs to save non-tensor, non-primitive state (e.g. custom Python objects), `weights_only=True` will reject it. Mitigation: `WeightComponent.state` is typed as `dict[str, Any]` but in practice must contain only torch-safe types. Document this constraint.

**[Component name collisions across brains]** → If two different brain types use the same component name (e.g. `"policy"`) with different semantics, loading weights from brain A into brain B would succeed but produce garbage. Mitigation: the `_metadata.brain_type` check warns on mismatch. Brain-specific shape validation in `load_state_dict(strict=True)` catches dimension mismatches.

**[Optimizer state shape errors on architecture mismatch]** → Optimizer state_dicts encode parameter shapes implicitly via `exp_avg` and `exp_avg_sq` tensors. Loading optimizer state from a different architecture would fail with shape errors. Mitigation: `load_state_dict(strict=True)` on the network catches mismatches first; optimizer state is only loaded if network state succeeds. Document that optimizer state is only valid when architectures match exactly.

**[Random projection matrix not persisted]** → MLPPPOBrain with `feature_expansion="random_projection"` creates a numpy `_projection_matrix` regenerated from `feature_expansion_seed`, not included in `state_dict()`. This works as long as the same brain config is used for both save and load (the brief's stated constraint: only environment parameters should change between curriculum stages). The generic `_metadata` does not capture brain-specific config; the shapes dict provides indirect validation (different projection dim = different input layer shape).

**[Manyworlds mode bypasses auto-save]** → The `manyworlds_mode` code path returns early and skips auto-save. Documented as known gap; manyworlds mode is a specialized execution path unlikely to need weight persistence.
