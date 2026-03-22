## 1. Core Module — `brain/weights.py`

- [ ] 1.1 Create `quantumnematode/brain/weights.py` with `WeightComponent` dataclass (name, state, metadata fields)
- [ ] 1.2 Define `WeightPersistence` protocol (`@runtime_checkable`) with `get_weight_components()` and `load_weight_components()` methods
- [ ] 1.3 Implement `save_weights()` free function — dispatch to `WeightPersistence` or `snapshot_brain_state()` fallback, build `_metadata` dict (brain_type, saved_at, components, shapes, episode_count), create parent dirs, call `torch.save()`
- [ ] 1.4 Implement `load_weights()` free function — `torch.load(path, weights_only=True)`, dispatch to `WeightPersistence` or `restore_brain_state()` fallback, component filtering, brain_type mismatch warning, FileNotFoundError on missing path
- [ ] 1.5 Export public API from `quantumnematode/brain/__init__.py` (or appropriate location)

## 2. Configuration — `BrainConfig` Base Class

- [ ] 2.1 Add `weights_path: str | None = None` field to `BrainConfig` in `brain/arch/dtypes.py`

## 3. MLP PPO Implementation

- [ ] 3.1 Implement `get_weight_components()` on `MLPPPOBrain` — return components: `policy` (actor state_dict), `value` (critic state_dict), `policy_optimizer`, `value_optimizer`, `training_state` (episode_count)
- [ ] 3.2 Implement `load_weight_components()` on `MLPPPOBrain` — load actor/critic state_dicts, restore optimizer states, restore `_episode_count`, recalculate LR scheduler from episode count
- [ ] 3.3 Add config-based weight loading in `MLPPPOBrain.__init__` — if `config.weights_path` is set, call `load_weights(self, Path(config.weights_path))`

## 4. Hybrid Brain Wrappers

- [ ] 4.1 Implement `get_weight_components()` on `HybridQuantumBrain` — components: `qsnn` (W_sh, W_hm, theta_hidden, theta_motor as detached CPU tensors), `cortex.policy` (cortex actor state_dict), `cortex.value` (cortex critic state_dict)
- [ ] 4.2 Implement `load_weight_components()` on `HybridQuantumBrain` — delegate to existing `_load_qsnn_weights` logic for qsnn component (with shape validation), `load_state_dict()` for cortex components
- [ ] 4.3 Implement `get_weight_components()` on `HybridClassicalBrain` — components: `reflex` (reflex MLP state_dict), `cortex.policy`, `cortex.value`
- [ ] 4.4 Implement `load_weight_components()` on `HybridClassicalBrain` — delegate to existing reflex/cortex load logic

## 5. CLI Integration — `run_simulation.py`

- [ ] 5.1 Add `--load-weights` and `--save-weights` argparse arguments (type=str, default=None)
- [ ] 5.2 After brain construction: if `--load-weights` provided, call `load_weights(brain, path)` (CLI overrides config.weights_path)
- [ ] 5.3 After training loop: auto-save `final.pt` to `exports/{session_id}/weights/` and log the path
- [ ] 5.4 After training loop: if `--save-weights` provided, also save to explicit path

## 6. Tests

- [ ] 6.1 Create `tests/quantumnematode_tests/brain/arch/test_weight_persistence.py` with test scaffolding and fixtures (MLPPPOBrain creation helper, `tmp_path` usage for all file I/O)
- [ ] 6.2 Test save/load round-trip for MLP PPO — verify actor and critic state_dicts match after round-trip
- [ ] 6.3 Test action probability consistency — saved and loaded brain produce identical outputs for same input tensor
- [ ] 6.4 Test training continues after load — verify weights update and loss changes after loading
- [ ] 6.5 Test component filtering — save all components, load only a subset, verify non-loaded components unchanged
- [ ] 6.6 Test architecture mismatch — load weights from different input_dim, verify clear error
- [ ] 6.7 Test parent directory creation — save to nested non-existent path
- [ ] 6.8 Test metadata contents — verify `_metadata` key contains brain_type, saved_at, components, shapes, episode_count
- [ ] 6.9 Test generic fallback — mock brain without `WeightPersistence` saves/loads via snapshot system
- [ ] 6.10 Test CLI flags accepted — argparse accepts `--load-weights` and `--save-weights`
- [ ] 6.11 Test hybrid `WeightPersistence` wrapper — HybridQuantumBrain round-trip with component filtering (partial load of qsnn only)
- [ ] 6.12 Test brain_type mismatch warning — load weights saved by different brain class, verify warning logged

## 7. Verification

- [ ] 7.1 Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_weight_persistence.py -v` — all pass
- [ ] 7.2 Run `uv run pytest -m "not nightly"` — no regressions
- [ ] 7.3 Run `uv run pre-commit run -a` — linting passes
