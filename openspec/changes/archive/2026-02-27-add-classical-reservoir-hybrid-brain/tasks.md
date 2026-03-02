## 1. ReservoirHybridBase Extraction

- [x] 1.1 Create `brain/arch/_reservoir_hybrid_base.py` — module docstring, imports, and `ReservoirHybridBaseConfig(BrainConfig)` with all shared config fields (readout dims, PPO params, LR scheduling, sensory modules)
- [x] 1.2 Move `_RolloutBuffer` class from `qrh.py` into `_reservoir_hybrid_base.py`
- [x] 1.3 Implement `ReservoirHybridBase(ClassicalBrain)` with abstract methods `_get_reservoir_features(sensory_features) -> np.ndarray` and `_compute_feature_dim() -> int`, and `_brain_name` class attribute for diagnostic log prefixes
- [x] 1.4 Move shared `__init__()` logic into base, accepting `feature_dim` as explicit argument from subclass (see design Decision 7): seeding (`ensure_seed`/`get_rng`/`set_global_seed`), sensory module setup (`input_dim` computation), actor/critic MLP construction via `build_readout_network()`, LayerNorm, optimizer, LR scheduling setup, PPO params, buffer, state tracking (`_pending_*`, `current_probabilities`, `last_value`, `history_data`, `latest_data`), episode tracking
- [x] 1.5 Move shared methods into base: `run_brain()`, `learn()`, `_perform_ppo_update()`, `preprocess()`, `prepare_episode()`, `post_process_episode()`, `update_memory()`, `_get_current_lr()`, `_update_learning_rate()`, `action_set` property/setter
- [x] 1.6 Move `copy()` into base using construct-then-copy-weights pattern: serialize config via `model_dump()`, call subclass `_create_copy_instance(config)` to build fresh instance, deep-copy actor/critic/feature_norm/optimizer state dicts and `_episode_count`

## 2. QRH Refactor

- [x] 2.1 Refactor `QRHBrainConfig` to inherit from `ReservoirHybridBaseConfig`, keeping only quantum-specific fields (`num_reservoir_qubits`, `reservoir_depth`, `reservoir_seed`, `shots`, `use_random_topology`, `num_sensory_qubits`)
- [x] 2.2 Refactor `QRHBrain` to inherit from `ReservoirHybridBase`, implementing `_get_reservoir_features()` and `_compute_feature_dim()` (refactor from module-level function to instance method using `self.num_qubits`). Set `_brain_name = "QRH"`. Compute `feature_dim` before calling `super().__init__(config, feature_dim, ...)`
- [x] 2.3 Keep quantum-specific code in `qrh.py`: topology constants, `_build_structured_reservoir()`, `_build_random_reservoir()`, `_generate_random_topology()`, `_encode_and_run()`, `_extract_features()`
- [x] 2.4 Implement `_create_copy_instance(config)` to construct a new `QRHBrain` (random topology is regenerated from seed in `__init__`)
- [x] 2.5 Run full QRH test suite to verify zero behavioral changes: `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qrh.py -v`

## 3. CRH Brain Implementation

- [x] 3.1 Create `brain/arch/crh.py` — module docstring, imports, and default/validation constants
- [x] 3.2 Define `FeatureChannel = Literal["raw", "cos_sin", "squared", "pairwise"]` type alias
- [x] 3.3 Implement `CRHBrainConfig(ReservoirHybridBaseConfig)` with ESN-specific fields: `num_reservoir_neurons`, `reservoir_depth`, `reservoir_seed`, `spectral_radius`, `input_connectivity`, `input_scale`, `feature_channels: list[FeatureChannel]`, `num_sensory_neurons`
- [x] 3.4 Add Pydantic validators for `feature_channels` (non-empty), `input_connectivity` (sparse/dense), `spectral_radius` (> 0), `num_reservoir_neurons` (>= 2), `num_sensory_neurons` (\<= `num_reservoir_neurons` when set)
- [x] 3.5 Implement `CRHBrain.__init__()` — W_in/W_res construction, spectral radius scaling, compute `feature_dim`, then call `super().__init__(config, feature_dim, ...)` which handles seeding, readout, optimizer, buffer (design Decision 7). Set `_brain_name = "CRH"`.
- [x] 3.6 Implement `_build_reservoir_matrices()` — W_in (sparse: zero rows for non-sensory neurons; dense: full matrix), W_res (random normal, eigenvalue-scaled to spectral_radius with epsilon guard for degenerate matrices)
- [x] 3.7 Implement `_get_reservoir_features(sensory_features)` — ESN forward pass: h_0 = tanh(W_in @ x), h_l = tanh(W_res @ h\_{l-1} + W_in @ x), then feature extraction
- [x] 3.8 Implement `_extract_features(activations)` — apply configured feature channels (raw, cos_sin, squared, pairwise) and concatenate
- [x] 3.9 Implement `_compute_feature_dim()` — sum dimensions from configured channels
- [x] 3.10 Implement `_create_copy_instance(config)` to construct a new `CRHBrain` (W_in/W_res are regenerated from seed in `__init__`)

## 4. Brain Type Registration

- [x] 4.1 Add `CRH = "crh"` to `BrainType` enum in `brain/arch/dtypes.py`; add `"crh"` to the `BRAIN_TYPES` Literal type alias; add `BrainType.CRH` to the `CLASSICAL_BRAIN_TYPES` set
- [x] 4.2 Add `CRHBrain`, `CRHBrainConfig`, `ReservoirHybridBase`, `ReservoirHybridBaseConfig` exports to `brain/arch/__init__.py`
- [x] 4.3 Add `CRHBrainConfig` to imports, union type, and `elif BrainType.CRH` branch in `utils/brain_factory.py`
- [x] 4.4 Add `CRHBrainConfig` to imports, `BrainConfigType` union, and `BRAIN_CONFIG_MAP` in `utils/config_loader.py`

## 5. Configuration Files

- [x] 5.1 Create `configs/examples/crh_thermotaxis_pursuit_predators_large.yml` — ablation mode (N=10, sparse, [raw, cos_sin, pairwise], 75 features) matching QRH R9 environment config
- [x] 5.2 Create `configs/examples/crh_thermotaxis_pursuit_predators_large_standalone.yml` — standalone mode (tuned N=14, all 4 channels, dense input) for MLPPPO comparison
- [x] 5.3 Create `configs/examples/crh_thermotaxis_stationary_predators_large.yml` — stationary predator ablation mode matching QRH stationary config for 2×2 comparison matrix (QRH/CRH × pursuit/stationary)

## 6. Tests

- [x] 6.1 Create `tests/.../brain/arch/test_reservoir_hybrid_base.py` — tests for base config defaults, base class instantiation (via QRH subclass), PPO buffer, shared methods (18 tests)
- [x] 6.2 Create `tests/.../brain/arch/test_crh.py` — `TestCRHBrainConfig`: default config, custom values, all validators (9 tests)
- [x] 6.3 Add `TestCRHReservoir`: W_in/W_res shapes, spectral radius scaling, seed reproducibility, sparse vs dense connectivity (8 tests)
- [x] 6.4 Add `TestCRHFeatureExtraction`: dimension for each channel combo, raw range [-1,1], cos_sin range, pairwise range, ablation mode = 75 features (12 tests)
- [x] 6.5 Add `TestCRHBrainReadout`: actor output shape, critic output shape (2 tests)
- [x] 6.6 Add `TestCRHBrainLearning`: `run_brain()` returns valid ActionData, PPO update changes weights, buffer management, full episode workflow (4 tests)
- [x] 6.7 Add `TestCRHBrainCopy`: copy independence, shared W_in/W_res values, independent readout weights (5 tests)
- [x] 6.8 Add `TestCRHBrainSensoryModules`: unified mode dimensions, legacy fallback, triple-objective (3 tests)
- [x] 6.9 Add CRH config to smoke test `SIMULATION_CONFIGS` list — `crh_foraging_small.yml`
- [x] 6.10 Run QRH regression: `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qrh.py -v` — 53 passed, zero modifications needed
- [x] 6.11 Run full test suite: 1567 passed (42 new CRH + 18 base + 53 QRH unchanged); CRH smoke test: 1 passed

## 7. Documentation

- [x] 7.1 Update `AGENTS.md` — add `crh` to brain architecture list, update count from 13 to 14 (QRH already added)
- [x] 7.2 Update `README.md` — add CRH to brain architecture section, update count from 13 to 14, add CLI example
- [x] 7.3 Update `CONTRIBUTING.md` — add CRH under "Classical" in architecture list, update count from 13 to 14

## 8. Verification

- [x] 8.1 Run lint: `uv run pre-commit run -a` — ruff check/format passed, yaml/toml/markdown passed (pyright/pytest hooks fail due to uv PATH in pre-commit sandbox — known infra issue)
- [x] 8.2 Run full test suite: `uv run pytest` — 1590 passed, 85% coverage
- [x] 8.3 Run smoke tests: `uv run pytest -m smoke -v` — 12 passed (including crh_foraging_small.yml)
- [x] 8.4 Manual smoke test: `uv run ./scripts/run_simulation.py --runs 2 --config ./configs/examples/crh_thermotaxis_pursuit_predators_large.yml` — 2 runs completed, no crashes
