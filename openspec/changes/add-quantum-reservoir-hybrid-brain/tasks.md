## 1. Shared Utility Extraction

- [x] 1.1 Create `brain/arch/_quantum_utils.py` with `get_qiskit_backend()` moved from `_qlif_layers.py`, plus `run_circuit_shots()` helper
- [x] 1.2 Update `_qlif_layers.py` to import and re-export `get_qiskit_backend` from `_quantum_utils` for backward compatibility
- [x] 1.3 Update `qrc.py` to import `get_qiskit_backend` from `_quantum_utils` instead of inline implementation
- [x] 1.4 Create `brain/arch/_quantum_reservoir.py` with `build_readout_network()` extracted from `qrc.py`
- [x] 1.5 Update `qrc.py` to use `build_readout_network()` from `_quantum_reservoir`
- [x] 1.6 Run full test suite to verify refactoring is backward-compatible: `uv run pytest`

## 2. Brain Type Registration

- [x] 2.1 Add `QRH = "qrh"` to `BrainType` enum in `brain/arch/dtypes.py`; add to `QUANTUM_BRAIN_TYPES` and `BRAIN_TYPES`
- [x] 2.2 Add `QRHBrain` and `QRHBrainConfig` exports to `brain/arch/__init__.py`
- [x] 2.3 Add `QRHBrainConfig` to imports, union type, and `elif BrainType.QRH` branch in `utils/brain_factory.py`
- [x] 2.4 Add `QRHBrainConfig` to imports, `BrainConfigType` union, and `BRAIN_CONFIG_MAP` in `utils/config_loader.py`

## 3. Core QRH Brain Implementation

- [x] 3.1 Create `brain/arch/qrh.py` — module docstring, imports, and default/validation constants
- [x] 3.2 Implement C. elegans topology constants: gap junction CZ pairs and chemical synapse CRY/CRZ angles for 8-qubit mapping (ASEL/ASER → AIYL/AIYR → AIAL/AIAR → AVAL/AVAR)
- [x] 3.3 Implement `QRHBrainConfig(BrainConfig)` with all config fields, Pydantic validators, and LR scheduling parameters
- [x] 3.4 Implement `QRHBrain.__init__()` — seeding, sensory modules, reservoir setup, actor/critic MLPs, single combined Adam optimizer, LayerNorm, rollout buffer, episode state, LR scheduling
- [x] 3.5 Implement `_build_structured_reservoir()` — structured CZ + CRY/CRZ gate sequence from topology constants
- [x] 3.6 Implement `_build_random_reservoir()` — random CZ + random CRY/CRZ variant for MI comparison
- [x] 3.7 Implement `_encode_and_run(features)` — Qiskit circuit construction (Hadamard → data re-uploading on sensory qubits → reservoir layers) with `Statevector` simulation
- [x] 3.8 Implement `_extract_features(statevector)` — X/Y/Z-expectations + ZZ-correlations from probability amplitudes (3N + N(N-1)/2 features)
- [x] 3.9 Implement `preprocess(params)` — legacy 2-feature and unified sensory module input modes
- [x] 3.10 Implement `run_brain()` — feature extraction → LayerNorm → actor forward → action sampling → critic value → buffer staging
- [x] 3.11 Implement `learn()` — reward storage, buffer management, PPO update trigger at buffer full or episode end
- [x] 3.12 Implement `_perform_ppo_update()` — GAE computation, multi-epoch minibatch PPO with clipped surrogate loss, value loss, entropy bonus, gradient clipping
- [x] 3.13 Implement remaining Brain protocol methods: `update_memory()`, `prepare_episode()`, `post_process_episode()`, `copy()`, `action_set` property
- [x] 3.14 Implement `num_sensory_qubits` auto-computation from sensory modules with explicit override support
- [x] 3.15 Implement LR scheduling: warmup phase (`lr_warmup_episodes`, `lr_warmup_start`) and decay phase (`lr_decay_episodes`, `lr_decay_end`)

## 4. Configuration Files

- [x] 4.1 Create `configs/examples/qrh_foraging_small.yml` — 8 qubits, grid 20, foraging environment
- [x] 4.2 Create `configs/examples/qrh_pursuit_predators_small.yml` — 8 qubits, pursuit predators, sensory_modules: [food_chemotaxis, nociception]
- [x] 4.3 Create `configs/examples/qrh_thermotaxis_pursuit_predators_small.yml` — 8 qubits, thermotaxis + pursuit predators (added during R3-R4 iteration)
- [x] 4.4 Create `configs/examples/qrh_thermotaxis_pursuit_predators_large.yml` — 10 qubits, 100x100 grid, 4 predators, thermotaxis, extended LR scheduling (primary training config for R5-R9)

## 5. Tests

- [x] 5.1 Create `tests/.../brain/arch/test_quantum_utils.py` — tests for shared `get_qiskit_backend()` and `run_circuit_shots()`
- [x] 5.2 Create `tests/.../brain/arch/test_quantum_reservoir.py` — tests for shared `build_readout_network()`
- [x] 5.3 Create `tests/.../brain/arch/test_qrh.py` — `TestQRHBrainConfig`: default config, custom values, all validators
- [x] 5.4 Add `TestQRHReservoirCircuit`: CZ gate presence, controlled rotations, seed reproducibility, structured vs random topology
- [x] 5.5 Add `TestQRHFeatureExtraction`: dimension (52 for 8q), X/Y/Z range [-1,1], ZZ range [-1,1], input sensitivity, determinism
- [x] 5.6 Add `TestQRHBrainReadout`: actor output shape, critic output shape
- [x] 5.7 Add `TestQRHBrainLearning`: `run_brain()` returns valid ActionData, PPO update changes weights, buffer management, full episode workflow
- [x] 5.8 Add `TestQRHBrainCopy`: copy independence, shared reservoir topology
- [x] 5.9 Add `TestQRHBrainSensoryModules`: unified mode dimensions, legacy fallback
- [x] 5.10 Add QRH foraging config to smoke test `SIMULATION_CONFIGS` list
- [x] 5.11 Run full test suite: `uv run pytest -m "not smoke and not nightly"` and `uv run pytest -m smoke -k qrh -v`
- [x] 5.12 Add `TestQRHEpisodeBoundaries`: episode state management, no cross-contamination, action validation
- [x] 5.13 Add `TestQRHLRScheduling`: warmup, decay, combined scheduling (8 tests)
- [x] 5.14 Add `TestQRHSensoryQubits`: auto-computation, explicit override, validation, wrapping

## 6. MI Decision Gate Script

- [x] 6.0 Add `scikit-learn` as project dependency (needed for `mutual_info_classif`)
- [x] 6.1 Create `scripts/qrh_mi_analysis.py` — oracle dataset generation using rule-based gradient-following policy
- [x] 6.2 Implement structured vs random vs classical MLP feature extraction comparison (three-way)
- [x] 6.3 Implement MI computation using `sklearn.feature_selection.mutual_info_classif`
- [x] 6.4 Implement permutation test (1000 permutations) for p-value significance
- [x] 6.5 Add CLI argument parsing and structured results output

## 7. Documentation

- [x] 7.1 Update `AGENTS.md` — add `qrh` to brain architecture list, update count to 13
- [x] 7.2 Update `README.md` — add QRH to brain architecture section, update "12 brain architectures" to 13
- [x] 7.3 Update `CONTRIBUTING.md` — add QRH entry under "Quantum" in architecture list, update count to 13

## 8. Verification

- [x] 8.1 Run lint: `/opt/homebrew/bin/uv run pre-commit run -a` — ruff check/format, mdformat, markdownlint all pass
- [x] 8.2 Run full test suite: `/opt/homebrew/bin/uv run pytest` — 1507 passed
- [x] 8.3 Run smoke tests: `/opt/homebrew/bin/uv run pytest -m smoke -k qrh -v` — 1 passed
- [x] 8.4 Manual smoke test: `/opt/homebrew/bin/uv run ./scripts/run_simulation.py --runs 2 --config ./configs/examples/qrh_foraging_small.yml`
