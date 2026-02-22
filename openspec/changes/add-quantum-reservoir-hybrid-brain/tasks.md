## 1. Shared Utility Extraction

- [x] 1.1 Create `brain/arch/_quantum_utils.py` with `get_qiskit_backend()` moved from `_qlif_layers.py`, plus `run_circuit_shots()` helper
- [x] 1.2 Update `_qlif_layers.py` to import and re-export `get_qiskit_backend` from `_quantum_utils` for backward compatibility
- [x] 1.3 Update `qrc.py` to import `get_qiskit_backend` from `_quantum_utils` instead of inline implementation
- [x] 1.4 Create `brain/arch/_quantum_reservoir.py` with `build_readout_network()` extracted from `qrc.py`
- [x] 1.5 Update `qrc.py` to use `build_readout_network()` from `_quantum_reservoir`
- [x] 1.6 Run full test suite to verify refactoring is backward-compatible: `uv run pytest`

## 2. Brain Type Registration

- [ ] 2.1 Add `QRH = "qrh"` to `BrainType` enum in `brain/arch/dtypes.py`; add to `QUANTUM_BRAIN_TYPES` and `BRAIN_TYPES`
- [ ] 2.2 Add `QRHBrain` and `QRHBrainConfig` exports to `brain/arch/__init__.py`
- [ ] 2.3 Add `QRHBrainConfig` to imports, union type, and `elif BrainType.QRH` branch in `utils/brain_factory.py`
- [ ] 2.4 Add `QRHBrainConfig` to imports, `BrainConfigType` union, and `BRAIN_CONFIG_MAP` in `utils/config_loader.py`

## 3. Core QRH Brain Implementation

- [ ] 3.1 Create `brain/arch/qrh.py` — module docstring, imports, and default/validation constants
- [ ] 3.2 Implement C. elegans topology constants: gap junction CZ pairs and chemical synapse RY/RZ angles for 8-qubit mapping (ASEL/ASER → AIYL/AIYR → AIAL/AIAR → AVAL/AVAR)
- [ ] 3.3 Implement `QRHBrainConfig(BrainConfig)` with all config fields and Pydantic field validators
- [ ] 3.4 Implement `QRHBrain.__init__()` — seeding, sensory modules, reservoir setup, actor/critic MLPs, Adam optimizers, rollout buffer, episode state
- [ ] 3.5 Implement `_build_structured_reservoir()` — structured CZ + fixed RY/RZ gate sequence from topology constants
- [ ] 3.6 Implement `_build_random_reservoir()` — random CZ + random rotation variant for MI comparison
- [ ] 3.7 Implement `_encode_and_run(features)` — Qiskit circuit construction (Hadamard → data re-uploading reservoir layers) with `Statevector` simulation
- [ ] 3.8 Implement `_extract_features(statevector)` — Z-expectations ⟨Z_i⟩ + ZZ-correlations ⟨Z_i Z_j⟩ from probability amplitudes
- [ ] 3.9 Implement `preprocess(params)` — legacy 2-feature and unified sensory module input modes
- [ ] 3.10 Implement `run_brain()` — feature extraction → actor forward → action sampling → critic value → buffer staging
- [ ] 3.11 Implement `learn()` — reward storage, buffer management, PPO update trigger at buffer full or episode end
- [ ] 3.12 Implement `_perform_ppo_update()` — GAE computation, multi-epoch minibatch PPO with clipped surrogate loss, value loss, entropy bonus, gradient clipping
- [ ] 3.13 Implement remaining Brain protocol methods: `update_memory()`, `prepare_episode()`, `post_process_episode()`, `copy()`, `action_set` property

## 4. Configuration Files

- [ ] 4.1 Create `configs/examples/qrh_foraging_small.yml` — 8 qubits, grid 20, foraging environment
- [ ] 4.2 Create `configs/examples/qrh_predators_small.yml` — 8 qubits, pursuit predators, sensory_modules: [food_chemotaxis, nociception, mechanosensation]

## 5. Tests

- [x] 5.1 Create `tests/.../brain/arch/test_quantum_utils.py` — tests for shared `get_qiskit_backend()` and `run_circuit_shots()`
- [x] 5.2 Create `tests/.../brain/arch/test_quantum_reservoir.py` — tests for shared `build_readout_network()`
- [ ] 5.3 Create `tests/.../brain/arch/test_qrh.py` — `TestQRHBrainConfig`: default config, custom values, all validators
- [ ] 5.4 Add `TestQRHReservoirCircuit`: CZ gate presence, fixed rotations, seed reproducibility, structured vs random topology
- [ ] 5.5 Add `TestQRHFeatureExtraction`: dimension (36 for 8q), Z range [-1,1], ZZ range [-1,1], input sensitivity, determinism
- [ ] 5.6 Add `TestQRHBrainReadout`: actor output shape, critic output shape
- [ ] 5.7 Add `TestQRHBrainLearning`: `run_brain()` returns valid ActionData, PPO update changes weights, buffer management, full episode workflow
- [ ] 5.8 Add `TestQRHBrainCopy`: copy independence, shared reservoir topology
- [ ] 5.9 Add `TestQRHBrainSensoryModules`: unified mode dimensions, legacy fallback
- [ ] 5.10 Add QRH foraging config to smoke test `SIMULATION_CONFIGS` list
- [ ] 5.11 Run full test suite: `uv run pytest` and `uv run pytest -m smoke -k qrh -v`

## 6. MI Decision Gate Script

- [ ] 6.0 Add `scikit-learn` as project dependency (needed for `mutual_info_classif`)
- [ ] 6.1 Create `scripts/qrh_mi_analysis.py` — oracle dataset generation using pre-trained MLPPPO
- [ ] 6.2 Implement structured vs random vs classical MLP feature extraction comparison (three-way)
- [ ] 6.3 Implement MI computation using `sklearn.feature_selection.mutual_info_classif`
- [ ] 6.4 Implement permutation test (1000 permutations) for p-value significance
- [ ] 6.5 Add CLI argument parsing and structured results output

## 7. Documentation

- [ ] 7.1 Update `AGENTS.md` — add `qrh` to brain architecture list, update count to 13
- [ ] 7.2 Update `README.md` — add QRH to "Hybrid (Quantum + Classical)" section, update "12 brain architectures" to 13, add CLI example
- [ ] 7.3 Update `CONTRIBUTING.md` — add QRH entry under "Hybrid" in architecture list, update count to 13, add example command

## 8. Verification

- [ ] 8.1 Run lint: `uv run pre-commit run -a`
- [ ] 8.2 Run full test suite: `uv run pytest`
- [ ] 8.3 Run smoke tests: `uv run pytest -m smoke -v`
- [ ] 8.4 Manual smoke test: `uv run ./scripts/run_simulation.py --runs 2 --config ./configs/examples/qrh_foraging_small.yml`
