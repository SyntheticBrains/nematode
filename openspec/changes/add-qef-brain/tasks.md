## 1. QEFBrainConfig and QEFBrain Implementation

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/qef.py` with module docstring, imports, and constants (MODALITY_PAIRED_CZ, MIN_QUBITS, defaults)
- [x] 1.2 Implement `QEFBrainConfig` extending `ReservoirHybridBaseConfig` with fields: `num_qubits`, `circuit_depth`, `circuit_seed`, `entanglement_topology`, `entanglement_enabled`, `trainable_entanglement`. Add field validators for num_qubits >= 2 and circuit_depth >= 1
- [x] 1.3 Implement `QEFBrain.__init__` — store quantum params, pre-compute `_signs` array for vectorized feature extraction, compute topology pairs, call super().__init__. Raise NotImplementedError if trainable_entanglement is True
- [x] 1.4 Implement `_compute_feature_dim()` returning `3 * N + N * (N - 1) // 2`
- [x] 1.5 Implement `_encode_and_run(features)` — H layer, then [RY(feature * pi) on all qubits + CZ topology]^depth, return statevector
- [x] 1.6 Implement topology builders: `_build_modality_paired_topology()`, `_build_ring_topology()`, `_build_random_topology()`. Skip CZ gates when `entanglement_enabled` is False
- [x] 1.7 Implement `_extract_features(statevector)` — compute Z expectations, ZZ correlations, cos/sin of Z expectations using pre-computed index arrays
- [x] 1.8 Implement `_get_reservoir_features(sensory_features)` calling `_encode_and_run` then `_extract_features`
- [x] 1.9 Implement `_create_copy_instance(config)` for checkpoint/copy support
- [x] 1.10 Add initialization logging: num_qubits, entanglement_topology, entanglement_enabled, circuit_depth, feature_dim, input_dim

## 2. Brain Type Registration

- [x] 2.1 Add `QEF = "qef"` to `BrainType` enum in `brain/arch/dtypes.py`, add to `BRAIN_TYPES` Literal and `QUANTUM_BRAIN_TYPES` set
- [x] 2.2 Add `QEFBrain` and `QEFBrainConfig` imports and `__all__` entries in `brain/arch/__init__.py`
- [x] 2.3 Import `QEFBrainConfig` in `utils/config_loader.py`, add to `BrainConfigType` union, add `"qef": QEFBrainConfig` to `BRAIN_CONFIG_MAP`
- [x] 2.4 Add `elif brain_type == BrainType.QEF:` case in `utils/brain_factory.py` following QRH pattern

## 3. Example YAML Configs

- [x] 3.1 Create `configs/examples/qef_foraging_small.yml` — 8 qubits, grid 20, modality_paired topology (default food_chemotaxis, 2 features on 8 qubits)
- [x] 3.2 Create `configs/examples/qef_pursuit_predators_small.yml` — 8 qubits, grid 20, modality_paired topology, sensory_modules: [food_chemotaxis, nociception] (4 features on 8 qubits)
- [x] 3.3 Create `configs/examples/qef_thermotaxis_pursuit_predators_large.yml` — 8 qubits, grid 100, modality_paired topology, sensory_modules: [food_chemotaxis, nociception, thermotaxis] (7 features on 8 qubits), readout_hidden_dim: 64
- [x] 3.4 Create `configs/examples/qef_thermotaxis_stationary_predators_large.yml` — 8 qubits, grid 100, modality_paired topology, sensory_modules: [food_chemotaxis, nociception, thermotaxis] (7 features on 8 qubits), readout_hidden_dim: 64
- [x] 3.5 Create `configs/examples/qef_foraging_small_separable.yml` — same as 3.1 but with entanglement_enabled: false (separable ablation control)

## 4. Unit Tests

- [x] 4.1 Create `tests/quantumnematode_tests/brain/arch/test_qef.py` with TestQEFBrainConfig: default values, custom values, validation errors (num_qubits < 2, circuit_depth < 1), trainable_entanglement NotImplementedError
- [x] 4.2 Add TestQEFFeatureExtraction: feature dimension = 52 for 8 qubits, Z in [-1,1], ZZ in [-1,1], cos/sin in [-1,1], determinism (same input = same output), feature vector ordering [z, zz, cos_z, sin_z], feature wrapping, minimum qubits, data re-uploading depth effect
- [x] 4.3 Add TestQEFTopology: modality_paired/ring/random produce different features for same input, separable vs entangled produce different features, separable has same feature dimension, unused qubits (input_dim < num_qubits) still participate in entanglement, modality-paired CZ pair verification, small qubit count filtering
- [x] 4.4 Add TestQEFBrainReadout: actor output shape (num_actions,), critic output shape (1,), run_brain returns valid ActionData
- [x] 4.5 Add TestQEFBrainLearning: PPO update changes weights, buffer management works across episodes
- [x] 4.6 Add TestQEFBrainCopy: copy has independent weights, copy preserves topology and config

## 5. Smoke Test Integration

- [x] 5.1 Add `"qef_foraging_small.yml"` to the smoke test config parametrize list

## 6. MI Decision Gate Script

- [x] 6.1 Create `scripts/qef_mi_analysis.py` — adapt `scripts/qrh_mi_analysis.py` for QEF: compare MI(entangled_features, optimal_action) vs MI(separable_features, optimal_action) vs MI(qrh_random_features, optimal_action)
- [x] 6.2 Add CLI arguments for topology selection, num_samples, and output format
- [x] 6.3 Add permutation test (1000 permutations) for p-value significance on MI differences

## 7. Documentation

- [x] 7.1 Update `AGENTS.md` — add `qef` to brain architecture list in `brain/arch/`, update count from 17 to 18
- [x] 7.2 Update `README.md` — add QEF to brain architectures section, update "17 brain architectures" to 18
- [x] 7.3 Update `CONTRIBUTING.md` — add QEFBrain entry under brain architectures list, update "17 brain architectures" to 18
- [x] 7.4 Update `openspec/config.yaml` — add `qef` to brain architecture list (updated all 18 architectures)

## 8. Validation

- [x] 8.1 Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_qef.py -v` — all 33 unit tests pass (98% coverage on qef.py)
- [x] 8.2 Run `uv run pre-commit run -a` — lint and format clean (ruff, pyright, mdformat, markdownlint)
- [ ] 8.3 Run `uv run pytest -m smoke -v -k qef` — smoke test passes
- [x] 8.4 Run `uv run pytest -m "not smoke and not nightly"` — 1696 tests pass (no regressions)
