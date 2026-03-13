## 1. QEFBrainConfig and QEFBrain Implementation

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/qef.py` with module docstring, imports, and constants (MODALITY_PAIRED_CZ, MIN_QUBITS, defaults)
- [ ] 1.2 Implement `QEFBrainConfig` extending `ReservoirHybridBaseConfig` with fields: `num_qubits`, `circuit_depth`, `circuit_seed`, `entanglement_topology`, `entanglement_enabled`, `trainable_entanglement`. Add field validators for num_qubits >= 2 and circuit_depth >= 1
- [ ] 1.3 Implement `QEFBrain.__init__` — store quantum params, pre-compute index arrays (\_signs, \_low_indices, \_high_indices) for vectorized feature extraction, compute topology pairs, call super().__init__. Raise NotImplementedError if trainable_entanglement is True
- [ ] 1.4 Implement `_compute_feature_dim()` returning `3 * N + N * (N - 1) // 2`
- [ ] 1.5 Implement `_encode_and_run(features)` — H layer, then [RY(feature × π) on all qubits + CZ topology]^depth, return statevector
- [ ] 1.6 Implement topology builders: `_build_modality_paired_topology(qc)`, `_build_ring_topology(qc)`, `_build_random_topology(qc)`. Skip CZ gates when `entanglement_enabled` is False
- [ ] 1.7 Implement `_extract_features(statevector)` — compute Z expectations, ZZ correlations, cos/sin of Z expectations using pre-computed index arrays
- [ ] 1.8 Implement `_get_reservoir_features(sensory_features)` calling `_encode_and_run` then `_extract_features`
- [ ] 1.9 Implement `_create_copy_instance(config)` for checkpoint/copy support

## 2. Brain Type Registration

- [ ] 2.1 Add `QEF = "qef"` to `BrainType` enum in `brain/arch/dtypes.py`, add to `BRAIN_TYPES` Literal and `QUANTUM_BRAIN_TYPES` set
- [ ] 2.2 Add `QEFBrain` and `QEFBrainConfig` imports and `__all__` entries in `brain/arch/__init__.py`
- [ ] 2.3 Import `QEFBrainConfig` in `utils/config_loader.py`, add to `BrainConfigType` union, add `"qef": QEFBrainConfig` to `BRAIN_CONFIG_MAP`
- [ ] 2.4 Add `elif brain_type == BrainType.QEF:` case in `utils/brain_factory.py` following QRH pattern

## 3. Example YAML Configs

- [ ] 3.1 Create `configs/examples/qef_foraging_small.yml` — basic foraging with modality_paired topology, sensory_modules: [food_chemotaxis, nociception]
- [ ] 3.2 Create `configs/examples/qef_pursuit_predators_small.yml` — pursuit predators, modality_paired topology
- [ ] 3.3 Create `configs/examples/qef_thermotaxis_pursuit_predators_large.yml` — thermotaxis + pursuit predators on large grid
- [ ] 3.4 Create `configs/examples/qef_thermotaxis_stationary_predators_large.yml` — thermotaxis + stationary predators on large grid
- [ ] 3.5 Create `configs/examples/qef_foraging_small_separable.yml` — separable ablation control (entanglement_enabled: false)

## 4. Unit Tests

- [ ] 4.1 Create `tests/quantumnematode_tests/brain/arch/test_qef.py` with TestQEFBrainConfig: default values, custom values, validation errors (num_qubits < 2, circuit_depth < 1), trainable_entanglement NotImplementedError
- [ ] 4.2 Add TestQEFFeatureExtraction: feature dimension = 52 for 8 qubits, Z in [-1,1], ZZ in [-1,1], cos/sin in [-1,1], determinism (same input = same output)
- [ ] 4.3 Add TestQEFTopology: modality_paired/ring/random produce different features for same input, separable vs entangled produce different features, separable has same feature dimension
- [ ] 4.4 Add TestQEFBrainReadout: actor output shape (num_actions,), critic output shape (1,), run_brain returns valid ActionData
- [ ] 4.5 Add TestQEFBrainLearning: PPO update changes weights, buffer management works across episodes
- [ ] 4.6 Add TestQEFBrainCopy: copy has independent weights, copy preserves topology and config

## 5. Smoke Test Integration

- [ ] 5.1 Add `"qef_foraging_small.yml"` to the smoke test config parametrize list

## 6. Documentation

- [ ] 6.1 Update `AGENTS.md` — add `qef` to brain architecture list in `brain/arch/`, update count from 17 to 18
- [ ] 6.2 Update `README.md` — add QEF to brain architectures section, update "17 brain architectures" to 18
- [ ] 6.3 Update `CONTRIBUTING.md` — add QEFBrain entry under brain architectures list, update "17 brain architectures" to 18
- [ ] 6.4 Update `openspec/config.yaml` — add `qef` to brain architecture list

## 7. Validation

- [ ] 7.1 Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_qef.py -v` — all unit tests pass
- [ ] 7.2 Run `uv run pre-commit run -a` — lint and format clean
- [ ] 7.3 Run `uv run pytest -m smoke -v -k qef` — smoke test passes
- [ ] 7.4 Run `uv run pytest` — full test suite passes (no regressions)
