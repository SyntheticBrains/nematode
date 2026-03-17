## 1. QEFBrainConfig and QEFBrain Core Implementation

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/qef.py` with module docstring, imports, and constants (MODALITY_PAIRED_CZ, MIN_QUBITS, defaults)
- [x] 1.2 Implement `QEFBrainConfig` extending `ReservoirHybridBaseConfig` with fields: `num_qubits`, `circuit_depth`, `circuit_seed`, `entanglement_topology`, `entanglement_enabled`, `trainable_entanglement`, `encoding_mode`, `gate_mode`, `feature_mode`. Add field validators for num_qubits >= 2 and circuit_depth >= 1
- [x] 1.3 Implement `QEFBrain.__init__` — store quantum params, pre-compute `_signs` array for vectorized feature extraction, compute topology pairs, call super().__init__. Raise NotImplementedError if trainable_entanglement is True
- [x] 1.4 Implement `_compute_feature_dim()` with support for ZZZ, cross-modal ZZ, and optional cos/sin
- [x] 1.5 Implement `_encode_and_run(features)` — H layer, then [encoding + entanglement topology]^depth, return statevector. Support uniform and sparse encoding modes
- [x] 1.6 Implement topology builders: `_build_modality_paired_topology()`, `_build_ring_topology()`, `_build_random_topology()`. Support CZ-only and CRY/CRZ gate modes. Skip gates when `entanglement_enabled` is False
- [x] 1.7 Implement `_extract_features(statevector)` — compute Z expectations, ZZ correlations (all or cross-modal), optional cos/sin, optional ZZZ three-body correlations
- [x] 1.8 Implement `_get_reservoir_features(sensory_features)` with hybrid input support (raw + quantum + optional polynomial concatenation)
- [x] 1.9 Implement `_create_copy_instance(config)` for checkpoint/copy support
- [x] 1.10 Add initialization logging: num_qubits, entanglement_topology, entanglement_enabled, circuit_depth, feature_dim, input_dim, hybrid_input, gating mode

## 2. Hybrid Input and Feature Gating

- [x] 2.1 Add `hybrid_input: bool` config field — concatenate raw sensory features with quantum features
- [x] 2.2 Add `hybrid_polynomial: bool` config field — add classical pairwise products to hybrid input
- [x] 2.3 Add `feature_gating: Literal["none", "static", "context", "mixed"]` config field
- [x] 2.4 Implement `_apply_feature_gating()` with static sigmoid gate, context MLP gate, and mixed average
- [x] 2.5 Implement `_compute_gate()` helper for gate value computation across modes
- [x] 2.6 Implement `_init_gating_and_critic()` — initialize gate weights/network and optional separate critic
- [x] 2.7 Override `run_brain()` for custom forward pass when gating or separate critic is active
- [x] 2.8 Override `_perform_ppo_update()` to apply gating during PPO minibatch training
- [x] 2.9 Implement `_collect_trainable_params()` for gradient clipping across all gating modes
- [x] 2.10 Add `separate_critic: bool` config field with model validator requiring hybrid_input
- [x] 2.11 Add `include_zzz: bool` config field for three-body correlations
- [x] 2.12 Add `zz_mode: Literal["all", "cross_modal"]` and `include_cossin: bool` config fields
- [x] 2.13 Implement `_get_cross_modal_pairs()` for cross-modal ZZ pair computation
- [x] 2.14 Implement `_quantum_feature_dim()` helper for quantum-only dimension calculation

## 3. Brain Type Registration

- [x] 3.1 Add `QEF = "qef"` to `BrainType` enum in `brain/arch/dtypes.py`, add to `BRAIN_TYPES` Literal and `QUANTUM_BRAIN_TYPES` set
- [x] 3.2 Add `QEFBrain` and `QEFBrainConfig` imports and `__all__` entries in `brain/arch/__init__.py`
- [x] 3.3 Import `QEFBrainConfig` in `utils/config_loader.py`, add to `BrainConfigType` union, add `"qef": QEFBrainConfig` to `BRAIN_CONFIG_MAP`
- [x] 3.4 Add `elif brain_type == BrainType.QEF:` case in `utils/brain_factory.py` following QRH pattern

## 4. Classical Ablation Infrastructure (MLP PPO)

- [x] 4.1 Add `feature_expansion: Literal["none", "polynomial", "polynomial3", "random_projection"]` to `MLPPPOBrainConfig`
- [x] 4.2 Add `feature_gating: bool` to `MLPPPOBrainConfig`
- [x] 4.3 Implement `_init_feature_expansion()` — compute expanded input_dim and initialize random projection matrix
- [x] 4.4 Implement `_apply_feature_expansion()` — compute polynomial, polynomial3, or random projection features
- [x] 4.5 Implement `_apply_torch_gating()` — sigmoid gate on expanded features in torch forward pass
- [x] 4.6 Update `forward_actor()`, `forward_critic()` to apply gating
- [x] 4.7 Update PPO training loop to apply gating on minibatch states

## 5. Example YAML Configs

- [x] 5.1 Create `configs/examples/qef_foraging_small.yml` — 8 qubits, grid 20, ring topology
- [x] 5.2 Create `configs/examples/qef_pursuit_predators_small.yml` — 8 qubits, grid 20, ring topology, hybrid + context gating, compact readout
- [x] 5.3 Create `configs/examples/qef_thermotaxis_pursuit_predators_large.yml` — 8 qubits, grid 100, ring topology, hybrid + context gating, bigbuf 1024
- [x] 5.4 Create `configs/examples/qef_thermotaxis_stationary_predators_large.yml` — 8 qubits, grid 100, ring topology, hybrid + curated cross-modal ZZ + static gating, bigbuf 1024
- [x] 5.5 Create `configs/examples/qef_foraging_small_separable.yml` — separable ablation control
- [x] 5.6 Create fair MLP PPO configs for all three evaluation tasks with matched buffer/minibatches
- [x] 5.7 Create modality_paired variant config for ablation reference

## 6. Unit Tests

- [x] 6.1 TestQEFBrainConfig: default values, custom values, validation errors, trainable_entanglement NotImplementedError
- [x] 6.2 TestQEFFeatureExtraction: feature dimensions, Z/ZZ/cos_sin ranges, determinism, ordering, wrapping, depth effect, sparse encoding
- [x] 6.3 TestQEFGateAndFeatureModes: CZ vs CRY/CRZ, z_cossin vs xyz, combined modes
- [x] 6.4 TestQEFTopology: modality_paired/ring/random differences, separable vs entangled, CZ pair verification
- [x] 6.5 TestQEFBrainReadout: actor/critic output shapes, run_brain returns valid ActionData
- [x] 6.6 TestQEFBrainLearning: PPO update changes weights, buffer management
- [x] 6.7 TestQEFHybridInput: feature dimensions, output composition, actor/critic compatibility, run_brain, copy
- [x] 6.8 TestQEFFeatureGating: config defaults, dimension preservation, run_brain, PPO update changes gate weights
- [x] 6.9 TestQEFSeparateCritic: requires hybrid, network dimensions, run_brain, PPO update, combined features
- [x] 6.10 TestQEFContextGating: network creation, output shape, input dependence, run_brain, sensory modules
- [x] 6.11 TestQEFZZZCorrelations: dimension computation, output shape, range, hybrid+gating integration
- [x] 6.12 TestQEFBrainCopy: independent weights, shared topology, preserved config
- [x] 6.13 TestFeatureExpansion (MLP PPO): polynomial/polynomial3/random_projection dims, output values, run_brain
- [x] 6.14 TestFeatureGating (MLP PPO): gate creation, gradient flow, scaling, PPO training loop

## 7. Smoke Test Integration

- [x] 7.1 Add `"qef_foraging_small.yml"` to the smoke test config parametrize list

## 8. MI Decision Gate Script

- [x] 8.1 Create `scripts/qef_mi_analysis.py` — compare MI across topologies and ablations
- [x] 8.2 Add CLI arguments for topology selection, num_samples, and output format
- [x] 8.3 Add permutation test for p-value significance on MI differences

## 9. Documentation

- [x] 9.1 Update `AGENTS.md` — add `qef` to brain architecture list
- [x] 9.2 Update `README.md` — add QEF to brain architectures section
- [x] 9.3 Update `CONTRIBUTING.md` — add QEFBrain entry
- [x] 9.4 Update `openspec/config.yaml` — add `qef` to brain architecture list

## 10. Validation

- [x] 10.1 Run unit tests — 78 QEF tests pass + 58 MLP PPO tests pass
- [x] 10.2 Run ruff — all checks pass on qef.py and mlpppo.py
- [x] 10.3 Run 12-seed validation across 3 tasks (stationary, pursuit, small PP) with QEF, MLP PPO fair, and A3 polynomial ablation
