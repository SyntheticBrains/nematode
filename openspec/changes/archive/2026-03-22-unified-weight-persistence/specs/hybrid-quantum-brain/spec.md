## ADDED Requirements

### Requirement: WeightPersistence Protocol Support

HybridQuantumBrain SHALL implement the `WeightPersistence` protocol as a thin wrapper around its existing private save/load methods.

#### Scenario: Hybrid Quantum Weight Components

- **WHEN** `get_weight_components()` is called on a HybridQuantumBrain
- **THEN** the returned dict SHALL contain these components:
  - `"qsnn"`: dict with keys `W_sh`, `W_hm`, `theta_hidden`, `theta_motor` (detached CPU tensors)
  - `"cortex.policy"`: cortex actor state_dict
  - `"cortex.value"`: cortex critic state_dict

#### Scenario: Hybrid Quantum Partial Load

- **WHEN** `load_weight_components()` is called with only `{"qsnn": component}`
- **THEN** only the QSNN weights SHALL be loaded (W_sh, W_hm, theta_hidden, theta_motor)
- **AND** cortex actor and critic weights SHALL remain unchanged
- **AND** shape validation SHALL be performed on QSNN tensors before loading

#### Scenario: Hybrid Quantum Full Load

- **WHEN** `load_weight_components()` is called with all three components
- **THEN** QSNN weights, cortex actor, and cortex critic SHALL all be loaded
- **AND** each component's shapes SHALL be validated before loading
- **AND** the cortex PPO rollout buffer SHALL be reset to prevent stale experience from corrupting the first update

#### Scenario: Existing Config Fields Continue Working

- **WHEN** HybridQuantumBrain is instantiated with `qsnn_weights_path` or `cortex_weights_path` set
- **THEN** existing `_load_qsnn_weights()` and `_load_cortex_weights()` methods SHALL be called as before
- **AND** behavior SHALL be identical to before this change

#### Scenario: Existing Auto-Save Unchanged

- **WHEN** `post_process_episode()` is called during training
- **THEN** the existing per-episode auto-save behavior SHALL continue using the old separate-file format (`qsnn_weights.pt`, `cortex_weights.pt`)
- **AND** the new `WeightPersistence` protocol SHALL NOT affect this behavior
