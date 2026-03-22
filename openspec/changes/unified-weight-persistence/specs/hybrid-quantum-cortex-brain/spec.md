## ADDED Requirements

### Requirement: WeightPersistence Protocol Support

HybridQuantumCortexBrain SHALL implement the `WeightPersistence` protocol as a thin wrapper around its existing private save/load methods.

#### Scenario: HybridQuantumCortex Weight Components

- **WHEN** `get_weight_components()` is called on a HybridQuantumCortexBrain
- **THEN** the returned dict SHALL contain these components:
  - `"reflex"`: reflex QSNN weights dict (W_sh, W_hm, theta_hidden, theta_motor as detached CPU tensors)
  - `"cortex"`: cortex QSNN weights dict (grouped sensory QLIF weights as detached CPU tensors)
  - `"critic"`: critic MLP state_dict

#### Scenario: HybridQuantumCortex Partial Load

- **WHEN** `load_weight_components()` is called with only `{"reflex": component}`
- **THEN** only the reflex QSNN weights SHALL be loaded
- **AND** cortex and critic weights SHALL remain unchanged
- **AND** shape validation SHALL be performed on reflex tensors before loading

#### Scenario: HybridQuantumCortex Full Load

- **WHEN** `load_weight_components()` is called with all three components
- **THEN** reflex, cortex, and critic weights SHALL all be loaded
- **AND** each component's shapes SHALL be validated before loading

#### Scenario: Existing Config Fields Continue Working

- **WHEN** HybridQuantumCortexBrain is instantiated with `reflex_weights_path`, `cortex_weights_path`, or `critic_weights_path` set
- **THEN** existing `_load_reflex_weights()`, `_load_cortex_weights()`, and `_load_critic_weights()` methods SHALL be called as before
- **AND** behavior SHALL be identical to before this change

#### Scenario: Existing Auto-Save Unchanged

- **WHEN** `post_process_episode()` is called during training
- **THEN** the existing per-episode auto-save behavior SHALL continue using the old separate-file format (`reflex_weights.pt`, `cortex_weights.pt`, `critic_weights.pt`)
- **AND** the new `WeightPersistence` protocol SHALL NOT affect this behavior
