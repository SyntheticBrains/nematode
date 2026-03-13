# qef-brain Specification

## Purpose

Define the QEF (Quantum Entangled Features) brain architecture — an entangled PQC feature extractor with configurable cross-modal entanglement topology and classical PPO actor-critic readout.

## ADDED Requirements

### Requirement: QEF Circuit Construction

The QEFBrain SHALL construct a parameterized quantum circuit with uniform RY encoding on all qubits and configurable entanglement topology.

#### Scenario: Initial superposition

- **WHEN** `_encode_and_run()` is called with sensory features
- **THEN** the circuit SHALL apply Hadamard gates to all `num_qubits` qubits before any encoding layers

#### Scenario: Uniform RY encoding on all qubits

- **WHEN** sensory features are encoded into the circuit
- **THEN** the circuit SHALL apply RY(feature × π) to ALL qubits (not a sensory subset)
- **AND** features SHALL be assigned to qubits in order (qubit index = feature index % num_qubits)
- **AND** when input_dim < num_qubits, unencoded qubits SHALL remain in their H-initialized superposition state and SHALL still participate in entanglement topology

#### Scenario: Data re-uploading

- **WHEN** `circuit_depth` is set to N
- **THEN** the circuit SHALL repeat the encoding + entanglement layer N times
- **AND** each layer SHALL re-encode the same sensory features

#### Scenario: Statevector simulation

- **WHEN** the circuit has been constructed
- **THEN** the brain SHALL compute the exact statevector via `Statevector.from_instruction()`
- **AND** SHALL NOT use shot-based measurement

### Requirement: QEF Entanglement Topology

The QEFBrain SHALL support three configurable entanglement topologies applied via CZ gates.

#### Scenario: Modality-paired topology (default)

- **WHEN** `entanglement_topology` is `"modality_paired"`
- **THEN** the circuit SHALL apply CZ gates between cross-modal qubit pairs: (0, 2), (1, 3), (4, 6), (5, 7)
- **AND** these pairs SHALL encode food_chemotaxis ↔ nociception and thermotaxis ↔ mechanosensation interactions

#### Scenario: Ring topology

- **WHEN** `entanglement_topology` is `"ring"`
- **THEN** the circuit SHALL apply CZ gates in a ring: (0,1), (1,2), ..., (N-2, N-1), (N-1, 0)

#### Scenario: Random topology

- **WHEN** `entanglement_topology` is `"random"`
- **THEN** the circuit SHALL apply seeded random CZ pairs using `circuit_seed`
- **AND** the number of CZ pairs SHALL equal the modality-paired count (4 for 8 qubits)
- **AND** the same seed SHALL produce identical topology across runs

### Requirement: QEF Separable Ablation

The QEFBrain SHALL support a separable (no-entanglement) mode for controlled ablation experiments.

#### Scenario: Entanglement disabled

- **WHEN** `entanglement_enabled` is False
- **THEN** the circuit SHALL NOT apply any CZ gates
- **AND** all other processing (encoding, feature extraction, readout) SHALL be identical to the entangled version

#### Scenario: Entanglement enabled (default)

- **WHEN** `entanglement_enabled` is True (default)
- **THEN** the circuit SHALL apply CZ gates per the configured topology

### Requirement: QEF Feature Extraction

The QEFBrain SHALL extract Z expectations, ZZ correlations, and cos/sin features from the statevector.

#### Scenario: Z expectations

- **WHEN** features are extracted from a statevector of N qubits
- **THEN** the brain SHALL compute N per-qubit Z expectations: ⟨Z_i⟩ = Σ_k (-1)^bit(k,i) |ψ_k|²
- **AND** each Z expectation SHALL be in the range [-1, 1]

#### Scenario: ZZ pairwise correlations

- **WHEN** features are extracted from a statevector of N qubits
- **THEN** the brain SHALL compute N(N-1)/2 pairwise ZZ correlations: ⟨Z_i Z_j⟩ for all i < j
- **AND** each ZZ correlation SHALL be in the range [-1, 1]

#### Scenario: Cos/sin features

- **WHEN** features are extracted from a statevector of N qubits
- **THEN** the brain SHALL compute 2N cos/sin features: cos(⟨Z_i⟩) and sin(⟨Z_i⟩) for each qubit
- **AND** each cos/sin feature SHALL be in the range [-1, 1]

#### Scenario: Feature vector ordering

- **WHEN** features are extracted from a statevector of N qubits
- **THEN** the feature vector SHALL be concatenated in the order: [z_0..z_N-1, zz_01..zz\_(N-1)N, cos_z_0..cos_z_N-1, sin_z_0..sin_z_N-1]

#### Scenario: Total feature dimension

- **WHEN** the brain is configured with N qubits
- **THEN** the total feature dimension SHALL be 3N + N(N-1)/2
- **AND** for 8 qubits this SHALL equal 52

#### Scenario: Feature determinism

- **WHEN** the same sensory input is provided twice
- **THEN** the extracted features SHALL be identical

### Requirement: QEF PPO Readout

The QEFBrain SHALL inherit PPO actor-critic readout from ReservoirHybridBase, identical to QRH.

#### Scenario: Actor-critic architecture

- **WHEN** QEFBrain is instantiated
- **THEN** it SHALL create an actor MLP: feature_dim → hidden_dim → num_actions
- **AND** SHALL create a critic MLP: feature_dim → hidden_dim → 1
- **AND** SHALL apply LayerNorm to features before the readout networks

#### Scenario: PPO training

- **WHEN** the PPO buffer is full
- **THEN** the brain SHALL perform PPO updates with clipped surrogate objective, value loss, and entropy bonus
- **AND** SHALL use a single combined Adam optimizer for actor + critic + LayerNorm parameters

#### Scenario: Inherited configuration

- **WHEN** QEFBrainConfig is instantiated
- **THEN** it SHALL support all ReservoirHybridBaseConfig fields: readout_hidden_dim, readout_num_layers, actor_lr, gamma, gae_lambda, ppo_clip_epsilon, ppo_epochs, ppo_minibatches, ppo_buffer_size, entropy_coeff, value_loss_coef, max_grad_norm, lr_warmup_episodes, lr_warmup_start, lr_decay_episodes, lr_decay_end, entropy_coeff_end, entropy_decay_episodes, sensory_modules

### Requirement: QEF Configuration

The QEFBrainConfig SHALL define quantum-specific fields with validation.

#### Scenario: Default configuration

- **WHEN** QEFBrainConfig is instantiated with no arguments
- **THEN** `num_qubits` SHALL default to 8
- **AND** `circuit_depth` SHALL default to 2
- **AND** `circuit_seed` SHALL default to 42
- **AND** `entanglement_topology` SHALL default to `"modality_paired"`
- **AND** `entanglement_enabled` SHALL default to True

#### Scenario: Validation constraints

- **WHEN** `num_qubits` is less than 2
- **THEN** a ValidationError SHALL be raised
- **AND** **WHEN** `circuit_depth` is less than 1
- **THEN** a ValidationError SHALL be raised

#### Scenario: Trainable entanglement reserved

- **WHEN** `trainable_entanglement` is set to True
- **THEN** a NotImplementedError SHALL be raised during brain construction

### Requirement: QEF Initialization Logging

The QEFBrain SHALL log its configuration at initialization for experiment tracking.

#### Scenario: Initialization log message

- **WHEN** QEFBrain is instantiated
- **THEN** it SHALL log: num_qubits, entanglement_topology, entanglement_enabled, circuit_depth, feature_dim, and input_dim

### Requirement: QEF Brain Copy

The QEFBrain SHALL support copy operations for checkpoint and population-based methods.

#### Scenario: Independent copy

- **WHEN** `_create_copy_instance()` is called
- **THEN** the copy SHALL have independent weights and state
- **AND** the copy SHALL share the same circuit topology and configuration
