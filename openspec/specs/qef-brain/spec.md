# qef-brain Specification

## Purpose

Define the QEF (Quantum Entangled Features) brain architecture — an entangled PQC feature extractor with configurable cross-modal entanglement topology, hybrid input, learnable feature gating, and classical PPO actor-critic readout.

## ADDED Requirements

### Requirement: QEF Circuit Construction

The QEFBrain SHALL construct a parameterized quantum circuit with configurable encoding and entanglement.

#### Scenario: Initial superposition

- **WHEN** `_encode_and_run()` is called with sensory features
- **THEN** the circuit SHALL apply Hadamard gates to all `num_qubits` qubits before any encoding layers

#### Scenario: Uniform RY encoding (default)

- **WHEN** `encoding_mode` is `"uniform"`
- **THEN** the circuit SHALL apply RY(feature × π) to ALL qubits
- **AND** features SHALL be assigned to qubits in order (qubit index = feature index % num_qubits)

#### Scenario: Sparse encoding

- **WHEN** `encoding_mode` is `"sparse"`
- **THEN** the circuit SHALL apply RY/RZ only on the first input_dim qubits
- **AND** remaining qubits SHALL remain in H-superposition as entanglement relays

#### Scenario: Data re-uploading

- **WHEN** `circuit_depth` is set to N
- **THEN** the circuit SHALL repeat the encoding + entanglement layer N times

#### Scenario: Gate mode

- **WHEN** `gate_mode` is `"cry_crz"`
- **THEN** entanglement gates SHALL use CRY/CRZ with seeded random angles
- **WHEN** `gate_mode` is `"cz"`
- **THEN** entanglement gates SHALL use CZ-only

### Requirement: QEF Entanglement Topology

The QEFBrain SHALL support three configurable entanglement topologies.

#### Scenario: Modality-paired topology

- **WHEN** `entanglement_topology` is `"modality_paired"`
- **THEN** the circuit SHALL apply entanglement gates between cross-modal qubit pairs: (0, 2), (1, 3), (4, 6), (5, 7)

#### Scenario: Ring topology

- **WHEN** `entanglement_topology` is `"ring"`
- **THEN** the circuit SHALL apply entanglement gates in a ring: (0,1), (1,2), ..., (N-2, N-1), (N-1, 0)

#### Scenario: Random topology

- **WHEN** `entanglement_topology` is `"random"`
- **THEN** the circuit SHALL apply seeded random pairs using `circuit_seed`

#### Scenario: Separable ablation

- **WHEN** `entanglement_enabled` is False
- **THEN** the circuit SHALL NOT apply any entanglement gates

### Requirement: QEF Feature Extraction

The QEFBrain SHALL extract configurable feature sets from the statevector.

#### Scenario: Z expectations

- **WHEN** features are extracted from a statevector of N qubits
- **THEN** the brain SHALL compute N per-qubit Z expectations in [-1, 1]

#### Scenario: ZZ pairwise correlations (all mode)

- **WHEN** `zz_mode` is `"all"`
- **THEN** the brain SHALL compute N(N-1)/2 pairwise ZZ correlations for all i < j

#### Scenario: ZZ cross-modal correlations

- **WHEN** `zz_mode` is `"cross_modal"`
- **THEN** the brain SHALL compute ZZ correlations only for cross-modal qubit pairs
- **AND** `_get_cross_modal_pairs()` SHALL determine pairs based on sensory module structure

#### Scenario: Cos/sin features

- **WHEN** `include_cossin` is True
- **THEN** the brain SHALL compute 2N cos/sin features: cos(⟨Z_i⟩) and sin(⟨Z_i⟩)
- **WHEN** `include_cossin` is False
- **THEN** cos/sin features SHALL be omitted

#### Scenario: ZZZ three-body correlations

- **WHEN** `include_zzz` is True
- **THEN** the brain SHALL compute N(N-1)(N-2)/6 three-body ZZZ correlations

#### Scenario: Feature determinism

- **WHEN** the same sensory input is provided twice
- **THEN** the extracted features SHALL be identical

### Requirement: QEF Hybrid Input

The QEFBrain SHALL support concatenating raw sensory features with quantum features.

#### Scenario: Hybrid input enabled

- **WHEN** `hybrid_input` is True
- **THEN** `_get_reservoir_features()` SHALL return `[raw_features, quantum_features]` concatenated
- **AND** the feature dimension SHALL be `raw_input_dim + quantum_dim`

#### Scenario: Hybrid polynomial

- **WHEN** `hybrid_polynomial` is True
- **THEN** classical pairwise products (x_i * x_j for i < j) SHALL be appended to the feature vector

### Requirement: QEF Feature Gating

The QEFBrain SHALL support learnable gating on quantum feature dimensions.

#### Scenario: Static gating

- **WHEN** `feature_gating` is `"static"`
- **THEN** the brain SHALL apply `sigmoid(w) * quantum_features` with learned per-dimension weights

#### Scenario: Context gating

- **WHEN** `feature_gating` is `"context"`
- **THEN** the brain SHALL apply `sigmoid(MLP(raw_features)) * quantum_features`
- **AND** the gate MLP SHALL be `raw_dim → 16 → quantum_dim`

#### Scenario: Mixed gating

- **WHEN** `feature_gating` is `"mixed"`
- **THEN** the brain SHALL apply `(static_gate + context_gate) / 2 * quantum_features`

#### Scenario: Gating preserves raw features

- **WHEN** hybrid_input is True and gating is active
- **THEN** raw features SHALL pass through unchanged; only quantum features SHALL be gated

#### Scenario: Gating in PPO training

- **WHEN** gating is active
- **THEN** `_perform_ppo_update()` SHALL apply gating to stored states during minibatch training
- **AND** gate parameters SHALL be included in the optimizer and gradient clipping

### Requirement: QEF Configuration

The QEFBrainConfig SHALL define quantum-specific fields with validation.

#### Scenario: Default configuration

- **WHEN** QEFBrainConfig is instantiated with no arguments
- **THEN** `num_qubits` SHALL default to 8, `circuit_depth` to 2, `circuit_seed` to 42
- **AND** `entanglement_topology` SHALL default to `"modality_paired"`, `entanglement_enabled` to True
- **AND** `hybrid_input` SHALL default to False, `feature_gating` to `"none"`
- **AND** `include_zzz` SHALL default to False, `zz_mode` to `"all"`, `include_cossin` to True

#### Scenario: Validation

- **WHEN** `num_qubits` < 2 or `circuit_depth` < 1 — ValidationError SHALL be raised
- **WHEN** `separate_critic` is True without `hybrid_input` — ValidationError SHALL be raised
- **WHEN** `hybrid_polynomial` is True without `hybrid_input` — ValidationError SHALL be raised
- **WHEN** `trainable_entanglement` is True — NotImplementedError SHALL be raised

### Requirement: Classical Ablation (MLP PPO Extensions)

MLPPPOBrain SHALL support feature expansion and gating for ablation experiments.

#### Scenario: Polynomial expansion

- **WHEN** `feature_expansion` is `"polynomial"`
- **THEN** `preprocess()` SHALL append all pairwise products (x_i * x_j for i < j)

#### Scenario: Degree-3 polynomial expansion

- **WHEN** `feature_expansion` is `"polynomial3"`
- **THEN** `preprocess()` SHALL append pairwise AND triple products

#### Scenario: Random projection

- **WHEN** `feature_expansion` is `"random_projection"`
- **THEN** `preprocess()` SHALL append features from a fixed random projection matrix

#### Scenario: Feature gating on MLP PPO

- **WHEN** `feature_gating` is True and expansion is active
- **THEN** `_apply_torch_gating()` SHALL apply sigmoid gate on expanded features during forward pass and PPO training
