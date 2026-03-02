# crh-brain Specification

## Purpose

Define the requirements for the Classical Reservoir Hybrid (CRH) brain architecture. CRH uses a fixed classical Echo State Network (ESN) reservoir with configurable feature channels and a PPO-trained classical actor-critic readout. It serves as both a quantum ablation control for QRH (matching feature dimension and architecture) and a standalone benchmark architecture filling the "classical fixed reservoir + PPO readout" niche.

## ADDED Requirements

### Requirement: CRH Brain Architecture

The system SHALL support a Classical Reservoir Hybrid brain architecture that uses a fixed ESN reservoir with a PPO-trained classical actor-critic readout, inheriting from ReservoirHybridBase.

#### Scenario: CRH Brain Instantiation

- **WHEN** a CRHBrain is instantiated with default configuration
- **THEN** the system SHALL create an ESN reservoir with 10 neurons, spectral radius 0.9, and 3 depth layers
- **AND** SHALL create a classical actor MLP for action logits (inheriting readout dims from base)
- **AND** SHALL create a classical critic MLP for value estimation (inheriting readout dims from base)
- **AND** SHALL initialize W_in and W_res matrices deterministically using the configured seed
- **AND** SHALL scale W_res so its largest eigenvalue magnitude equals the configured spectral radius
- **AND** SHALL compute feature dimension based on the configured feature channels

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain crh --config config.yml`
- **THEN** the system SHALL initialize a CRHBrain instance
- **AND** the simulation SHALL proceed using the ESN reservoir for feature extraction
- **AND** SHALL train the readout via PPO

### Requirement: Echo State Network Reservoir

The CRHBrain SHALL implement a fixed classical reservoir using an Echo State Network that transforms sensory input through nonlinear dynamics without trainable reservoir parameters.

#### Scenario: ESN Reservoir Construction

- **WHEN** a CRHBrain constructs its reservoir
- **THEN** the system SHALL create an input weight matrix W_in of shape (num_neurons, input_dim)
- **AND** SHALL create a reservoir weight matrix W_res of shape (num_neurons, num_neurons)
- **AND** SHALL initialize W_in with uniform random values in [-input_scale, input_scale]
- **AND** SHALL initialize W_res with random normal values, then scale so the largest eigenvalue magnitude equals spectral_radius (if max eigenvalue magnitude < 1e-10, skip scaling and log a warning)
- **AND** SHALL use the configured seed for deterministic initialization

#### Scenario: Sparse Input Connectivity

- **WHEN** `input_connectivity` is "sparse" (default)
- **THEN** the system SHALL create W_in with shape (num_neurons, input_dim) where rows beyond `num_sensory_neurons` are zeroed out
- **AND** only the first `num_sensory_neurons` neurons SHALL receive direct input; remaining neurons receive signal exclusively through W_res connections
- **AND** this SHALL match QRH's pattern where only sensory qubits receive direct input encoding

#### Scenario: Dense Input Connectivity

- **WHEN** `input_connectivity` is "dense"
- **THEN** the system SHALL route input features to all reservoir neurons via a full W_in matrix

#### Scenario: ESN Forward Pass

- **WHEN** sensory features are processed through the ESN reservoir
- **THEN** the system SHALL compute `h_0 = tanh(W_in @ x)` for the initial layer
- **AND** SHALL compute `h_l = tanh(W_res @ h_{l-1} + W_in @ x)` for layers 1 through depth-1
- **AND** SHALL use the final layer activations for feature extraction
- **AND** SHALL NOT carry state between successive `run_brain()` calls (stateless)

#### Scenario: ESN Reproducibility

- **WHEN** two CRHBrain instances are created with the same `reservoir_seed`
- **THEN** both instances SHALL produce identical W_in and W_res matrices
- **AND** SHALL generate identical feature vectors for the same input

#### Scenario: ESN Immutability

- **WHEN** the CRHBrain is trained over multiple episodes
- **THEN** W_in and W_res SHALL remain unchanged
- **AND** only the classical readout network parameters SHALL be updated

### Requirement: Configurable Feature Channels

The CRHBrain SHALL extract features from ESN activations using a configurable set of feature channels, enabling both ablation-matched and standalone-optimized modes.

#### Scenario: Raw Channel

- **WHEN** `feature_channels` includes "raw"
- **THEN** the system SHALL include the raw activation values h_i (N features)
- **AND** values SHALL be in the range [-1, 1] (output of tanh)

#### Scenario: Cos/Sin Channel

- **WHEN** `feature_channels` includes "cos_sin"
- **THEN** the system SHALL include cos(pi * h_i) and sin(pi * h_i) (2N features)
- **AND** this SHALL serve as the classical analog to QRH's X/Y Pauli expectations

#### Scenario: Squared Channel

- **WHEN** `feature_channels` includes "squared"
- **THEN** the system SHALL include h_i^2 (N features)
- **AND** values SHALL be in the range [0, 1]

#### Scenario: Pairwise Channel

- **WHEN** `feature_channels` includes "pairwise"
- **THEN** the system SHALL include h_i * h_j for all pairs i < j (N(N-1)/2 features)
- **AND** this SHALL serve as the classical analog to QRH's ZZ-correlations

#### Scenario: Feature Dimension Computation

- **WHEN** computing the feature dimension for the readout
- **THEN** the system SHALL sum the dimensions contributed by each channel:
  - "raw": N
  - "cos_sin": 2N
  - "squared": N
  - "pairwise": N(N-1)/2
- **AND** SHALL pass this total dimension to the base class for actor/critic construction

#### Scenario: Ablation Mode Feature Matching

- **WHEN** CRH is configured with `num_reservoir_neurons=10`, `feature_channels=[raw, cos_sin, pairwise]`
- **THEN** the feature dimension SHALL be 10 + 20 + 45 = 75
- **AND** this SHALL match QRH's feature dimension of 3\*10 + C(10,2) = 75 for 10 qubits

### Requirement: CRH Configuration Schema

The configuration system SHALL support CRH-specific parameters via Pydantic BaseModel, extending ReservoirHybridBaseConfig.

#### Scenario: CRH Configuration Parameters

- **WHEN** parsing a CRHBrain configuration
- **THEN** the system SHALL accept `num_reservoir_neurons` (int, default 10)
- **AND** SHALL accept `reservoir_depth` (int, default 3)
- **AND** SHALL accept `reservoir_seed` (int, default 42)
- **AND** SHALL accept `spectral_radius` (float, default 0.9)
- **AND** SHALL accept `input_connectivity` (str, default "sparse")
- **AND** SHALL accept `input_scale` (float, default 1.0)
- **AND** SHALL accept `feature_channels` (list[FeatureChannel], default ["raw", "cos_sin", "pairwise"]) where `FeatureChannel = Literal["raw", "cos_sin", "squared", "pairwise"]`
- **AND** SHALL accept `num_sensory_neurons` (int | None, default None)
- **AND** SHALL inherit all PPO/readout parameters from ReservoirHybridBaseConfig

#### Scenario: CRH Configuration Validation

- **WHEN** validating CRHBrain configuration
- **THEN** the system SHALL require `num_reservoir_neurons` >= 2
- **AND** SHALL require `reservoir_depth` >= 1
- **AND** SHALL require `spectral_radius` > 0
- **AND** SHALL require `input_connectivity` in {"sparse", "dense"}
- **AND** SHALL require all items in `feature_channels` to be valid channel names ("raw", "cos_sin", "squared", "pairwise")
- **AND** SHALL require `feature_channels` to be non-empty
- **AND** SHALL require `num_sensory_neurons` >= 1 and \<= `num_reservoir_neurons` when explicitly set

### Requirement: ClassicalBrain Protocol Compliance

The CRHBrain SHALL implement the ClassicalBrain protocol via ReservoirHybridBase for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods (Inherited)

- **WHEN** CRHBrain is used in a simulation
- **THEN** it SHALL inherit `run_brain()`, `learn()`, `update_memory()`, `prepare_episode()`, `post_process_episode()` from ReservoirHybridBase
- **AND** SHALL implement `_get_reservoir_features(sensory_features)` returning ESN-extracted features
- **AND** SHALL implement `_compute_feature_dim()` returning the feature dimension for configured channels

#### Scenario: Brain Copy Independence

- **WHEN** `crh_brain.copy()` is called
- **THEN** the system SHALL return an independent copy of the CRHBrain
- **AND** the copy SHALL have identical W_in and W_res matrices
- **AND** the copy SHALL have independent readout network weights
- **AND** modifications to the copy's readout SHALL NOT affect the original

### Requirement: Sensory Module Integration (Inherited)

The CRHBrain SHALL support both legacy and unified sensory module input modes, inherited from ReservoirHybridBase.

#### Scenario: Legacy Input Mode

- **WHEN** `sensory_modules` is None (default)
- **THEN** the system SHALL use 2 features: gradient_strength [0, 1] and relative_angle [-1, 1]

#### Scenario: Unified Sensory Input Mode

- **WHEN** `sensory_modules` is configured with a list of ModuleName values
- **THEN** the system SHALL use `extract_classical_features()` from `brain/modules.py`
- **AND** SHALL compute input dimension as the sum of each module's classical_dim
