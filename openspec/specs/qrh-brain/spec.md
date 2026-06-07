# qrh-brain Specification

## Purpose

Refactor QRHBrain to inherit from ReservoirHybridBase, extracting shared PPO readout infrastructure into the base class. No behavioral changes to QRH.

## Requirements

### Requirement: QRH Inheritance Refactor

QRHBrain SHALL inherit from `ReservoirHybridBase` and QRHBrainConfig SHALL inherit from `ReservoirHybridBaseConfig`, with shared PPO readout infrastructure provided by the base classes while preserving identical behavior to the pre-refactor version.

#### Scenario: QRH Inherits from ReservoirHybridBase

- **WHEN** QRHBrain is instantiated
- **THEN** it SHALL inherit from `ReservoirHybridBase` instead of directly from `ClassicalBrain`
- **AND** SHALL implement `_get_reservoir_features(sensory_features)` calling `_encode_and_run()` and `_extract_features()`
- **AND** SHALL implement `_compute_feature_dim()` returning `3*N + N*(N-1)//2`
- **AND** SHALL inherit PPO training, rollout buffer, LR scheduling, and readout construction from the base class

#### Scenario: QRH Config Inherits from ReservoirHybridBaseConfig

- **WHEN** QRHBrainConfig is defined
- **THEN** it SHALL inherit from `ReservoirHybridBaseConfig` instead of `BrainConfig`
- **AND** SHALL add quantum-specific fields: `num_reservoir_qubits`, `reservoir_depth`, `reservoir_seed`, `shots`, `use_random_topology`, `num_sensory_qubits`
- **AND** all existing config fields SHALL retain their default values

#### Scenario: QRH Behavioral Regression

- **WHEN** the refactored QRHBrain is used in simulation
- **THEN** it SHALL produce identical behavior to the pre-refactor version for the same seed and configuration
- **AND** all existing QRH tests SHALL pass without modification
- **AND** all existing QRH configs SHALL continue to work without changes
