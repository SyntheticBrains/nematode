# brain-architecture Specification (Delta)

## Purpose

Extend the brain architecture system to support the CRH (Classical Reservoir Hybrid) brain type and the ReservoirHybridBase base class.

## MODIFIED Requirements

### Requirement: Brain Type Registry

#### Scenario: CRH Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.CRH` SHALL exist with value `"crh"` in the BrainType enum
- **AND** CRH SHALL be included in `CLASSICAL_BRAIN_TYPES`
- **AND** `"crh"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: CRH Brain Factory

- **WHEN** `create_brain(BrainType.CRH, config)` is called
- **THEN** the factory SHALL return a `CRHBrain` instance
- **AND** SHALL accept `CRHBrainConfig` for configuration

#### Scenario: CRH Config Loading

- **WHEN** a YAML config specifies `brain.name: crh`
- **THEN** the config loader SHALL parse brain config using `CRHBrainConfig`
- **AND** SHALL support all CRH-specific fields plus inherited ReservoirHybridBase fields

#### Scenario: QLIF-LSTM Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QLIF_LSTM` SHALL exist with value `"qliflstm"` in the BrainType enum
- **AND** QLIF_LSTM SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `BrainType.QLIF_LSTM` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QLIF-LSTM Config Loading

- **WHEN** a YAML config specifies `brain.name: qliflstm`
- **THEN** the config loader SHALL parse brain config using `QLIFLSTMBrainConfig`
- **AND** SHALL support all QLIF-LSTM-specific fields

#### Scenario: QLIF-LSTM Brain Factory Instantiation

- **WHEN** the brain factory receives a `QLIFLSTMBrainConfig`
- **THEN** the factory SHALL instantiate a `QLIFLSTMBrain` with the provided configuration

### Requirement: Module Exports

#### Scenario: CRH Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `CRHBrain` and `CRHBrainConfig` SHALL be importable
- **AND** `ReservoirHybridBase` and `ReservoirHybridBaseConfig` SHALL be importable

#### Scenario: QLIF-LSTM Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QLIFLSTMBrain` and `QLIFLSTMBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list
