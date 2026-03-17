# brain-architecture Specification

## Purpose

Extend the brain architecture system to support the CRH (Classical Reservoir Hybrid) brain type with ReservoirHybridBase base class, the QLIF-LSTM (Quantum LIF Long Short-Term Memory) brain type, and the QEF (Quantum Entangled Features) brain type.

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
- **AND** `"qliflstm"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QLIF-LSTM Config Loading

- **WHEN** a YAML config specifies `brain.name: qliflstm`
- **THEN** the config loader SHALL parse brain config using `QLIFLSTMBrainConfig`
- **AND** SHALL support all QLIF-LSTM-specific fields

#### Scenario: QLIF-LSTM Brain Factory Instantiation

- **WHEN** the brain factory receives a `QLIFLSTMBrainConfig`
- **THEN** the factory SHALL instantiate a `QLIFLSTMBrain` with the provided configuration

#### Scenario: QEF Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QEF` SHALL exist with value `"qef"` in the BrainType enum
- **AND** QEF SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `"qef"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QEF Brain Factory

- **WHEN** `setup_brain_model(BrainType.QEF, config)` is called
- **THEN** the factory SHALL return a `QEFBrain` instance
- **AND** SHALL accept `QEFBrainConfig` for configuration
- **AND** SHALL raise ValueError if config is not QEFBrainConfig

#### Scenario: QEF Config Loading

- **WHEN** a YAML config specifies `brain.name: qef`
- **THEN** the config loader SHALL parse brain config using `QEFBrainConfig`
- **AND** SHALL support all QEF-specific fields plus inherited ReservoirHybridBaseConfig fields

### Requirement: Module Exports

#### Scenario: CRH Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `CRHBrain` and `CRHBrainConfig` SHALL be importable
- **AND** `ReservoirHybridBase` and `ReservoirHybridBaseConfig` SHALL be importable

#### Scenario: QLIF-LSTM Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QLIFLSTMBrain` and `QLIFLSTMBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list

#### Scenario: QEF Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QEFBrain` and `QEFBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list
