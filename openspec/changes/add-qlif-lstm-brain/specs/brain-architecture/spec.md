# brain-architecture Specification (Delta)

## Purpose

Extend the brain architecture system to support the QLIF-LSTM brain type.

## MODIFIED Requirements

### Requirement: Brain Type Registry

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

#### Scenario: QLIF-LSTM Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QLIFLSTMBrain` and `QLIFLSTMBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list
