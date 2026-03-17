# brain-architecture Specification (Delta)

## Purpose

Register the QEF (Quantum Entangled Features) brain type in the brain architecture system.

## MODIFIED Requirements

### Requirement: Brain Type Registry

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

#### Scenario: QEF Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QEFBrain` and `QEFBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list
