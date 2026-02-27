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

### Requirement: Module Exports

#### Scenario: CRH Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `CRHBrain` and `CRHBrainConfig` SHALL be importable
- **AND** `ReservoirHybridBase` and `ReservoirHybridBaseConfig` SHALL be importable
