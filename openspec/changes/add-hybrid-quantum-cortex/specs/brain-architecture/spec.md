# brain-architecture Delta Specification

## MODIFIED Requirements

### Requirement: Brain Factory Extension

The brain factory method SHALL support spiking neural network and hybrid quantum cortex brain instantiation.

#### Scenario: Brain Type Resolution

- **GIVEN** a configuration specifies brain type as "spiking"
- **WHEN** the brain factory creates a brain instance
- **THEN** it SHALL return a SpikingBrain object
- **AND** SHALL pass through all spiking-specific configuration parameters

#### Scenario: HybridQuantumCortex Brain Type Resolution

- **GIVEN** a configuration specifies brain type as "hybridquantumcortex"
- **WHEN** the brain factory creates a brain instance
- **THEN** it SHALL return a HybridQuantumCortexBrain object
- **AND** SHALL pass through all HybridQuantumCortexBrainConfig parameters

### Requirement: CLI Argument Extension

The command-line interface SHALL accept "spiking" and "hybridquantumcortex" as valid brain type options.

#### Scenario: Argument Validation

- **GIVEN** a user specifies `--brain spiking`
- **WHEN** command-line arguments are parsed
- **THEN** the system SHALL recognize "spiking" as a valid brain type
- **AND** SHALL pass the selection to the brain factory

#### Scenario: HybridQuantumCortex Argument Validation

- **GIVEN** a user specifies `--brain hybridquantumcortex`
- **WHEN** command-line arguments are parsed
- **THEN** the system SHALL recognize "hybridquantumcortex" as a valid brain type
- **AND** SHALL pass the selection to the brain factory
