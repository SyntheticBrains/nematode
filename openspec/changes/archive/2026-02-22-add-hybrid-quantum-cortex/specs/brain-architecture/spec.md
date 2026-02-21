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
- **AND** SHALL validate the config is an instance of `HybridQuantumCortexBrainConfig`
- **AND** SHALL pass through all HybridQuantumCortexBrainConfig parameters

### Requirement: BrainType Enum Extension

The BrainType enum and associated type sets SHALL include the hybrid quantum cortex brain type.

#### Scenario: Enum Value Registration

- **WHEN** the BrainType enum is defined
- **THEN** it SHALL include `HYBRID_QUANTUM_CORTEX = "hybridquantumcortex"`
- **AND** `HYBRID_QUANTUM_CORTEX` SHALL be included in the `QUANTUM_BRAIN_TYPES` set

#### Scenario: BRAIN_TYPES Literal

- **WHEN** the BRAIN_TYPES Literal type is defined
- **THEN** it SHALL include `"hybridquantumcortex"` as a valid value

### Requirement: Configuration Loader Extension

The configuration loader SHALL support resolving hybrid quantum cortex brain configurations from YAML.

#### Scenario: Config Import and Type Union

- **WHEN** the config loader module is loaded
- **THEN** it SHALL import `HybridQuantumCortexBrainConfig` from `quantumnematode.brain.arch`
- **AND** the `BrainConfigType` union SHALL include `HybridQuantumCortexBrainConfig`

#### Scenario: Config Map Registration

- **WHEN** a YAML configuration specifies brain type as `"hybridquantumcortex"`
- **THEN** the `BRAIN_CONFIG_MAP` dict SHALL resolve this to `HybridQuantumCortexBrainConfig`

### Requirement: Module Export Extension

The brain architecture package SHALL export hybrid quantum cortex brain classes.

#### Scenario: Package Exports

- **WHEN** `quantumnematode.brain.arch` package is imported
- **THEN** `HybridQuantumCortexBrain` and `HybridQuantumCortexBrainConfig` SHALL be available in `__all__`
- **AND** SHALL be importable via explicit import from the package

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
