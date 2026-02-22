# brain-architecture Delta Specification

## ADDED Requirements

### Requirement: QRH Brain Factory Support

The brain factory SHALL support instantiation of Quantum Reservoir Hybrid brains.

#### Scenario: Brain Type Resolution for QRH

- **WHEN** a configuration specifies brain type as "qrh"
- **THEN** the brain factory SHALL return a QRHBrain instance
- **AND** SHALL pass through all QRH-specific configuration parameters
- **AND** SHALL validate the configuration against QRHBrainConfig schema

#### Scenario: CLI Argument Extension for QRH

- **WHEN** a user specifies `--brain qrh` on the command line
- **THEN** the system SHALL recognize "qrh" as a valid brain type
- **AND** SHALL pass the selection to the brain factory

### Requirement: QRH Brain Type Classification

The QRHBrain SHALL be classified as a quantum brain type for benchmarking purposes.

#### Scenario: Quantum Brain Type Registration

- **WHEN** QRH is registered in the brain type system
- **THEN** the system SHALL add `QRH = "qrh"` to the BrainType enum
- **AND** SHALL include `BrainType.QRH` in the `QUANTUM_BRAIN_TYPES` set
- **AND** SHALL include `BrainType.QRH` in the `BRAIN_TYPES` literal type

### Requirement: Shared Quantum Utilities

The brain architecture SHALL provide shared quantum utility functions accessible to all quantum and reservoir-based brains.

#### Scenario: Qiskit Backend Initialization

- **WHEN** any brain requires a Qiskit AerSimulator backend
- **THEN** the system SHALL provide a shared `get_qiskit_backend(device, seed)` function in `_quantum_utils.py`
- **AND** the function SHALL be importable from both `_quantum_utils` and `_qlif_layers` (backward compatibility)
- **AND** existing brains (QSNNReinforce, QSNNPPO, HybridQuantum, HybridQuantumCortex) SHALL continue to function without changes

#### Scenario: Shared Readout Network Builder

- **WHEN** a brain requires a classical readout network with orthogonal initialization
- **THEN** the system SHALL provide a shared `build_readout_network()` function in `_quantum_reservoir.py`
- **AND** the function SHALL support configurable input dimension, hidden dimension, output dimension, readout type, and number of layers
- **AND** SHALL apply orthogonal weight initialization
