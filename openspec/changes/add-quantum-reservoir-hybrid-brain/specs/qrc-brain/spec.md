# qrc-brain Delta Specification

## MODIFIED Requirements

### Requirement: QRC Brain Architecture

The system SHALL support a Quantum Reservoir Computing brain architecture that uses a fixed quantum reservoir with a trainable classical readout layer.

#### Scenario: QRC Brain Instantiation

- **WHEN** a QRCBrain is instantiated with default configuration
- **THEN** the system SHALL create a quantum reservoir with configurable qubits (default 8, recommended 4 for efficiency) and 3 entangling layers
- **AND** SHALL create a trainable MLP readout network with configurable hidden units (default 32)
- **AND** SHALL initialize the reservoir with a deterministic seed for reproducibility
- **AND** SHALL use orthogonal weight initialization for better gradient flow
- **AND** SHALL use the shared `get_qiskit_backend()` from `_quantum_utils.py` for Qiskit backend initialization
- **AND** SHALL use the shared `build_readout_network()` from `_quantum_reservoir.py` for readout construction

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qrc --config config.yml`
- **THEN** the system SHALL initialize a QRCBrain instance
- **AND** the simulation SHALL proceed using the quantum reservoir for feature extraction
