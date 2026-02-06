# brain-architecture Delta Specification

## ADDED Requirements

### Requirement: QRC Brain Factory Support

The brain factory SHALL support instantiation of Quantum Reservoir Computing brains.

#### Scenario: Brain Type Resolution for QRC

- **WHEN** a configuration specifies brain type as "qrc"
- **THEN** the brain factory SHALL return a QRCBrain instance
- **AND** SHALL pass through all QRC-specific configuration parameters
- **AND** SHALL validate the configuration against QRCBrainConfig schema

#### Scenario: CLI Argument Extension for QRC

- **WHEN** a user specifies `--brain qrc` on the command line
- **THEN** the system SHALL recognize "qrc" as a valid brain type
- **AND** SHALL pass the selection to the brain factory

### Requirement: QRC Brain Copying

The QRCBrain SHALL support the brain copying interface required for certain simulation modes.

#### Scenario: Brain Copy Independence

- **WHEN** `qrc_brain.copy()` is called
- **THEN** the system SHALL return an independent copy of the QRCBrain
- **AND** the copy SHALL produce identical reservoir circuits for the same input (same seed, same structure)
- **AND** the copy SHALL have independent readout network weights
- **AND** modifications to the copy's readout SHALL NOT affect the original

Note: With data re-uploading, circuits are built dynamically per input rather than stored as a fixed circuit reference.
