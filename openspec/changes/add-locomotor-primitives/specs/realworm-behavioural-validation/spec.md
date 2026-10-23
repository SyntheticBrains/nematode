## ADDED Requirements

### Requirement: Capture records the primitive

Under `locomotion: primitives`, each captured step SHALL record which primitive moved the worm: run,
reversal, omega or delta. Under free locomotion the field SHALL be absent, and the capture file
byte-identical to before. No grading readout SHALL read the field.

#### Scenario: A primitive step is labelled

- **WHEN** a worm under primitives reverses with behaviour capture on
- **THEN** each of that bout's captured steps SHALL carry `primitive: reversal`
