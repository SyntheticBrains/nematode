## ADDED Requirements

### Requirement: A kinematic segmented body

The continuous environment SHALL offer `continuous.body_model: kinematic` beside the default `point`. A
kinematic body SHALL have 12 segments whose curvature a body-level generator sets from a 25-number drive
action: a head relaxation switch for the phase, a front-to-back relay for propagation that runs
tail-to-head when the direction channel is negative, and per-segment amplitude and bias from the drives.
Within each step the body SHALL move by resistive-force theory, so that the net drag force and torque on
it vanish. Its head SHALL be the agent's position, and its heading SHALL be the direction from the body's midpoint
to the head. Under `point` every output SHALL be
identical to before.

#### Scenario: A forward wave moves the body forward and a backward wave backward

- **WHEN** the body is driven with symmetric drives and a positive direction for several steps
- **THEN** its head SHALL advance along its heading
- **AND** with a negative direction it SHALL move the other way

#### Scenario: A dorsal–ventral bias turns the body

- **WHEN** the dorsal drives exceed the ventral drives
- **THEN** the heading SHALL turn, the opposite way to the reverse bias

#### Scenario: The head's period is the configured period

- **WHEN** the generator runs freely
- **THEN** the head's curvature SHALL switch sign at intervals giving the configured period

#### Scenario: The body stays in the arena

- **WHEN** a step would carry the head past the arena edge
- **THEN** the head SHALL be clamped to the edge, without error

#### Scenario: The point worm is unchanged

- **WHEN** `body_model` is `point`
- **THEN** every motion and output SHALL be identical to before this option existed
