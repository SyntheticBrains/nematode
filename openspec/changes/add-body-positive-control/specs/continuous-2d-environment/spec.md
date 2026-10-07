## ADDED Requirements

### Requirement: A reversal threshold on the body's direction channel

A kinematic body's wave SHALL run tail-to-head only when the drive's direction channel is below a fixed
threshold, −0.5 by default, the same for every brain.

#### Scenario: A small negative direction does not reverse the body

- **WHEN** the direction channel is between −0.5 and 0
- **THEN** the wave SHALL run head-to-tail

#### Scenario: A strongly negative direction reverses it

- **WHEN** the direction channel is below −0.5
- **THEN** the wave SHALL run tail-to-head

### Requirement: Sub-step posture capture

A kinematic body SHALL, when asked, record each sub-step's time, segment curvatures and head position,
without changing the motion.

#### Scenario: Recording does not change the motion

- **WHEN** a body is stepped with and without a recorder under the same drives
- **THEN** its head positions SHALL be identical
