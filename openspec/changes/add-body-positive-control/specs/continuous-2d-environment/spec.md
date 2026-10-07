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

### Requirement: Episode posture capture, sub-step count and steering gain

The environment SHALL hold an optional posture log, off by default, to which every kinematic step
appends its drive and its sub-steps' posture. The body's sub-step count SHALL be configurable, 20 by
default. Its steering gain SHALL be configurable as an override, and when unset the body's calibrated
default SHALL apply.

#### Scenario: Capture is off unless an evaluation asks for it

- **WHEN** an environment steps a kinematic body without a posture log set
- **THEN** nothing SHALL be recorded

#### Scenario: Capture does not change the motion

- **WHEN** two environments step the same drives, one capturing
- **THEN** their agents' positions SHALL be identical and the capturing one SHALL hold one entry per
  step, with one posture per sub-step

#### Scenario: The sub-step count is configurable

- **WHEN** the configured sub-step count is 40
- **THEN** each captured step SHALL hold 40 postures
