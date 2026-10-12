# realistic-replay Specification

## Purpose

Record kinematic-body runs at the body's sub-step resolution, and replay them as a worm on an agar
plate, in worm time at a chosen speed, with a label that names the model and the speed.

## Requirements

### Requirement: Recording a kinematic-body run for replay

`run_simulation.py` SHALL accept `--record-replay <dir>` and, for each run, write a compressed
recording of every body sub-step (time, segment curvatures, head, frame angle) and every step's
status and lawn densities. It SHALL be refused without the kinematic body, with `--manyworlds`, and
for multi-agent runs. Recording SHALL NOT change the simulation.

#### Scenario: A run is recorded at sub-step resolution

- **WHEN** a 3-step kinematic-body run on lawns is recorded at 20 sub-steps per step
- **THEN** its recording SHALL hold 60 sub-steps, 3 steps of status, and each step's lawn densities

#### Scenario: Recording does not change the run

- **WHEN** a seeded kinematic-body run is made with and without recording
- **THEN** both runs SHALL take the same actions and end in the same state

#### Scenario: A point-worm run is refused

- **WHEN** `--record-replay` is given for a point-worm configuration
- **THEN** the command SHALL exit with an error naming the kinematic body

### Requirement: Replaying in worm time

The replay player SHALL advance a worm-time clock by wall time × the selected speed (1×, 10×, 60×),
interpolate the body between the sub-steps around the clock, and support pause, single-step and
restart.

#### Scenario: Interpolation between sub-steps

- **WHEN** the clock sits halfway between two sub-steps
- **THEN** the drawn head SHALL lie halfway between their heads, and each curvature halfway between
  theirs

#### Scenario: Speed scales the clock

- **WHEN** the player advances by 0.05 s of wall time at 10×
- **THEN** the clock SHALL advance by 0.5 worm-seconds

### Requirement: A realistic, honest worm

The player SHALL draw the body as a smooth tube through its joints with an elliptical width profile,
lawns as a film whose opacity follows each cell's density, and an always-on label naming the model
and the playback speed.

#### Scenario: The tube passes through the joints

- **WHEN** a body is drawn
- **THEN** its smoothed midline SHALL pass through each of the body's 13 joints

#### Scenario: The width profile is elliptical with open tips

- **WHEN** the width profile is computed
- **THEN** it SHALL be widest at the middle, symmetric, and above zero at both tips

#### Scenario: The label names the model and speed

- **WHEN** a frame is drawn at 10×
- **THEN** it SHALL carry a label naming the kinematic body and "10×"
