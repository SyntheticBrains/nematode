## MODIFIED Requirements

### Requirement: Roaming and dwelling from speed and angular speed

The validation package SHALL classify behaviour into roaming and dwelling from each 10-second window's
speed and angular speed, measured from positions sampled once per 5-second step. A track sampled at a
shorter step that divides 5 seconds SHALL be downsampled to 5-second steps before it is measured, and a
track at any other step SHALL be refused. Every consumer SHALL take a track's step from its capture or
its config, never from a constant. A two-state hidden Markov model with Gaussian emissions over each
window's log speed and angular speed SHALL decode the windows within each run of on-food windows. It
SHALL be fitted on vendored real-worm tracks, measured the same way, against their authors' labels, and
applied unchanged to simulated tracks.

#### Scenario: Real and simulated worms are read alike

- **WHEN** a real track resampled to 5-second steps and a simulated track are classified
- **THEN** both SHALL be measured by the same functions and decoded with the same model, with no
  refitting on simulated data

#### Scenario: The calibration is checked on held-out animals

- **WHEN** the model is fitted on half of the real animals
- **THEN** its agreement with the authors' labels on the other half SHALL be reported as Cohen's
  kappa, and the instrument SHALL NOT be used if kappa is below 0.6

#### Scenario: Off-food windows are not classified

- **WHEN** a simulated worm is off every lawn
- **THEN** its windows SHALL be reported as off-food and SHALL NOT count toward either state

#### Scenario: A 1-second track is read at 5 seconds

- **WHEN** a track captured at a 1-second step is classified
- **THEN** every fifth position SHALL be kept for speed and turning, and the windows decoded with the
  unchanged model
- **AND** a window SHALL be on a lawn only when every 1-second step it spans is
- **AND** its eaten-cell density SHALL be the mean of its 1-second bites, each divided by the
  1-second step's own intake fraction times quality

#### Scenario: A step that does not divide 5 seconds is refused

- **WHEN** a track captured at a 2-second step is classified
- **THEN** the classifier SHALL refuse it, naming the step

## ADDED Requirements

### Requirement: Captures record their step

A behaviour capture file SHALL carry its run's step duration as a top-level `step_seconds` when that
step is not 5 seconds, and SHALL be byte-identical to before when it is. A reader SHALL take an absent
key to mean 5 seconds.

#### Scenario: A 1-second capture is labelled

- **WHEN** a 1-second lawn run captures behaviour
- **THEN** its `behaviour_capture.json` SHALL carry `"step_seconds": 1.0`

#### Scenario: A default capture is unchanged

- **WHEN** a run at the default step captures behaviour
- **THEN** its `behaviour_capture.json` SHALL carry no `step_seconds` key

### Requirement: The bias-curve harness reads 5-second captures only

The klinokinesis and weathervane harness SHALL refuse a capture whose step is not 5 seconds. Its
statistics are per transition, calibrated at that step.

#### Scenario: A 1-second capture is refused

- **WHEN** the harness is given a capture carrying `"step_seconds": 1.0`
- **THEN** it SHALL exit with an error naming the step, and write no output
