## ADDED Requirements

### Requirement: Posture instruments for the segmented body

The kinematic instruments SHALL report, from captured postures:

- the variance the first four eigenworms capture, with each posture's segment angles interpolated to
  100 mean-removed tangent angles head to tail and projected on the vendored basis;
- each posture's peak |κL|;
- omega turns: the line from the midline at 0.2 body lengths to the head turning, net, by more than
  135° between two consecutive crossings of the head's swing;
- the share of episodes with a forward bout of at least 20 worm-seconds.

#### Scenario: The basis captures a real posture

- **WHEN** a vendored real posture is projected on the basis
- **THEN** the first four modes SHALL capture most of its variance, as the reference set reports

#### Scenario: A straight body has no omega turn

- **WHEN** the body crawls straight
- **THEN** no omega turn SHALL be counted

#### Scenario: A sharp turn is an omega turn

- **WHEN** the head line turns by more than 135° within one head swing
- **THEN** one omega turn SHALL be counted, with its heading change

### Requirement: Behaviour capture in the evaluation harness

The evaluation harness SHALL, when asked, record each step's behaviour as the simulation's behaviour
capture does, so that the bias-curve harness can read evaluation episodes.

#### Scenario: An evaluation writes a behaviour capture the bias-curve harness reads

- **WHEN** a run is evaluated with behaviour capture
- **THEN** it SHALL write a capture the bias-curve harness loads, one series per episode
