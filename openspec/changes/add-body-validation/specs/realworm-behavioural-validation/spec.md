## ADDED Requirements

### Requirement: Posture instruments for the segmented body

The kinematic instruments SHALL report, from captured postures:

- the variance the first four eigenworms capture, with each posture's segment angles interpolated to
  100 mean-removed tangent angles head to tail and projected on the vendored basis;
- each posture's undulation amplitude, its radius in the plane of the first two eigenworms;
- omega turns: across one head swing, the interval between consecutive zero-crossings of the head
  segment's curvature, the line from the midline at 0.2 body lengths to the head turning, net, by more
  than 135° in the world frame, each with its swing's bounds so that the posture across the turn can
  be read;
- the share of episodes with a forward bout of at least 20 worm-seconds.

#### Scenario: The basis captures a real posture

- **WHEN** a vendored real posture is projected on the basis
- **THEN** the first four modes SHALL capture at least 96% of the variance pooled over the 6,655 real
  postures, as the reference set reports (96.46%)

#### Scenario: A straight body has no omega turn

- **WHEN** the body crawls straight
- **THEN** no omega turn SHALL be counted

#### Scenario: A sharp turn is an omega turn

- **WHEN** the head line turns by more than 135° within one head swing
- **THEN** one omega turn SHALL be counted, with its heading change and the swing's first and last
  sub-steps

### Requirement: Behaviour capture in the evaluation harness

The evaluation harness SHALL, when asked, record each step's behaviour as the simulation's behaviour
capture does, so that the bias-curve harness can read evaluation episodes.

#### Scenario: An evaluation writes a behaviour capture the bias-curve harness reads

- **WHEN** a run is evaluated with behaviour capture
- **THEN** it SHALL write a capture the bias-curve harness loads, one series per episode
