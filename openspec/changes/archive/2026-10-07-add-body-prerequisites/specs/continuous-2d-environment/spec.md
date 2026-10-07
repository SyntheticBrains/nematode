## ADDED Requirements

### Requirement: Signed speed

The continuous environment SHALL offer `continuous.allow_reversal`, default false. When true, a step's
speed SHALL be clamped to `[-max_step_mm, max_step_mm]` and the worm SHALL move along its heading by the
signed speed, the heading unchanged by the sign. When false every motion, sensed feature, capture file
and random draw SHALL be identical to the environment before this option existed.

#### Scenario: Reversal is off by default

- **WHEN** a configuration does not set `allow_reversal`
- **THEN** a negative speed SHALL be clamped to zero as before, and every output SHALL be unchanged

#### Scenario: A negative speed moves the worm backward

- **WHEN** reversal is on and a step's speed is negative
- **THEN** the worm SHALL move opposite to its heading by that distance, and its heading SHALL not flip

#### Scenario: Sensing follows the head and the displacement

- **WHEN** a worm reverses
- **THEN** its lateral head-sweep sample SHALL be taken across its heading as when moving forward
- **AND** its rate-of-change feature SHALL reflect the actual displacement
- **AND** a predator behind its heading SHALL still read as posterior

#### Scenario: Capture records the signed speed only under reversal

- **WHEN** behaviour capture is on and reversal is on
- **THEN** each captured step SHALL record its signed displacement along the heading
- **AND** with reversal off the capture file SHALL be byte-identical to before

### Requirement: The step's duration in worm time

The environment SHALL record the duration of a full-speed step in worm-seconds as `max_step_mm` divided
by the crawl speed, with the crawl speed and the undulation period as named constants beside it.

#### Scenario: Block V's step is five worm-seconds

- **WHEN** the constant is evaluated at `max_step_mm: 1.0`
- **THEN** it SHALL be 5.0 worm-seconds, about three undulation periods
