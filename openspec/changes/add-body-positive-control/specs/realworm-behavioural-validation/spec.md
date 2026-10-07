## ADDED Requirements

### Requirement: Kinematic instruments

The kinematic instruments SHALL read captured episodes of a segmented body. They SHALL report:

- undulation frequency, from mid-body curvature crossings of its mean that count only after leaving a
  ±0.31 κL band;
- wavelength, from crossing delays between adjacent segments summed down the body;
- speed, in body lengths per second;
- reversal fraction.

Frequency and wavelength SHALL be read only on undulating steps: forward-running, and with every
segment's wave amplitude at least a quarter of the peak. Each unbroken stretch of them SHALL be read on
its own, and their count SHALL be reported. Reversal fraction SHALL count the steps the body executed
tail-to-head. Steps whose head lies within a margin of a wall SHALL be excluded and counted.

#### Scenario: A known wave is recovered

- **WHEN** the instruments read a generated travelling wave of known frequency and wavelength
- **THEN** they SHALL recover the frequency within 3% and the wavelength within 5%

#### Scenario: Jitter is not an undulation

- **WHEN** the curvature stays inside the band
- **THEN** no crossing SHALL be counted

#### Scenario: A silenced segment keeps a step out of the wave

- **WHEN** every step's drive silences one segment's wave
- **THEN** no frequency or wavelength SHALL be reported, and speed SHALL still be

#### Scenario: The executed direction is counted

- **WHEN** a step requested a reversal the body did not run
- **THEN** it SHALL count as forward

#### Scenario: Wall-proximal steps are excluded

- **WHEN** every step's head lies within the margin of a wall
- **THEN** no speed, frequency or wavelength SHALL be reported, and every step SHALL be counted as
  excluded

### Requirement: Evaluating a trained body-drive run

The evaluation harness SHALL rebuild a run's brain the way the simulation entry point builds it,
load its final weights, or none for the untrained policy, freeze learning, and capture postures over
episodes seeded away from any training run's. It SHALL accept a sub-step override for the half-step
check, and SHALL refuse a configuration without the kinematic body.

#### Scenario: Saved weights round-trip

- **WHEN** a brain's saved weights are evaluated at 20 and at 40 sub-steps
- **THEN** every captured step SHALL be either used or counted as excluded

#### Scenario: A point-worm configuration is refused

- **WHEN** the configuration's body model is the point worm
- **THEN** the harness SHALL refuse it
