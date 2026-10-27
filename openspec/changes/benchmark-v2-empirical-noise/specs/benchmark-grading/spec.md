## ADDED Requirements

### Requirement: The lawn rig's tracker noise is resampled real noise

B v2's lawn measurement model SHALL draw each simulated track's position noise from a library of the real tracker's readings of a still worm: the offsets from their own mean of PD1074's paused stretches on half of its band dates. The draw SHALL splice stretches at the nearest stored offset (the best of 50 drawn stretches), from the animal's own stream, with no other transformation. The model SHALL pass the jitter check on the other half of the band dates, which the library never saw, before any further B v2 reading. The check's gate SHALL be calibrated on the tracker's own date-to-date variation.

#### Scenario: A still simulated worm reads like a still real worm

- **WHEN** a still point is measured through the lawn model and read on the check half's paused-stretch statistics
- **THEN** each of its onsets per paused minute and its 10-s window speed's median and 90th percentile SHALL differ from the real values by no more than the central 95% of the differences between random halves of the real band dates

#### Scenario: The library never sees the check

- **WHEN** the library is built
- **THEN** it SHALL hold stretches from the library half's dates only, and the jitter check SHALL read the check half's only

#### Scenario: An animal is measured the same way alone or in a population

- **WHEN** an animal with seed s is measured alone and within a population
- **THEN** its noise track SHALL be identical, drawn from `default_rng([s, 2033])`

### Requirement: The reachability oracle's worm can reach its bar

The reachability oracle SHALL read an open-loop two-state worm whose per-state speeds, speed spreads (per 10-s window), curvature and hazards are fitted through the measurement model on discarded seeds. Before registration, the fitted worm SHALL reach the five gated statistics at a median RMS z of at most 1 on fresh discarded populations. If it does not, B v2 SHALL stop before registration, with the statistics it misses as the diagnosis.

#### Scenario: The matched worm misses with no noise

- **WHEN** B.4's matched worm reads outside the bands with no measurement noise
- **THEN** it SHALL NOT be the reachability oracle's worm, and the fitted worm SHALL take its place

#### Scenario: No open-loop worm reaches a statistic

- **WHEN** the fitted worm's median RMS z on fresh populations exceeds 1
- **THEN** the change SHALL stop before registration, and the statistic it misses SHALL be recorded as unreachable by an open-loop point worm

### Requirement: The on-lawn reversal rate is reported above the tracker's noise, not graded

The kinematics family's on-lawn reversal rate SHALL be read with the off-food detector at a step floor of 0.03 mm: the smallest 0.01-mm floor at which the real tracker's readings of a still worm, on the library half, detect reversals at under 1% of the real on-lawn rate. Because no open-loop point worm reaches it through the tracker, it SHALL be reported beside the grade and not graded, and B v2's bar SHALL be Bonferroni over 19 statistics. Off-food uses of the detector SHALL be unchanged.

#### Scenario: Tracker noise does not read as reversals

- **WHEN** a still worm's real readings are passed to the on-lawn reversal detector
- **THEN** it SHALL detect reversals at under 1% of the real on-lawn rate

### Requirement: Only Scheer's rig is measured by the lawn model

The lawn measurement model SHALL apply to the families graded against Scheer's rig (the small lawn, the uniform well and kinematics). Simulated local-search tracks SHALL stay exact, since López-Cruz's reorientation counts come from a rig whose tracker no track measures.

#### Scenario: Local search stays exact

- **WHEN** a simulated local-search population is graded on v2
- **THEN** its tracks SHALL NOT pass through the lawn model
