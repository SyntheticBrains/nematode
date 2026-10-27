## ADDED Requirements

### Requirement: The lawn rig's tracker noise is resampled real noise

B v2's lawn measurement model SHALL draw each simulated track's position noise from a library of the real tracker's readings of a still worm: the offsets from their own mean of PD1074's paused stretches on half of its band dates. The draw SHALL join whole stretches end to end, from the animal's own stream, by a joining rule chosen on the library's half alone. The model SHALL pass the jitter check on the other half of the band dates, which the library never saw, before any further B v2 reading.

#### Scenario: A still simulated worm reads like a still real worm

- **WHEN** a still point is measured through the lawn model and read on the check half's paused-stretch statistics
- **THEN** its onsets per paused minute and its 10-s window speed's median and 90th percentile SHALL each lie within one n = 200 band scale of the real values

#### Scenario: The library never sees the check

- **WHEN** the library is built
- **THEN** it SHALL hold stretches from the library half's dates only, and the jitter check SHALL read the check half's only

#### Scenario: An animal is measured the same way alone or in a population

- **WHEN** an animal with seed s is measured alone and within a population
- **THEN** its noise track SHALL be identical, drawn from `default_rng([s, 2033])`
