## ADDED Requirements

### Requirement: Patch-leaving readings

The lawn analysis SHALL read each run's lawn exits from its behaviour capture (a step off a lawn
that stays off for at least one full 10-second window), the roaming share of its on-lawn windows 30
to 180 s before each exit, its pre-exit roaming enrichment (that share minus the run's overall
on-lawn roaming share), the same for the last on-lawn window alone, its leaving rate, and, for on-lawn dwelling windows, how often the
next window is roaming where the cell under the worm is grazed against where it is fresh.

#### Scenario: An exit is found and its preceding state read

- **WHEN** a worm roams on a lawn for three minutes and then steps off it for a minute
- **THEN** one exit SHALL be counted, and its band of 30 to 180 s before the exit SHALL be all
  roaming

#### Scenario: A flicker at the edge is not an exit

- **WHEN** a worm steps off a lawn for less than a full window and back on
- **THEN** no exit SHALL be counted

#### Scenario: Enrichment compares with the run's own baseline

- **WHEN** a run's windows 30 to 180 s before its exits are all roaming and a quarter of its
  on-lawn windows are
- **THEN** its enrichment SHALL be 0.75

### Requirement: A matched real reference for leaving

The leaving reference SHALL be read from the vendored real tracks with the same calibrated
classifier and windows as simulated worms, with the authors' own labels reported beside.

#### Scenario: Real and simulated leaving are read alike

- **WHEN** the reference is derived
- **THEN** its enrichment SHALL use the calibrated classifier's states on windows measured at the
  5-second step, and the authors'-label enrichment SHALL be reported beside it
