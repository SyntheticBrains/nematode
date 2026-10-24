## ADDED Requirements

### Requirement: Reversal onsets from positions alone

`quantumnematode.benchmark.readouts.reversal_onsets` SHALL detect reversal-initiated reorientations
from positions sampled each second, by design Decision A, without reading any primitive label,
posture or source flag. Its minimum step SHALL be relative to the track's own median step, with a
0.01 mm floor where the units are mm, so the same parameters read pixels and millimetres. A retrace
test, tuned on or off, SHALL be available to require that an onset's step moves the worm back toward
where it was W seconds earlier. A step touching a missing position SHALL never be a candidate.

#### Scenario: A simulated track is read without its labels

- **WHEN** the detector reads a simulated capture
- **THEN** it SHALL use the capture's positions only

#### Scenario: A forward turn is not a reversal

- **WHEN** a synthetic track turns 150° and runs off in a new direction without moving back along its
  path
- **THEN** the detector with retrace on SHALL report no onset, and SHALL report one for a track that
  moves back along its path

#### Scenario: Missing positions

- **WHEN** a track has a missing position
- **THEN** no onset SHALL be placed on a step touching it, and the onset rate SHALL be per second of
  valid steps

#### Scenario: A track in pixels

- **WHEN** a track's units are pixels
- **THEN** the detector SHALL apply no absolute floor, and its relative threshold SHALL be the same as
  for a track in mm

### Requirement: The detector tuned on Roberts and gated elsewhere

The detector's parameters SHALL be chosen by exhaustive search over design Decision A's grid, on the
three Roberts cohorts that `default_rng(2028)` assigns to calibration, maximising F1 against the
tracker's reversal onsets. Without retuning, it SHALL be read once against every gate of design
Decision B. The gates are fixed before tuning. A failed gate SHALL be reported, and the benchmark SHALL
NOT register on the detector until the maintainer decides.

#### Scenario: The preflight protects the held-out cohorts

- **WHEN** the best in-sample recall or precision on the calibration cohorts is below the Roberts
  held-out bars (recall 0.55 and precision 0.60, as amended on 2026-10-24 after the first run stopped
  at 0.62 and 0.68 against 0.70)
- **THEN** the script SHALL stop and report without reading any held-out cohort

#### Scenario: Gates fixed before tuning

- **WHEN** the calibration script runs
- **THEN** it SHALL read its gates from constants fixed in this change, and SHALL write every gate's
  value and verdict to its record

#### Scenario: The dead worms

- **WHEN** the detector reads Roberts's dead-worm tracks
- **THEN** it SHALL report detected onsets per minute against the 0.1 gate

### Requirement: The local-search curve and its decay

`local_search_curve` SHALL count each animal's detected onsets in 2-min bins of time since removal,
in bins where at least 90% of its positions are present. `fit_local_search` SHALL fit αe^(−γ(t − t₀)) + β
over a registered window. López-Cruz's deposited bin `[a, a + 2)` min SHALL be read as `[a + 4, a + 6)`
min after removal. The window SHALL start at the deposited wild type's peak bin (recording 4–6 min)
and end at recording 38–40 min. The deposited Fig. 1C wild type's fit SHALL be recorded as the
reference, with fits from the adjacent start bins beside it.

#### Scenario: The same window for real and simulated curves

- **WHEN** a simulated assay timed from removal and López-Cruz's wild type are fitted
- **THEN** both SHALL be fitted over 8–44 min after removal

### Requirement: Roaming and dwelling on 1-s tracks

The benchmark SHALL classify 1-s tracks by downsampling them to the classifier's 5-s step, with the
calibrated model unchanged.

#### Scenario: A simulated 1-s capture

- **WHEN** a 1-s primitives capture is classified
- **THEN** it SHALL go through `downsample_to_reference` and the calibrated model, as real 1-s tracks
  do

### Requirement: The pirouette quintile ratio

`quintile_ratio` SHALL return the detected-onset rate in the worm's lowest dC/dt quintile over its
highest, with quintiles taken within each worm and rates pooled over worms. Its reference SHALL be a
committed digitisation of Pierce-Shimomura 1999's pirouette rate against dC/dt, with the calibration
points, each digitised point and the rank-only units mapping, and Lee 2026's ratio cited beside it.

#### Scenario: Units cancel

- **WHEN** a track's dC/dt is rescaled by any positive factor
- **THEN** its quintile ratio SHALL be unchanged
