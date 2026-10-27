## ADDED Requirements

### Requirement: Simulated tracks are measured as the real tracker measures

Before windowing, every simulated track SHALL receive per-axis position noise at the tracker's measured jitter, and SHALL lose seconds at the tracker's measured rates, conditioned on whether the simulated primitive that second is an omega. Both SHALL be measured from band cohorts and recorded with the bands. Real and imported tracks SHALL NOT be altered.

#### Scenario: An omega second can go missing

- **WHEN** a simulated animal is in an omega bout for a second
- **THEN** that second SHALL be missing with the measured omega-second probability, from the animal's own stream

#### Scenario: Real tracks are untouched

- **WHEN** imported or vendored real tracks are graded
- **THEN** no noise SHALL be added and no sample SHALL be dropped

### Requirement: The kinematics family

B v2 SHALL grade a kinematics family on the small lawn: the median and 90th-percentile 10-s window speed in dwelling and in roaming windows, detected reversal onsets per on-lawn minute from positions alone, and Kaplan–Meier roaming-bout survival at 5 min. It SHALL be a fourth family of equal weight in D, and part of D_fit for every controller.

#### Scenario: A worm that roams too fast

- **WHEN** a controller's roaming windows are much faster than the band cohorts'
- **THEN** its roaming speed statistics SHALL fail, whatever its roaming fraction

### Requirement: Local search's rise

B v2's local-search family SHALL also grade the rise: the mean reorientation rate over 2–6 min after removal divided by the 0–2 min rate, computed the same way for López-Cruz's band batches and for simulated curves.

#### Scenario: A curve that only decays

- **WHEN** a simulated search curve falls from its first bin
- **THEN** its rise SHALL be below 1 and be graded against the band batches' rise

### Requirement: Versioned bands and bar

B v1's bands and rule SHALL remain available for every past verdict. B v2's bands SHALL be built from band cohorts only, with the new statistics' generators after v1's, and its bar SHALL be Bonferroni over its 18 wild-type statistics.

#### Scenario: A past verdict is reproduced

- **WHEN** grading is asked for v1
- **THEN** it SHALL reproduce v1's values exactly
