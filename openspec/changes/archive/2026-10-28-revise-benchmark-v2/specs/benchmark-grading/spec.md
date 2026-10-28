## ADDED Requirements

### Requirement: Simulated tracks are measured as the real tracker measures

Before windowing, every simulated track graded against a measured rig SHALL receive that rig's position noise (on Scheer's rig, resampled from the real tracker's readings of a still worm), and SHALL lose seconds at the tracker's measured rates, conditioned on whether the simulated primitive that second is an omega. Both SHALL be measured from band cohorts and recorded with the bands. Real and imported tracks SHALL NOT be altered. Checks SHALL show the model reproduces the tracker without a smoothed truth: real tracks with their gaps filled, passed through the model's losses, SHALL give every kinematics statistic within |z| ≤ 1 of the raw tracks' value; the model's jitter on a still worm SHALL reproduce the real tracker's readings on the authors' paused stretches to within one graded-size band scale of the statistics they bias; and Chen's smoothed fragments with the model's jitter SHALL give every chemotaxis statistic within |z| ≤ 1. Local search, graded against counts from a rig no track measures, SHALL stay exact.

#### Scenario: An omega second can go missing

- **WHEN** a simulated animal is in an omega bout for a second
- **THEN** that second SHALL be missing with the measured omega-second probability, from the animal's own stream

#### Scenario: Real tracks are untouched

- **WHEN** imported or vendored real tracks are graded
- **THEN** no noise SHALL be added and no sample SHALL be dropped

### Requirement: The kinematics family

B v2 SHALL grade a kinematics family on the small lawn: the median and 90th-percentile 10-s window speed in dwelling and in roaming windows, and Kaplan–Meier roaming-bout survival at 5 min. Detected reversal onsets per on-lawn minute SHALL be computed and reported beside the grade, not graded, since no open-loop point worm reaches it through the tracker. It SHALL be a family of equal weight in D, and part of D_fit for every controller.

#### Scenario: A worm that roams too fast

- **WHEN** a controller's roaming windows are much faster than the band cohorts'
- **THEN** its roaming speed statistics SHALL fail, whatever its roaming fraction

### Requirement: Versioned bands and bar

B v1's bands and rule SHALL remain available for every past verdict. B v2's bands SHALL be built from band cohorts only, with the new statistics' generators after v1's, and its bar SHALL be Bonferroni over its 19 wild-type statistics.

#### Scenario: A past verdict is reproduced

- **WHEN** grading is asked for v1
- **THEN** it SHALL reproduce v1's values exactly

### Requirement: The chemotaxis family

B v2 SHALL grade a chemotaxis family on a linear-gradient assay matched to Chen et al. 2025's naive N2 butanone plates, from positions and the known gradient alone, with bands from its six standard-protocol recordings validated leave-one-recording-out, registered before any statistic is computed. It SHALL be held out of D_fit (a prediction for C.2 and C.3) and fitted by C.1, and simulated tracks on that assay SHALL receive that tracker's measured jitter and fragmentation. The simulated arena SHALL be the full plate, with samples outside the imaged field missing, so no statistic reads a wall. Like local search, chemotaxis SHALL pass only where it is present: up-gradient drift and the log quintile ratio each above zero at the 2.5% quantile of 2,000 resamples of the population's units (animals for a simulated population, fragments for a real recording).

#### Scenario: A worm that ignores the gradient

- **WHEN** a controller's up-gradient drift and turning bias match a random walk's
- **THEN** its chemotaxis statistics SHALL fail
- **AND** this SHALL hold even where its values lie inside the bands, through the presence test
