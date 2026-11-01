## ADDED Requirements

### Requirement: C.1 is refitted on benchmark v2

C.1's model SHALL be refitted to the real wild type on benchmark v2 by sep-CMA from the box centre, 8 restarts, ranking each candidate by the definedness-first score against the bands at n = 200. Every registered reading SHALL use the benchmark's own grade, graded with presence.

#### Scenario: The objective is the bands'

- **WHEN** a candidate is scored
- **THEN** its score SHALL be 1,000 × the statistics it leaves undefined + the weighted RMS of 10·asinh(z/10), for z its statistics' distances from the band centres on the band scales

### Requirement: The plateau stop waits for a defined window

A C.1b restart SHALL stop on the plateau rule only once σ ≤ σ₀ and no generation of the two windows the rule compares has a median score of 1,000 or more.

#### Scenario: An undefined median holds a restart

- **WHEN** the plateau rule and σ hold, but a median in the last 40 generations is 1,000 or more
- **THEN** the restart SHALL continue

### Requirement: Restarts are chosen on separate populations

The chosen restart SHALL be chosen on 3 selection populations per plateaued restart, graded with presence, and disjoint from the 5 registered populations its verdict is read on. A restart stopped by the cap SHALL NOT be chosen.

#### Scenario: Selection and verdict share no animals

- **WHEN** the restarts are selected and the verdict is read
- **THEN** no animal SHALL appear in both a selection population and a registered population

#### Scenario: The most passing restart is chosen

- **WHEN** two plateaued restarts pass 3 and 2 of their selection populations
- **THEN** the one passing 3 SHALL be chosen, and a tie SHALL go to the lower median D

### Requirement: A reachability pilot precedes the fits

Before Part B, one restart SHALL run 300 generations on discarded seeds to set the cap and project the cost, and its answer SHALL be graded with presence on 5 discarded populations. If it passes fewer than 3, the change SHALL return to the maintainer before any registered fit.

#### Scenario: The pilot cannot reach the wild type

- **WHEN** the pilot's answer passes 2 of its 5 discarded populations
- **THEN** no registered fit SHALL run until the maintainer decides

## MODIFIED Requirements

### Requirement: C.1 is fitted to the wild type by sep-CMA

On benchmark v1 (Logbook 113), C.1 SHALL be fitted by sep-CMA with 8 restarts from the box centre (σ₀ 0.2, population 16, bounds [0, 1]). The objective SHALL be the benchmark's D on the 11 wild-type statistics against the band cohorts at n = 200, for 200 fresh animals per evaluation. Each restart SHALL stop at the plateau rule or at 210 generations. A restart stopped by the cap SHALL NOT be chosen. The chosen restart SHALL be the plateaued restart with the lowest mean median objective over its last 20 generations. C.1's refit on benchmark v2 follows "C.1 is refitted on benchmark v2".

#### Scenario: No restart plateaus

- **WHEN** every restart stops at the cap
- **THEN** no verdict SHALL be read, and the run SHALL return to the maintainer

### Requirement: The wild-type verdict is the benchmark's rule

Every restart's answer SHALL be graded on five registered populations of 200 animals. On benchmark v1, a population SHALL pass when every family passes: each statistic at |z| ≤ 2.84, and local search only where its presence test holds. On benchmark v2, a population SHALL pass when every family passes: each statistic at |z| ≤ 3.0078, and local search and chemotaxis only where present. The chosen restart SHALL pass the wild type when at least 3 of its 5 populations pass. Every restart's verdict and D, the restart spread and the parameters SHALL be reported beside the verdict.

#### Scenario: C.1 fails the wild type

- **WHEN** fewer than 3 of the chosen restart's 5 populations pass
- **THEN** the failing statistics SHALL be recorded as the diagnosis, and the roadmap's pivot SHALL go to the maintainer before C.2

#### Scenario: One unlucky population does not fail a faithful controller

- **WHEN** one of five populations fails a family and the other four pass
- **THEN** the chosen restart SHALL pass the wild type
