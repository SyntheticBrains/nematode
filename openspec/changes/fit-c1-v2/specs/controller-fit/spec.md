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
