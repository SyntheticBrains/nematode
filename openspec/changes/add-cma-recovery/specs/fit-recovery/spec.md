## ADDED Requirements

### Requirement: sep-CMA's recovery is registered in two parts

A launch record SHALL be committed in two dated parts.

- **Part A, before the preflight**, SHALL fix:
  - the objective, the CMA settings and the seeds;
  - the plateau rule;
  - the truth-selection rule;
  - the preflight's thresholds;
  - the gate;
  - the stop conditions.
- **Part B, before the registered fits**, SHALL add the truths, the preflight's numbers, the pilot's plateau generation, the cap and the projected cost. It SHALL NOT change any gate or threshold.

#### Scenario: The read is guarded

- **WHEN** `read` runs while the launch record, the preflight record or the pilot record is uncommitted or differs from its version in the commit that last touched the launch record
- **THEN** it SHALL refuse to read the gate

### Requirement: Truths are stable worms near the wild type

Candidates SHALL be F.0a's prior simulations with every statistic defined, D at most 5 to the bands and a presence bound above 1.5, in ascending D. A candidate SHALL be kept only if none of 20 fresh re-simulations changes any statistic's definedness, and its prior D exceeds its floor by at least 1.0. The first kept candidate SHALL be the pilot truth and the next four the registered truths.

#### Scenario: A truth whose statistics flip

- **WHEN** a re-simulation leaves a statistic undefined that the observation defines
- **THEN** that candidate SHALL be skipped, and the record SHALL say so

### Requirement: The oracle passes and the unfitted start fails before launch

Each truth's repeats 11–20, scored as a fit, SHALL reach a median gain fraction of at least 0.85 over the registered truths. The box centre, re-simulated and scored the same way, SHALL have a median gain of at most 0.25. If either check fails, or the pilot's gain after 300 generations is below 0.75, no registered fit SHALL run.

#### Scenario: A fit that never moves

- **WHEN** the scored point is the starting point
- **THEN** its gain fraction SHALL be near 0, and the gate SHALL fail

### Requirement: sep-CMA recovers C.1's model near the wild type

Each of the 4 truths SHALL be fitted twice by sep-CMA from the box centre (σ₀ = 0.2, population 16, bounds [0, 1]), stopping at the registered plateau rule or the cap. For each truth, the restart with the lower final objective SHALL be re-simulated 10 times, and its gain fraction is (prior D − its median D) / (prior D − floor). The median gain over the truths SHALL be at least 0.75. The other restart's gain, the restarts' distance to each other, the stop generations and the parameter errors SHALL be reported beside the gate.

#### Scenario: The best restart is chosen by the fit's own objective

- **WHEN** both restarts of a truth have stopped
- **THEN** the one with the lower mean of its last 20 generations' median objectives SHALL be scored, without reference to its re-simulated gain
