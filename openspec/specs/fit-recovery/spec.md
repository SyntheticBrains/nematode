# fit-recovery Specification

## Purpose

Parameter-recovery positive controls for the fitting methods. Statistics generated at known
parameters must be recovered, under rules registered before the run. NPE is gated on calibration and
informativeness. sep-CMA is gated on how close each truth's best restart comes to that truth's own
noise floor, with the oracle and the unfitted start checked before launch.

## Requirements

### Requirement: Registered before the recovery runs

A launch record SHALL fix the model's parameters and priors, the summary, the simulation and seed
scheme, the NPE configuration and the gate before any training simulation runs. It SHALL include the
pilot's measured cost.

#### Scenario: The cost is measured first

- **WHEN** the launch record is committed
- **THEN** it SHALL state the pilot's seconds per simulation and the projected wall time

### Requirement: NPE recovers C.1's parameters with honest calibration

NPE SHALL be trained on 10,000 simulations drawn from the prior, twice with different seeds. Each
flow SHALL pass two gates:

- **calibrated:** on 100 held-out truths from the prior, the pooled central-90% interval coverage over
  truths and parameters SHALL lie in [0.80, 0.97];
- **informative:** over 20 of those truths, the median over truths of the median D between a truth's
  simulation and simulations at 20 posterior draws SHALL be at most 1.42.

Each parameter's coverage and contraction, the same D at prior draws, and coverage on the quarter of
truths nearest the wild type SHALL be reported beside the gates.

#### Scenario: An overconfident posterior fails

- **WHEN** a flow's pooled coverage is below 0.80
- **THEN** F.0a SHALL fail, and NPE SHALL not be used for C.1 until a new change fixes it

#### Scenario: A posterior equal to the prior fails

- **WHEN** a flow returns the prior for every observation
- **THEN** it SHALL pass calibration but fail the informative gate, and F.0a SHALL fail

#### Scenario: An unconstrained parameter is named

- **WHEN** a parameter's contraction is near zero
- **THEN** the record SHALL report it as not constrained by the benchmark's statistics

### Requirement: Undefined statistics enter the summary explicitly

Each statistic SHALL be z-scored over the training simulations where it is defined. An undefined
entry SHALL then be set to −5, with a 0/1 flag beside it, so the summary is finite and an undefined
value carries information.

#### Scenario: A worm that never leaves

- **WHEN** a simulation's pre-exit enrichment is undefined
- **THEN** its summary SHALL hold the sentinel and a flag of 1

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

Candidates SHALL be F.0a's prior simulations with every statistic defined, D at most 5 to the bands and a presence bound above 1.5, in ascending D. A candidate SHALL be kept only if none of its first 10 fresh re-simulations changes any statistic's definedness, and its prior D exceeds its floor by at least 1.0. The first kept candidate SHALL be the pilot truth and the next four the registered truths.

#### Scenario: A truth whose statistics flip

- **WHEN** one of the first 10 re-simulations leaves a statistic undefined that the observation defines
- **THEN** that candidate SHALL be skipped, and the record SHALL say so

### Requirement: The oracle passes and the unfitted start fails before launch

Each truth's repeats 11–20, scored as a fit, SHALL reach a median gain fraction of at least 0.85 over the registered truths. The box centre, re-simulated and scored the same way, SHALL have a median gain of at most 0.25. If either check fails, or the pilot's gain after 300 generations is below 0.75, no registered fit SHALL run.

#### Scenario: A fit that never moves

- **WHEN** the scored point is the starting point
- **THEN** its gain fraction SHALL be near 0, and the gate SHALL fail

### Requirement: sep-CMA recovers C.1's model near the wild type

Each of the 4 truths SHALL be fitted twice by sep-CMA from the box centre (σ₀ = 0.2, population 16, bounds [0, 1]), stopping at the registered plateau rule or the cap. A fit's answer SHALL be its distribution's mean mapped into the box. For each truth, the answer of the restart with the lower final objective SHALL be re-simulated 10 times, and its gain fraction is (prior D − its median D) / (prior D − floor). The median gain over the truths SHALL be at least 0.75. The other restart's gain, the restarts' distance to each other, the stop generations and the parameter errors SHALL be reported beside the gate.

#### Scenario: The best restart is chosen by the fit's own objective

- **WHEN** both restarts of a truth have stopped
- **THEN** the one with the lower mean of its last 20 generations' median objectives SHALL be scored, without reference to its re-simulated gain

#### Scenario: The answer stays in the box

- **WHEN** the distribution's mean has left the unit cube
- **THEN** the answer SHALL be its image in the box, and every parameter SHALL lie within its prior range

#### Scenario: An early plateau does not shorten the cap

- **WHEN** the pilot's answer gains more than 0.05 between ⌈1.5·G_p⌉ and generation 300
- **THEN** the registered cap SHALL be 300
