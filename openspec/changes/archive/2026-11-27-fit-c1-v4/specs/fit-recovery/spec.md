## ADDED Requirements

### Requirement: A warm-started fit's recovery is measured from its start

Before C.1's registered v4 fit, sep-CMA SHALL recover C.1 (v4) at one known truth, the pilot's answer, fitted twice by the registered procedure from the registered start. Every check and the gate SHALL be read on two distances: the combined distance, the registered capped D on the six families with each mutant's RMS capped z over its graded statistics combined at half, and the wild type's distance, that capped D alone. On each, the truth's floor SHALL be the median distance from its observation to re-simulations 1–10, its oracle the same over 11–20, and its start D the median distance to the registered start re-simulated 10 times. The gain SHALL be (start D − D) / (start D − floor), with the prior-relative gain, the band reference's gain from the start and the gate line reported beside it. Before any recovery fit, on each distance the start D SHALL exceed the floor by at least 1.0 and the oracle's gain SHALL be at least 0.85, and no statistic's definedness SHALL differ between the observation and re-simulations 1–10. The restart with the lower final objective SHALL be re-simulated 10 times, and the registered fit SHALL run only if its gain is at least 0.75 on both distances, or an amended Part A records the maintainer's decision to run despite the gate.

#### Scenario: The gain is read from the start

- **WHEN** the truth's start D is 3.0, its floor 1.0 and a recovered answer's D 1.4
- **THEN** its gain SHALL be 0.8

#### Scenario: The start is too close to the truth

- **WHEN** the truth's start D exceeds its floor by 0.6
- **THEN** no recovery fit SHALL run, and the recovery SHALL return to the maintainer

#### Scenario: A statistic's definedness flips

- **WHEN** a mutant difference is defined in the observation and undefined in one of re-simulations 1–10
- **THEN** the preflight SHALL fail

#### Scenario: The gate passes on the mutants alone

- **WHEN** the better restart's gain is 0.86 on the combined distance and 0.40 on the wild type's
- **THEN** the gate SHALL fail

#### Scenario: The recovery falls short

- **WHEN** the better restart's gain is 0.70 on either distance
- **THEN** no registered fit SHALL run unless an amended Part A records the maintainer's decision to run despite the gate
