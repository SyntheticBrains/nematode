## ADDED Requirements

### Requirement: F.0b-ii recovers C.2's procedure at one truth

Before C.2's registered fit, the registered procedure SHALL recover one known truth, the readiness pilot's answer, from the box centre.

- **The observation** SHALL be the truth simulated once on all six assays and the three mutants at 200 animals. It SHALL replace the bands' centres and the mutant bands' centres in the objective, with the scales unchanged.
- **The recovery distance** SHALL be the registered capped D on the six families, with the mutants entering as the objective enters them, together half.
- **The preflight, before the recovery fit,** SHALL require:
  - the prior D (200 uniform draws of the box) to exceed the floor (re-simulations 1–10) by at least 1.0;
  - the oracle's gain (re-simulations 11–20) to be at least 0.85;
  - the box centre's gain (10 re-simulations) to be at most 0.25.
- **The truth's stability:** if any of re-simulations 1–10 flips a statistic's definedness against the observation, the recovery SHALL return to the maintainer.
- **The gate:** the recovery's answer SHALL be re-simulated 10 times, and the fit SHALL run only if the gain at its median D, (prior D − D) / (prior D − floor), is at least 0.75.
- **The wild-type reference:** a record at the bands' centres and the mutant bands' centres SHALL be scored by the same gain and reported beside the gate. The record SHALL state that the gate cannot tell recovering the truth from fitting the wild type.

#### Scenario: The preflight fails

- **WHEN** the oracle's gain is 0.80
- **THEN** the recovery fit SHALL NOT run, and the change SHALL return to the maintainer

#### Scenario: The recovery falls short

- **WHEN** the recovery's gain is 0.70
- **THEN** no registered C.2 fit SHALL run until the maintainer decides

#### Scenario: A truth whose statistic flips

- **WHEN** re-simulation 4 leaves a statistic undefined that the observation defines
- **THEN** the recovery SHALL NOT run, and the change SHALL return to the maintainer

#### Scenario: The preflight runs once

- **WHEN** `recovery-preflight` is invoked and its record already exists
- **THEN** it SHALL refuse
