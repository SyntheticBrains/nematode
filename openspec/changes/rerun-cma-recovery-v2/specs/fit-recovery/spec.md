## ADDED Requirements

### Requirement: sep-CMA's recovery is re-read on benchmark v2

Before C.1 is refitted on B v2, sep-CMA's recovery of C.1's model SHALL be re-run under v2's objective: the distance over v2's 19 graded statistics on its n = 200 band scales with its five-family weights, between records of simulations passed through v2's measurement models. Truths SHALL be re-selected on v2 by F.0b-i's rule, the cap SHALL be set again by a 300-generation pilot, and the gate SHALL be F.0b-i's: the median of the four truths' best-restart gains at least 0.75.

#### Scenario: The objective is the one C.1b fits

- **WHEN** a fit's candidate is scored
- **THEN** the score SHALL be the v2 distance over 19 statistics, and the on-lawn reversal rate, which v2 reports and does not grade, SHALL take no part

#### Scenario: The gates are read before they are registered

- **WHEN** the oracle's or the unfitted start's gain, read on discarded seeds before registration, lies within 0.05 of its bar or on its wrong side
- **THEN** the change SHALL return to the maintainer before Part A is committed

#### Scenario: A truth on v2 is stable and near the wild type

- **WHEN** a candidate's v2 repeats change any statistic's definedness, or its prior distance exceeds its floor by less than 1.0
- **THEN** it SHALL be skipped and recorded with the reason

### Requirement: The plateau stop is guarded by sigma

A recovery fit on v2 SHALL stop on the plateau rule only once its σ is at most σ₀, and otherwise at its cap, as C.1b will.

#### Scenario: A false plateau at wide sigma

- **WHEN** the plateau rule holds at a generation where σ exceeds σ₀
- **THEN** the fit SHALL continue
