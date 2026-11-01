## ADDED Requirements

### Requirement: B v3 is valid only if real worms pass, the exploits fail, and the new statistics are reachable

Benchmark v3 SHALL be VALID only if every registered control passes:

- the Chen round trip for the two new statistics;
- the reachability oracle;
- all 10 held-out López-Cruz batches;
- Chen's leave-one-out within its allowance;
- the adversaries;
- the negatives.

Otherwise it SHALL be VOID, and v2 SHALL stay the default.

#### Scenario: A reversal-dominated answer is caught

- **WHEN** C.1b's restart 2, 4 or 6, or the κ probe's answer, is graded on 5 fresh populations
- **THEN** it SHALL fail at least one of log early/late, the onset rate and the direction-flip fraction in at least 3 of them, or v3 SHALL be VOID

#### Scenario: The new statistics are unreachable

- **WHEN** the oracle's fitted worm fails a new statistic's mean-z or 90% rule on its 40 populations
- **THEN** the run SHALL stop before Part B, and return to the maintainer

#### Scenario: v3 becomes the default

- **WHEN** `controls_v3.json` reads VALID with `bands_v3.json`'s SHA-256
- **THEN** `benchmark_grade.py` SHALL grade on v3 by default, and `--bands v2` SHALL reproduce every v2 reading
