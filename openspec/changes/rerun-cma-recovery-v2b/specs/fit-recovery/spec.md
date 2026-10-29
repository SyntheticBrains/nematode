## ADDED Requirements

### Requirement: A fit ranks definedness first, then a compressed distance

A recovery fit on v2 SHALL rank its candidates by a score of 1,000 × u + D_c. Here u counts the graded statistics undefined in exactly one of the truth and the candidate, and D_c is the weighted RMS of 10·asinh(z/10) over the statistics defined in both, for z uncapped on the bands' n = 200 scales. Every quantity read against a gate SHALL use the registered, capped distance.

#### Scenario: Undefined never ranks ahead of defined

- **WHEN** one candidate leaves a single statistic undefined and is otherwise exact, and another is fully defined with three statistics 1,700 band scales from the truth
- **THEN** the fully defined candidate SHALL rank ahead

#### Scenario: Near the truth the terms are the registered ones

- **WHEN** a statistic lies within 3 band scales of the truth
- **THEN** its ranking term SHALL be within 1.5% of its z

### Requirement: The objective is probed on the truths that failed, on pinned observations

Before registration, the ranking score SHALL be fitted on discarded seeds to the two truths attempt 1 failed and to one it recovered. If any of their gains is below 0.80, the change SHALL return to the maintainer before registering. The reused observations' file SHALL be pinned by its SHA-256, and every attempt-2 stage SHALL refuse a file that differs.

#### Scenario: A failed truth still fails

- **WHEN** truth 1164's or 6985's probe gain is below 0.80
- **THEN** no registered fit SHALL run

#### Scenario: The observations changed

- **WHEN** `candidates.jsonl` no longer matches its registered SHA-256
- **THEN** every attempt-2 stage SHALL refuse to run
