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

Before registration, the ranking score SHALL be fitted on discarded seeds to the two truths attempt 1 failed and to one it recovered. If any of their gains is below 0.80, the change SHALL return to the maintainer before registering, and no registered fit SHALL run for a truth below the bar unless the maintainer's decision to accept it is registered. The reused observations' file SHALL be pinned by its SHA-256, and every attempt-2 stage SHALL refuse a file that differs.

#### Scenario: A failed truth still fails

- **WHEN** truth 1164's or 6985's probe gain is below 0.80, and the maintainer has not accepted that truth below the bar
- **THEN** no registered fit SHALL run

#### Scenario: A truth accepted below the bar

- **WHEN** the maintainer accepts a probed truth below 0.80, and the decision is registered with the launch record
- **THEN** the registered fits MAY run, and the launch record SHALL disclose the truth's probe gains

#### Scenario: The observations changed

- **WHEN** `candidates.jsonl` no longer matches its registered SHA-256
- **THEN** every attempt-2 stage SHALL refuse to run

### Requirement: Each truth is fitted with four restarts

The registered re-run SHALL fit each of the four truths with 4 restarts of sep-CMA from the box centre. Each truth's best restart SHALL be the one with the lowest final objective, and the gate SHALL read its gain. Every restart's gain SHALL be reported.

#### Scenario: The best of four restarts

- **WHEN** a truth's four restarts end with different final objectives
- **THEN** the gate SHALL read the gain of the restart with the lowest final objective, and the record SHALL report the runner-up's gain, every restart's gain, and, as description, the gain of the better of the first two restarts alone

## REMOVED Requirements

### Requirement: A fit ranks on the uncapped distance

**Reason**: attempt 1's registered gate failed on this objective (median gain 0.690): a statistic undefined in only one record counted a flat 10, less than a defined statistic far from the truth, so fits escaped into parameters that leave statistics undefined.

**Migration**: recovery fits on v2 rank on the definedness-first score of "A fit ranks definedness first, then a compressed distance". Attempt 1's records keep their objective and are read as registered.
