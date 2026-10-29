## ADDED Requirements

### Requirement: A fit ranks on a compressed distance that never rewards undefined statistics

A recovery fit on v2 SHALL rank its candidates on the v2 distance with each defined statistic's term 10·asinh(z/10), for the uncapped z on the bands' n = 200 scales, and a statistic undefined in only one record costing 100. Every quantity read against a gate SHALL use the registered, capped distance.

#### Scenario: Far is never cheaper than undefined

- **WHEN** a candidate's statistic is defined but 1,700 band scales from the truth, and another candidate's is undefined
- **THEN** the defined candidate SHALL rank ahead (58.3 against 100)

#### Scenario: Near the truth the distance is the registered one in shape

- **WHEN** every statistic of two candidates lies within a few band scales of the truth
- **THEN** their order SHALL be the order the uncapped distance gives

### Requirement: The objective is probed on the truths that failed

Before registration, the compressed objective SHALL be fitted on discarded seeds to the two truths attempt 1 failed and to one it recovered. If any of their gains is below 0.75, the change SHALL return to the maintainer before registering.

#### Scenario: A failed truth still fails

- **WHEN** truth 1164's or 6985's probe gain is below 0.75
- **THEN** no registered fit SHALL run
