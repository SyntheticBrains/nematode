## ADDED Requirements

### Requirement: A panel's contrast reads both components of its outcome

Where a panel's outcome is bimodal — a seed reaching a competent policy or a dead one — the
registered contrast SHALL read both the frequency with which an arm reaches competence and the
level it reaches when it does, because a test of either alone reports no effect when only the other
moves. The registered family SHALL comprise a paired competent-fraction discordance at the
committed competence threshold, a paired contrast on the level among competent seeds, and the
all-seeds paired rank test the committed record was scored with, corrected together.

The competence threshold SHALL be the committed one and SHALL NOT be re-chosen with the outcome known. The level contrast SHALL compare each arm's mean over its **own** competent seeds, so that a seed competent in one arm only contributes to that arm's level and a frequency difference is not read as a level difference; where either arm has no competent seed the contrast SHALL be reported as undefined rather than as a null result. Every member SHALL be one-sided in the arm-improves direction at the committed significance level, corrected together.

#### Scenario: A level-only effect is not reported as no effect

- **GIVEN** two arms reaching competence about equally often, where one arm's competent seeds score
  higher than the other's
- **WHEN** the family is computed
- **THEN** the level contrast SHALL report the difference, and the verdict SHALL NOT be the
  no-effect branch

#### Scenario: A frequency difference is not read as a level difference

- **GIVEN** two arms whose competent seeds score the same, where one arm reaches competence on
  more seeds than the other
- **WHEN** the family is computed
- **THEN** the frequency contrast SHALL report the difference and the level contrast SHALL NOT,
  and the verdict SHALL be the frequency-only branch

#### Scenario: A seed competent in one arm counts toward that arm's level

- **GIVEN** a seed competent in one arm and not the other
- **WHEN** the level contrast is computed
- **THEN** its value SHALL enter that arm's level and SHALL NOT enter the other's

#### Scenario: An undefined level contrast is not a null

- **GIVEN** a panel in which one arm has no competent seed
- **WHEN** the family is computed
- **THEN** the level contrast SHALL be reported as undefined, and SHALL NOT contribute a passing or
  failing result to the family

### Requirement: A graded metric is read beside the full-clear metric

A panel SHALL be read on a graded measure of task progress in addition to the full-clear rate, so
that learning short of a full clear is visible. The graded reading SHALL use the level and all-seeds members of the family over the competence the primary metric defines, corrected within itself, SHALL NOT choose a competence threshold of its own, and the full-clear metric SHALL remain the primary one in which registered verdicts are expressed.

#### Scenario: Progress short of a clear is visible

- **GIVEN** an arm whose full-clear rate is at its floor while its graded measure exceeds the
  comparator's
- **WHEN** both readings are computed
- **THEN** the graded reading SHALL report the difference, and the record SHALL carry both

### Requirement: A bimodal outcome has a name that licenses nothing

The outcome map SHALL name every combination of the frequency and level contrasts' directions, and
SHALL include a branch for a two-directional result: the two contrasts significant against each
other. That branch SHALL license no follow-on work and SHALL require its own registration to act
on.

A per-seed spread — some seeds improved and some degraded — SHALL NOT decide that branch, and a
panel with neither contrast significant SHALL be the no-effect branch whatever its spread. The
counts MAY be recorded descriptively. The reason is that the null of such a panel is itself
bimodal, so two arms drawn from one law routinely place a seed high in one and low in the other;
no threshold on that statistic distinguishes a mixed response from noise at these panel sizes.

#### Scenario: A split outcome is named rather than discovered

- **GIVEN** a panel whose frequency and level contrasts are significant in opposite directions
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the split branch, and the record SHALL state that it licenses nothing

#### Scenario: A per-seed spread is not a mixed response

- **GIVEN** a panel with neither contrast significant, some seeds improved and some degraded
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the no-effect branch

#### Scenario: Opposed significant contrasts are the split branch, not degradation

- **GIVEN** a panel where fewer seeds reach competence and those that do score higher, both
  significant
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the split branch

#### Scenario: Missing significance alone is not the split branch

- **GIVEN** a panel with neither contrast significant and no seed improved above the comparator
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the no-effect branch

### Requirement: The re-read of committed tables cannot change a committed verdict

Committed results SHALL be re-read under the registered family from their committed per-seed
tables. Each committed verdict SHALL stand as registered, in the units and under the rule it was
registered with; the re-read SHALL be reported beside it as a second, pre-specified reading and
SHALL be an input to the ladder re-read alone.

#### Scenario: A disagreement leaves the verdict standing

- **GIVEN** a committed result whose re-read points the other way
- **WHEN** the re-read is recorded
- **THEN** both readings SHALL be reported, the committed verdict SHALL be unchanged, and the record
  SHALL state which is the verdict

#### Scenario: An assay is re-read in its own protocol

- **GIVEN** a committed result whose protocol is an assay against a per-seed comparator rather than
  a panel against a paired arm
- **WHEN** it is re-read
- **THEN** the contrasts SHALL be computed against each seed's own committed comparator, and results
  SHALL NOT be pooled across protocols
