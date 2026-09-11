## ADDED Requirements

### Requirement: A panel's contrast reads both components of its outcome

Where a panel's outcome is bimodal — a seed reaching a competent policy or a dead one — the
registered contrast SHALL read both the frequency with which an arm reaches competence and the
level it reaches when it does, because a test of either alone reports no effect when only the other
moves. The registered family SHALL comprise a paired competent-fraction discordance at the
committed competence threshold, a paired contrast on the level among competent seeds, and the
all-seeds paired rank test the committed record was scored with, corrected together.

The competence threshold SHALL be the committed one and SHALL NOT be re-chosen with the outcome
known. A pair SHALL enter the level contrast where **either** arm is competent, and where no pair
qualifies the contrast SHALL be reported as undefined rather than as a null result.

#### Scenario: A level-only effect is not reported as no effect

- **GIVEN** two arms reaching competence about equally often, where one arm's competent seeds score
  higher than the other's
- **WHEN** the family is computed
- **THEN** the level contrast SHALL report the difference, and the verdict SHALL NOT be the
  no-effect branch

#### Scenario: The level contrast does not condition on both arms succeeding

- **GIVEN** a pair in which exactly one arm is competent
- **WHEN** the level contrast is computed
- **THEN** that pair SHALL be included

#### Scenario: An undefined level contrast is not a null

- **GIVEN** a panel in which no seed is competent in either arm
- **WHEN** the family is computed
- **THEN** the level contrast SHALL be reported as undefined, and SHALL NOT contribute a passing or
  failing result to the family

### Requirement: A graded metric is read beside the full-clear metric

A panel SHALL be read on a graded measure of task progress in addition to the full-clear rate, so
that learning short of a full clear is visible. The graded reading SHALL use the same contrast
family, corrected within itself, and the full-clear metric SHALL remain the primary one in which
registered verdicts are expressed.

#### Scenario: Progress short of a clear is visible

- **GIVEN** an arm whose full-clear rate is at its floor while its graded measure exceeds the
  comparator's
- **WHEN** both readings are computed
- **THEN** the graded reading SHALL report the difference, and the record SHALL carry both

### Requirement: A bimodal outcome has a name that licenses nothing

The outcome map SHALL include a branch for an arm that improves at least one seed above the
comparator while degrading at least one below it beyond the hold band, with neither contrast
significant. That branch SHALL license no follow-on work and SHALL require its own registration to
act on, and SHALL fire only on such a split rather than wherever significance is missed.

#### Scenario: A split outcome is named rather than discovered

- **GIVEN** a panel improving some seeds and degrading others, with neither contrast significant
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the split branch, and the record SHALL state that it licenses nothing

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
