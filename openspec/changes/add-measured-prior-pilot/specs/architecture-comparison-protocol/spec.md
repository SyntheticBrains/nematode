## ADDED Requirements

### Requirement: A pin is chosen on the learner's own gate, never on the contrast it will carry

Where a pilot sweeps a setting in order to fix the value a later contrast runs at, the record SHALL
register the selection rule before the pilot runs, and that rule SHALL depend only on whether the
learner learns at each level — its arms against their own frozen floors — and on a stated default. It
SHALL NOT depend on the contrast the later campaign exists to read. A value picked because it shows
the contrast best has built the answer into the operating point, and fresh seeds downstream do not
undo that: they re-measure the contrast at a point chosen for its size.

#### Scenario: The rule is fixed before the pilot and names its default

- **GIVEN** a pilot that fixes a later contrast's setting
- **WHEN** it is registered
- **THEN** its launch record SHALL state the selection rule, the default value, and the tie-break,
  before any pilot seed runs
- **AND** the rule SHALL name, for the case where no level passes, what the later campaign does

#### Scenario: The contrast is recorded but not consulted

- **GIVEN** a pilot that runs both arms of the later contrast at every level
- **WHEN** its value is selected
- **THEN** the contrast at each level SHALL be recorded descriptively, with its interval
- **AND** the selection SHALL NOT read it, and the record SHALL say so beside the chosen value

#### Scenario: A contrast that moves across the pilot is carried, not resolved

- **GIVEN** a pilot level whose contrast interval excludes zero on the side opposite to the default
  level's
- **WHEN** the record draws consequences
- **THEN** the movement SHALL be carried as a registered condition of the later contrast
- **AND** it SHALL NOT be reported as a finding at pilot scale
