## ADDED Requirements

### Requirement: A control repeated on another cell is sized and judged on that cell's own effect

When a registered control is repeated on a cell other than the one it first ran on, its minimum SHALL be
a fraction of that cell's own committed effect at the same point, and its sensitivity SHALL be taken
from that cell's own committed per-seed spread. Neither SHALL be carried over from the first cell, and
neither SHALL be taken from the repeated control's own data.

#### Scenario: The first cell's minimum is carried over

- **GIVEN** a control registered on one cell with a minimum taken from that cell's effect
- **WHEN** it is repeated on a second cell whose committed effect differs
- **THEN** the repeat SHALL register a minimum from the second cell's effect
- **AND** SHALL size its seed count from the second cell's per-seed spread

#### Scenario: A follow-up's minimum would come from its own campaign

- **GIVEN** two interactions scored in one campaign, one of which on the first cell took its minimum
  from the other's committed result
- **WHEN** both are registered together on a new cell
- **THEN** both SHALL be read against a minimum fixed before the campaign from committed data, never
  against a fraction of a result the same campaign produces
