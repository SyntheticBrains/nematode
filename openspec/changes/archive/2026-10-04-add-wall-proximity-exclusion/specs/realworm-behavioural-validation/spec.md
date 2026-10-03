## ADDED Requirements

### Requirement: Wall-proximal transitions can be excluded

The validation harness SHALL accept a wall margin and the arena's side. When given, it SHALL drop every
transition with a step closer than the margin to any edge of the square arena, SHALL never form a
transition across a dropped stretch, and SHALL record the margin, the arena and the transitions kept
before and after in its summary. When not given, its summary SHALL be unchanged.

#### Scenario: A run that visits a wall

- **WHEN** a run crosses the arena, slides along an edge and returns
- **THEN** the steps within the margin of the edge SHALL be dropped
- **AND** the stretches before and after SHALL be analysed as separate runs, with no transition joining
  them

#### Scenario: The exclusion is off

- **WHEN** the harness runs without a wall margin
- **THEN** its summary SHALL carry no wall-exclusion block and SHALL equal what it produced before this
  requirement existed

#### Scenario: A margin is given without the arena

- **WHEN** a wall margin is given without the arena's side, or the reverse
- **THEN** the harness SHALL refuse to run
