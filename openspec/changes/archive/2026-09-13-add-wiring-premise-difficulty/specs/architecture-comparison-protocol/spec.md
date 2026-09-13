## ADDED Requirements

### Requirement: An effect present on one cell and absent on another is attributed only after the cells' differences are separated

Where a wiring contrast is positive on one cell and null on another, the record SHALL enumerate every
respect in which the two cells differ, and SHALL NOT attribute the difference in outcome to one of
them while the others stand untested. Where a respect cannot be isolated by the arms available, the
record SHALL name it as unseparated rather than leaving the attribution implied.

#### Scenario: Cells differing in several respects are enumerated before attribution

- **GIVEN** a wiring contrast positive on one cell and null on another
- **WHEN** the difference is recorded
- **THEN** every respect in which the cells differ SHALL be stated
- **AND** no single respect SHALL be named as the cause while others are untested

#### Scenario: An unseparated factor is named rather than implied

- **GIVEN** two candidate explanations that the available arms cannot distinguish
- **WHEN** the result is recorded
- **THEN** both SHALL be reported as live, and the arm that would separate them SHALL be named
- **AND** the record SHALL NOT present either as established

#### Scenario: A difficulty manipulation that fails to discriminate is a fact about the manipulation

- **GIVEN** a cell whose difficulty was raised to make a contrast measurable
- **WHEN** both arms reach the registered ceiling on that cell
- **THEN** the result SHALL be recorded as a property of the manipulation, not of the wiring, and the
  registered remedy SHALL be applied once without tuning the recipe further
