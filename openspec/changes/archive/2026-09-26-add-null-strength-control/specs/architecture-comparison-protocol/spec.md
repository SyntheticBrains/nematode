## ADDED Requirements

### Requirement: A null states every structural property it does not preserve

Where a result compares a real structure against a null, the record SHALL state which structural
properties the null preserves and which it does not — at minimum degree, weight or strength per unit,
and self-connections — and SHALL NOT describe the null by the one property it was built to preserve.
A null that preserves degree and moves strength compares the real structure against a different
distribution of strength as well as a different placement, and a result read as "placement matters"
then has a second explanation the record never named.

#### Scenario: The unpreserved properties are named where the null is introduced

- **GIVEN** a campaign that reads a structure contrast against a null
- **WHEN** its launch record describes the null
- **THEN** it SHALL list the properties the null preserves and the properties it does not, including
  per-unit strength and self-connections
- **AND** SHALL name the result any unpreserved property could explain

#### Scenario: An unpreserved property found after a result ships becomes a standing condition

- **GIVEN** a shipped structure result whose null is found to move a property the record did not name
- **WHEN** the finding is recorded
- **THEN** the result SHALL carry it as a standing condition at every citation site
- **AND** a control holding that property at the real structure's value SHALL be scheduled before the
  result is cited in a synthesis or a publication

#### Scenario: A combined control is read as combined

- **GIVEN** a control that holds several unpreserved properties at their real values at once
- **WHEN** it moves the contrast
- **THEN** the record SHALL attribute the move to the properties jointly
- **AND** SHALL NOT name any one of them as the cause without a control that separates them
