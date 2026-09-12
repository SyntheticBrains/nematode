## ADDED Requirements

### Requirement: A wiring contrast is run on a cell matched to the behaviour under claim

Where a comparison asks whether a biological wiring is load-bearing, the cell it is measured on
SHALL be matched to the behaviour the claim is about, and the record SHALL state which behaviours
the cell demands. A contrast measured only on a multi-objective cell SHALL NOT be reported as a
general statement about the wiring, and where the substrate's measured deficit is concentrated in
one component of such a cell, that component SHALL be separated before the contrast is read as
evidence about the wiring.

#### Scenario: A multi-objective result is scoped to its cell

- **GIVEN** a wiring contrast measured on a cell demanding several behaviours
- **WHEN** the result is recorded
- **THEN** it SHALL be scoped to that cell's demand, and SHALL NOT be reported as a general
  statement about the wiring

#### Scenario: The component carrying the deficit is separated

- **GIVEN** a substrate whose measured deficit on a multi-objective cell is concentrated in one
  component
- **WHEN** a wiring contrast on that cell returns a null
- **THEN** the contrast SHALL be re-run on a cell without that component before the null is read as
  evidence about the wiring

### Requirement: A wiring contrast is gated on the cell showing learning

Each arm of a wiring contrast SHALL be paired with its own floor on the same seeds — the same
substrate with the optimiser disabled — and the contrast SHALL NOT be read unless the arm carrying
the claim beats that floor by the registered test. Where it does not, the result SHALL be recorded
as a finding about the platform and SHALL license no conclusion about the wiring.

A ceiling threshold SHALL be registered before any data exists, and where both arms of a contrast
reach it the contrast SHALL be recorded as unresolvable on that cell, with the remedy named in
advance rather than chosen after the outcome is known.

#### Scenario: A contrast whose arm did not learn is not read

- **GIVEN** a wiring contrast whose claim-carrying arm does not beat its own frozen floor
- **WHEN** the family is scored
- **THEN** the cell's verdict SHALL be that no learning occurred, and the contrast SHALL NOT be
  assigned a wiring verdict

#### Scenario: A saturated cell is named, not re-tuned

- **GIVEN** both arms of a contrast at or above the registered ceiling threshold
- **WHEN** the family is scored
- **THEN** the cell SHALL be recorded as unresolvable and the registered remedy applied, and the
  recipe SHALL NOT be adjusted until the arms separate

#### Scenario: A minimum effect is registered beside significance

- **GIVEN** a paired contrast at a sample size where a rank test fires on the consistency of the
  sign rather than the size of the shift
- **WHEN** the primary is registered
- **THEN** a minimum effect SHALL be registered with it, and a significant result below that minimum
  SHALL be recorded as such and SHALL license nothing on its own
