## ADDED Requirements

### Requirement: A feature ablation on a positive structure result registers a minimum effect as a decision rule

Where an experiment removes one feature of a substrate to ask whether a previously established
structure effect survives, the record SHALL register, before the campaign runs, the smallest
reduction in that effect that will be read as the feature carrying it — stated as a fraction of the
established effect — and SHALL NOT read a significant reduction below that minimum as the feature
carrying the effect.

#### Scenario: The minimum is a fraction of the effect being ablated

- **GIVEN** an established structure effect of a known size
- **WHEN** an ablation is registered against it
- **THEN** the minimum reduction SHALL be stated as a fraction of that size, with the power to detect
  it computed from observed spread and registered beside it

#### Scenario: Significant below the minimum is not carrying

- **GIVEN** an ablation whose interaction is significant but smaller than the registered minimum
- **WHEN** the reading is assigned
- **THEN** it SHALL be reported as inconclusive at the panel's sensitivity, with the observed size and
  the minimum both stated
- **AND** it SHALL NOT be reported as the feature carrying the effect

#### Scenario: A removal that moves the frozen substrate is qualified

- **GIVEN** an ablation whose frozen floors differ from the baseline's frozen floors
- **WHEN** the ablation reads as carrying the effect
- **THEN** the reading SHALL state that the substrate's operating point moved and SHALL be reported
  as carrying-or-saturating rather than as carrying
- **AND** the floor comparison SHALL be registered before the campaign runs

#### Scenario: Ablations are read separately

- **GIVEN** more than one ablation of the same established effect in one campaign
- **WHEN** their readings are assigned
- **THEN** each SHALL be read on its own, and a difference between them SHALL be reported as the
  finding rather than averaged into one reading

### Requirement: A committed baseline is reused only under a byte-identity check

Where a campaign reuses committed runs from an earlier campaign as one cell of a contrast, the record
SHALL establish that a run produced now reproduces a committed run on every field the analysis reads,
and SHALL re-run the baseline in full where any field differs.

#### Scenario: Reuse is licensed by a re-run, not by argument

- **GIVEN** committed runs proposed as a baseline for a new contrast
- **WHEN** anything in the execution path has changed since they ran — code, flags, environment
- **THEN** at least one seed per reused arm SHALL be re-run under the new path and compared to the
  committed log on every parsed field
- **AND** the record SHALL state the comparison's result as the evidence for reuse

#### Scenario: There is no partial reuse

- **GIVEN** a byte-identity check in which any field differs for any reused arm
- **WHEN** the campaign is planned
- **THEN** the whole baseline SHALL be re-run under the new path
- **AND** no committed run SHALL be mixed with re-run ones in the same contrast

### Requirement: A mechanism probe is registered before its correlation with the outcome is computed

Where a result is left unexplained and a structural statistic is proposed to explain it, the record
SHALL register the statistic, the direction of the predicted relationship, and a minimum effect before
the relationship to the outcome is computed, and SHALL read a positive as licensing a hypothesis
rather than as establishing a mechanism.

#### Scenario: Feasibility may be checked, the correlation may not

- **GIVEN** a candidate structural statistic
- **WHEN** it is being registered
- **THEN** its computability and whether it discriminates the conditions MAY be checked
- **AND** its relationship to the outcome variable SHALL NOT be computed until the test is registered

#### Scenario: A positive probe licenses a follow-up

- **GIVEN** a registered probe that meets its direction and minimum
- **WHEN** it is recorded
- **THEN** it SHALL be reported as a hypothesis licensed for a follow-up that manipulates the
  statistic directly
- **AND** it SHALL NOT be reported as the mechanism of the unexplained result
