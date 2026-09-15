## ADDED Requirements

### Requirement: A wiring contrast under a learner that does not write the wiring states what it is about

Where a structure contrast is run under a learner that leaves the substrate's own weights fixed, the
record SHALL state that the substrate enters as **fixed features** and not as something the rule
adapts, and SHALL state which claim the result bears on. Such a result SHALL NOT be reported as
satisfying a deliverable whose condition is that the substrate's own weights are plastic.

#### Scenario: The learner's relationship to the substrate is recorded with the contrast

- **GIVEN** a wiring contrast run under a learner that does not write the substrate's weights
- **WHEN** the result is recorded
- **THEN** the record SHALL state which tensors the learner writes and which it leaves fixed
- **AND** it SHALL state that the contrast is about the substrate as fixed features

#### Scenario: A positive result is not read as the plastic-substrate deliverable

- **GIVEN** a deliverable whose condition is that the substrate's own weights are plastic
- **WHEN** a contrast under a learner that leaves them fixed returns positive
- **THEN** the record SHALL state that the deliverable's condition remains unmet
- **AND** the positive SHALL be reported as a claim about the substrate's fixed features

### Requirement: A campaign whose null carries a registered consequence states its power in advance

Where a null result would trigger a registered consequence — closing a phase, retiring a programme,
or standing as a claim about the substrate — the registration SHALL state, before the campaign runs,
what effect size the design can detect and against which comparator. Where a comparator's own effect
size is on the record, the power against it SHALL be computed and stated.

#### Scenario: The power arithmetic is registered before the run

- **GIVEN** a campaign whose null outcome carries a registered consequence
- **WHEN** its protocol is registered
- **THEN** the registration SHALL state the seed count, the detectable effect size, and the power
  against the comparator the result will be read beside
- **AND** where the design is underpowered against that comparator, the registration SHALL say so
  rather than leave it to be discovered in the result

#### Scenario: An underpowered null is not recorded as a clean null

- **GIVEN** a campaign underpowered against its comparator
- **WHEN** it returns a null
- **THEN** the record SHALL state the null as underpowered against that comparator
- **AND** it SHALL NOT be reported as evidence that the effect is absent
