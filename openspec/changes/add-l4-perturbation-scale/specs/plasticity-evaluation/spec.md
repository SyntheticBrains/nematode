## ADDED Requirements

### Requirement: A stochastic-gradient result records the perturbation dimension it was measured at

Where a learning rule estimates a gradient from a scalar signal and per-unit noise, every recorded
result SHALL state the number of perturbed units and the number of perturbation draws per scored
decision. Where results from different platforms are compared, the record SHALL state that dimension
for each, so a difference in the dimension is visible beside the difference in outcome.

#### Scenario: The dimension is reported with the outcome

- **GIVEN** a result produced by a rule whose eligibility carries per-unit perturbation
- **WHEN** the result is recorded
- **THEN** the record SHALL state the perturbed-unit count and the draws per scored decision
- **AND** where the two differ, because units are perturbed more than once per decision, both SHALL be
  stated rather than one standing for the other

#### Scenario: A cross-platform comparison states each platform's dimension

- **GIVEN** a success on one platform and a failure on another under the same rule
- **WHEN** the two are compared in a record
- **THEN** the perturbation dimension of each SHALL be stated
- **AND** the record SHALL NOT attribute the difference in outcome to the difference in task while the
  dimensions differ and are untested

### Requirement: A scale-dependent mechanism is tested where the rule works before a failure is attributed to it

Where a mechanism predicts that a rule's performance depends on a platform dimension, that dimension
SHALL be varied on a platform where the rule is demonstrated to learn, before any failure elsewhere is
attributed to the dimension or cleared of it. The varying platform SHALL be one on which the dimension
is isolated from capacity, or the record SHALL carry a capability control per level and report a level
that fails it as uninterpretable rather than as a null.

#### Scenario: The sweep runs first where the rule learns

- **GIVEN** a proposed explanation of a failure in terms of a platform dimension
- **WHEN** the dimension is investigated
- **THEN** it SHALL be swept on a platform where the rule passes its positive control
- **AND** the sweep SHALL report the rate the mechanism makes a claim about, not only a pass or fail

#### Scenario: A level without capacity is uninterpretable, not negative

- **GIVEN** a sweep whose levels vary capacity as well as the dimension under test
- **WHEN** a level shows no effect
- **THEN** the record SHALL consult that level's capability control
- **AND** a level failing that control SHALL be reported as uninterpretable rather than as evidence
  against the mechanism

#### Scenario: A budget read outside the fitted range is labelled an extrapolation

- **GIVEN** a fit of performance against a platform dimension over a bounded grid
- **WHEN** the fit is used to state what a platform outside that grid would require
- **THEN** the figure SHALL be labelled an extrapolation wherever it appears
- **AND** it SHALL NOT be reported as a measurement of that platform
