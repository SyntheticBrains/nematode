## ADDED Requirements

### Requirement: A diagnostic that borrows a gradient-trained component states what it cannot establish

Where a plausibility claim is under test and a diagnostic supplies one of the learner's components from a
gradient-trained source, the record SHALL state that the diagnostic cannot satisfy the plausibility
deliverable whatever it returns, and SHALL name what a positive result would license instead. A positive
result SHALL NOT be reported as the plausible learner working.

#### Scenario: The borrowed component is disclosed with the result

- **GIVEN** an arm whose component is taken from a gradient-trained run
- **WHEN** its result is recorded
- **THEN** the record SHALL name the borrowed component and its source
- **AND** it SHALL state that the arm is a diagnostic and not a candidate for the plausibility
  deliverable

#### Scenario: A positive diagnostic names its plausible follow-up

- **GIVEN** a diagnostic in which the borrowed component removes the failure
- **WHEN** the consequence is recorded
- **THEN** the record SHALL name a follow-up that does not depend on the gradient-trained source
- **AND** it SHALL NOT treat the diagnostic itself as satisfying the deliverable

### Requirement: A substituted component is tested against a same-magnitude random control

Where a component is replaced to test whether it limits learning, the comparison SHALL include a control
of the same magnitude and arbitrary direction, so that "this particular replacement helps" is separable
from "the default was bad and any change helps". The control's position SHALL be reported in every
branch of the verdict, including branches where the substitution shows no effect.

#### Scenario: The random control separates the two readings

- **GIVEN** a substituted component that outperforms the default
- **WHEN** the result is recorded
- **THEN** the same-magnitude random control's result SHALL be reported beside it
- **AND** the record SHALL state which reading the ordering supports, and the follow-up each implies

#### Scenario: Only the component under test is substituted

- **GIVEN** a component harvested from a source that also carries other trained tensors
- **WHEN** the arm is prepared
- **THEN** every tensor other than the one under test SHALL be at the arm's own initialisation
- **AND** the preparation SHALL be checked against the unmodified arm so a second simultaneous change
  fails rather than being reported as one change

#### Scenario: A harvested component is taken under the conditions it will be used in

- **GIVEN** a component harvested from a source run whose configuration differs from the arm's
- **WHEN** that difference affects what the component is adapted to
- **THEN** the harvest SHALL be configured to match the arm on that setting, or the mismatch SHALL be
  recorded as a limit on the result

#### Scenario: An arm reusing an earlier result as its comparator establishes they are comparable

- **GIVEN** a comparator taken from an earlier campaign that did not go through this change's setup path
- **WHEN** it is used as the baseline
- **THEN** the equivalence of the two paths SHALL be established by test rather than by argument
- **AND** where it cannot be established, the comparator SHALL be re-run through the same path
