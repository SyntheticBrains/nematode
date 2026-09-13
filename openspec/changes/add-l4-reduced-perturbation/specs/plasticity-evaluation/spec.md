## ADDED Requirements

### Requirement: A perturbation that cannot reach the scored outcome is not counted as exploration

Where a rule forms its learning signal from per-unit perturbation, and the scored outcome depends on a
strict subset of units through a forward pass of finite depth, the record SHALL state what share of
perturbation draws can reach that outcome. A claim about the perturbation dimension SHALL distinguish
draws that can influence the outcome from draws that cannot, and SHALL NOT report the total as though
every draw contributed exploration.

#### Scenario: The causally connected share is reported with the dimension

- **GIVEN** a substrate whose readout reads a subset of its units at a finite settling depth
- **WHEN** the perturbation dimension is recorded
- **THEN** the record SHALL state the draws per scored decision and the share of them that can reach the
  readout
- **AND** where the two differ, both SHALL be stated rather than the total standing for the dimension

#### Scenario: Removing unreachable draws is recorded as a correctness change

- **GIVEN** a mask that removes only draws which cannot influence the scored outcome
- **WHEN** its effect is recorded
- **THEN** it SHALL be described as removing variance without removing signal
- **AND** it SHALL NOT be reported as a reduction in the dimension available to the learner, since the
  units and synapses it can adapt are unchanged

#### Scenario: A restricted set states what it gives up

- **GIVEN** a perturbation set restricted beyond causal reach
- **WHEN** a result is recorded for it
- **THEN** the record SHALL state the synapses that set can no longer adapt
- **AND** a positive result SHALL be reported as a result for a restricted learner
