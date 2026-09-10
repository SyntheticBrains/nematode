## ADDED Requirements

### Requirement: An eligibility variant clears the clone assay before any panel arm

A variant of the three-factor rule that has passed the rule's positive control SHALL be run
through the clone assay before any arm of it enters a registered panel. The assay SHALL be the one
registered with the clone-destruction diagnostic, unchanged in arms, comparator, budget, metric and
pass rule, so that the variant's result is directly comparable with the mechanisms already screened
by it.

The variant SHALL be run at the parameter value its positive control pinned. That value SHALL NOT
be re-tuned against the assay's outcome: a gate whose parameter is chosen by its own result is a
search, and the record SHALL state the value it was run at.

The result SHALL be reported with the endpoint cosine to the clone the run started from, since a
variant can hold the metric while having rewritten the policy underneath it.

#### Scenario: The assay is unchanged

- **WHEN** an eligibility variant is screened
- **THEN** the arms, comparator, budget, metric and pass rule SHALL be those registered with the
  clone-destruction diagnostic

#### Scenario: The parameter comes from the control

- **WHEN** the variant is run
- **THEN** it SHALL use the parameter value its positive control pinned
- **AND** the record SHALL state that value

#### Scenario: Failing the assay does not stop at the metric

- **GIVEN** a variant that does not pass
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the variant learns from random weights without holding a
  competent policy, and that a panel remains gated
