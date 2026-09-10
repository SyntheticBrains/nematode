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

A variant whose mechanism perturbs the substrate SHALL additionally be reported with **a trajectory
annotation** — its plateau tail over the final quarter of the run against the first quarter — and
against **a frozen control**: the same arm with updates frozen and the perturbation applied. A
perturbing mechanism costs a competent policy something before any question of retention arises,
and without these two a fail cannot be attributed to the rule rather than to the exploration.
Neither SHALL change the verdict, which remains whatever the registered pass rule gives.

#### Scenario: The assay is unchanged

- **WHEN** an eligibility variant is screened
- **THEN** the arms, comparator, budget, metric and pass rule SHALL be those registered with the
  clone-destruction diagnostic

#### Scenario: The parameter comes from the control

- **WHEN** the variant is run
- **THEN** it SHALL use the parameter value its positive control pinned
- **AND** the record SHALL state that value

#### Scenario: A perturbing variant is read against its own frozen control

- **GIVEN** a variant whose mechanism perturbs the substrate
- **WHEN** its assay result is recorded
- **THEN** the frozen control's result and the trajectory annotation SHALL be reported beside it
- **AND** neither SHALL change the verdict the registered pass rule gives

#### Scenario: Failing the assay does not stop at the metric

- **GIVEN** a variant that does not pass
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the variant learns from random weights without holding a
  competent policy, and that a panel remains gated
