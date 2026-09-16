## ADDED Requirements

### Requirement: A capacity manipulation crossed with a structure contrast is read as an interaction

Where an experiment changes a learner's capacity — the parameter count of a readout, a layer width,
the number of adapted tensors — in order to ask whether a structure effect was hidden by that
capacity, the record SHALL read the **interaction** between capacity and structure as the primary,
and SHALL NOT read a capacity main effect as evidence about the structure. A larger parameter set
learning faster is a fact about the parameter set.

#### Scenario: The capacity change is crossed rather than compared across campaigns

- **GIVEN** a structure contrast that returned null at one capacity
- **WHEN** the question is whether that capacity hid the structure effect
- **THEN** the design SHALL run both capacities against **both** levels of the structure contrast
- **AND** the primary contrast SHALL be the interaction, stated as such before the campaign runs
- **AND** a capacity main effect SHALL be reported beside the interaction and never in place of it

#### Scenario: The capacities are matched at initialisation

- **GIVEN** two capacities of the same learner compared on the same seeds
- **WHEN** the arms are constructed
- **THEN** the record SHALL state what differs between them and what does not, covering the random
  draws consumed, the initial policy, and any control arm claimed to be shared
- **AND** where an arm is claimed to be unaffected by the capacity change, that invariance SHALL be
  asserted by test rather than assumed, and a failure SHALL stop the campaign

#### Scenario: An analytic equivalence is not treated as run-level identity

- **GIVEN** two configurations shown to compute the same function in exact arithmetic
- **WHEN** that equivalence is used to justify running one of them in place of the other
- **THEN** the record SHALL establish that the two agree at the precision the runs use, not only
  analytically
- **AND** where they differ at that precision in a system whose trajectory depends on the difference,
  the configurations SHALL be run separately and the equivalence SHALL be reported as a statement
  about the computed function rather than about the runs

#### Scenario: A null interaction states the panel's sensitivity

- **GIVEN** an interaction contrast, whose per-seed variance exceeds that of either single contrast
- **WHEN** the interaction is not significant
- **THEN** the record SHALL report it as no interaction detected **at the panel's sensitivity**, with
  that sensitivity computed from observed per-seed spread and registered before the campaign ran
- **AND** it SHALL NOT be reported as excluding an interaction of unspecified size

### Requirement: A metric is chosen for the contrast it must support, and a departure is registered with its reason

Where a campaign departs from the metric a committed instrument used for a comparable contrast, the
record SHALL state the departure and its reason **before the campaign runs**, and SHALL report the
committed instrument's metric beside the chosen one.

#### Scenario: A censored metric is not used for a difference of differences

- **GIVEN** a metric that is right-censored at a horizon
- **AND** a contrast formed as a difference between two differences across cells of a design
- **WHEN** the censoring rate is not known to be equal across those cells
- **THEN** that metric SHALL NOT be the primary for that contrast
- **AND** where it is reported, its censoring SHALL be counted **per cell** rather than pooled

#### Scenario: The departed-from metric is reported beside the chosen one

- **GIVEN** a campaign that changed its primary metric from the one a committed instrument used
- **WHEN** the result is recorded
- **THEN** both metrics SHALL be reported
- **AND** where they disagree, the record SHALL state the disagreement rather than reporting only the
  primary
