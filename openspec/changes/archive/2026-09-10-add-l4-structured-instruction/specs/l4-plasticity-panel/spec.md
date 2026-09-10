## ADDED Requirements

### Requirement: The structured-instruction test asks whether routing the third factor matters

The routed third factor SHALL be tested against the global scalar on both wirings: four arms —
wild-type and rewired-null under each routing mode — on paired seeds 1–16, the panel's 3000-episode
plastic budget, the committed plateau-tail full-clear metric, and the panel's registered extension
of a fresh run at 1.5× replacing a shorter log the plateau detector marks non-converged.

The global-scalar arms SHALL be **re-run concurrently** rather than read from the first panel's
committed table, whose plastic arm carries eight seeds at this budget; the committed values SHALL
be reported beside the new global arm as a consistency check and SHALL NOT be used as a comparator.

The confirmatory family SHALL be exactly four one-sided paired tests corrected together under
BH-FDR: **S1** the wild-type pathway arm over the wild-type global arm, **S2** the rewired-null
pathway arm over the rewired-null global arm, **S3** wild-type over rewired null under pathway
routing, and **S4** the same under the global scalar.

The verdict SHALL be assigned in order as `insufficient_seeds`; then **`no_routing_effect`** when
neither S1 nor S2 confirms; then `routing_helps_both`, `routing_helps_wild_type_only` and
`routing_helps_rewired_only`. S3 and S4 SHALL annotate the verdict and never change it. A launch
record SHALL be committed before any run.

The record SHALL report **each arm's** instructed fraction — the rewired null derives its own
pathway from its own edges — and each arm's mean instructed share of the update, and SHALL state
that the pathway is a model of aminergic reach by synaptic connectivity — a lower bound, since
aminergic transmission in this animal is substantially extrasynaptic — so that a negative is
recorded as refuting this proxy rather than structured instruction itself.

#### Scenario: Routing changing nothing is a nameable outcome

- **GIVEN** neither S1 nor S2 confirming
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `no_routing_effect`
- **AND** the record SHALL state that the global scalar was not the limitation, and that the
  pathway tested was the synaptic proxy rather than expressed-receptor reach

#### Scenario: Helping the scramble as much as the animal is not a win

- **GIVEN** both S1 and S2 confirming
- **WHEN** the result is written up
- **THEN** the record SHALL state that a routed third factor helping the rewired null as much as
  the wild type is a fact about running two learning regimes in one network and not about this
  animal's connectivity

#### Scenario: The proxy's coverage is reported with the result

- **WHEN** the test is written up
- **THEN** each arm's instructed fraction SHALL be reported
- **AND** each arm's mean instructed share of the update SHALL be reported

#### Scenario: The panel stays gated

- **WHEN** routing confirms
- **THEN** the 2×2 panel SHALL still require a passing clone assay before it is re-run
