## ADDED Requirements

### Requirement: An eligibility derived from a substrate's own dynamics states what its approximation drops

Where a learning rule builds its eligibility from the substrate's forward dynamics rather than from
injected perturbation, the record SHALL state which terms of the true derivative the trace keeps and
which it drops, and SHALL state what structure of the substrate the dropped terms correspond to. Where
the dropped terms are systematically larger for some units than others, the record SHALL name the
asymmetry before the result is read.

#### Scenario: The truncation is stated with the mechanism

- **GIVEN** an eligibility that keeps the direct path from a weight to its post-synaptic unit
- **WHEN** the mechanism is recorded
- **THEN** the record SHALL state that the paths through other units are dropped
- **AND** it SHALL state what that costs on this substrate rather than describing the trace as the
  derivative

#### Scenario: The asymmetry the truncation implies is registered as a prediction

- **GIVEN** a substrate whose scored outcome depends on a subset of its units
- **WHEN** an eligibility that drops multi-hop paths is used on it
- **THEN** the record SHALL state, before the result is read, where the weight change is expected to
  concentrate
- **AND** the result SHALL report where it actually landed

### Requirement: A rule whose update needs a per-unit signal registers how that signal reaches each unit

Where a rule's eligibility is unsigned with respect to the outcome, the per-unit signal that supplies
the sign SHALL be recorded as a declared mechanism rather than assumed, and the comparison SHALL
include an arm in which that signal is removed. A result SHALL NOT attribute an effect to the
eligibility while the per-unit signal is uncontrolled.

#### Scenario: The signal-free ablation is run and reported

- **GIVEN** a rule whose update is the product of an unsigned eligibility and a per-unit signal
- **WHEN** the arms are registered
- **THEN** an arm with the per-unit signal removed SHALL be among them
- **AND** its position SHALL be reported in every branch of the verdict, including branches where the
  rule shows no effect

#### Scenario: An ablation matching the full rule changes what the result is about

- **GIVEN** an ablation arm that performs as well as the arm carrying the per-unit signal
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the effect is attributable to the eligibility alone
- **AND** it SHALL NOT be reported as a result about the per-unit signal

#### Scenario: A broadcast signal is distinguished from a transported one

- **GIVEN** a per-unit signal formed by projecting one shared error vector
- **WHEN** the plausibility of the rule is described
- **THEN** the record SHALL state whether the projection is fixed and arbitrary or taken from the
  network's own forward weights
- **AND** an arm using the forward weights SHALL be reported as a yardstick rather than as the
  plausible learner

#### Scenario: A signal whose reach the mechanism constrains carries a reach-matched control

- **GIVEN** a routing of the per-unit signal that can reach only a subset of the substrate's units
- **WHEN** it is compared against a routing that reaches all of them
- **THEN** a control matched on which units the signal reaches SHALL be among the arms
- **AND** a difference between the two routings SHALL NOT be attributed to the signal's source while the
  units it reaches also differ

#### Scenario: A cell the mechanism forbids is recorded as forbidden

- **GIVEN** a design crossing the signal's source with the units it reaches
- **WHEN** one combination cannot be realised by the mechanism
- **THEN** the record SHALL state that it is forbidden rather than omitted
- **AND** it SHALL state what property of the mechanism forbids it

#### Scenario: A fixed projection is persisted with the policy

- **GIVEN** a learning signal routed through a projection drawn once per run
- **WHEN** the policy is checkpointed
- **THEN** the projection SHALL be persisted with it
- **AND** a reloaded policy SHALL learn against the projection it was trained with

### Requirement: A mechanism whose positive control has a closed-form optimum states which arm must fail

Where a new learning mechanism is validated on a control task with a known optimum, the registration
SHALL name both an arm that must pass and an arm that must fail, and SHALL treat either expectation
being violated as voiding the control rather than as a result.

#### Scenario: Both directions of the control are registered

- **GIVEN** a control task whose optimum and floor are closed-form
- **WHEN** the arms are registered
- **THEN** one arm SHALL be required to pass and one SHALL be required to fail
- **AND** a violation of either SHALL void the control

#### Scenario: A voided control stops the campaign it gates

- **GIVEN** a control that gates a campaign on another substrate
- **WHEN** the gated campaign is about to launch
- **THEN** both the arm required to pass and the arm required to fail SHALL have been evaluated and
  their results recorded
- **AND** where either expectation is not met the control SHALL be recorded as void and no run of the
  gated campaign SHALL be launched
- **AND** the record SHALL state which expectation was violated
