## ADDED Requirements

### Requirement: Registered before the recovery runs

A launch record SHALL fix the model's parameters and priors, the summary, the simulation and seed
scheme, the NPE configuration and the gate before any training simulation runs. It SHALL include the
pilot's measured cost.

#### Scenario: The cost is measured first

- **WHEN** the launch record is committed
- **THEN** it SHALL state the pilot's seconds per simulation and the projected wall time

### Requirement: NPE recovers C.1's parameters with honest calibration

NPE SHALL be trained on 10,000 simulations drawn from the prior, twice with different seeds. On 100
held-out truths from the prior, each flow's pooled central-90% interval coverage over truths and
parameters SHALL lie in [0.80, 0.97]. Each parameter's coverage and contraction, and a
posterior-predictive check in statistic space, SHALL be reported beside the gate.

#### Scenario: An overconfident posterior fails

- **WHEN** a flow's pooled coverage is below 0.80
- **THEN** F.0a SHALL fail, and NPE SHALL not be used for C.1 until a new change fixes it

#### Scenario: An unconstrained parameter is named

- **WHEN** a parameter's contraction is near zero
- **THEN** the record SHALL report it as not constrained by the benchmark's statistics

### Requirement: Undefined statistics enter the summary explicitly

An undefined statistic SHALL be replaced by a fixed out-of-range sentinel, with a 0/1 flag beside it,
so the summary is finite and an undefined value carries information.

#### Scenario: A worm that never leaves

- **WHEN** a simulation's pre-exit enrichment is undefined
- **THEN** its summary SHALL hold the sentinel and a flag of 1
