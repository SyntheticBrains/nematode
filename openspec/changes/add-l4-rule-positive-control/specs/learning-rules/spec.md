## ADDED Requirements

### Requirement: The three-factor rule has a positive control

The project SHALL provide a positive control for the reward-modulated three-factor rule: a task on
which reward-modulated Hebbian learning is expected to work, with an optimum and a cue-blind floor
both computable in closed form, driving the committed rule over the committed plastic-topology
seam with synthetic observations and rewards. The control SHALL NOT use an environment, a runner,
a reward-shaping configuration or a plateau metric, so that a null result cannot be attributed to
any of them.

The task SHALL present a cue drawn uniformly from a fixed set, map it to a continuous action
through the seam's topology with exploration noise at the action, and reward the negative squared
distance between the action and that cue's fixed target. The targets SHALL NOT appear in the
observation, so that the association is discoverable only from reward.

#### Scenario: The control isolates the rule

- **WHEN** the control runs
- **THEN** it SHALL drive the same `ThreeFactorRule` implementation the panels use, over the same
  plastic-topology seam
- **AND** it SHALL NOT construct an environment, an episode runner or a connectome

#### Scenario: The floor and the optimum are computed, not measured

- **WHEN** the control reports a result
- **THEN** the cue-blind floor SHALL be the negative variance of the targets and the optimum the
  negative exploration variance, both derived from the task's own parameters

#### Scenario: The answer is reachable only through reward

- **WHEN** the observation is inspected
- **THEN** it SHALL carry the cue and SHALL NOT carry that cue's target

### Requirement: The control is bounded by a reference arm and a floor arm

The control SHALL run three arms: the modulated three-factor rule under test, the **unmodulated**
rule as a floor, and an **analytic reference** that descends the task's exact gradient through the
same topology. The reference arm establishes that the task is learnable in this setup and that the
topology can express the answer; the unmodulated arm establishes that the task's answer is not
available without reward.

The control SHALL be reported as **void**, drawing no conclusion about the three-factor rule, when
the reference arm does not pass the same bar or the unmodulated arm does pass it. A void control
SHALL NOT be reported as a negative result about the rule.

#### Scenario: A control whose reference fails concludes nothing

- **GIVEN** an analytic reference arm that does not clear the pass bar
- **WHEN** the result is assigned
- **THEN** it SHALL be `void`
- **AND** the record SHALL state that the task, topology or optimiser is at fault and that the
  three-factor arm's result carries no information

#### Scenario: A control the floor solves concludes nothing

- **GIVEN** an unmodulated arm that clears the pass bar
- **WHEN** the result is assigned
- **THEN** it SHALL be `void`
- **AND** the record SHALL state that the task leaks its answer without reward

### Requirement: The control reports the alignment between its update and the gradient

The control SHALL report, per run, the effective modulator, the eligibility magnitude, and the
**cosine between the weight update the rule applies and the analytic gradient of the same step**,
and SHALL retain them whatever the outcome. The alignment distinguishes a rule that learns slowly
from one whose updates are unrelated to reward, which a pass/fail bar alone cannot.

#### Scenario: A failure carries its diagnosis

- **GIVEN** a control the three-factor arm does not pass
- **WHEN** the result is recorded
- **THEN** the modulator, the eligibility magnitude and the gradient alignment SHALL be reported
  beside it
