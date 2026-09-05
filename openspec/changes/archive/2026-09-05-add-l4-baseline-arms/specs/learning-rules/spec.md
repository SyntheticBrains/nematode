# Spec: learning-rules

## ADDED Requirements

### Requirement: Unmodulated Hebbian mode

The plasticity rule SHALL support an **unmodulated** mode in which the neuromodulatory third factor is not applied: the weight change is the plasticity rate times the eligibility trace, with the stabilisation terms unchanged.

This mode exists as the ablation that isolates the rule's central claim. An arm that beats an untrained network has learned something; an arm that beats unmodulated Hebbian has learned something **from reward**. Without the comparison, an advantage attributable to correlation structure alone is indistinguishable from one attributable to reward-driven learning.

The mode SHALL differ from the modulated rule in the modulator alone. Eligibility accumulation, masking, decay, clamping, and reporting SHALL be identical, so that a difference between the two arms is attributable to the third factor and not to an incidentally different code path.

The reward prediction error and its baseline SHALL still be computed and reported in this mode, even though the update does not apply them. Both arms then record what the reward stream was doing, and only one records having used it, making the ablation visible in telemetry rather than inferable only from configuration.

#### Scenario: The update omits the modulator

- **WHEN** the rule steps in unmodulated mode with a known trace
- **THEN** the weight change SHALL equal the plasticity rate times the trace, before stabilisation terms
- **AND** it SHALL NOT depend on the reward

#### Scenario: Reward changes nothing in this mode

- **GIVEN** two rules in unmodulated mode with identical traces and identical initial weights
- **WHEN** one is stepped with a large reward and the other with a small one
- **THEN** their resulting weights SHALL be identical

#### Scenario: The reward stream is still observed

- **WHEN** the rule steps in unmodulated mode
- **THEN** its report SHALL carry the prediction error and baseline the reward stream implies
- **AND** those values SHALL match what the modulated rule would have reported for the same rewards

#### Scenario: Only the modulator differs from the modulated rule

- **GIVEN** a modulated and an unmodulated rule over identical topologies with identical traces
- **WHEN** each is stepped once with a reward whose prediction error is exactly one
- **THEN** their weight changes SHALL be identical

#### Scenario: Stabilisation still applies

- **GIVEN** an unmodulated rule driven until weights reach the magnitude bound
- **WHEN** it steps again
- **THEN** no weight magnitude SHALL exceed the bound
- **AND** the reported weight change SHALL be the effective change, which is zero once saturated
