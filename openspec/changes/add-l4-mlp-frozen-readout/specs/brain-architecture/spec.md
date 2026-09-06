## ADDED Requirements

### Requirement: Selectable plastic depth on the MLP brain

The MLP brain configuration SHALL accept `plastic_layers`, `all` (default) or `hidden`. Under
`all` every `Linear` weight of the actor SHALL be plastic, as today. Under `hidden` every
`Linear` weight but the output layer's SHALL be plastic and the output layer SHALL NOT appear on
the plastic-topology seam: no eligibility trace, no mask, no fan-in axis, no homeostatic target,
and no plastic update. The topology's forward SHALL still run the whole actor, bitwise-equal to
the actor's own forward, and SHALL credit eligibility to the plastic layers only. The option
SHALL have no effect on the gradient rule, which trains every actor parameter through the
optimiser. With the default the build SHALL be bit-identical to the brain without this
requirement.

#### Scenario: Default is bit-identical

- **WHEN** an MLP brain is built with the default `plastic_layers`
- **THEN** its seam lists, trace buffers and forward SHALL be exactly today's

#### Scenario: A hidden-only substrate keeps its readout fixed

- **GIVEN** an MLP brain with `plastic_layers: hidden` under a plastic rule
- **WHEN** the rule steps with non-trivial traces and modulators
- **THEN** the output layer's weight and bias SHALL be bit-identical to their initial values
- **AND** at least one hidden weight SHALL have changed
- **AND** the seam SHALL expose one plastic tensor per hidden layer and none for the output layer

#### Scenario: The forward is unchanged by the option

- **WHEN** the topology's forward runs with traces enabled under either setting
- **THEN** its output SHALL be bitwise-equal to `actor(features)`
- **AND** under `hidden` no trace SHALL exist for the output layer while every hidden layer's
  trace accumulates

#### Scenario: The gradient rule ignores the option

- **WHEN** an MLP brain with `plastic_layers: hidden` trains under the gradient rule
- **THEN** every actor parameter SHALL remain trainable and the update SHALL be bit-identical to
  the frozen reference
