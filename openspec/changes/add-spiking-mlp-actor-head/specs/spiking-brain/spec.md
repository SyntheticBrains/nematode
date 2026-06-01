## ADDED Requirements

### Requirement: Configurable Spiking Actor Head

The `SpikingPPOBrain` SHALL support a configurable actor head selected by an `actor_head` config field.
With `actor_head: "spike"` (the default) the action logits SHALL be produced by the non-spiking
leaky-integrator readout (the default head specified in the Spiking Brain Architecture requirement). With
`actor_head: "mlp"` the action logits SHALL instead be produced by a small actor MLP — `actor_hidden_dim`
wide and `actor_num_layers` deep — that maps the recurrent layer's hidden membrane potential to the
`num_actions` logits, giving the policy added capacity for hard multi-objective cells. In both modes the
recurrent spiking core SHALL be identical and the critic SHALL read the detached hidden membrane.

#### Scenario: Default spike head builds no actor MLP

- **WHEN** a brain config specifies `name: spikingppo` (with `actor_head` defaulting to `"spike"`)
- **THEN** the action logits SHALL be produced by the leaky-integrator readout
- **AND** no actor MLP SHALL be constructed
- **AND** the forward-pass contract SHALL be unchanged from the architecture's default behaviour

#### Scenario: MLP actor head maps the hidden membrane to logits

- **WHEN** a brain config specifies `name: spikingppo` and `actor_head: "mlp"`
- **THEN** the brain SHALL construct an actor MLP mapping the recurrent layer's hidden membrane potential
  to the `num_actions` logits
- **AND** the recurrent spiking core SHALL be identical to the spike-head mode
- **AND** the forward-pass contract — a finite `num_actions`-vector sampled by softmax + categorical —
  SHALL be identical to the spike-head mode
- **AND** the actor MLP parameters SHALL be included in the actor optimizer and persisted via
  `WeightPersistence`

#### Scenario: Actor head selection is validated

- **WHEN** a brain config sets `actor_head` to a value other than `"spike"` or `"mlp"`
- **THEN** construction SHALL raise a clear error naming the allowed values

#### Scenario: PPO gradient flows through the spiking core under the MLP head

- **GIVEN** a `SpikingPPOBrain` with `actor_head: "mlp"`
- **WHEN** a PPO update runs over a recorded rollout
- **THEN** gradient SHALL flow from the policy loss through the actor MLP into the recurrent spiking core
  (the recurrent weights and learnable decay receive finite gradient)
- **AND** the network parameters SHALL remain finite
