# spiking-brain Specification

## Purpose

This specification defines the **spiking neural-network brain** capability — a recurrent adaptive leaky-integrate-and-fire (LIF) brain (`SpikingPPOBrain`, registry name `spikingppo`) whose core carries spiking membrane state across env-steps (learnable per-neuron decay, adaptive firing threshold, a recurrent spike-feedback current, soft reset, sigmoid-derivative surrogate gradient). A learnable direct-current encoder feeds the core, a non-spiking leaky-integrator reads out the action logits (the spiking actor), a plain-ANN critic estimates value from the detached membrane state, and training is PPO over truncated-BPTT sequence chunks. It provides the neuromorphic / event-driven row of the cross-architecture comparison, alongside the wild-type-connectome and continuous-time (CfC) brains; it is built by extending the in-repo spiking substrate with no SNN-library dependency. The capability was introduced by the archived change `add-spiking-ppo-brain`; this document is its canonical record of required behaviour — the architecture, weight persistence, and plugin-registry registration. Audience: developers extending the brain-architecture plugin layer and authors of the cross-architecture comparison.

## Requirements

### Requirement: Spiking Brain Architecture

The system SHALL provide a recurrent spiking brain `SpikingPPOBrain` (config `name: spikingppo`,
`brain_type: SPIKING_PPO`) whose core is a **recurrent adaptive leaky-integrate-and-fire (LIF)** layer —
a learnable per-neuron membrane decay, an adaptive firing threshold, a recurrent spike-feedback current,
and a sigmoid-family surrogate gradient — that carries its membrane state across env-steps (one LIF update
per step). A learnable direct-current encoder SHALL feed the core; a **non-spiking leaky-integrator**
readout SHALL produce the action logits over the 4-action `DEFAULT_ACTIONS` set (the spiking actor); and
a plain-ANN critic SHALL estimate state value from the detached membrane state. The brain SHALL be
trained by PPO over truncated-BPTT sequence chunks. Input dimensionality SHALL be derived from the active
sensory-module configuration at construction time, matching the other classical PPO brains.

#### Scenario: Brain construction builds the recurrent spiking core

- **WHEN** a brain config specifies `name: spikingppo`
- **THEN** the brain SHALL construct a learnable direct-current input encoder, a recurrent adaptive-LIF
  hidden layer (with learnable decay, adaptive threshold, and a recurrent spike-feedback connection), a
  non-spiking leaky-integrator readout to the action logits, and a critic MLP over the hidden membrane
- **AND** the input dimensionality SHALL be derived from the active sensory modules at construction time
- **AND** no SNN-library dependency SHALL be required (the spiking primitives are in-repo)

#### Scenario: Forward pass produces finite logits and a value over the action set

- **WHEN** `run_brain()` is called with any well-formed `BrainParams`
- **THEN** the returned action logits SHALL be a 4-vector matching the `DEFAULT_ACTIONS` order
  ([FORWARD, LEFT, RIGHT, STAY])
- **AND** the logits and the critic value SHALL be finite (no NaN, no Inf)
- **AND** the brain SHALL sample the action by applying softmax to the logits and drawing from the
  resulting categorical distribution

#### Scenario: Recurrent membrane state carries within an episode and resets at boundaries

- **GIVEN** the brain has processed at least one step within an episode
- **WHEN** the brain processes the next step in the same episode
- **THEN** the neuron state (hidden membrane, adaptation, last spikes, and readout membrane) from the
  previous step SHALL be carried into the current step's recurrence
- **AND** because the core is recurrent, identical sensory input at two different points in an episode
  MAY yield different logits (the carried state differs) — distinguishing it from a memoryless
  feedforward LIF
- **AND WHEN** `prepare_episode()` is called
- **THEN** the carried neuron state SHALL be reset to zeros

#### Scenario: Forward pass produces non-degenerate variance across an episode

- **WHEN** the brain is run for at least one episode on the smoke config with non-zero environmental
  gradients
- **THEN** the variance across the 4-action logits over a sample of ≥ 100 forward passes SHALL be
  strictly greater than zero
- **AND** the logits SHALL not collapse to a constant action across that sample

#### Scenario: PPO update runs over truncated-BPTT sequence chunks with surrogate gradients

- **GIVEN** a full rollout buffer of recorded steps
- **WHEN** the PPO update runs
- **THEN** it SHALL replay the spiking core over stored sequence chunks (of length `bptt_chunk_length`),
  recomputing action log-probabilities and values with fresh gradient flowing through the spike
  surrogate
- **AND** apply the clipped PPO objective plus an entropy bonus and a value loss, with gradient-norm
  clipping at `max_grad_norm`, over `num_epochs`
- **AND** complete without error and leave the network parameters finite

#### Scenario: Surrogate-slope schedule fields are validated as a pair

- **WHEN** a brain config sets exactly one of `surrogate_slope_end` / `surrogate_slope_anneal_episodes`
  (one set, the other unset)
- **THEN** construction SHALL raise a clear error stating the two fields must be set together (both to
  schedule the slope, or neither for a flat slope)
- **AND WHEN** neither is set
- **THEN** the surrogate slope SHALL be flat at `surrogate_slope` for the whole run

### Requirement: WeightPersistence Protocol Conformance

The `SpikingPPOBrain` SHALL implement the `WeightPersistence` protocol from
[`brain/weights.py`](../../../../packages/quantum-nematode/quantumnematode/brain/weights.py) so its
weights can be serialised and restored (checkpoint round-trip). The per-step recurrent neuron state
(hidden membrane, adaptation, last spikes, readout membrane) SHALL NOT be part of the persisted weights
(it is reset at `prepare_episode()`).

#### Scenario: Brain exposes its weights as named WeightComponents

- **GIVEN** an instance of `SpikingPPOBrain`
- **WHEN** `brain.get_weight_components()` is called
- **THEN** the returned dict SHALL contain entries for the learnable components, including at minimum the
  input encoder, the recurrent spiking core (recurrent weights + learnable decay/adaptation parameters),
  the readout, and the critic
- **AND** each entry SHALL be a `WeightComponent` carrying a `state` dict of named `torch.Tensor`
  parameters

#### Scenario: Brain restores weights from a WeightComponents dict

- **GIVEN** two `SpikingPPOBrain` instances with identical topology
- **WHEN** the second loads the first's `get_weight_components()` output via `load_weight_components()`
- **THEN** the two brains SHALL produce identical action logits for the same input and carried neuron
  state (within floating-point tolerance)

### Requirement: Plugin Registry Registration

The `SpikingPPOBrain` SHALL register through the `@register_brain` plugin registry with a unique
`BrainType` (`SPIKING_PPO`) and SHALL load through the standard simulation launcher without any
per-architecture conditional branch.

#### Scenario: Brain loads through the plugin registry

- **GIVEN** the `quantumnematode.brain.arch` package is imported
- **WHEN** a simulation config names brain `spikingppo`
- **THEN** the launcher SHALL instantiate `SpikingPPOBrain` via the registry lookup
- **AND** no per-architecture conditional SHALL be required in the launcher or brain factory

### Requirement: Configurable Spiking Actor Head

The `SpikingPPOBrain` SHALL support a configurable actor head selected by an `actor_head` config field.
This requirement refines the Spiking Brain Architecture requirement's readout clause: the non-spiking
leaky-integrator readout is the **default** head (`actor_head: "spike"`), not the only one. With
`actor_head: "spike"` (the default) the action logits SHALL be produced by that leaky-integrator readout.
With `actor_head: "mlp"` the action logits SHALL instead be produced by a small actor MLP —
`actor_hidden_dim` wide and `actor_num_layers` deep — that maps the recurrent layer's hidden membrane
potential to the `num_actions` logits, giving the policy added capacity for hard multi-objective cells. In
both modes the recurrent spiking core SHALL be identical and the critic SHALL read the detached hidden
membrane. Weight round-trip assumes the same `actor_head` mode; `load_weight_components` SHALL tolerate an
absent `actor_mlp` component (e.g. restoring into spike mode) without error.

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
- **THEN** gradient SHALL flow from the policy loss through the actor MLP into the recurrent spiking core —
  the learnable membrane decay SHALL receive a finite, non-zero gradient (it flows through the smooth
  surrogate independently of spike density), and the recurrent weights SHALL receive finite gradient
- **AND** the network parameters SHALL remain finite
