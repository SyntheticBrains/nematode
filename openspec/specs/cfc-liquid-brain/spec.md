# cfc-liquid-brain Specification

## Purpose

TBD - created by archiving change add-cfc-liquid-brain. Update Purpose after archive.

## Requirements

### Requirement: CfC Liquid Brain Architecture

The system SHALL provide a continuous-time recurrent brain `CfCPPOBrain` (config `name: cfcppo`, `brain_type: CFC_PPO`) whose recurrent core is a Closed-form Continuous-time (CfC) network wired with a Neural Circuit Policy `AutoNCP` wiring (a sparse sensory → interneuron → command → motor graph), trained by PPO. The actor head SHALL be selectable via an `actor_head` config field: with `actor_head: "motor"` (the default) the `AutoNCP` motor neurons SHALL serve as the action logits over the 4-action `DEFAULT_ACTIONS` set — scaled by a learnable logit temperature so the bounded motor activations do not cap policy decisiveness — with no separate actor MLP; with `actor_head: "mlp"` a small actor MLP SHALL map the recurrent hidden state to the action logits. A critic MLP SHALL estimate state value from the recurrent hidden state in both modes. Input dimensionality SHALL be derived from the active sensory-module configuration at construction time, matching the other classical PPO brains.

#### Scenario: Brain construction builds the CfC + AutoNCP topology (default motor head)

- **WHEN** a brain config specifies `name: cfcppo` (with `actor_head` defaulting to `"motor"`)
- **THEN** the brain SHALL construct a `CfC` recurrent core wired with `AutoNCP(units, num_actions)`
- **AND** the action logits SHALL be the CfC output (the `num_actions` motor neurons), with no separate actor MLP constructed
- **AND** the brain SHALL construct a critic MLP mapping the `units`-dimensional recurrent hidden state to a scalar value
- **AND** the input dimensionality SHALL be derived from the active sensory modules at construction time

#### Scenario: Brain construction with the MLP actor head

- **WHEN** a brain config specifies `name: cfcppo` and `actor_head: "mlp"`
- **THEN** the brain SHALL construct the same `CfC` + `AutoNCP` recurrent core
- **AND** the action logits SHALL be produced by an actor MLP mapping the `units`-dimensional recurrent hidden state to `num_actions`
- **AND** the forward-pass contract (a finite `num_actions`-vector sampled by softmax + categorical) SHALL be identical to the motor-head mode

#### Scenario: Motor head applies a learnable logit temperature

- **GIVEN** a `CfCPPOBrain` in `actor_head: "motor"` mode (whose `AutoNCP` motor activations are bounded)
- **WHEN** the learnable `logit_scale` parameter is raised to a large value
- **THEN** the motor head SHALL be able to produce a peaked policy (at least one action's softmax probability exceeding 0.9)
- **AND** the `logit_scale` parameter SHALL be among the actor optimizer's trained parameters

#### Scenario: Construction rejects too-small units

- **WHEN** a brain config specifies `units` not greater than `num_actions + 2`
- **THEN** construction SHALL raise a clear error stating the `AutoNCP` minimum-units requirement (`units > num_actions + 2`)

#### Scenario: Forward pass produces finite logits and a value over the action set

- **WHEN** `run_brain()` is called with any well-formed `BrainParams`
- **THEN** the returned action logits SHALL be a 4-vector matching the `DEFAULT_ACTIONS` order ([FORWARD, LEFT, RIGHT, STAY])
- **AND** the logits and the critic value SHALL be finite (no NaN, no Inf)
- **AND** the brain SHALL sample the action by applying softmax to the logits and drawing from the resulting categorical distribution

#### Scenario: Recurrent hidden state carries within an episode and resets at boundaries

- **GIVEN** the brain has processed at least one step within an episode
- **WHEN** the brain processes the next step in the same episode
- **THEN** the hidden state from the previous step SHALL be carried into the current step's recurrence
- **AND WHEN** `prepare_episode()` is called
- **THEN** the recurrent hidden state SHALL be reset to zeros

#### Scenario: Forward pass produces non-degenerate variance across an episode

- **WHEN** the brain is run for at least one episode on the smoke config with non-zero environmental gradients
- **THEN** the variance across the 4-action logits over a sample of ≥ 100 forward passes SHALL be strictly greater than zero
- **AND** the logits SHALL not collapse to a constant action across that sample

#### Scenario: PPO update runs over truncated-BPTT sequence chunks

- **GIVEN** a full rollout buffer of recorded steps
- **WHEN** the PPO update runs
- **THEN** it SHALL replay the CfC over stored sequence chunks (of length `bptt_chunk_length`) to recompute action log-probabilities and values with fresh gradient
- **AND** apply the clipped PPO objective plus an entropy bonus and a value loss, with gradient-norm clipping at `max_grad_norm`, over `num_epochs`
- **AND** complete without error and leave the network parameters finite

### Requirement: WeightPersistence Protocol Conformance

The `CfCPPOBrain` SHALL implement the `WeightPersistence` protocol from [`brain/weights.py`](../../../../packages/quantum-nematode/quantumnematode/brain/weights.py) so its weights can be serialised and restored (checkpoint round-trip). The per-episode recurrent hidden state SHALL NOT be part of the persisted weights (it is reset at `prepare_episode()`).

#### Scenario: Brain exposes its weights as named WeightComponents

- **GIVEN** an instance of `CfCPPOBrain`
- **WHEN** `brain.get_weight_components()` is called
- **THEN** the returned dict SHALL contain entries for the learnable components, including at minimum `"cfc"` (the recurrent core), `"critic"`, and `"feature_norm"`
- **AND** each entry SHALL be a `WeightComponent` carrying a `state` dict of named `torch.Tensor` parameters

#### Scenario: Brain restores weights from a WeightComponents dict

- **GIVEN** two `CfCPPOBrain` instances with identical topology
- **WHEN** the second loads the first's `get_weight_components()` output via `load_weight_components()`
- **THEN** the two brains SHALL produce identical action logits for the same input and hidden state (within floating-point tolerance)

### Requirement: Plugin Registry Registration

The `CfCPPOBrain` SHALL register through the `@register_brain` plugin registry with a unique `BrainType` (`CFC_PPO`) and SHALL load through the standard simulation launcher without any per-architecture conditional branch.

#### Scenario: Brain loads through the plugin registry

- **GIVEN** the `quantumnematode.brain.arch` package is imported
- **WHEN** a simulation config names brain `cfcppo`
- **THEN** the launcher SHALL instantiate `CfCPPOBrain` via the registry lookup
- **AND** no per-architecture conditional SHALL be required in the launcher or brain factory
