# Spec: brain-architecture

## MODIFIED Requirements

### Requirement: Brain Topology Protocol

The system SHALL provide a `BrainTopology` Protocol that exposes the structural seam a learning rule needs — the weight-mask projector and the learnable parameters — factored out from learning-rule concerns (optimisers, replay buffers, value heads). Forward-pass signatures are topology-specific and SHALL NOT be part of the Protocol: a rule receives replayed activations and features via its `batch`, not by invoking the topology's forward itself. `ConnectomeTopology` SHALL satisfy the Protocol at runtime (`isinstance` under `runtime_checkable`). *(Contract reconciled 2026-08-28: the previous `n_inputs`/`n_outputs`/`n_hidden` + `forward(x)` mandate had zero conforming implementations and zero consumers.)*

#### Scenario: Topology exposes shape attributes

- **WHEN** a `BrainTopology` implementation is inspected
- **THEN** topology-specific shape metadata MAY be exposed (e.g. `ConnectomeTopology.n_neurons` / `n_food_features`)
- **AND** the Protocol SHALL NOT mandate `n_inputs` / `n_outputs` / `n_hidden` — no implementation ever satisfied them and the rule seam does not need them

#### Scenario: Topology computes a forward pass

- **WHEN** a topology's forward pass is invoked by its owning brain
- **THEN** the call SHALL be free of optimiser, replay-buffer, or value-head state changes
- **AND** the forward signature SHALL NOT be part of the Protocol — it is topology-specific, and a rule never calls it directly

#### Scenario: Topology exposes its learnable parameters

- **WHEN** a `BrainTopology` implementation is inspected
- **THEN** the instance SHALL expose `learnable_parameters` returning the parameters a learning rule may update
- **AND** the returned set SHALL reflect the topology's enabled optional blocks (e.g. predator / thermotaxis projections, continuous `log_std`)

#### Scenario: Topology applies weight mask

- **WHEN** `topology.apply_weight_mask(weights)` is called with a candidate weight tensor
- **THEN** the call SHALL return the weight tensor projected onto the topology's allowed connectivity manifold
- **AND** for dense topologies the default implementation SHALL be the identity function
- **AND** for sparse / strict-mask topologies (e.g. connectome-constrained) the call SHALL zero out weights along non-existent edges

#### Scenario: Focal implementation conforms at runtime

- **WHEN** `isinstance(connectome_brain.topology, BrainTopology)` is evaluated
- **THEN** it SHALL return `True`
- **AND** a conformance test SHALL pin this (the check the L1 change promised but did not ship)

### Requirement: Learning Rule Protocol

The system SHALL provide a `LearningRule` Protocol that encapsulates how a topology's weights are updated from experience, factored out from topology concerns — and the Protocol SHALL have at least one genuine code consumer: the connectome brain's PPO update SHALL route through a `LearningRule` implementation (`ConnectomePPORule`, in the `learning_rules` package).

#### Scenario: Rule owns the optimiser state

- **WHEN** a `LearningRule` implementation is constructed
- **THEN** the rule instance SHALL own its optimiser, value head (if any), update hyperparameters, advantage estimator (if any), and gradient clipper (if any)
- **AND** these objects SHALL NOT be exposed on the `BrainTopology` interface
- **AND** an experience-collection buffer owned by the brain MAY be surfaced to the rule through `batch` rather than owned by the rule *(ownership softened 2026-08-28)*

#### Scenario: Rule advances the topology weights

- **WHEN** `rule.step(topology, batch)` is called with an experience batch
- **THEN** the rule SHALL compute updates to `topology` parameters and apply its optimiser
- **AND** SHALL apply `topology.apply_weight_mask(...)` to any updated weights that are subject to a topology mask
- **AND** SHALL return a `RuleStepReport` summarising the update (loss components, gradient norms)

#### Scenario: Rule resets per-episode state

- **WHEN** `rule.reset_episode()` is called at the start of a new episode
- **THEN** the rule SHALL clear any per-episode state (advantage-estimator buffers, recurrent-state caches owned by the rule)
- **AND** SHALL NOT clear the optimiser state or any persistent replay buffer
