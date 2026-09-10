## ADDED Requirements

### Requirement: A plastic topology may perturb its units and expose the perturbation

A topology satisfying the plastic-topology seam MAY support **per-unit perturbation**. When it is
enabled, a trace-accumulating forward pass SHALL draw an independent perturbation
`ξ ~ N(0, σ_node²)` for each plastic post-synaptic unit from a generator dedicated to perturbation and seeded from the run seed, **add it to that unit's pre-activation so the network acts on it through its own nonlinearity**, and expose the drawn perturbations on the seam as `plastic_perturbations`,
aligned with the plastic weights as the other seam members are.

Adding the perturbation to the activity the network uses is required rather than incidental: an
eligibility built from a perturbation the unit did not act on describes a counterfactual the
network never took, and the resulting estimator is biased.

A recurrent topology that settles over several steps SHALL draw an independent perturbation at
every settling step, since a perturbation applied to the settled state alone leaves a synapse's
effect through later steps uncredited.

Perturbation state SHALL be transient: cleared with the traces at the start of every episode,
never persisted with weights, and absent from the topology's persisted state, so that a checkpoint
written before perturbation existed loads unchanged.

When perturbation is disabled the topology SHALL behave exactly as before, allocate no perturbation
state, draw nothing from any generator, and expose no perturbations, so that every existing result
stands unchanged.

#### Scenario: The unit acts on its own perturbation

- **GIVEN** perturbation enabled with a positive scale
- **WHEN** a trace-accumulating forward pass runs
- **THEN** each plastic unit's pre-activation SHALL differ from its unperturbed value by the perturbation exposed for that unit, and its activity SHALL be the nonlinearity of that perturbed pre-activation

#### Scenario: Disabled perturbation is byte-identical

- **WHEN** perturbation is disabled
- **THEN** the forward pass and the accumulated trace SHALL be bit-identical to the topology
  without this requirement

#### Scenario: Enabling perturbation leaves the rest of the random stream alone

- **GIVEN** two runs at the same seed, one with perturbation enabled and one without
- **WHEN** their action-noise draws are compared
- **THEN** they SHALL be identical, the perturbation having come from its own generator

#### Scenario: Perturbation state does not survive an episode or a checkpoint

- **WHEN** an episode begins, or weights are saved
- **THEN** the perturbations SHALL have been cleared, and SHALL be absent from the saved topology

#### Scenario: The perturbations are aligned with the weights

- **WHEN** the seam's perturbations are read
- **THEN** there SHALL be one vector per plastic tensor, indexed along the axis complementary to
  that tensor's fan-in axis
