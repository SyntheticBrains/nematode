## ADDED Requirements

### Requirement: A gap-only rewired null

The brain SHALL offer `wiring: rewired_gap_junctions_only`: the gap junctions rewired by the
degree-preserving undirected swap with their counts, and the chemical graph, autapses included, held at
the input's exactly. Every other wiring value SHALL be unchanged.

#### Scenario: The chemical graph is the wild type's

- **WHEN** the gap-only null is drawn at any seed
- **THEN** its chemical synapses SHALL equal the wild type's, edge for edge and count for count
- **AND** every neuron's gap degree SHALL equal the wild type's while the gap pairs differ

#### Scenario: Existing nulls are unchanged

- **WHEN** the default or any existing rewired wiring is drawn at a pinned seed
- **THEN** its graph SHALL equal the graph drawn before this option existed

### Requirement: Plastic gap junctions under PPO

The brain SHALL offer `plastic_gaps`, default false. When true, every existing gap pair SHALL carry a
learnable strength multiplier, positive and symmetric by construction and starting at 1, that PPO
updates with the rest of the brain; no gap pair SHALL be created. When false every output, optimiser
state and random draw SHALL be identical to the brain before this option existed. `plastic_gaps` SHALL
be refused under leaky dynamics and under any learning rule other than PPO.

#### Scenario: The coupling stays symmetric, positive and on existing pairs

- **WHEN** a plastic-gap brain's multipliers take any values
- **THEN** the coupling its forward pass uses SHALL be symmetric, positive on every existing gap pair,
  and zero on every other pair

#### Scenario: The multipliers receive a gradient

- **WHEN** a plastic-gap brain takes a PPO update
- **THEN** its multipliers SHALL receive a non-zero gradient and change

#### Scenario: Off is byte-identical

- **WHEN** `plastic_gaps` is false
- **THEN** the learnable parameters, their order and every output SHALL equal the brain's before this
  option existed
