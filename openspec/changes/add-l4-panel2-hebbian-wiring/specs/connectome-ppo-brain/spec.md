## ADDED Requirements

### Requirement: Synapse-count-scaled chemical weight initialisation

The connectome brain configuration SHALL accept `weight_init`, `degree_scaled` (default) or
`count_scaled`. Under `degree_scaled` the chemical weights SHALL be initialised exactly as before:
each edge into a post-synaptic neuron with `k` chemical inputs drawn from `N(0, 1/√k)`. Under
`count_scaled` each edge's weight SHALL be `z · n / sqrt(Σ n²)`, where `n` is the edge's synapse
count from the connectome data, the sum runs over the post-synaptic neuron's incoming chemical
edges, and `z ~ N(0, 1)` is drawn in the same order and from the same generator as the
degree-scaled draw, so that every neuron with inputs has expected squared incoming norm 1 under
both settings and only the relative magnitudes within a neuron's inputs differ. Signs SHALL remain
random draws under both settings. Under the degree-preserving rewiring each edge's count SHALL
travel with its pre-synaptic endpoint and the normalisation SHALL be recomputed on the rewired
edge set. With the default, construction SHALL be bit-identical to the brain without this
requirement.

#### Scenario: Default initialisation is bit-identical

- **WHEN** a connectome brain is built with the default `weight_init`
- **THEN** its chemical weights SHALL be bit-identical to today's at the same seed

#### Scenario: Count-scaled magnitudes follow the counts within a neuron

- **GIVEN** a connectome brain built with `weight_init: count_scaled`
- **WHEN** the incoming weights of a neuron with several inputs are inspected
- **THEN** their magnitudes divided by the standard-normal draws SHALL be proportional to the edges'
  synapse counts
- **AND** the mean over neurons with inputs of the squared incoming norm SHALL be 1 within tolerance

#### Scenario: The rewired arm carries counts with the moved edges

- **GIVEN** the wild-type and rewired-null brains built at one seed with `weight_init: count_scaled`
- **WHEN** their chemical weights are compared
- **THEN** the multiset of counts incident on each neuron MAY differ while every in/out degree is
  preserved
- **AND** the motor readout, every sensory-projection gain and `log_std` SHALL be bit-identical
  between the two brains
