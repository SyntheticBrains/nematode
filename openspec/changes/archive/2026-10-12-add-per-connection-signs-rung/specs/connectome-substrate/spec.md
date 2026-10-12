## ADDED Requirements

### Requirement: The rewiring records each edge's origin

The degree-preserving rewiring SHALL be able to report, for each chemical edge of the null, the
wild-type edge whose place it took, without changing the rewired graph or the draws that produce it.
An edge the swap does not touch SHALL map to itself.

#### Scenario: Recording origins leaves the null unchanged

- **WHEN** a null is rewired at one seed with and without its origins recorded
- **THEN** the two rewired connectomes SHALL be identical

#### Scenario: An origin keeps its presynaptic neuron

- **WHEN** a null's origins are recorded
- **THEN** every null edge and its origin SHALL share their presynaptic neuron, every wild-type edge
  SHALL be the origin of exactly one null edge, and each null edge's synapse count SHALL equal its
  origin's
