## ADDED Requirements

### Requirement: A boundary-preserving rewired null

The brain SHALL offer `wiring: rewired_boundary_held`: the chemical-only rewired null with every
chemical edge leaving a neuron any sensory projection injects into, or entering a neuron the motor
readout pools, held at the wild type's with its count, and only the remaining interior edges rewired by
the degree-preserving swap. Gap junctions and autapses SHALL be held as in the chemical-only null. Every
other wiring value SHALL be unchanged.

#### Scenario: The boundary is held exactly

- **WHEN** the boundary-preserving null is drawn at any seed
- **THEN** every chemical edge out of a boundary sensory neuron or into a boundary motor neuron SHALL be
  present with the wild type's count, and no other edge SHALL touch the boundary
- **AND** every neuron's chemical in- and out-degree, gap junctions and autapses SHALL equal the wild
  type's

#### Scenario: Shortcuts are not manufactured

- **WHEN** the one- and two-hop chemical routes from boundary sensory neurons to boundary motor neurons
  are counted
- **THEN** the boundary-preserving null SHALL have exactly the wild type's routes

#### Scenario: Existing nulls are unchanged

- **WHEN** the default or any existing rewired wiring is drawn at a pinned seed
- **THEN** its graph SHALL equal the graph drawn before this option existed
