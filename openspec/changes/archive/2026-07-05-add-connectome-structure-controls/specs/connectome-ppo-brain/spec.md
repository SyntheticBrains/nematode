# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Degree-preserving rewired-null wiring option

`ConnectomePPOBrainConfig` SHALL expose a `wiring` selector with values `wild_type` (default) and
`rewired_degree_preserving`, plus a `rewire_seed` (integer, or unset to derive from the run seed).
When `wiring` is `rewired_degree_preserving`, the loaded `Connectome` SHALL be transformed **before**
the topology is constructed: its chemical-synapse edge set SHALL be replaced by a **directed**
degree-preserving edge-swapped set (each neuron's out-degree and in-degree preserved exactly) and its
gap-junction edge set by an **undirected** degree-preserving edge-swapped set (each neuron's gap degree
preserved exactly). The neuron set and ordering SHALL be unchanged, so per-post fan-in — and hence the
`w_chem` initialisation scale and the `g_gap` fan-in normalisation — are preserved; only *which*
neurons are connected changes. The transform SHALL be deterministic given `rewire_seed`, SHALL reject
self-loops and duplicate edges, and SHALL NOT silently reseed on a pathological draw.

When `wiring` is `wild_type` the transform SHALL be a no-op, leaving the built strict-mask, weight
initialisation, and gap-junction buffer byte-identical to the pre-change connectome brain.

#### Scenario: Rewiring preserves each neuron's in/out degree

- **WHEN** a connectome is rewired with `wiring: rewired_degree_preserving`
- **THEN** every neuron's chemical out-degree and in-degree, and its gap-junction degree, SHALL equal its wild-type values, while the connected pairs differ

#### Scenario: Rewiring produces a simple graph

- **WHEN** a connectome is rewired
- **THEN** the rewired chemical and gap-junction edge sets SHALL contain no self-loops and no duplicate edges

#### Scenario: Node set and ordering are preserved

- **WHEN** a connectome is rewired
- **THEN** the neuron set and its sorted ordering SHALL be identical to wild-type, so the strict-mask, `w_chem` init scale, and `g_gap` normalisation derive from the same per-neuron fan-in

#### Scenario: Rewiring is deterministic under the seed

- **WHEN** two connectomes are rewired with the same `rewire_seed`
- **THEN** their rewired edge sets SHALL be identical; different seeds SHALL (with overwhelming probability) differ

#### Scenario: Wild-type wiring is byte-identical

- **WHEN** the brain is built with `wiring: wild_type`
- **THEN** the strict-mask `m_chem`, the `w_chem` initialisation, and the `g_gap` buffer SHALL be identical to the pre-change connectome brain (no behaviour change to the existing ranking cell)
