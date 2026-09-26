## MODIFIED Requirements

### Requirement: Degree-preserving rewired-null wiring option

`ConnectomePPOBrainConfig` SHALL expose a `wiring` selector with values `wild_type` (default),
`rewired_degree_preserving`, `rewired_chemical_only` and `rewired_gap_junctions_held`, plus a
`rewire_seed` (integer, or unset to derive from the run seed). Under any rewired value the loaded `Connectome` SHALL be transformed
**before** the topology is constructed, deterministically given `rewire_seed`, and SHALL NOT silently
reseed on a pathological draw. The neuron set and ordering SHALL be unchanged, so per-post fan-in —
and hence the `w_chem` initialisation scale — is preserved.

Under `rewired_degree_preserving` the chemical-synapse edge set SHALL be replaced by a **directed**
degree-preserving edge-swapped set (each neuron's out-degree and in-degree preserved exactly) and the
gap-junction edge set by an **undirected** degree-preserving edge-swapped set (each neuron's gap
degree preserved exactly). The swap SHALL NOT create a self-loop or a duplicate edge. It MAY remove an
existing self-loop, so the wild type's autapses are not preserved. Synapse and gap-junction counts
travel with their edges, so a neuron's total gap-junction strength is not preserved.

Under `rewired_chemical_only` the chemical edges that are not self-loops SHALL be replaced by the same
directed degree-preserving swap. Every self-loop (autapse) SHALL be kept with its count, and the
gap-junction edges SHALL be kept with their counts, identical to the wild type's. Each neuron's
chemical out-degree and in-degree SHALL be preserved exactly.

Under `rewired_gap_junctions_held` the chemical edge set SHALL be the one `rewired_degree_preserving`
produces at the same `rewire_seed`, edge for edge and count for count — the same directed swap over the
same list, so autapses are handled exactly as there — and the gap-junction edges SHALL be kept with
their counts, identical to the wild type's. The two values therefore differ in their gap junctions
alone.

When `wiring` is `wild_type` the transform SHALL be a no-op, leaving the built strict-mask, weight
initialisation, and gap-junction buffer byte-identical to the pre-change connectome brain. The
`rewired_degree_preserving` transform SHALL be identical, edge for edge and count for count, to the
transform before `rewired_chemical_only` existed.

#### Scenario: Rewiring preserves each neuron's in/out degree

- **WHEN** a connectome is rewired under any rewired value
- **THEN** every neuron's chemical out-degree and in-degree, and its gap-junction degree, SHALL equal
  its wild-type values, while the connected chemical pairs differ

#### Scenario: Rewiring produces a simple graph

- **WHEN** a connectome is rewired
- **THEN** the swap SHALL have created no self-loop and no duplicate edge
- **AND** under `rewired_chemical_only` the only self-loops SHALL be the wild type's autapses, with
  their counts

#### Scenario: The chemical-only null holds gap junctions and autapses at the wild type

- **WHEN** a connectome is rewired under `rewired_chemical_only`
- **THEN** its gap-junction edges and counts SHALL equal the wild type's
- **AND** every neuron's total gap-junction strength SHALL equal its wild-type total
- **AND** the built `g_gap` buffer SHALL be bit-identical to the wild type's

#### Scenario: Node set and ordering are preserved

- **WHEN** a connectome is rewired
- **THEN** the neuron set and its sorted ordering SHALL be identical to the wild type's, so the
  strict-mask, `w_chem` initialisation scale, and `g_gap` normalisation derive from the same
  per-neuron fan-in

#### Scenario: Rewiring is deterministic under the seed

- **WHEN** two connectomes are rewired with the same `rewire_seed` and wiring value
- **THEN** their rewired edge sets SHALL be identical; different seeds SHALL (with overwhelming
  probability) differ

#### Scenario: The existing null is unchanged

- **WHEN** a connectome is rewired under `rewired_degree_preserving` at a given seed
- **THEN** its chemical and gap-junction edges and counts SHALL equal those the transform produced
  before `rewired_chemical_only` existed

#### Scenario: Wild-type wiring is byte-identical

- **WHEN** the brain is built with `wiring: wild_type`
- **THEN** the strict-mask `m_chem`, the `w_chem` initialisation, and the `g_gap` buffer SHALL be
  identical to the pre-change connectome brain

#### Scenario: The gap-held null pairs exactly with the degree-preserving null

- **GIVEN** a connectome rewired under `rewired_degree_preserving` and under
  `rewired_gap_junctions_held` at the same `rewire_seed`
- **WHEN** their edge sets are compared
- **THEN** their chemical edges and counts SHALL be identical, autapses included
- **AND** the gap-held null's gap-junction edges and counts SHALL equal the wild type's, so the built
  `g_gap` buffer is bit-identical to the wild type's while `m_chem` is bit-identical to the
  degree-preserving null's
