## ADDED Requirements

### Requirement: Measured synaptic weights from a vendored fitted model

The substrate SHALL vendor the Creamer, Leifer & Pillow fitted synaptic weights
(`quick_start_examples/model_weights.csv` from `Nondairy-Creamer/Creamer_LDS_2026`) under
`data/connectome/`, byte-for-byte, with the upstream licence notice beside it and a `PROVENANCE.md`
entry in the format the directory's existing entries use, recording the upstream commit and SHA256.
A loader SHALL read that file, SHALL refuse it if its SHA256 differs from the recorded digest, SHALL
validate every neuron name against the canonical classification, and SHALL return the signed weight
of each directed `(pre, post)` pair it lists. A coverage report SHALL join the table to a connectome's
chemical synapses and state the covered edges at full scope and at head scope — the edges whose
endpoints both lie in the table's neuron set, with self-loops counted separately because the table
has no diagonal and cannot cover them — together with the table entries that fall on a gap junction
only or on no connection, which SHALL be reported and not applied. Gap junctions SHALL be treated as
undirected for that report.

#### Scenario: The vendored file is the one recorded

- **WHEN** the measured table is read
- **THEN** its SHA256 SHALL equal the digest recorded in `PROVENANCE.md`
- **AND** a file with any other digest SHALL be refused

#### Scenario: Every name is a known neuron

- **WHEN** the loader reads the table
- **THEN** every pre- and post-synaptic name SHALL be one of the 302 canonical neuron names

#### Scenario: Coverage on the Cook 2019 substrate is the documented one

- **WHEN** the table is joined to the Cook 2019 hermaphrodite chemical synapses
- **THEN** the covered edges, the head-scope edges and the self-loops among them, the positive and
  negative counts, and the entries on a gap junction only or on no connection SHALL equal the
  figures the change records
