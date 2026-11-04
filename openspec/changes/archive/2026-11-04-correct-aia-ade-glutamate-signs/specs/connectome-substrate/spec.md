## MODIFIED Requirements

### Requirement: Per-connection chemical signs from vendored sources

The substrate SHALL vendor, under `data/connectome/` with `PROVENANCE.md` entries in the directory's
format, the S1 Data and S5 Data files of Fenyves et al. 2020 (*PLoS Comput Biol* 16:e1007974) byte for
byte, and a table of cited per-connection signs. A loader SHALL sign every chemical edge of a given
connectome by the first of four steps that gives a sign: a cited physiology override; the expression-based
polarity Fenyves et al. predict, used only where every transmitter that prediction rests on is one of
the presynaptic cell's release identities in the package's classification table; the per-neuron transmitter
rule; otherwise no fast sign. The loader SHALL accept a set of citation keys whose overrides it ignores,
empty by default. No brain SHALL read the table unless its configuration asks for it.

#### Scenario: The vendored files are the ones recorded

- **WHEN** the loader reads either Fenyves file
- **THEN** the file's SHA256 SHALL equal the digest recorded in `PROVENANCE.md`
- **AND** a file with any other digest SHALL be refused, as SHALL a missing file

#### Scenario: The Cook 2019 table has the recorded composition

- **WHEN** the Cook 2019 hermaphrodite's 3,709 chemical edges are signed
- **THEN** 67 SHALL be signed by physiology, 1,676 by expression, 1,433 by the rule, and 533 SHALL have no
  fast sign
- **AND** AWCL → AIYL SHALL be inhibitory, by physiology, where the per-neuron rule makes it excitatory

#### Scenario: The local-search pathways onto AIA and ADE are inhibitory

- **WHEN** the table signs the glutamatergic edges from the tax-4 chemosensory cells ASE, ASG, ASK and
  AWC onto AIAL and AIAR, and from the mec-3 mechanosensory cells FLP and AVM onto ADEL and ADER
- **THEN** each of the 16 SHALL be inhibitory, by physiology, citing López-Cruz et al. 2019
- **AND** the 13 other glutamatergic edges onto AIA and ADE SHALL keep the rule's excitatory sign

#### Scenario: The earlier table is reproducible

- **WHEN** the table is built with the López-Cruz overrides excluded
- **THEN** it SHALL equal, edge for edge, the table before those overrides were added: 51 by physiology,
  1,676 by expression, 1,449 by the rule and 533 with no fast sign

#### Scenario: The table matches Wormlight's

- **WHEN** the table with the López-Cruz overrides excluded is compared with the chemical signs in
  Wormlight's export at commit `1190b3e`
- **THEN** every edge SHALL carry the same sign and the same source, except 23 whose expression
  prediction rests on a secondary transmitter the atlas does not give the cell, which this table signs by
  the rule
- **AND** the full table SHALL differ from the excluded one on exactly the 16 López-Cruz edges

#### Scenario: A prediction resting on a transmitter the cell does not release is set aside

- **WHEN** Fenyves et al.'s primary transmitter for a cell is not one of its release identities, or their
  secondary transmitter is not and the primary alone would not give the same polarity
- **THEN** the connection SHALL fall through to the per-neuron rule

#### Scenario: The sources disagree or point at nothing

- **WHEN** the two Fenyves files predict opposite signs for one edge, or an override names an edge the
  connectome lacks, names an edge twice, or cites a source the table does not list, or the excluded keys
  name a citation the table does not list
- **THEN** the loader SHALL refuse with an error naming the edge, row or key

#### Scenario: Nothing an experiment reads changes

- **WHEN** a committed configuration that does not read the table, or one that excludes the López-Cruz
  overrides, is loaded
- **THEN** its brain SHALL sign its synapses exactly as before the López-Cruz overrides were added
