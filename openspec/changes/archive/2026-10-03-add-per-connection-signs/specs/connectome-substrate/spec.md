## ADDED Requirements

### Requirement: Per-connection chemical signs from vendored sources

The substrate SHALL vendor, under `data/connectome/` with `PROVENANCE.md` entries in the directory's
format, the S1 Data and S5 Data files of Fenyves et al. 2020 (*PLoS Comput Biol* 16:e1007974) byte for
byte, and a table of cited per-connection signs. A loader SHALL sign every chemical edge of a given
connectome by the first of four steps that gives a sign: a cited physiology override; the expression-based
polarity Fenyves et al. predict, used only where every transmitter that prediction rests on is one of
the presynaptic cell's release identities in the package's classification table; the per-neuron transmitter
rule; otherwise no fast sign. No brain SHALL read the table unless its configuration asks for it.

#### Scenario: The vendored files are the ones recorded

- **WHEN** the loader reads either Fenyves file
- **THEN** the file's SHA256 SHALL equal the digest recorded in `PROVENANCE.md`
- **AND** a file with any other digest SHALL be refused, as SHALL a missing file

#### Scenario: The Cook 2019 table has the recorded composition

- **WHEN** the Cook 2019 hermaphrodite's 3,709 chemical edges are signed
- **THEN** 51 SHALL be signed by physiology, 1,676 by expression, 1,449 by the rule, and 533 SHALL have no
  fast sign
- **AND** AWCL → AIYL SHALL be inhibitory, by physiology, where the per-neuron rule makes it excitatory

#### Scenario: The table matches Wormlight's

- **WHEN** the table is compared with the chemical signs in Wormlight's export at commit `1190b3e`
- **THEN** every edge SHALL carry the same sign and the same source, except 23 whose expression
  prediction rests on a secondary transmitter the atlas does not give the cell, which this table signs by
  the rule

#### Scenario: A prediction resting on a transmitter the cell does not release is set aside

- **WHEN** Fenyves et al.'s primary transmitter for a cell is not one of its release identities, or their
  secondary transmitter is not and the primary alone would not give the same polarity
- **THEN** the connection SHALL fall through to the per-neuron rule

#### Scenario: The sources disagree or point at nothing

- **WHEN** the two Fenyves files predict opposite signs for one edge, or an override names an edge the
  connectome lacks, names an edge twice, or cites a source the table does not list
- **THEN** the loader SHALL refuse with an error naming the edge or row

#### Scenario: Nothing an experiment reads changes

- **WHEN** any committed configuration is loaded
- **THEN** its brain SHALL sign its synapses exactly as before this change
