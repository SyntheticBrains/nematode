## ADDED Requirements

### Requirement: A neuromodulator layer derived from licensed data

The repository SHALL vendor `data/neuromodulators/layer.json`, derived only from CC BY sources:
Bentley et al. 2016's S1 Dataset and Ripoll-Sánchez et al. 2023's figshare data. Its producers,
receptor cells and couples SHALL come from those sources alone, every cell tagged with the source
that places it. Monoamine producers SHALL be each transmitter's synthesis-marker cells, with the
source's excluded rows dropped. The provenance SHALL record:

- each source's URL, checksum and licence;
- what the MIT-only 2024 corrections would change and why they are left out;
- each name the connectome lacks.

#### Scenario: Every cell names its source

- **WHEN** the layer file is loaded
- **THEN** every producer and receptor cell SHALL carry at least one source tag, `bentley2016` or
  `ripollsanchez2023`

#### Scenario: Ripoll-Sánchez's couples come from the CC BY networks

- **WHEN** the extraction reads a long-range individual network
- **THEN** it SHALL confirm the network is the outer product of its producer and receptor sets, and
  SHALL raise otherwise

### Requirement: The data layer behind a byte-identical default

The connectome brain SHALL accept `modulator_layer` with values `hand_placed` (the default) and
`data`, read only when `modulators` is true. With `hand_placed`, every result SHALL be byte-identical
to before. With `data`:

- the brain SHALL keep one concentration per ligand, each moving toward its producers' mean positive
  settled rate with its own timescale (60 s unless `modulator_timescales_s` sets it);
- the held current SHALL be the sum over couples of the ligand's concentration times the couple's
  strength on its receptor cells, plus the drive rows;
- fixed couples SHALL carry their published sign with a softplus strength starting at 1;
- fitted couples SHALL carry a free signed strength starting at 0.

#### Scenario: The default is unchanged

- **WHEN** a config with `modulators: true` sets no `modulator_layer`
- **THEN** the brain SHALL build the hand-placed field, and its outputs SHALL equal those before this
  change

#### Scenario: Unverified signs are fitted

- **WHEN** a monoamine receptor's coupling has no verified citation in `receptor_signs.json`
- **THEN** its couples SHALL be fitted, and the layer's provenance SHALL list it

### Requirement: Knockouts by gene

With the data layer, `modulator_knockout_genes` SHALL name genes to remove. A producer gene SHALL hold
its ligand's concentration at zero, and a receptor gene SHALL remove every couple through it. A gene
the layer does not contain SHALL be refused at load.

#### Scenario: pdfr-1 removes both PDF couples

- **WHEN** `modulator_knockout_genes` is `["pdfr-1"]`
- **THEN** no current SHALL flow through any couple whose receptor is pdfr-1, and every other couple
  SHALL be unchanged

### Requirement: The machinery reproduces the hand-placed field

A layer spec built from the hand-placed circuit, with its four pathways as two fixed couples and two
drive rows, SHALL reproduce the hand-placed field's currents and concentrations exactly, with no
knockout and under each of `tph1` and `pdfr1`.

#### Scenario: The restricted layer matches bitwise

- **WHEN** both are evaluated on the same random settled states and inputs
- **THEN** their currents and next concentrations SHALL be equal

### Requirement: The data compared with the verified circuit

The change SHALL publish a table of every link the hand-placed circuit verified, saying for each
source whether it contains the link. Disagreements SHALL be reported, not patched.

#### Scenario: A link the data lacks

- **WHEN** neither source places PDFR-1 on a cell the hand-placed circuit verified
- **THEN** the table SHALL record that cell as absent from both, with the circuit's citation beside it

### Requirement: Batched evaluation of the data layer

`BatchedConnectome` SHALL carry the data layer. Its features, action means and concentrations SHALL
match the per-step brain within 1e-5 over 50 steps, with nonzero fitted strengths.

#### Scenario: Parity with fitted strengths set

- **WHEN** a data-layer brain with nonzero fitted strengths is evaluated batched and per step on the
  same seeds
- **THEN** the batched concentrations SHALL match the per-step ones within 1e-5 over 50 steps

### Requirement: A receptor-placement null with its audit

`modulator_layer_null: receptor_placement` with a seed SHALL redraw each receptor gene's expressing
classes from classes of the same size, holding producers, drives and signs. It SHALL keep every
receptor's class count and cell count exactly, and one placement per receptor gene across its
couples. An audit over seeds 7501–7564 SHALL publish the matched-statistics table of design
Decision G and the shortcut audit's modulator term.

#### Scenario: Counts are kept exactly

- **WHEN** any null draw is built
- **THEN** each receptor gene SHALL have the same number of classes and of cells as in the data layer

#### Scenario: The draw is reproducible

- **WHEN** the same seed is drawn twice
- **THEN** the two layers SHALL be identical
