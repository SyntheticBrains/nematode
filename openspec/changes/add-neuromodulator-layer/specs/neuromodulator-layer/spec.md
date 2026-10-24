## ADDED Requirements

### Requirement: A neuromodulator layer derived from licensed data

The repository SHALL vendor `data/neuromodulators/layer.json`, derived only from CC BY sources:
Bentley et al. 2016's S1 Dataset and Ripoll-Sánchez et al. 2023's figshare data. Its producers,
receptor cells and couples SHALL come from those sources alone, every cell tagged with the source
that places it. Gene names SHALL be canonicalised to WormBase public names through a committed
synonym table before the sources are joined. Monoamine producers SHALL be each transmitter's
synthesis-marker cells, with the source's excluded rows dropped. The union SHALL apply no common
potency cut, and each couple SHALL record its EC50 range and source. The provenance SHALL record:

- each source's URL, checksum and licence;
- what the MIT-only 2024 corrections would change and why they are left out;
- each name the connectome lacks.

#### Scenario: Every cell names its source

- **WHEN** the layer file is loaded
- **THEN** every producer and receptor cell SHALL carry at least one source tag, `bentley2016` or
  `ripollsanchez2023`

#### Scenario: One gene under two names

- **WHEN** Bentley's pdf-2 and Ripoll-Sánchez's NLP-37 are joined
- **THEN** they SHALL form one ligand with one pool, and a couple still matching under two names after
  canonicalisation SHALL be refused

#### Scenario: The potency sensitivity

- **WHEN** the layer is built with the `ec50_500nM` sensitivity
- **THEN** every couple whose recorded EC50 range lies above 500 nM SHALL be dropped, including
  pdf-1 → pdfr-1

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
- the held current SHALL be the sum over couples and links of the ligand's concentration times the
  strength on their cells, plus the drive rows, which are sensory only;
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

With the data layer, `modulator_knockout_genes` SHALL name genes to remove; `modulator_knockout`
SHALL be refused there, and `modulator_knockout_genes` with the hand-placed layer. A producer gene SHALL hold
its ligand's concentration at zero, and a receptor gene SHALL remove every couple through it. A gene
the layer does not contain SHALL be refused at load.

#### Scenario: pdfr-1 removes both PDF couples

- **WHEN** `modulator_knockout_genes` is `["pdfr-1"]`
- **THEN** no current SHALL flow through any couple whose receptor is pdfr-1, and every other couple
  SHALL be unchanged

### Requirement: The machinery reproduces the hand-placed field

A layer spec built from the hand-placed circuit, with its four pathways as two fixed couples, one link
tagged to pdfr-1 and one drive, SHALL reproduce the hand-placed field's currents and concentrations exactly, with no
knockout and under each of `tph1` and `pdfr1`.

#### Scenario: The restricted layer matches bitwise

- **WHEN** both are evaluated on the same random settled states and inputs
- **THEN** their currents and next concentrations SHALL be equal

### Requirement: The layer holds a slow state

A ligand's concentration SHALL decay with its own timescale, and a fixed couple's current SHALL follow
it.

#### Scenario: A nine-minute timescale

- **WHEN** a ligand's τ is 540 s at a 1-s step and its release stops
- **THEN** its concentration SHALL reach 1/e of its starting value within one step of step 540, and
  its receptor cells' current SHALL scale with it

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

`modulator_layer_null: receptor_placement` with a seed SHALL move each receptor gene's cells, grouped
by anatomical class, to classes of the same size on the same member positions, holding producers,
links, drives and signs. Classes SHALL come from `quantumnematode.connectome.neuron_classes`, which
SHALL assign the 302 neurons to WormAtlas's 118 classes. It SHALL keep every
receptor's class count and cell count exactly, and one placement per receptor gene across its
couples. An audit over seeds 7501–7564 SHALL publish the matched-statistics table of design
Decision G and the shortcut audit's modulator term.

#### Scenario: Counts are kept exactly

- **WHEN** any null draw is built
- **THEN** each receptor gene SHALL have the same number of classes and of cells as in the data layer

#### Scenario: The class table

- **WHEN** the class table is built over the 302 neurons
- **THEN** it SHALL give 118 classes, with OLQ of 4 cells, IL1 of 6, VD of 13 and SAB of 3

#### Scenario: A class of unique size

- **WHEN** a receptor sits on a class whose size no other class shares
- **THEN** those cells SHALL stay in place, and the audit SHALL report the receptor's pinned share

#### Scenario: The draw is reproducible

- **WHEN** the same seed is drawn twice
- **THEN** the two layers SHALL be identical

### Requirement: A data-layer config and its smoke run

The repository SHALL ship a data-layer variant of the 1-s primitives lawn config, and the smoke tier
SHALL run one short episode of it through the CLI.

#### Scenario: The smoke run

- **WHEN** the smoke tier runs
- **THEN** the data-layer config SHALL complete one episode through `run_simulation.py`
