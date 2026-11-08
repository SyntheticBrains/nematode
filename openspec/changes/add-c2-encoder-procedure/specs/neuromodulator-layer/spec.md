## ADDED Requirements

### Requirement: Cited receptor signs behind an empty default

`ConnectomePPOBrainConfig` SHALL accept `modulator_receptor_signs`, a list of receptor genes, empty by default. With it empty, every output SHALL be byte-identical to the brain before the option.

- **A named receptor's fitted couples SHALL take its sign** from `data/neuromodulators/receptor_couplings.json`, under `receptor_signs.json`'s rule: Gαi/o and Cl⁻ channels −1, Gαq and Gαs +1.
- **Refused at load:** a receptor not in the table, with a null sign, not verified against its source, or with no fitted couple in the layer. The option also requires `modulator_layer: data`.
- **Couples `receptor_signs.json` already fixes SHALL keep their sign.**
- **Under the receptor-placement null,** a couple's sign SHALL travel with the couple.

#### Scenario: Empty is byte-identical

- **WHEN** a data-layer brain is built with `modulator_receptor_signs: []`
- **THEN** its layer rows and outputs SHALL equal those of the brain before the option

#### Scenario: PDFR-1 and SER-5 fix their couples

- **WHEN** `modulator_receptor_signs: [pdfr-1, ser-5]` is set
- **THEN** pdf-1:pdfr-1, pdf-2:pdfr-1 and serotonin:ser-5 SHALL carry sign +1
- **AND** every other couple's sign SHALL be unchanged

#### Scenario: The sign travels with its couple under the null

- **WHEN** `modulator_receptor_signs: [pdfr-1]` is set under `modulator_layer_null: receptor_placement`
- **THEN** the moved pdf-1:pdfr-1 and pdf-2:pdfr-1 couples SHALL carry sign +1 and stay their own group under per-ligand tying, with the wild type's entry names

#### Scenario: An unresolved receptor is refused

- **WHEN** a receptor whose table sign is null, or which is not in the table, is named
- **THEN** loading SHALL be refused, naming the receptor

### Requirement: The receptor-coupling table is cited and checked

`data/neuromodulators/receptor_couplings.json` SHALL list the receptors of the layer's fitted couples. For each it SHALL record the coupling, the sign under the rule, the evidence class, the citation, and a quote from the source, or null with the reason why. Each entry SHALL also record whether its quote was checked against the source, and when. `PROVENANCE.md` SHALL describe it.

#### Scenario: Every signed entry is checked

- **WHEN** the table is read
- **THEN** every entry with a non-null sign SHALL be marked verified, with a date and a quote

### Requirement: Table-signed couples are their own groups under per-ligand tying

Under `encoder_modulator_tying: per_ligand`, a couple signed through `modulator_receptor_signs` SHALL form a fixed-sign group of its own, one per ligand and receptor, named `modulator.fixed.<ligand>.<receptor>`. Its magnitude SHALL NOT be shared with the ligand's other fixed couples.

#### Scenario: ser-5 keeps its own magnitude

- **WHEN** `modulator_receptor_signs: [ser-5]` is set under per-ligand tying
- **THEN** `modulator.fixed.serotonin.ser-5` SHALL be an entry distinct from `modulator.fixed.serotonin`
- **AND** the per-ligand count SHALL stay 132
