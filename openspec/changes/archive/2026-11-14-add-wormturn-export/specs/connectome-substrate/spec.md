## ADDED Requirements

### Requirement: A versioned export of the signed connectome and the neuromodulator layer for Wormturn

`scripts/export_wormturn.py` SHALL write one JSON file with the schema `wormturn.connectome/1`
holding: the neurons and the gap junctions, as the Wormlight export writes them; every chemical
synapse of the Emmons 2024 connectome with its section count and the per-connection sign the
substrate gives it on that connectome, with no citation excluded, together with the step that gave
the sign and, for a physiology sign, its citation key; the full citation for each key; the
neuromodulator layer's ligands, couples, links and drives as `data/neuromodulators/layer.json`
holds them; and the receptor couplings and their rule as `receptor_couplings.json` holds them. A
layer value SHALL be carried unchanged; only a mapping keyed by name MAY become a list of records
carrying the name. The export SHALL NOT carry muscles or neuromuscular connections. Its provenance
SHALL record the commit exported from, whether tracked files had uncommitted changes, and the path,
SHA256 and role of each input, including both Fenyves files, the physiology table, `layer.json` and
`receptor_couplings.json`, and the script SHALL refuse an input that is a Git LFS pointer. The output
SHALL be deterministic, with one record per line. The script SHALL refuse a tree whose tracked files
have uncommitted changes unless `--allow-dirty` is passed, in which case the provenance SHALL say the
tree was dirty.

#### Scenario: The export carries the signed wiring

- **WHEN** the export is built
- **THEN** it SHALL hold 302 neurons, 3,709 chemical synapses with 20,965 sections and 1,095 gap
  junctions with 5,864 sections, and no muscle or neuromuscular entry
- **AND** every chemical synapse SHALL carry exactly one sign and one source: 67 by physiology, 1,676
  by expression, 1,433 by the rule, and 533 with no fast sign and sign 0
- **AND** every physiology sign, and no other, SHALL carry a citation key that the citation table
  lists, and every key in the table SHALL sign at least one edge

#### Scenario: The signs are the wiring's own

- **WHEN** the Emmons 2024 and Cook 2019 connectomes are loaded
- **THEN** their chemical edges SHALL be the same, with the same section counts
- **AND** the export's signs SHALL equal the per-connection table built on either

#### Scenario: The layer round-trips against its files

- **WHEN** the export's ligands, couples, links and drives are compared with `layer.json`
- **THEN** they SHALL hold the same ligands with the same kinds and the same producers and source
  tags, and the same couples, cells, source tags and signs, in the files' order
- **AND** the receptor couplings and their rule SHALL equal `receptor_couplings.json`'s, entry for
  entry, including each sign, `verified` value, citation and note

#### Scenario: The rendering is stable and parses back

- **WHEN** an export is rendered
- **THEN** parsing the text SHALL give back the same object
- **AND** each neuron, connection, citation, ligand, couple, link, drive and receptor coupling SHALL sit
  on its own line

#### Scenario: A dirty tree is refused unless allowed

- **WHEN** tracked files have uncommitted changes and `--allow-dirty` is not passed
- **THEN** the script SHALL exit non-zero without writing a file
- **AND** with `--allow-dirty` it SHALL write the file and record `nematodeDirty: true`

#### Scenario: The Wormlight export is unchanged

- **WHEN** the Wormlight export is built from a tree with this exporter added
- **THEN** it SHALL hold the records it held before, rendered the same way
