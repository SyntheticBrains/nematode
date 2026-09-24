## ADDED Requirements

### Requirement: The Emmons 2024 release of the Cook 2019 connectome is vendored and loadable

The substrate SHALL vendor the S1 File of Emmons 2024 (*PLoS Biology* 22:e3002939), the Cook et al.
2019 adjacency matrices released under CC BY 4.0 with the lab's 2020 corrections and 2023 additions,
under `data/connectome/`, byte for byte, with a `PROVENANCE.md` entry in the format the directory's
existing entries use. The 2019 file SHALL stay vendored, and every loader that reads it today SHALL
return the same connectome. A loader SHALL read the new file, SHALL refuse it if its SHA256 differs from
the recorded digest, and SHALL return a `Connectome` built by the same parse as the Cook 2019 loader's,
with `source` set to `emmons_2024_hermaphrodite`.

#### Scenario: The vendored file is the one recorded

- **WHEN** the Emmons 2024 loader reads its file
- **THEN** the file's SHA256 SHALL equal the digest recorded in `PROVENANCE.md`
- **AND** a file with any other digest SHALL be refused, as SHALL a missing file

#### Scenario: The chemical wiring is the 2019 file's

- **WHEN** the Emmons 2024 connectome is loaded beside the Cook 2019 one
- **THEN** it SHALL have the same 302 neurons and exactly the same chemical synapses: 3,709, with
  20,965 sections and 38 autapses

#### Scenario: The gap junctions differ only by the 2023 addition

- **WHEN** the two connectomes' gap junctions are compared
- **THEN** the Emmons 2024 connectome SHALL have 1,095 pairs and 5,864 sections
- **AND** it SHALL differ from the Cook 2019 connectome in exactly ALML–BDUL and ALMR–BDUR (absent, then
  23 sections each) and BDUL–PLML and BDUR–PLMR (23, then 37)

### Requirement: Chemical synapses onto the body wall muscles are loadable

The substrate SHALL name the 95 body wall muscles as the adjacency matrices do: `dBWML1`–`dBWML24`,
`dBWMR1`–`dBWMR24`, `vBWML1`–`vBWML23` and `vBWMR1`–`vBWMR24`. A loader SHALL return every non-zero
entry of the vendored Emmons 2024 file's hermaphrodite chemical sheet from one of the 302 neurons onto
one of those muscles, as a typed record of the neuron, the muscle and the section count, sorted for
determinism. The parse SHALL refuse a sheet that does not list each body wall muscle exactly once. The
`Connectome` model SHALL be unchanged.

#### Scenario: Every body wall muscle is innervated

- **WHEN** the neuromuscular connections are loaded
- **THEN** there SHALL be 956 entries from 162 neurons, with 5,515 sections
- **AND** every one of the 95 muscles SHALL receive at least one

#### Scenario: The 2019 file holds the same entries

- **WHEN** the same parse reads the Cook 2019 file's chemical sheet
- **THEN** it SHALL return exactly the entries it returns for the Emmons 2024 file

#### Scenario: A sheet that does not list every muscle is refused

- **WHEN** the parse is given a sheet that omits a body wall muscle, or lists one twice
- **THEN** it SHALL raise, naming the muscle

### Requirement: A versioned export of the connectome for Wormlight

`scripts/export_wormlight.py` SHALL write the Emmons 2024 connectome as one JSON file with the schema
`wormlight.connectome/1`: the neurons, each with its class, its release identities with the primary
first, and the sign `TRANSMITTER_SIGN` gives the primary identity or null; the body wall muscles; the
chemical synapses; the gap junctions, each pair once in canonical order; the neuromuscular connections;
and provenance recording the commit exported from, whether tracked files had uncommitted changes, and
the path, SHA256 and role of each file the values come from. The output SHALL be deterministic, with
one record per line. The script SHALL refuse a tree whose tracked files have uncommitted changes unless
`--allow-dirty` is passed, in which case the provenance SHALL say the tree was dirty.

#### Scenario: The export carries the loaded connectome

- **WHEN** the export is built
- **THEN** it SHALL hold 302 neurons, 95 muscles, 3,709 chemical synapses, 1,095 gap junctions and 956
  neuromuscular connections, with the section totals the loaders give
- **AND** each neuron's identities SHALL equal the atlas reader's for that neuron, primary first

#### Scenario: The rendering is stable and parses back

- **WHEN** an export is rendered
- **THEN** parsing the text SHALL give back the same object
- **AND** each neuron, muscle and connection SHALL sit on its own line

#### Scenario: A dirty tree is refused unless allowed

- **WHEN** tracked files have uncommitted changes and `--allow-dirty` is not passed
- **THEN** the script SHALL exit non-zero without writing a file
- **AND** with `--allow-dirty` it SHALL write the file and record `nematodeDirty: true`
