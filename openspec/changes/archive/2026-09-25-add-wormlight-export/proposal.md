## Why

[Wormlight](https://github.com/chrisjz/wormlight) is a browser simulation of *C. elegans* in which
the connectome, run as a graded conductance-based neural model, drives a physically simulated body.
It takes its wiring, neuron classes and release identities from this repository, so the two projects
share one curated source instead of maintaining two. Three things stand in the way.

- **A licensed copy of the wiring.** The vendored Cook et al. 2019 SI 5 is supplementary information
  to a subscription article and states no licence; WormWiring shows only "Emmons Lab Copyright (c)
  2020". Emmons 2024 (*PLoS Biology* 22:e3002939), by Cook et al.'s senior author, republishes the
  same matrices as its S1 File under **CC BY 4.0**, carrying the lab's July 2020 corrections and a
  2023 addition of gap junctions between BDU and the touch cells ALM and PLM.
- **Neuromuscular connections.** The loader keeps the 302 neurons and drops every other cell, but the
  chemical sheet also holds all 95 body wall muscles: 956 non-zero entries from 162 neurons.
- **An export.** Wormlight is TypeScript and reads one versioned JSON file, pinned by commit and
  digest.

**Reading the new file before designing for it settled what changes.** Among the 302 neurons its
chemical matrix is identical to the 2019 file's, including every neuromuscular entry. Its gap
junctions differ only by the 2023 addition: ALML–BDUL and ALMR–BDUR are new at 23 sections each, and
BDUL–PLML and BDUR–PLMR rise from 23 to 37. The 2020 corrections do touch three neuron cells whose two
directions disagreed in 2019 (PVCR–VA9 at 3 and 2, and PDB→DD6 with no mirror), but they set each to
the larger value, which is what the loader's fold already took. So the loaded wiring gains two pairs,
1,093 to 1,095, and 74 sections, 5,790 to 5,864.

## What Changes

### 1. Vendored data

`data/connectome/emmons_2024_s1_connectome_adjacency.xlsx`, the publisher's file byte for byte, with a
`PROVENANCE.md` entry in the house shape. The 2019 file stays, and everything that reads it today
keeps reading it.

### 2. Two loaders

- `load_emmons_2024_hermaphrodite()` returns a `Connectome` shaped exactly as the Cook 2019 loader's,
  sharing its parse. The two files differ only in how they name the gap-junction sheets.
- `load_emmons_2024_neuromuscular()` returns a `NeuromuscularJunction` for every non-zero entry from
  a neuron onto a body wall muscle, with the 95 muscles named in a new `connectome.muscles` module.

Both refuse a file whose SHA256 differs from the recorded one, as the measured-weight loader does.

### 3. The exporter

`scripts/export_wormlight.py --out <path>` writes `wormlight.connectome/1`: neurons (name, class,
release identities with the primary first, and the sign the primary implies), the muscles, chemical
synapses, gap junctions, neuromuscular connections, and provenance (the commit, and the digest of
every input). The output is deterministic, one record per line, and the script refuses a tree with
uncommitted changes to tracked files unless told otherwise, in which case it records that the tree was
dirty.

## Capabilities

**Modified**: `connectome-substrate`, with three added requirements: the vendored release and its
loader, the neuromuscular parse, and the export.

## Impact

- `data/connectome/`: the spreadsheet (Git LFS, per the existing `*.xlsx` rule) and `PROVENANCE.md`
- `packages/quantum-nematode/quantumnematode/connectome/`: `loader.py` (the shared parse and the two
  loaders), `model.py` (`NeuromuscularJunction`), `muscles.py` (new), `__init__.py`
- `scripts/export_wormlight.py`: new
- Tests: the loaders against both files, and the exporter's content, format and refusals
- Docs: `CHANGELOG.md` and a line in `AGENTS.md`'s commands

Nothing under `brain/`, `env/`, `agent/` or `configs/` changes, and no experiment's input changes.

## Breaking Changes

None. The Cook 2019 loader's output is unchanged, and every addition is new API.
