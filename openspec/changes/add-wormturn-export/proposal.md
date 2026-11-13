## Why

Wormturn is a separate research project that, like Wormlight, takes its connectome data from this
repository as one versioned file, pinned by commit and digest, instead of curating its own. The
Wormlight export (`wormlight.connectome/1`) carries the wiring, the neuron classes, the release
identities and the per-neuron rule's sign. Wormturn needs two things it does not carry:

- **Per-connection signs.** The package signs each chemical connection separately
  (`per_connection_signs`): a cited physiology measurement first, then the expression-based polarity
  Fenyves et al. 2020 predict, then the per-neuron transmitter rule, else no fast sign. A consumer
  needs each sign together with the step that gave it, and the citation behind a physiology sign.
- **The neuromodulator layer.** `data/neuromodulators/layer.json` places 56 ligands, their producer
  cells and 121 ligand–receptor couples from Bentley et al. 2016 and Ripoll-Sánchez et al. 2023, and
  `receptor_couplings.json` holds the cited G-protein coupling and sign of each receptor of the
  layer's fitted couples.

The maintainer approved the exporter on 2026-11-14.

**Reading the data before designing for it settled three things.**

- The Emmons 2024 release's chemical edges are those of Cook 2019, in the same order with the same
  section counts, so the sign table built on either is the same, edge for edge. The export signs the
  Emmons 2024 wiring it exports, passed explicitly, rather than relying on the default.
- Over the 3,709 edges: **67 by physiology, 1,676 by expression, 1,433 by the rule, and 533 with no
  fast sign**. Every one of the nine cited sources signs at least one edge. The 533 come from 32
  presynaptic cells whose primary release identity the rule does not sign, among them the
  dopaminergic ADE, CEP and PDE, the octopaminergic RIC, the serotonergic NSM and the peptidergic
  RID. Several of them are producers in the layer, so the two halves of the export meet there.
- The two layer files fit a record-per-line layout almost unchanged, with three quirks kept as they
  are (design Decision D): `verified` is `false` or an object naming the date and the source read,
  not a flag; flp-24 has no producer, so its one couple has no pool; and `links` is empty.

## What Changes

### 1. The exporter

`scripts/export_wormturn.py --out <path>` writes `wormturn.connectome/1`:

- `provenance`: the exporter, the commit, whether tracked files had uncommitted changes, and the
  path, SHA256 and role of every input;
- `neurons` and `gap`, exactly as the Wormlight export writes them;
- `chemical`: each Emmons 2024 chemical synapse with its section count, its sign, the step that gave
  the sign, and the citation key of a physiology sign;
- `signCitations`: the full citation for each key;
- `ligands`, `couples`, `links` and `drives`, carried from `layer.json`;
- `receptorCouplingRule` and `receptorCouplings`, carried from `receptor_couplings.json`.

No muscles and no neuromuscular connections. The output is deterministic, one record per line, and
the script refuses a dirty tree unless `--allow-dirty` is passed, which is then recorded.

### 2. Shared helpers in the Wormlight exporter

`scripts/export_wormlight.py` exposes what both exporters need: its git check, its input-digest
record, its neuron and gap-junction records, and a renderer that takes the list of record keys.
Its output is unchanged, byte for byte.

## Capabilities

**Modified**: `connectome-substrate`, with one added requirement: the export for Wormturn.

## Impact

- `scripts/export_wormturn.py`: new
- `scripts/export_wormlight.py`: helpers made public and the renderer parameterised; same output
- Tests: `entrypoints/test_export_wormturn.py`, new
- Docs: `CHANGELOG.md` and a line in `AGENTS.md`'s commands

No package code, no data file and no configuration changes. Nothing under `brain/`, `env/`, `agent/`
or `configs/` changes, and no experiment's input changes.

## Breaking Changes

None. Every addition is new, and the Wormlight export writes the same bytes.
