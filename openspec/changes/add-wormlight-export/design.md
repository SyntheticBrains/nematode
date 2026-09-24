## Overview

This change gives an external consumer, Wormlight, a licensed and versioned copy of the connectome
this repository curates. It adds data and read paths only: no brain, environment or experiment reads
anything new. The decisions below were taken from the data itself, and the counts they cite are the
ones the tests pin.

## Design Decisions

### Decision A: Vendor the Emmons 2024 release beside the 2019 file, not in place of it

| source | licence found | decision |
|---|---|---|
| Cook et al. 2019, *Nature* 571:63, SI 5 (vendored today via the cect mirror) | none stated; a subscription article's supplement | stays, and every current reader keeps it |
| Cook et al. 2019 SI 5, "corrected July 2020" (WormWiring) | none stated; "Emmons Lab Copyright (c) 2020" | not vendored |
| Emmons 2024, *PLoS Biology* 22:e3002939, S1 File | **CC BY 4.0**, the article's licence | vendored |

The new file is the licensed route to the same data, from the same lab. Replacing the 2019 file would
move every logbook's input by two gap-junction pairs, so the two sit side by side and only new code
reads the new one.

Compared cell by cell among the 302 neurons:

| | 2019 SI 5 | Emmons 2024 S1 |
|---|---|---|
| chemical synapses | 3,709 (20,965 sections, 38 autapses) | identical |
| neuromuscular entries onto the 95 body wall muscles | 956 from 162 neurons (5,515 sections) | identical |
| gap-junction pairs after the loader's fold | 1,093 (5,790 sections) | 1,095 (5,864 sections) |

The gap-junction difference is the 2023 addition alone: ALML–BDUL and ALMR–BDUR at 23 sections each,
and BDUL–PLML and BDUR–PLMR from 23 to 37. The legend's July 2020 corrections make each gap-junction
table agree across its diagonal. Among neurons the 2019 symmetric sheet disagreed in three cells,
PVCR–VA9 (3 against 2) and PDB→DD6 (2, with no mirror entry), and the correction set each pair to the
larger value, which the loader's fold (the maximum over both directions and both sheets) already took.

### Decision B: The loaders live in the package, the export in a script

The parse is the package's: the file has the 2019 layout exactly, so `load_emmons_2024_hermaphrodite()`
reuses `_parse_cook_2019_adjacency_sheet` and `_to_gap_junctions` through one shared function, and the
Cook 2019 loader moves onto it unchanged. Only the gap-junction sheet names differ:
`hermaphrodite gap jn symmetric` and `hermaphrodite gap jn asymmetric` against `herm gap jn …`.

The export is one consumer's format, so it lives in `scripts/`, as `generate_neuron_transmitters.py`
does, and reads only public package API.

### Decision C: The neuromuscular parse covers the body wall muscles, from neurons

Besides the 302 neurons and the body wall muscles, the chemical sheet's columns name 57 other cells:
glia, pharyngeal, vulval and anal muscles, gland and marginal cells, the hypodermis, the intestine and
the excretory cells. The parse keeps the 95 body wall muscles (`dBWML1`–`24`, `dBWMR1`–`24`,
`vBWML1`–`23`, `vBWMR1`–`24`), numbered from head to tail: IL1, SMD and SAB synapse onto position 1,
and DA9, VD13 and VA12 onto 23 and 24. It refuses a sheet that does not list each of them exactly once,
so a revision that renamed a quadrant fails loudly rather than exporting fewer muscles. Rows are
presynaptic cells, as for the neuron edges; no row outside the 302 neurons has an entry onto body wall
muscle, so none is dropped.

`NeuromuscularJunction` is a separate type and `Connectome` is unchanged, so no existing consumer sees
a new field. Phase 8's C.1a, the anatomical muscle readout, can build its motor-neuron-to-muscle tensor
on this parse. This change is not a Phase 8 task and does not tick or edit the tracker.

### Decision D: A deterministic export that knows where it came from

- **Content.** Neurons carry the class and release identities the classification table holds, the
  primary identity first, and the sign `TRANSMITTER_SIGN` gives the primary identity, or null. For all
  302 neurons the primary plus the co-transmitters equals what the atlas reader returns, so the export
  reads the committed literals and never opens the atlas.
- **Provenance.** The commit exported from, whether the tree was dirty, and the path, SHA256 and role
  of each file the values come from: the Emmons spreadsheet, the classification module, and the atlas
  it was generated from.
- **Refusing a dirty tree.** A recorded commit only describes the export if the tracked files match
  it, so the script refuses uncommitted changes to tracked files. `--allow-dirty` exports anyway and
  records `nematodeDirty: true`, which a consumer can refuse in turn. Untracked files, such as run
  logs, are ignored.
- **Format.** Top-level keys in a fixed order, and one compact record per line, so a re-export diffs
  line by line. The same commit gives the same bytes.

## Risks

- **Two wirings in one repository.** A future reader could load the wrong one. The loader names say
  which, `Connectome.source` differs (`cook_2019_hermaphrodite`, `emmons_2024_hermaphrodite`), and
  `PROVENANCE.md` records why both are there.
- **LFS.** A fresh clone smudges `data/**`, so the file is present by default. A loader handed an LFS
  pointer fails the digest check with a message that names `git lfs pull`.
