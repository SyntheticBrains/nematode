## Overview

This change gives a second external consumer, Wormturn, a versioned copy of the signed wiring and the
neuromodulator layer this repository curates. It adds one script and its tests: no package code, data
file or configuration changes, and no brain, environment or experiment reads anything new. The
decisions below were taken from the data, and the counts they cite are the ones the tests pin.

## Design Decisions

### Decision A: Sign the wiring that is exported, with the table as it stands

`per_connection_signs` signs the Cook 2019 wiring unless given another. The export passes the Emmons
2024 connectome it exports, so the signs and the wiring cannot come from different files. Read side
by side, the two files' chemical edges are the same 3,709 in the same order with the same section
counts, and the two sign tables are equal, edge for edge; a test pins both.

The table is built with no citation excluded, so it carries the López-Cruz et al. 2019 overrides.
Over the Emmons 2024 edges:

| source | edges | excitatory | inhibitory | no sign | sections |
|---|---|---|---|---|---|
| physiology | 67 | 19 | 48 | | 1,080 |
| expression | 1,676 | 1,291 | 385 | | 10,962 |
| rule | 1,433 | 1,308 | 125 | | 6,868 |
| none | 533 | | | 533 | 2,055 |
| **total** | **3,709** | 2,618 | 558 | 533 | 20,965 |

Physiology agrees with the per-neuron rule on 19 edges and signs 48 inhibitory that the rule signs
excitatory. Expression reverses the rule on 310 edges, 303 of them from excitatory to inhibitory. The
nine cited sources sign 16 (López-Cruz 2019), 10, 9, 9, 8, 7, 4, 2 and 2 edges; none is unused.

The surprise is the size of the unsigned share: 533 edges, 14%, from 32 presynaptic cells whose
primary release identity the rule does not sign: every cell of ADE, ASI, AVF, AVH, AVK, BDU, CEP,
NSM, PDE, PVQ, PVW, RIC and RMG, and I6, PVM, PVT and RID. Among them are the layer's producers of
dopamine (ADE, CEP, PDE), octopamine (RIC) and serotonin (NSM), so those cells appear in both halves
of the export: unsigned in `chemical`, as producers in `ligands`.

The committed per-connection configs behind recorded runs exclude `lopezcruz2019`. Without them the
16 edges fall to the rule (51 physiology, 1,449 rule), and each carries the citation key, so a
consumer can find them.

### Decision B: The records

- **`neurons`** and **`gap`** are the Wormlight export's records, built by the same functions: name,
  class, release identities with the primary first, and `ruleSign`; and each gap-junction pair once
  with `a < b`.
- **`chemical`**: `{"pre", "post", "sections", "sign", "signSource", "citation"}`, in the
  connectome's order. `sign` is 1, -1, or 0 for `signSource: "none"`; `signSource` is `physiology`,
  `expression`, `rule` or `none`; `citation` is the key of a physiology sign and null otherwise.
  Every record has all six keys.
- **`signCitations`**: `{"key", "citation"}` for each key of `PHYSIOLOGY_CITATIONS`, in the package's
  order. The overrides' evidence column, the quoted sentence behind each sign, is not carried: the
  sign function does not return it, and the file is in the provenance.

The `ruleSign` on each neuron and the per-connection `sign` sit side by side, so a consumer can read
where the table departs from the rule without re-deriving it.

### Decision C: No muscles

The export has no body wall muscles and no neuromuscular connections. This is a deliberate scope
choice: Wormturn models a navigation-level point agent, with no body to drive. The Wormlight export at
the same commit carries the muscles for any consumer that needs them.

### Decision D: The layer, carried as the files hold it

The layer is restructured, never transformed. Each list keeps its file's order.

| list | from | record |
|---|---|---|
| `ligands` | `layer.json` `ligands` (keyed by name, 56) | `{"name", "kind", "producers"}`; `producers` maps each cell to its source tags |
| `couples` | `layer.json` `couples` (121) | the file's object, verbatim: `id`, `ligand`, `receptor`, `cells` (cell to source tags), `sources`, `ec50`, `within_500nM`, `sign`, `sign_class` |
| `links` | `layer.json` `links` | verbatim; empty at this commit |
| `drives` | `layer.json` `drives` (1) | verbatim: `name`, `cells`, `source` |
| `receptorCouplingRule` | `receptor_couplings.json` `rule` (keyed by coupling, 4) | `{"coupling", "sign"}` |
| `receptorCouplings` | `receptor_couplings.json` `receptors` (keyed by gene, 54) | `{"receptor", "coupling", "sign", "evidence", "citation", "quote", "note", "verified"}` |

A key the files name keeps their spelling (`sign_class`, `within_500nM`), so a record compares with its
source field for field; a key the exporter adds follows the Wormlight export's camelCase. Text
outside ASCII is written as JSON escapes, as the Wormlight renderer writes it, and parses back to the
same strings.

What does not fit cleanly is kept as it is, and said here:

- **`verified` is not a flag.** It is `false` (32 receptors) or an object with the `date` and the
  `source` read (22). Every signed receptor is verified; three with a coupling and no sign, frpr-8,
  npr-22 and ntr-1, are unverified, npr-22 and ntr-1 with signs withdrawn on checking.
- **flp-24 has no producer.** Its ligand has empty `producers`, and its one couple, flp-24:npr-17,
  is kept. The package's layer loader drops both, since a ligand with no producer has no pool.
- **`ec50` is free text by source.** It is an empty object for the 17 Bentley-only monoamine couples.
- **Fitted couples have a null `sign`.** 16 couples are `fixed` (7 at +1, 9 at −1); 105 are `fitted`.
  `receptorCouplings` signs 22 of the fitted couples' 54 receptors. The export does not apply them to
  the couples, because the package applies one only when a brain names its receptor; the two lists
  sit side by side.
- **The fixed couples' signs** were written into `layer.json` from `receptor_signs.json` when the layer
  was built. The export reads only `layer.json`, so that file is not an input.

Not carried: `layer.json`'s `description`, `areas` (each neuron's body areas, kept for a later model),
`skipped_names` (empty) and `fetched` (the source downloads' digests), and `receptor_couplings.json`'s
`description`. None is part of the layer the package loads into a brain, and the provenance records
both files' digests. The export also does not apply the layer loader's options (the 500 nM
sensitivity, a source filter, the network's cells): it carries every couple and cell, and a consumer
applies its own. Every cell the layer names is one of the 302 neurons.

### Decision E: Reuse the Wormlight exporter, by import

The Wormturn script imports `scripts/export_wormlight.py`, which sits beside it, as other scripts
import their siblings. To make that clean, the Wormlight exporter:

- renames `_input` to `input_entry`, the path, digest and role of one input;
- moves its neuron and gap-junction records into `neuron_records` and `gap_records`;
- gives `render` a `record_lists` argument, the record keys after `schema` and `provenance`,
  defaulting to its own.

`git_state` is shared as it is. The Wormlight export writes the same bytes before and after, checked
on one tree. A third, shared module would layer more cleanly, but it would move more of the Wormlight
exporter for no difference in what either writes. The package is the wrong home, because an export
is one consumer's format.

Sharing the record functions ties the two exports' neuron and gap-junction shapes together. A change
to either would change both; the Wormturn tests pin those records' keys, so it would fail them and
need a new schema version.

### Decision F: Provenance

As the Wormlight export's: the exporter, the commit, whether tracked files had uncommitted changes, and
for each input its repository-relative path, SHA256 and role. The inputs, in order:

1. the Emmons 2024 S1 File: the wiring;
2. `connectome/neurons.py`: classes and release identities;
3. the Wang et al. 2024 atlas those identities were generated from;
4. `sign_overrides_physiology.csv`: the physiology signs;
5. the Fenyves et al. 2020 S1 Data;
6. the Fenyves et al. 2020 S5 Data: the expression signs;
7. `connectome/signs.py`: the sign procedure and the citation table, as committed values, in the way
   the Wormlight export lists `neurons.py`;
8. `layer.json`;
9. `receptor_couplings.json`.

The Cook 2019 file is not read, so it is not listed. Unlike the Wormlight export, the script refuses
an input that is a Git LFS pointer. The Emmons and Fenyves loaders already refuse one by digest, but
the atlas is only hashed, so without the check its pointer's digest would be recorded as the atlas's.

### Decision G: Deterministic, and refusing a dirty tree

As the Wormlight export: top-level keys in a fixed order, one compact record per line, the same bytes
from the same commit, and a refused dirty tree unless `--allow-dirty`, which records
`nematodeDirty: true`. Untracked files are ignored.

## Risks

- **The table moves when the physiology table does.** A new override changes an edge's sign and
  source with no schema change. The digests in the provenance say which table it was, and the export
  records the commit.
- **Two exporters, one set of helpers.** A change to the Wormlight exporter's helpers reaches this
  export. The Wormturn tests catch a change of shape; a change of value, such as a neuron's class,
  is a change to the data both exports carry, which is intended.
- **LFS.** On a clone without the spreadsheets the loaders refuse their pointers, and this script
  refuses the atlas's.
