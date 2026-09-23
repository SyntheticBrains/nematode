## Overview

B.1a lands the measured-weight data and the switch that reads it. Roadmap decision **D17** is
authoritative for what B.1 must support; this document records what the data turned out to be, the
licence findings, and the design decisions taken with the maintainer before implementation.

## Design Decisions

### Decision A: Vendor Creamer's fitted weights; cite Randi, do not vendor it

| source | licence found | decision |
|---|---|---|
| Creamer, Leifer & Pillow, `quick_start_examples/model_weights.csv` in `Nondairy-Creamer/Creamer_LDS_2026` | **MIT**, the repository's single LICENSE ("Copyright (c) 2026 Matthew S. Creamer"); no separate data licence | vendor, MIT notice beside it |
| Creamer model pickles, ~20 MB each | MIT | not vendored: unpickling executes code, and the CSV holds what B.1 reads |
| Creamer OSF deposit `qxhjd` (reformatted Randi recordings) | **none stated** | not vendored |
| Randi et al. 2023, OSF `e2syt` (the primary atlas) | **none stated** | not vendored |
| Randi atlas as `funatlas.h5` in `wormneuroatlas` | GPL-3.0 | not vendored: GPL data in an Apache-2.0 repository |

That the MIT licence covers the weight file is an interpretation — the repository has one LICENSE,
worded for "the Software", and the file ships inside it. It is the only licensed route to the data,
and the record says so. Randi is cited in `PROVENANCE.md` as Creamer's raw source; nothing in B.1c
reads it directly. The roadmap and tracker say Randi is vendored, and each carries a dated correction.

### Decision B: Read the table at load time, SHA-checked

The transmitter atlas is an LFS spreadsheet that can be a pointer in a fresh clone, so its values are
generated into a committed literal. This table is 42 KB of plain text in plain git — `.gitattributes`
routes only `*.xlsx` through LFS under `data/connectome/` — so it is always present, and reading it
directly with a pinned SHA256 is simpler and loses nothing. The loader refuses a file whose digest
differs, which is the property the literal's round-trip test exists to give the atlas.

### Decision C: Coverage is what the join finds, reported at two scopes

The table is fitted on a White 1986 + Witvliet 2020 mask that unions chemical and gap-junction edges,
so it is joined to Cook 2019's chemical edges by `(pre, post)` name and the result reported rather
than assumed:

| | edges |
|---|---|
| Cook chemical, full scope | 3,709 |
| **head scope** — both endpoints in the table's neuron set | 1,386 |
| **covered** | 1,049 (28.3% full, 75.7% head); 635 positive, 414 negative |
| on a Cook gap junction only | 265 — reported, not applied |
| on no Cook connection | 697 — reported, not applied |
| onto the 39 body motor neurons | 0 |

The repository has no notion of a head, so head scope is the table's own neuron set. Gap-junction
entries are not applied because this brain's gap junctions are fixed constants; whether measured
values belong there is B.2b's question.

### Decision D: Variance-matched values, with a multiplier as the pin

The values are dynamics coefficients at 2 Hz on calcium signals, median magnitude about 0.01, in no
unit this rate model shares. Using them raw would need an arbitrary scale and would start a measured
arm at a very different overall magnitude from its random comparator, confounding structure with
size. So covered values are divided by their **RMS** over the wild type's covered edges (sign kept),
multiplied by the per-post-neuron `1/sqrt(in-degree)` the random draw uses, and by
`measured_weight_scale`, default 1.0. At the default the measured and random arms share magnitude;
B.1b sweeps the multiplier, which is D17's pin.

### Decision E: Four modes, with the sign-only arm built now

| mode | covered edge | uncovered edge |
|---|---|---|
| `random` | the draw | the draw |
| `measured` | normalised value × scale × multiplier | the draw |
| `measured_signs` | `abs(draw)` × measured sign | the draw |
| `measured_shuffled` | `measured`, values permuted among the wild type's covered edges | the draw |

`measured_signs` is B.1b's sign-only arm; building it here keeps B.1b to configs and lets one test
suite assert all four modes touch only the chemical weights. The shuffle is global over covered edges,
as D17 words it, drawn from the brain's dedicated draw generator.

### Decision F: The rewired null receives each neuron's wild-type multiset

A degree-preserving rewiring keeps labelled endpoints, so most of its `(pre, post)` pairs do not exist
in the wild type and a name-keyed lookup would miss them. D17 resolves this with A.1's second
definition of shared initialisation: for each post-synaptic neuron, the multiset of its wild-type
incoming measured values is assigned to its incoming edges on the null **in pre-synaptic-index
order**, the first *k* edges where *k* is its wild-type covered count, the rest keeping the draw.
In-degree is preserved, so every neuron receives exactly its wild-type multiset. Because it is keyed by
post-synaptic neuron it needs nothing from the rewiring beyond the graph it produced.

### Decision G: The shared generator yields what it always has

A.1's defect was a mode that took a different number of values from a generator the rollout buffer
also consumes. The edge loop therefore still takes exactly one `rng.normal` per edge whatever the
prior, the prior overwrites afterwards, and the shuffle draws only from the dedicated draw generator.
A test asserts the shared generator ends where it started under every mode.

### Decision H: Refuse the pairings nothing has tested

A non-`random` prior is refused with `synapse_signs: atlas` (two sign sources for one edge), with a
non-default `weight_draw`, and with `weight_init: count_scaled` — at validation and again at
construction, because `model_copy`, which the campaign runner uses, skips validators.

## Risks

- **Measured weights may leave the klinotaxis pathway unlearnable** (Lee 2026, registered on B.1).
  B.1b's sign-only pilot is where that would show.
- **The table is fitted on another connectome.** Coverage is reported, not assumed, and 962 of its
  2,011 entries have no Cook chemical edge to land on.
- **A preprint.** The weights come from an unreviewed fit; the record never lets them stand alone.
