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
| **head scope** — both endpoints in the table's neuron set | 1,386, of which 23 are self-loops |
| head scope the table can cover (no diagonal) | 1,363 |
| **covered** | 1,049 (28.3% full, 77.0% of coverable head scope); 635 positive, 414 negative |
| on a Cook gap junction only | 265 — reported, not applied |
| on no Cook connection | 697 — reported, not applied |
| onto the 39 body motor neurons | 0 |

The repository has no notion of a head, so head scope is the table's own neuron set. Cook 2019 has
38 self-loops and the table has no diagonal, so the 23 self-loops inside head scope are reported apart
from the coverable denominator rather than counted as misses. Gap junctions are treated as undirected
when a table entry is checked against them. Gap-junction entries are not applied because this brain's gap junctions are fixed constants; whether measured
values belong there is B.2b's question.

### Decision D: Variance-matched values, with a multiplier as the pin

The values are dynamics coefficients at 2 Hz on calcium signals, median magnitude about 0.01, in no
unit this rate model shares. Using them raw would need an arbitrary scale and would start a measured
arm at a very different overall magnitude from its random comparator, confounding structure with
size. So each covered value is multiplied by the per-post-neuron `1/sqrt(in-degree)` the random draw
uses, by `measured_weight_scale` (default 1.0), and by **one constant** chosen so that over the wild
type's covered edges the placed values' RMS equals the draw's *expected* RMS on those same edges. One
constant keeps the fitted values' relative sizes and every sign; the expectation rather than a
realised draw keeps it independent of the seed. B.1b sweeps the multiplier, which is D17's pin.

*(**Corrected during implementation, 2026-09-23.**)* As first specified, the values were divided by
their own RMS and *then* scaled per neuron. Measured on constructed brains, that left the covered
edges **1.275 times** the draw's expected magnitude — deterministic, not seed noise (1.19-1.37
realised over eight seeds). The large fitted values sit disproportionately on neurons with few inputs,
whose `1/sqrt(in-degree)` is large, so normalising the values alone does not normalise the weights.
The constant now has the per-neuron scale inside it, which is what "the measured arm has the random
arm's magnitude" was meant to guarantee; a test pins it.

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

### Decision F: The rewired null receives each neuron's wild-type values, under every prior

A degree-preserving rewiring keeps labelled endpoints, so most of its `(pre, post)` pairs do not exist
in the wild type and a name-keyed lookup would miss them. D17 resolves this with A.1's second
definition of shared initialisation, and it has to be defined for all three measured priors, because
B.1c's 2×3 runs each of them on both wirings.

**Both orders are stated.** For each post-synaptic neuron, its wild-type incoming values are taken in
order of their own wild-type pre-synaptic index and placed on the null's incoming edges in
pre-synaptic-index order, the first *k* edges where *k* is its wild-type covered count; the rest keep
the draw. What is placed depends on the prior:

| prior | placed on the null's first *k* edges |
|---|---|
| `measured` | the neuron's wild-type normalised measured values |
| `measured_shuffled` | the values the global permutation assigned to its wild-type edges |
| `measured_signs` | the signs of its wild-type measured values, each on the receiving edge's own draw magnitude |

In-degree is preserved, so every neuron receives exactly its wild-type multiset.

**The wild type is computed before rewiring.** The brain rewires its connectome before building the
topology (`connectome_ppo.py:2217-2222`), so the topology only ever sees the rewired graph. The
per-neuron wild-type values — including the permutation, which draws from the dedicated draw
generator exactly as on the wild type — are therefore computed from the table and the **unrewired**
connectome first, and handed to the topology. A rewired brain and a wild-type brain at one seed use
identical wild-type values.

### Decision G: The shared generator yields what it always has

A.1's defect was a mode that took a different number of values from a generator the rollout buffer
also consumes. The edge loop therefore still takes exactly one `rng.normal` per edge whatever the
prior, the prior overwrites afterwards, and the shuffle draws only from the dedicated draw generator.
A test asserts the shared generator ends where it started under every mode.

### Decision G2: A multiplier no prior reads is refused

`measured_weight_scale` is read only by `measured` and `measured_shuffled`. Under `random` and
`measured_signs` it would be accepted, validated and ignored — an arm that looks swept and is not,
which is exactly what the requirement A.2 added forbids. A non-default multiplier is refused under
those two priors, at validation and at construction.

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
