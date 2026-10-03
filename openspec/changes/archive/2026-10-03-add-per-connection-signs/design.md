## Overview

This change makes per-connection chemical signs available to the package without changing anything a
brain reads. It vendors the primary sources rather than Wormlight's derived export, ports Wormlight's
derivation, and pins that the two agree.

## Decisions

### Decision A: Vendor the sources, not the derived table

Wormlight's export carries a sign and a source on every chemical edge, and copying it would be one file.
But it is a derived artefact from a private repository, so its provenance chain would end at a commit
nobody else can read. Vendoring the publisher's Fenyves files by digest, plus the physiology table,
keeps the derivation reproducible from this repository alone, which is the house standard
(`PROVENANCE.md`). Wormlight's export is used once, as a cross-check: a test compares every edge.

### Decision B: The precedence is Wormlight's, unchanged

Physiology, then expression, then rule, then none. The expression step is used only where the transmitter
Fenyves's prediction rests on is one of the presynaptic cell's release identities in Wang et al. 2024,
the atlas the package's classification table already carries; otherwise the prediction is set aside
(40 edges) and the edge falls through to the rule. Where both Fenyves files predict a sign for one edge
they must agree, and a disagreement is an error rather than a choice. Changing any of this would break the
comparability the roadmap asks the two projects to keep.

### Decision C: Read the cached formula values, and refuse a copy without them

Every prediction cell in the Fenyves sheets is a spreadsheet formula. The loader reads the values the
publisher's file was saved with. A re-saved copy whose cache is empty would read as blank predictions and
silently become the rule; the digest check refuses it first, and the polarity check refuses any value
the sheet's formula cannot produce.

### Decision D: No brain key in this change

The tracker asks for the table to be loadable beside the per-neuron rule and byte-identical-when-off.
Nothing reads it, so every committed configuration is unchanged by construction. How a brain applies
these signs — as fixed signs on drawn magnitudes, as an enforcement constraint, or only on the circuit
M.1 places plasticity on — is a design question for the change that first uses it.

### Decision E: The table is per wiring

`per_connection_signs()` takes a connectome and signs its chemical edges, the Cook 2019 hermaphrodite by
default. The Emmons 2024 release has the same chemical edges, so it gives the same table. A rewired null
has edges the sources say nothing about, so the loader signs whatever edges it is given, and an override
naming an absent edge is refused; what a null's signs should be is the using change's decision.

## Pinned figures

Over Cook 2019's 3,709 chemical edges (20,965 sections):

| source | edges | + | − | sections |
|---|---|---|---|---|
| physiology | 51 | 19 | 32 | 850 |
| expression | 1,699 | 1,301 | 398 | 11,048 |
| rule | 1,426 | 1,307 | 119 | 7,012 |
| none | 533 | — | — | 2,055 |

353 edges carry a sign opposite to the per-neuron rule's: 32 from physiology and 321 from expression. The
S1 sheet predicts for 3,516 of the edges and names 122 others the wiring lacks; the S5 sheet predicts for
3,237 and names 5 others; 3,117 edges are in both.

## Risks

- **Two sign systems in one package.** A reader could take the rule's signs for these or the reverse. The
  module docstring states the difference, and nothing reads the new table unless asked.
- **The physiology table will grow.** Wormlight's sign audit added rows over time. A row added there is
  not added here until vendored again, so the cross-check test names the Wormlight commit it compares
  against rather than tracking it.
