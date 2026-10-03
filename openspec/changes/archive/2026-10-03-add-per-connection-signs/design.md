## Overview

This change makes per-connection chemical signs available to the package without changing anything a
brain reads. It vendors the primary sources rather than Wormlight's derived export, ports Wormlight's
derivation, and pins that the two agree.

## Decisions

### Decision A: Vendor the sources, not the derived table

Wormlight's export carries a sign and a source on every chemical edge, and copying it would be one file.
But it is a derived artefact, so its provenance chain would run through another repository's build
rather than ending at the publisher's bytes. Vendoring the publisher's Fenyves files by digest, plus the physiology table,
keeps the derivation reproducible from this repository alone, which is the house standard
(`PROVENANCE.md`). Wormlight's export is used once, as a cross-check: a test compares every edge, and
names the 23 where this table deliberately differs (Decision B).

### Decision B: Wormlight's precedence, with one correction found at review

Physiology, then expression, then rule, then none. The expression step is used only where every
transmitter Fenyves's prediction rests on is one of the presynaptic cell's release identities in Wang et
al. 2024, the atlas the package's classification table already carries; otherwise the prediction is set
aside and the edge falls through to the rule. Where both Fenyves files predict a sign for one edge they
must agree, and a disagreement is an error rather than a choice.

Fenyves et al. name a primary and sometimes a secondary transmitter per cell, and their polarity counts
receptor matches for both. Wormlight checks only the primary against the atlas. The branch review found 67
kept predictions whose secondary is not one of the cell's identities; recomputed from the primary alone,
44 give the same polarity and 23 give none, so those 23 signs rested entirely on a release the atlas does
not record. This table sets them aside — the module's own stated rule — so they fall to the per-neuron
rule, and 11 of them change sign. They include AVA → AVD and AVB → AVD in the command circuit. The
difference from Wormlight is those 23 edges exactly, and the test names them.

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
| expression | 1,676 | 1,291 | 385 | 10,962 |
| rule | 1,449 | 1,324 | 125 | 7,098 |
| none | 533 | — | — | 2,055 |

342 edges carry a sign opposite to the per-neuron rule's: 32 from physiology and 310 from expression. The
S1 sheet predicts for 3,516 of the edges and names 122 others the wiring lacks; the S5 sheet predicts for
3,237 and names 5 others; 3,117 edges are in both.

## Risks

- **Two sign systems in one package.** A reader could take the rule's signs for these or the reverse. The
  module docstring states the difference, and nothing reads the new table unless asked.
- **The physiology table will grow.** Wormlight's sign audit added rows over time. A row added there is
  not added here until vendored again, so the cross-check test names the Wormlight commit it compares
  against rather than tracking it.
