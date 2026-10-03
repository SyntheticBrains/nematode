## Why

The connectome brain signs a chemical synapse by what its presynaptic neuron releases: acetylcholine
and glutamate excite, GABA inhibits. A synapse's sign is set by the postsynaptic receptor, so the rule
is wrong wherever one transmitter acts through receptors of both kinds. The food-sensing entry to the
klinotaxis circuit is the standing example: AWC is glutamatergic, so the rule signs AWC → AIY positive,
but AIY is inhibited through a glutamate-gated chloride channel (Chalasani et al. 2007). The roadmap
records this as a condition on Logbooks 044 and 067, and B.3 and the placed-plasticity MAY (M.1) both
need per-connection signs before they can run.

Wormlight, the sister project, already builds such a table. It combines a cited physiology table, the
expression-based predictions of Fenyves et al. 2020 (*PLoS Comput Biol* 16:e1007974, CC BY 4.0) and the
per-neuron rule. Phase 8b's re-plan scheduled vendoring it here as housekeeping item H.2, ahead of B.3.

## What Changes

### 1. Vendored data

- `data/connectome/fenyves_2020_s1_data.xlsx` and `fenyves_2020_s5_data.xlsx`, the publisher's S1 Data
  and S5 Data files byte for byte, through Git LFS by the existing rule.
- `data/connectome/sign_overrides_physiology.csv`, Wormlight's table of 51 cited per-connection signs.
  Wormlight is Apache-2.0 under the same maintainer, so it is vendored under this repository's licence
  with attribution.
- `PROVENANCE.md` entries in the house shape for all three.

### 2. A loader

`quantumnematode.connectome.signs.per_connection_signs()` signs every chemical edge by the first of four
steps that gives a sign: physiology; Fenyves's expression prediction, where the transmitter it rests on
is one of the cell's release identities in the atlas the package already reads; the per-neuron rule;
else no fast sign. It refuses a vendored file whose digest differs, a disagreement between the two
Fenyves files, an override naming an edge the wiring lacks, and an override with an unknown citation.

The vendored inputs are the primary sources, not Wormlight's derived export, so the table is re-derivable
here. A test pins that the derivation reproduces Wormlight's export at `1190b3e` on every edge.

## Capabilities

**Modified**: `connectome-substrate`, with one added requirement: per-connection chemical signs from
vendored sources.

## Impact

- `data/connectome/`: three files and their `PROVENANCE.md` entries
- `packages/quantum-nematode/quantumnematode/connectome/signs.py`: new; exported from the package
- Tests: the loader's counts, precedence, refusals and agreement with Wormlight
- Docs: `CHANGELOG.md`; the Phase 8 tracker (H.1 and H.2 ticked) and a roadmap note on C.1e

Nothing under `brain/`, `env/`, `agent/` or `configs/` changes. No brain reads the new table, so no
experiment's input changes; the configuration key that applies it belongs to the change that first
uses it (B.3 or M.1).

## Breaking Changes

None.
