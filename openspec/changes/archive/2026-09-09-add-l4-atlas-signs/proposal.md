# Grounding synapse signs in the neurotransmitter atlas (7a-ii B.1)

## Why

Four registered panels and one probe (Logbooks 040–043 and the clone-destruction diagnostic)
found the wild-type wiring inert or worse under every learning rule tried, and the roadmap's
2026-09-08 reframing named the likely reason: as modelled, the wiring is a sparse graph with
**random synapse signs**, no receptor biology, no neuromodulation and rate units without
intrinsic dynamics — close to a degree distribution, which is what Logbook 034 measured. The
fidelity ladder makes each restored piece of the wiring's biology a registered rung of the same
question. This change is the first rung and the first item of 7a-ii.

Every one of the 302 entries in the project's neuron table carries an unpopulated
`neurotransmitter` slot, plumbed through the loader since Phase 6 and never filled. The Wang et
al. 2024 CRISPR knock-in neurotransmitter atlas fills it: a per-neuron release identity for all
302 hermaphrodite neurons, mirrored as machine-readable Supplementary File 2 in the same
MIT-licensed OpenWorm ConnectomeToolbox repository this project already vendors Cook 2019 from.
Recon against that file confirms exact 302-neuron coverage (one naming difference to map) and
that a transmitter-only sign rule would ground **85.6% of the 3,709 chemical synapses** — 79.9%
excitatory, 5.8% inhibitory — leaving 533 (monoaminergic, orphan and uptake-only sources) on
today's random draw.

That is a large, biologically-directed change to the substrate: today every sign is a coin flip,
so the network is half inhibitory by construction; grounded, it becomes the mostly-excitatory
network the animal is. It also makes Dale's law enforceable for the first time. The rule review
withdrew Dale's law in A.3 on the explicit grounds that constraining random signs would only
freeze noise; grounding removes that objection, and the diagnostic showed both local rules flip
35–46% of synapse signs, so enforcement is a candidate consolidation constraint in its own right.

Ratified with Chris 2026-09-08: a transmitter-only sign model first with the receptor layer
deferred to B.3; both initialisation-only and enforced-during-plasticity as arms; and the
registered test re-running panel 2's frozen prior sweep and the panels 2–3 Hebbian wiring
contrast under grounded signs.

## What Changes

- **Vendored data**: `elife-95402-supp2-v1.xlsx` under `data/connectome/` with a full
  `PROVENANCE.md` entry (source URL, SHA256, retrieval date, licence rationale, citation), LFS
  tracked like its siblings.
- **A curated sign table**: a loader that normalises the atlas's transmitter column (stripping
  its `*` and `- NEW` annotations, mapping `DB1/3`/`DB3/1` to `DB1`/`DB3`, distinguishing uptake
  from release), populates `NEURON_CLASSIFICATION`'s transmitter slot for all 302 neurons, and
  derives a documented per-transmitter sign: acetylcholine and glutamate excitatory, GABA
  inhibitory, monoamines and orphans **unknown**.
- **Sign grounding on the connectome brain**: `synapse_signs`, `random` (default,
  byte-identical) or `atlas` — each chemical weight keeps the magnitude of today's draw and takes
  its pre-synaptic neuron's sign where the atlas gives one; unknown sources keep the drawn sign.
- **Dale's law**: `enforce_synapse_signs`, default off — when on, every plastic update is
  projected back onto its synapse's sign before the bound is applied.
- **The registered test**: panel 2's 64-seed frozen prior sweep and the Hebbian wiring contrast
  re-run under grounded signs, against the committed random-sign values, as a four-test BH-FDR
  family with a verdict map that includes the substrate failing outright.
- Configs, a harness, the launch record, the runs, records under `supporting/044-l4-atlas-signs/`,
  tests, docs.

Out of scope: receptor classes and per-synapse sign resolution (B.3), the diffusible layer, any
rule change, the consolidation mechanism (B.4).

## Capabilities

**Modified**: `connectome-substrate` (transmitter identities and the vendored atlas),
`connectome-ppo-brain` (sign grounding and Dale's law), `l4-plasticity-panel` (the registered
sign-grounding test).

## Impact

- New: the vendored atlas and its provenance entry, an atlas loader module, four configs, an
  analysis harness, the supporting directory. Edited: `connectome/neurons.py` (302 transmitter
  slots), `connectome/loader.py`, `brain/arch/connectome_ppo.py`, `learning_rules/three_factor.py`
  (the sign projection), `docs/architectures.md`, `configs/README.md`, `CHANGELOG.md`.
- Defaults are byte-identical: `synapse_signs: random` takes today's code path, and the populated
  transmitter field is metadata no default build reads.
