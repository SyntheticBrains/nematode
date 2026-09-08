# Design: grounding synapse signs in the neurotransmitter atlas

## Context

`NEURON_CLASSIFICATION` maps each of 302 neuron names to `(cell_class, neurotransmitter)` and the
loader already passes the second element into `Neuron.neurotransmitter`; every entry is `None`.
Chemical weights are initialised `w[pre, post] = N(0, 1/√k)`, so sign and magnitude come from one
draw and the network is half inhibitory by construction. The three-factor rule's A.3 review
withdrew Dale's law precisely because those signs were noise.

Recon against the vendored-candidate atlas file (`elife-95402-supp2-v1.xlsx`, OpenWorm cect
mirror, downloaded and inspected 2026-09-08): sheet `Supp File 2`, 302 neuron rows, a curated
`Neurotransmitter(s)` column plus two unnamed columns carrying secondary identities, sixteen
reporter-allele columns behind them, and prior-report staining columns. Normalising the primary
column gives ACh 161, Glu 76, GABA 31, DA 8, orphan/unknown 16, betaine-uptake 4, GABA-uptake 2,
5-HT 2, octopamine 2.

## Goals / Non-Goals

**Goals**

- Populate the transmitter slot for all 302 neurons from a citable, vendored source.
- Ground the sign of every synapse whose pre-synaptic transmitter implies one, changing nothing
  else about initialisation.
- Make Dale's law available and measurable, as initialisation-only and as an enforced constraint.
- Ask, as a registered test, whether the sign structure alone changes the prior over policies or
  the Hebbian wiring contrast.

**Non-Goals**

- Per-synapse sign from post-synaptic receptors (B.3). A transmitter-only model is a documented
  approximation, not the finished biology.
- Any change to the rule, the modulator, or the panel protocols being re-run.

## Decisions

### D1. The vendored atlas

`data/connectome/elife-95402-supp2-v1.xlsx`, retrieved from
`https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/elife-95402-supp2-v1.xlsx`,
SHA256 `0013e4b5f366b82a6b0ec0d682c3bace4027545c957823841293de93feafc0e2`, 73,537 bytes. Same
mirror, licence and redistribution rationale as the Cook 2019 file already vendored there;
`PROVENANCE.md` gains an entry in the existing format citing Wang et al., *eLife* 13:RP95402
(2024), DOI `10.7554/eLife.95402`. LFS-tracked by the existing `data/connectome/**/*.xlsx`
pattern.

### D2. Normalisation and the sign table

The atlas's primary column carries editorial annotation. The loader normalises: strip a leading
`*`, strip a trailing `- NEW`, case-fold the `Unknown`/`unknown` variants, and split a
parenthetical qualifier from the base label. Two neuron names differ from the project's canonical
set — the atlas writes `DB1/3` and `DB3/1` for `DB1` and `DB3` — and are mapped explicitly, with
the mapping asserted in a test so a future atlas revision cannot silently drop them.

`(uptake)` marks a neuron that takes up a transmitter rather than releasing it and **does not**
give a release identity; `GABA (uptake)` and `betaine (uptake)` therefore ground no sign.

| normalised transmitter | sign | reasoning |
|---|---|---|
| ACh | excitatory | the animal's dominant fast excitatory transmitter |
| Glu | excitatory | excitatory through non-NMDA receptors; see the caveat |
| GABA | inhibitory | inhibitory at the great majority of sites; see the caveat |
| DA, 5-HT, octopamine, tyramine | unknown | modulatory rather than fast; the diffusible layer's business (B.3) |
| orphan / unknown / any uptake-only | unknown | no release identity |

**The caveat, stated in the code and the record.** A transmitter does not determine a synapse's
sign; the post-synaptic receptor does. Glutamate is inhibitory through glutamate-gated chloride
channels and GABA is excitatory at some neuromuscular junctions. This model is a per-neuron
approximation that B.3's receptor classes refine to a per-synapse one, and the test below is
registered as a question about the approximation, not a claim that it is the biology.

Coverage under this table, computed during recon: 268 of 302 neurons signed, grounding 3,176 of
3,709 chemical synapses (85.6%) — 2,962 excitatory (79.9%), 214 inhibitory (5.8%) — with 533
left on the random draw.

### D3. Sign grounding on the brain

`synapse_signs: Literal["random", "atlas"] = "random"`. Under `atlas`, the same draw is taken in
the same order from the same generator and the sign is replaced where the atlas gives one:
`w = |draw| · sign(pre)`, with the drawn sign kept for unknown sources. Magnitudes, the per-neuron
scale, the RNG stream and every other parameter are therefore untouched, so the arms differ in
sign structure and nothing else. `random` is the existing code path, byte-identical.

### D4. Dale's law

`enforce_synapse_signs: bool = False` on the plasticity mixin. When on, the rule projects each
plastic weight back onto its synapse's sign after the update and before the magnitude clamp:
positive synapses are floored at zero, negative synapses ceilinged at zero, unknown ones
unconstrained. Ordering is update → decay → **projection** → clamp → homeostasis, so the
homeostatic rescale acts on the projected weights and cannot reintroduce a forbidden sign.
Enforcement without grounding is refused at construction: signs must come from the atlas before
they can be law.

### D5. The registered test

Two campaigns, both re-runs of committed protocols with one key changed, compared against the
committed values rather than re-running the random-sign arms:

- **Prior sweep**: the frozen arms on both wirings under `synapse_signs: atlas`, seeds 1–64, 600
  episodes — panel 2's protocol exactly. Compared against panel 2's `wt_frozen` / `rn_frozen`
  per-seed table. Enforcement is irrelevant to a frozen arm and is not run. 128 runs.
- **Hebbian contrast**: the unmodulated-Hebbian arms on both wirings under grounded signs,
  init-only and enforced, seeds 1–16, 1000 episodes — panel 2's protocol exactly. Compared
  against panel 2's `wt_hebbian` / `rn_hebbian`. 64 runs.

About ninety minutes on 16 workers. The metric and statistics are the committed ones; pairing is
by seed against the committed tables.

**The family** (four one-sided paired tests, one BH-FDR family at α = 0.05):

| id | test | direction | reads |
|---|---|---|---|
| **G1** | grounded `wt_frozen` vs panel 2's random `wt_frozen`, seeds 1–64 | grounded > random | does the sign structure alone change the prior over untrained policies? |
| **G2** | grounded `wt_hebbian` vs grounded `rn_hebbian`, init-only, seeds 1–16 | wild-type > rewired | **the primary**: is the wiring contrast confirmable once signs are real? |
| **G3** | the same under enforcement | wild-type > rewired | does Dale's law change the contrast? |
| **G4** | enforced `wt_hebbian` vs init-only `wt_hebbian` | enforced > init-only | is sign enforcement a useful constraint on the drift? |

**Descriptive**: every grounded arm against its committed random-sign counterpart; the rewired
prior; the competent fractions and distributions of panel 2's protocol; the fraction of synapses
whose sign the rules flip under init-only (the diagnostic's 35–46%) against enforcement's zero;
mean absolute weight and saturation telemetry, because a mostly-excitatory network through tanh
units may saturate.

**The verdict map**, in order: `insufficient_seeds`; **`substrate_fail`** — the grounded frozen
arms' competent fraction is below a fifth of panel 2's random-sign 0.22, i.e. grounding breaks
the substrate rather than informing it, and nothing downstream is interpretable; then from G2 as
in panels 2–3: `specific_wiring`, `rewired_beats_wild_type`, `degree_statistics`,
`inconclusive`. G1, G3 and G4 annotate and never change the verdict.

`substrate_fail` is registered because it is a real possibility: an 80% excitatory network of
tanh units may settle at saturation and express nothing, and that outcome must be named in
advance rather than rationalised afterwards.

**Power, stated in advance.** G1 at n = 64 is well powered for a prior shift of a few points. G2
and G3 at n = 16 are the sample that failed to confirm this contrast in panel 2 and were replicated
inconclusively at n = 48 in panel 3; they are asked here only as "does grounding change the
answer", and a null is reported as unconfirmed at n = 16, not as evidence the signs did nothing.
The descriptive comparison against panel 2's committed values carries the rest.

### D6. Harness and records

`scripts/analysis/l4_atlas_signs.py`: the eight arm stems, per-arm seed ranges, panel 2's
committed per-seed table as the random-sign comparator, G1–G4 as one family, the verdict map
including `substrate_fail`, the descriptive layer and the sign-flip telemetry, per-seed CSV and
curves. Tested on synthetic values as its predecessors were.

## Risks / Trade-offs

- **The approximation may be the story.** A transmitter-only sign model can be wrong in exactly
  the places that matter (GluCl-inhibited pathways). B.3's receptor layer is the fix and this
  change is explicitly its first half; the record will say so.
- **A mostly-excitatory network may saturate.** Registered as `substrate_fail` with telemetry.
- **n = 16 on a contrast that has already failed twice.** Framed as a change-detection question
  with the prior sweep at n = 64 carrying the weight.
- **Enforcement interacts with homeostasis.** Ordering is pinned in D4 and tested directly.

## Open Questions

None.
