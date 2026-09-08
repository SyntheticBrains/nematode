# 044: Grounding Synapse Signs in the Neurotransmitter Atlas (7a-ii B.1 / Phase 7)

**Status**: completed — **`degree_statistics`** under the pre-registered verdict map, with the
registered `substrate_fail` outcome **not** triggered. This is the fidelity ladder's first rung:
four panels had read the wild-type connectome as inert under every local rule, and the ladder's
premise was that the substrate, not the rule, might be why — as modelled, every synapse sign was
a coin flip, so the network was half inhibitory where the animal is mostly excitatory. Grounding
3,176 of the 3,709 chemical synapses in the Wang et al. 2024 CRISPR atlas, with magnitudes,
incoming norms, readout, gains and RNG stream untouched, answers that: the prior over untrained
policies is essentially unchanged (wild-type competent fraction **0.23** against the committed
**0.22**, G1 **+0.8** with the interval spanning zero), the wiring contrast is neither rescued
nor reversed (G2 **+4.9**, q = 0.55), and reward-free Hebbian learning gets **substantially
worse** — wild-type **31.5 → 14.0**, rewired null **17.4 → 9.1**, both intervals clear of zero.
Enforcing Dale's law through plasticity makes the wild-type worse again (G4 **−4.8**, reverse),
and the endpoint telemetry says why: the unenforced rule ends with about a seventh of grounded
synapses carrying the opposite sign to their transmitter, and enforcement converts those
attempted flips into **silences** rather than into respect for the sign. The rung's answer is
that **the substrate's random signs were not what limited the rule**; the failure is the rule's,
and a purely potentiating rule on an 80%-excitatory network has no inhibitory brake.

**Branch**: `feat/l4-atlas-signs` (PR #329).

**Date**: 2026-09-09.

**OpenSpec change**: `add-l4-atlas-signs` (archived; vendored atlas and provenance, the curated
sign table, `synapse_signs` and `enforce_synapse_signs` on the connectome brain, the registered
test; extends capabilities `connectome-substrate`, `connectome-ppo-brain`,
`l4-plasticity-panel`).

## Objective

Ask, as a registered test rather than an assumption, whether the modelled substrate's arbitrary
synapse signs were what kept the wild-type wiring from being legible to a local rule. Concretely:
populate the neuron table's release identities from the atlas, derive per-synapse signs from
them, and re-run panel 2's frozen prior sweep and the panels 2–3 Hebbian wiring contrast under
grounded signs against the committed random-sign values — with Dale's law, now enforceable for
the first time, as its own pair of arms.

## Background

Logbooks [040](040-l4-panel.md)–[043](043-l4-warm-start.md) and the
[clone-destruction diagnostic](supporting/043-l4-warm-start/destruction-diagnostic.md) resolved
the same way four times: the local rule beats none of its floors, outcomes are fixed points set
by the random initial weights, and the rule takes a cloned competent policy apart at a
near-constant drift. The roadmap's 2026-09-08 reframing named the candidate confound. As
modelled, the wiring is a sparse graph with random synapse signs, no receptor biology, no
neuromodulation and rate units without intrinsic dynamics — close to the degree distribution
Logbook 034 measured. Each restored piece of biology becomes a rung of the same question, and
signs are the first because the neuron table has carried an unpopulated `neurotransmitter` slot
since Phase 6 and the atlas fills it for all 302 hermaphrodite neurons.

Dale's law had been **withdrawn at A.3's spec review** on the explicit grounds that constraining
arbitrary signs would only freeze noise. Grounding removes that objection, and the diagnostic's
finding that both local rules flip 35–46% of signs made enforcement a candidate consolidation
constraint in its own right. Ratified with Chris 2026-09-08: transmitter-only signs first
(receptor classes deferred to B.3), both initialisation-only and enforced arms, and panel 2's
protocol as the test.

## Hypothesis

Pre-registered before either campaign ran
(`supporting/044-l4-atlas-signs/launch.md` committed first). Four one-sided paired tests
corrected together under BH-FDR at α = 0.05: **G1** grounded wild-type frozen over the committed
random-sign wild-type frozen on the same seeds (does the prior move); **G2** grounded wild-type
over grounded rewired null under the Hebbian rule (the primary — is the wiring contrast rescued);
**G3** the same contrast under Dale's law; **G4** enforced over unenforced grounded wild-type
Hebbian (does enforcement help).

The verdict is assigned in order: `insufficient_seeds`; then **`substrate_fail`** when the
grounded frozen arms' competent fraction falls below half the committed random-sign value —
registered in advance so that a substrate broken by grounding would be named rather than
rationalised, with nothing downstream of it interpretable; then from G2 alone as
`specific_wiring`, `rewired_beats_wild_type`, `degree_statistics` or `inconclusive`. G1, G3 and
G4 annotate the verdict and can never change it.

## Method

**What was grounded.** The Wang et al. 2024 atlas (eLife 13:RP95402, Supplementary File 2) was
vendored from the same MIT-licensed OpenWorm mirror this project already takes Cook 2019 from,
with a full provenance entry (URL, SHA256, retrieval date, licence, citation). A loader
normalises the transmitter column — stripping the `*` and `- NEW` annotations, mapping `DB1/3`
and `DB3/1` to `DB1` and `DB3`, distinguishing uptake from release — and a generator script
writes the identities into the neuron table, with a `--check` mode that fails if the table drifts
from the atlas. **280 of 302 neurons** carry a release identity (ACh 161, Glu 76, GABA 31, DA 8,
5-HT 2, octopamine 2; 22 remain unknown). The sign rule is documented and deliberately narrow:
acetylcholine and glutamate excitatory, GABA inhibitory, monoamines and orphan or uptake-only
identities **unknown**.

**How it enters the brain.** `synapse_signs: atlas` keeps each chemical weight's drawn magnitude
and takes its pre-synaptic neuron's sign where the atlas gives one; unknown sources keep the sign
they drew. That grounds **3,176 of 3,709 synapses** — 2,962 excitatory, 214 inhibitory — leaving
533 on the draw, and takes the network from **48% inhibitory to 13%**. Magnitudes, per-neuron
incoming norms, the readout, the gains and the RNG stream are identical to the random-sign build,
so the arms differ in sign structure alone. `enforce_synapse_signs` adds Dale's law: every
plastic update is projected back onto its synapse's sign after decay and before the bound, so the
clamp still holds last. The sign vector is a wiring buffer and the sign model is recorded in the
saved training state, so a checkpoint from one sign model is refused by the other rather than
silently reinterpreted.

**The runs.** Prior sweep: `wt_frozen_atlas`, `rn_frozen_atlas`, seeds 1–64, 600 episodes
(enforcement is irrelevant to a frozen arm and was not run) — 128 runs. Hebbian contrast:
`wt_hebbian_atlas`, `rn_hebbian_atlas`, `wt_hebbian_dale`, `rn_hebbian_dale`, seeds 1–16, 1000
episodes — 64 runs. Every value is panel 1's registered pin, unchanged; there was no pilot. The
comparators are panel 2's committed per-seed table and **no random-sign arm was re-run**. The
single registered extension for a run the plateau detector marks non-converged is a fresh run at
1.5× replacing the shorter log; **eight were applied** (`wt_frozen_atlas` 35, 48;
`rn_frozen_atlas` 9, 11, 24, 27; `wt_hebbian_atlas` 2; `rn_hebbian_atlas` 4), and the family and
verdict were unchanged by them. All 192 runs exited 0 with no tracebacks.

## Results

### The registered family (BH-FDR α = 0.05)

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| G1 | grounded wt frozen − random wt frozen | +0.77 | −2.06 … +3.60 | 0.260 | 41/64 | fail |
| G2 | grounded wt − rn Hebbian (primary) | +4.87 | −1.70 … +11.80 | 0.551 | 8/16 | fail |
| G3 | the same under Dale's law | −2.47 | −8.51 … +3.56 | 0.986 | 7/16 | fail |
| G4 | enforced − unenforced wt Hebbian | −4.75 | −8.89 … −1.27 | 0.986 | 4/16 | **reverse** |

**Verdict: `degree_statistics`** — G2 does not confirm and its interval spans zero. The substrate
gate **did not fire**. Annotations: `prior_changed` false, `prior_worsened` false,
`contrast_holds_under_dale` false, `enforcement_helps` false.

### What grounding did to each arm

Plateau-tail full-clear success (%), against panel 2's committed values on the same seeds.

| arm | grounded | panel 2, random signs | paired Δ (descriptive) |
|---|---|---|---|
| wt_frozen_atlas | mean 12.0, median 7.3, competent 0.23 | mean 11.3, median 3.7, competent 0.22 | +0.8 (CI −2.1 … +3.6, 41/64) |
| rn_frozen_atlas | mean 8.7, median 3.3, competent 0.12 | mean 8.0, median 2.3, competent 0.16 | +0.8 (CI −2.0 … +3.4, 32/64) |
| wt_hebbian_atlas | mean 14.0, median 6.6, competent 0.25 | mean 31.5, median 24.6, competent 0.56 | −17.5 (CI −29.3 … −4.9, 6/16) |
| rn_hebbian_atlas | mean 9.1, median 4.4, competent 0.12 | mean 17.4, median 15.2, competent 0.31 | −8.3 (CI −14.9 … −1.0, 6/16) |
| wt_hebbian_dale | mean 9.2, median 2.4, competent 0.12 | mean 31.5, median 24.6, competent 0.56 | −22.3 (CI −33.6 … −10.3, 5/16) |
| rn_hebbian_dale | mean 11.7, median 6.8, competent 0.19 | mean 17.4, median 15.2, competent 0.31 | −5.7 (CI −11.1 … −0.4, 7/16) |

### What the rules did to the sign structure

Read from each run's own auto-saved endpoint, which carries both the final weights and the sign
buffer they were grounded to. Two quantities kept apart, because they are what separates the
enforced arms from the rest: **violated** is the share of grounded synapses ending with the
opposite sign, **silenced** the share driven to exactly zero.

| arm | violated (mean / max) | silenced (mean / max) |
|---|---|---|
| wt_frozen_atlas | 0.000 / 0.000 | 0.000 / 0.000 |
| rn_frozen_atlas | 0.000 / 0.000 | 0.000 / 0.000 |
| wt_hebbian_atlas | **0.153** / 0.330 | 0.000 / 0.000 |
| rn_hebbian_atlas | **0.136** / 0.274 | 0.000 / 0.000 |
| wt_hebbian_dale | 0.000 / 0.000 | **0.131** / 0.306 |
| rn_hebbian_dale | 0.000 / 0.000 | **0.135** / 0.262 |

## Analysis

- **The substrate is fine, and the prior barely moves.** Taking the network from 48% inhibitory
  to 13% leaves the distribution of untrained policies where it was: the wild-type competent
  fraction is 0.23 against the committed 0.22 and G1 is +0.8 with the interval spanning zero. The
  registered `substrate_fail` outcome did not occur, which matters because it was the outcome
  that would have made everything downstream uninterpretable. What did change is the *shape*: the
  wild-type median doubles (7.3 against 3.7) while the mean holds, and the rewired null's
  competent fraction falls (0.12 against 0.16). Real signs leave the animal's own wiring where it
  was and cost its scramble something — descriptive, and the direction the wiring hypothesis
  would predict, but not a confirmed contrast.
- **Reward-free Hebbian learning gets substantially worse, and that is this panel's real
  finding.** The wild-type Hebbian arm falls from 31.5 to 14.0 and the rewired from 17.4 to 9.1,
  both intervals clear of zero. The mechanism is legible: a purely potentiating co-activity rule
  on a network that is 80% excitatory has no inhibitory brake. Random signs had handed the rule
  an accidentally balanced substrate in which drift could settle on useful fixed points; the
  animal's sign structure removes that balance and the rule has nothing to replace it with.
  Biological circuits pair excitatory wiring with anti-Hebbian and inhibitory plasticity (Perks
  et al., *Nature*, 2026-09-02); this rule has neither.
- **Dale's law does not supply the missing brake — it silences synapses.** Enforcement takes the
  wild-type Hebbian arm from 14.0 to 9.2 and G4 reverses. The endpoint telemetry explains it. Left
  free, the rule ends with about a seventh of grounded synapses carrying the opposite sign to
  their source's transmitter: it does not respect Dale's law and never did, on random signs or
  real ones. Enforcement holds every sign exactly as registered, and the cost sits in the same
  row — it converts those attempted flips into **silences**, switching roughly the same seventh
  of synapses off. Dale's law as a constraint does not make the rule respect the biology; it
  removes the synapses the rule wanted to invert.
- **The wiring contrast is neither rescued nor reversed.** G2 is +4.9 with the interval spanning
  zero, against panel 2's +14.1 at the same seeds, itself unconfirmed. The estimate is smaller
  and no better resolved. Grounded signs do not move this question, so the verdict is
  `degree_statistics` for the fifth time.

## Conclusions

- **The first rung's answer is negative in the informative direction.** The substrate's random
  signs were not what limited the rule. Grounding them leaves the prior alone and makes the
  learning worse, so the rule's failure is the rule's.
- The result gives the next rule variant a sharp, falsifiable prediction: on a grounded,
  mostly-excitatory substrate, a rule with an anti-Hebbian or decorrelating term should recover
  what the purely potentiating rule loses here. That is a stronger test than the one the variant
  would otherwise have faced.
- Enforcing a biological constraint on an unbiological rule does not make the rule biological. It
  is worth knowing that Dale's law is *available* on this substrate now; it is not a consolidation
  mechanism.
- The receptor layer (B.3) is worth having for fidelity and should not be expected to rescue
  learning on its own.
- Sign grounding is now a config flag on a substrate that is otherwise byte-identical, so every
  later rung can be run with signs on without re-deriving anything.

## Limitations

- The sign model is transmitter-only. A synapse's actual sign depends on the post-synaptic
  receptor, and some cholinergic and glutamatergic synapses are inhibitory in the animal; those
  are counted excitatory here. B.3's receptor layer is the correction, and it will move some of
  the 2,962 excitatory synapses to inhibitory.
- 533 synapses (monoaminergic, orphan and uptake-only sources) keep the sign they drew, so the
  grounded substrate is not fully grounded.
- n = 16 on the Hebbian arms, matching panel 2's protocol by design; G2's interval is wide and
  the same bimodal outcome that defeated panels 2–3 is present here. G1's 64 seeds are the
  well-powered part of this panel, and it is the one that answers the rung's question.
- The comparators are a committed table, not a concurrent re-run. That is what the registration
  specified, and it means any drift in the surrounding code between panel 2 and now would appear
  as an effect; the substrate build is byte-identical under `synapse_signs: random`, which is the
  guard against it.
- The endpoint telemetry reads the auto-saved final weights, so it describes where each run
  ended, not the trajectory it took to get there.

## Next Steps

**B.4 consolidation**, with the anti-Hebbian and decorrelating rule variant **pulled forward
ahead of structured routing (B.4b)** on this panel's evidence — the missing inhibitory brake is
now a measured gap, not a hypothesis, and the variant has a prediction to be held to. Any variant
is cleared through the [clone assay](supporting/043-l4-warm-start/destruction-diagnostic.md#the-clone-assay)
before a panel is run. B.3's receptor layer stays queued as fidelity work with its expectations
lowered accordingly.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-09-add-l4-atlas-signs/`;
  capabilities `openspec/specs/connectome-substrate/spec.md`,
  `openspec/specs/connectome-ppo-brain/spec.md`, `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the panel produced:
  [supporting/044-l4-atlas-signs/](supporting/044-l4-atlas-signs/details.md) — `launch.md`,
  `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`, `details.md`.
- Vendored data and provenance: `data/connectome/` (`PROVENANCE.md`).
- Tooling: `connectome/neurotransmitters.py`, `connectome/neurons.py`,
  `scripts/generate_neuron_transmitters.py`, `brain/arch/connectome_ppo.py`,
  `learning_rules/three_factor.py`, `scripts/analysis/l4_atlas_signs.py`; arm configs under
  `configs/scenarios/foraging_predator_thermal/`.
