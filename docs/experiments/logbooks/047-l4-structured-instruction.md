# 047: Structured Instruction — Routing the Third Factor Through the Wiring (7a-ii B.4b / Phase 7)

**Status**: completed — **`no_routing_effect`** under the pre-registered verdict map, and the last
rule item of 7a-ii. Every panel before this one broadcast a single reward-prediction error to every
plastic synapse; the standing argument against that, from the roadmap's reframing and from the
electrosensory-lobe connectome, is that credit assignment is a property of the circuit rather than
a number in the air. The atlas made the alternative derivable: 18 aminergic release neurons synapse
onto **169 of 302 neurons**, covering **71.1%** of the wild-type's chemical synapses. Routing the
modulator to exactly those synapses, with the unmodulated Hebbian term everywhere else, made **both
wirings worse** — the wild type by 3.2 points, its scramble by 7.3 — and the damage sits in the
good seeds: the best wild-type seed falls from 70.5 to 24.3 and competent seeds from 3 of 16 to 1
of 16. Routing did not lower a floor; it removed a ceiling. **This panel must be read against the
rule's positive control**, which the rule failed the same day: it measures what a non-learning rule
does under two routing regimes, not whether aminergic pathways matter in the animal. Two results
outlast the verdict — the substrate has not drifted across six panels, verified bit-for-bit, and a
prediction committed before the run was right about the outcome and wrong about its direction.

**Branch**: `feat/l4-structured-instruction` (PR #337).

**Date**: 2026-09-10.

**OpenSpec change**: `add-l4-structured-instruction` (archived; the co-transmitter reader, the
derived pathway, the routed third factor and the registered test; extends capabilities
`connectome-substrate`, `learning-rules`, `l4-plasticity-panel`).

## Objective

Test whether the third factor's *global scalar* form was what limited four panels' worth of null
results, by routing it through the wiring that carries the modulatory signal and comparing against
the broadcast form on both wirings at n = 16.

## Background

[Logbook 044](044-l4-atlas-signs.md) grounded the substrate's synapse signs and found the wiring
contrast unrescued; [045](045-l4-consolidation.md) screened three consolidation mechanisms and none
held a cloned competent policy; [046](046-l4-decorrelation.md) built the decorrelating term 044
predicted and found it could not be built at transmitter-only fidelity. Structured,
pathway-specific instruction was the last mechanism in 7a-ii's queue, added after Logbook 040 and
Perks et al. (*Nature*, 2026-09-02), where anti-Hebbian depression sits at identified sites and the
wiring decides which synapses receive instructive input.

## The pathway, and a loader bug it exposed

Deriving the pathway required reading which neurons release an amine, and that surfaced a real gap
in the sign-grounding work. The atlas sheet's "Neurotransmitter(s)" heading spans **three** columns
and the vendored reader took only the first, so ADF and HSN read as purely cholinergic and RIM as
purely glutamatergic while their serotonergic and tyraminergic release identities sat unread. With
every identity column read, the aminergic release set is **18 neurons**, not 12:

| source | neurons | reach |
|---|---|---|
| dopamine | `ADEL/R`, `CEPDL/R`, `CEPVL/R`, `PDEL/R` | 104 |
| serotonin | `NSML/R`, `ADFL/R`, `HSNL/R` | 84 |
| octopamine | `RICL/R` | 31 |
| tyramine | `RIML/R` | 31 |
| **all four** | **18** | **169 of 302** |

Keyed on the post-synaptic neuron — a modulator gates a synapse by acting on the cell that owns it
— that is **2,636 of 3,709 synapses (71.1%)**. Uptake-only annotations (`AIM`, `RIH`), the atlas's
own "alternative synthesis" hedges (`I5`, `VC4`, `VC5`), its precursor entry (`MI`) and its
male-only note (`PVW`) are excluded by a stated rule; counting them would give 24 neurons and
74.1%. Primary identities and every grounded synapse sign are unchanged by the extra columns, which
is asserted by test.

**The pathway is a proxy and the record says so.** Aminergic transmission in this animal is
substantially extrasynaptic — released by volume onto receptors expressed by cells that need not be
synaptic partners (Bentley et al., *PLoS Comput Biol*, 2016). Wired reach is a lower bound and a
modelling choice, so a negative here refutes that proxy rather than structured instruction, and the
receptor layer is what would replace it.

## Hypothesis

Pre-registered before any run (`supporting/047-l4-structured-instruction/launch.md` committed
first). Four one-sided paired tests corrected together under BH-FDR at α = 0.05: **S1** wild-type
routed over wild-type global (the primary); **S2** the same on the rewired null; **S3** and **S4**
the wiring contrast under each routing mode.

Verdict in order: `insufficient_seeds`; **`no_routing_effect`** when neither S1 nor S2 confirms;
then `routing_helps_both`, `routing_helps_wild_type_only`, `routing_helps_rewired_only`. S3 and S4
annotate and never decide. **`routing_helps_both` was named in advance as not a win**: a routed
third factor helping the scramble as much as the animal would be a fact about running two learning
regimes in one network, not about *C. elegans* connectivity.

## Method

Four arms — `wt_pathway`, `rn_pathway`, `wt_global`, `rn_global` — seeds 1–16 paired, the panel's
3000-episode plastic budget, plateau-tail full-clear success, one registered extension of a fresh
run at 1.5× for a run the plateau detector marks non-converged. 64 runs.

Instructed synapses see the modulator; everywhere else the third factor is `1.0`, which is the
**unmodulated Hebbian floor's** update. The arm is therefore an interpolation between two arms the
panels already measured and introduces **no hyperparameter** — the split is read off the wiring.

The global arms were **re-run concurrently** rather than read from panel 1's committed table, whose
plastic arm carries eight seeds at this budget; panel 1's values serve as a consistency check and
never as a comparator.

## Results

### The registered family

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| S1 | wt routed − wt global | −3.18 | −9.74 … +2.83 | 0.975 | 8/16 | fail |
| S2 | rn routed − rn global | −7.30 | −11.77 … −3.01 | 0.975 | 6/16 | fail |
| S3 | wt − rn under routing | +0.87 | −3.21 … +5.02 | 0.975 | 8/16 | fail |
| S4 | wt − rn under the global scalar | −3.24 | −10.85 … +5.19 | 0.975 | 6/16 | fail |

**Verdict `no_routing_effect`** — and the name understates what was seen. All four tests are
one-sided, but they ask different questions. **S1 and S2** ask whether routing *helps*, each arm
against its own broadcast control, so a large negative effect produces a high q and reads as "no
confirmed benefit" rather than "no effect"; descriptively the effect was negative on both wirings
and S2's interval lies entirely below zero. **S3 and S4** are the wiring contrast in its registered
direction — wild type over rewired null, under routing and under the global scalar respectively —
and neither confirms; S4's estimate is negative, meaning the scramble scored above the animal at
n = 16. The verdict is kept as registered rather than renamed after the fact; the
honest reading is **no confirmed benefit, with observed degradation**, and a future map of this
shape should carry an explicit harm branch.

### Per arm

| arm | mean | median | max | competent (≥20%) | instructed fraction | instructed share |
|---|---|---|---|---|---|---|
| wt_global | 10.8 | 3.2 | 70.5 | 3/16 | — | 1.000 |
| wt_pathway | 7.7 | 4.5 | 24.3 | 1/16 | 0.711 † | 0.924 † |
| rn_global | 14.1 | 6.0 | 52.9 | 4/16 | — | 1.000 |
| rn_pathway | 6.8 | 3.1 | 33.1 | 1/16 | 0.781 † | 0.904 † |

† Averaged over the runs whose telemetry exports were retained — 10 of 16 and 7 of 16 respectively.

## Analysis

- **Routing removed ceilings rather than lowering floors.** The medians barely move (3.2 → 4.5 on
  the wild type, 6.0 → 3.1 on its scramble) while the maxima collapse and the competent counts fall
  to one seed each. Whatever the routed arm did, it did it to the runs that were working.
- **The intervention was real.** The instructed *share* of the update's magnitude is 0.92 against an
  instructed *fraction* of 0.71, so the routed synapses carry disproportionately more of the
  learning than their count suggests. This is a null with something having happened, not a no-op.
- **The rewired null derives a wider pathway and lost more.** Degree-preserving rewiring spreads the
  aminergic neurons' targets over more of the network — 78.1% against 71.1% — so the scramble
  received *less* of the unmodulated Hebbian term and fell further. Descriptive, and the opposite of
  what the pre-run prediction expected.
- **The wiring contrast points the wrong way and does not confirm**, at −3.24 under the global
  scalar. Across six panels it has now been positive, null and negative without ever confirming —
  which is what one would expect of a measurement made with an instrument that does not learn.

## The substrate has not drifted in six panels

The registered consistency check compares the concurrently re-run global arms against panel 1's
committed values on the eight seeds they share:

| arm | panel 1 committed | re-run here | difference |
|---|---|---|---|
| wt_global | 17.82778 | 17.82778 | 0 |
| rn_global | 11.2 | 11.2 | 0 |

**Bit-for-bit.** Everything added since panel 1 — the trace substrate, sign grounding, three
consolidation mechanisms, two decorrelating terms, the routing seam — is byte-identical on its
default path, as each change asserted individually and as nothing had checked end to end until now.
Every committed comparator in Logbooks 040–046 rests on the substrate it was written against. This
is the most reusable thing the panel produced.

## The prediction, and where it failed

Committed before the runs: `no_routing_effect` at ~65%, `routing_helps_wild_type_only` at ~15%,
`routing_helps_both` at ~12%. **The verdict was right and the direction was wrong.** The argument
was that panel 1's committed table shows the unmodulated Hebbian floor beating the modulated arm
(30.2 against 17.8 on the wild type), so replacing the modulator with `1.0` on ~29% of synapses
should move the arm toward the better floor — a weighted blend of about **+3.6**. Routing instead
cost 3.2 points, and cost the scramble 7.3.

The error was assuming outcomes combine linearly in the mixing fraction. They do not: running two
learning regimes in one network is not a weighted average of running each alone, and on a bimodal
outcome the arithmetic of means says almost nothing about what happens to the seeds that were
working. Recorded because a prediction committed in advance should be scored in public, including
when it is wrong.

## Conclusions

- **Routing the third factor through the aminergic wiring does not rescue the rule**, and at this
  fidelity it makes things worse on both wirings.
- **The result's weight is limited by the instrument.** The rule failed its positive control the
  same day, so this panel characterises what a non-learning rule does under two routing regimes.
  Structured instruction is not refuted by it; the synaptic-reach proxy for aminergic instruction,
  applied to *this rule*, is.
- **The last mechanism in 7a-ii's queue is now spent.** Sign grounding, consolidation,
  decorrelation and structured instruction have all been built and tested, and none moved the
  wild-type connectome off its floors — which is what made the instrument the thing to question.
- The atlas reader's missing co-transmitter columns were a real defect in the sign-grounding work,
  found only because this rung needed the amines. It changed no synapse sign, but it would have
  silently limited any later work keyed on release identity.

## Limitations

- The pathway is aminergic reach by **synaptic connectivity** and a lower bound; expressed-receptor
  reach is what B.3 would supply, and a negative here does not speak to it.
- The verdict map had no harm branch, so a substantial negative effect is reported under a name that
  reads as "no effect". Fixed in the prose here; a future map should fix it in the map.
- The instructed fraction and share are means over the runs whose exports were retained (10 and 7 of
  16), not over the arms.
- n = 16 with a bimodal outcome that one seed can dominate — the sixth panel in a row where that is
  true, and still without a registered statistic matched to it (block I's I.2).
- Seven runs were lost to a mid-campaign branch switch and one log to a bad copy; all were re-run,
  the analysis refused the incomplete arm both times rather than scoring it, and the panel as scored
  is 16 seeds per arm with nothing imputed.

## Next Steps

**Block I** — the instrument. I.0's positive control has already run and the rule failed it, so
**I.1**, an eligibility with the exploration noise inside it, is the critical path; it clears on the
MLP yardstick before any connectome arm. I.2's statistic and graded metric come before any further
panel, and I.4's re-read will state which of Logbooks 040–046 and this one survive as findings about
the wiring. B.3's receptor layer and B.5's panel stay queued behind block I.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-10-add-l4-structured-instruction/`;
  capabilities `openspec/specs/connectome-substrate/spec.md`,
  `openspec/specs/learning-rules/spec.md`, `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the test produced:
  [supporting/047-l4-structured-instruction/](supporting/047-l4-structured-instruction/details.md)
  — `launch.md` (the derived pathway, the exclusion rule, the family, the verdict map and the
  extrasynaptic caveat, all before the runs), `panel.json`, `per-seed.csv`, `curves.csv`,
  `_manifest.txt`, `details.md`.
- The rule's positive control, which this panel is read against:
  [supporting/048-l4-rule-positive-control/](supporting/048-l4-rule-positive-control/details.md).
- Tooling: `connectome/neurotransmitters.py`, `connectome/neurons.py` (`NEURON_CO_TRANSMITTERS`),
  `brain/arch/connectome_ppo.py`, `learning_rules/three_factor.py`,
  `scripts/analysis/l4_structured_instruction.py`; arm configs under
  `configs/scenarios/foraging_predator_thermal/`.
