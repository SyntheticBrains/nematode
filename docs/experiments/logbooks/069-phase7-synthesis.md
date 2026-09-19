# 069: Phase 7 Synthesis — What the Connectome's Wiring Turned Out to Be Worth (SPLIT close)

**Status**: **Phase 7 CLOSES as SPLIT.** The flagship MUST is **unmet and was unmeetable within the
closing scope**: no biologically plausible rule that *writes* the connectome learns this substrate to
any benefit, so "does plastic wild-type beat plastic rewired-null" has no interpretable answer here.
What the phase delivered instead is three citable results and a diagnosed negative:

1. **The wiring is learning-speed-relevant under gradient descent.** Under PPO the wild-type
   connectome reaches competence **35.4%** and **23.5%** sooner than its degree-preserving rewired
   null on two cells, replicated on fresh rewirings at **+55.3%** and **+40.1%**
   ([057](057-wiring-premise-contrast.md), [058](058-wiring-premise-difficulty.md),
   [065](065-wiring-fresh-rewiring.md)). **Standing condition: rewiring varies with initialisation in
   every one of those panels**, and the control that separates them is Phase 8's first act.
2. **The wiring is not legible as fixed features through a four-class readout, and this is
   rate-robust.** L.0 found no advantage ([064](064-l4-frozen-features.md)); the pooled null leads or
   levels at both learning rates since tested, so that null does not depend on the pin that moved the
   next result.
3. **Widening the readout to one weight per motor neuron makes the wiring legible — at one learning
   rate only.** L.1's interaction is **+0.2818** at `plasticity_rate` 0.001
   ([066](066-l4-readout-width.md)) and **−0.0657** a decade below it, a three-way of **+0.3475** on
   81 of 96 seeds ([068](068-l1b-rate-calibration.md)). **Standing condition: the positive is a
   positive at 0.001.** At 0.0001 the dominant effect is capacity, not wiring.
4. **The negative has a diagnosed cause, not an absence.** Node perturbation at 302 units is near its
   theoretical worst case; the minimal three-factor rule never passed a positive control; e-prop
   reaches competence **with the chemical matrix frozen**, and every arm that writes it does worse by
   3.9 to 15.9 foods ([048](048-l4-rule-positive-control.md), [060](060-l4-perturbation-scale.md)–[063](063-l4-eprop.md)).

**Every exit criterion carries a status below, from a fixed five-word vocabulary.** Nothing is left
unmarked, and *deferred* is not used where the honest word is *unreachable*.

**Date**: 2026-09-19.

**Scope**: Phase 7 in full — 7a-i (the D10 panel), 7a-ii (the rule programme, the instrument block,
block V, the substrate ladder), and 7b (deferred). Rolls Logbooks 040–068 into the exit-criterion
walkthrough, the terminal reading of the pre-registered 2×2, and what Phase 8 opens on.

**OpenSpec change**: `add-phase7-synthesis` (extends `plasticity-evaluation`: a close assigns every
criterion a status from a fixed vocabulary; a shipped result with an uncontrolled confound carries it
as a standing condition).

## Objective

Phase 7 asked one load-bearing question, registered as a 2×2 before any rule ran:

> *Under a biologically plausible three-factor rule — a member of the rule family available to the
> real animal — does the wild-type wiring become load-bearing? Does plastic wild-type beat plastic
> rewired-null?*

This synthesis states the answer, assigns every exit criterion a status, and separates what the phase
established from what it merely failed to detect.

## The five-status vocabulary, and why it matters

| status | what it asserts |
|---|---|
| **met** | the criterion's own test ran and passed |
| **unmet-with-reason** | the test ran and did not pass; the diagnosis is named |
| **deferred-with-destination** | not attempted, still a live question, destination named |
| **superseded-by-result** | not attempted, and a committed result removed the question |
| **unreachable-with-reason** | not attempted, and nothing in the phase could have made it attemptable |

*Deferred* and *superseded* are not interchangeable. A deferred criterion is still a question; a
superseded one is not. This synthesis's own first draft called the learnable-gap-junction ablation
*superseded* and the review caught it — one of that criterion's two recorded destinations is
untouched — so the distinction is load-bearing rather than decorative.

## Exit-criterion walkthrough

### Required (MUST)

| criterion | status | evidence |
|---|---|---|
| **7a-i** — substrate, rule seam, minimal three-factor rule, D10 2×2 panel | **met** (the panel ran; its verdict was negative) | [040](040-l4-panel.md): `sanity_floor_fail`. The criterion asked that the panel *resolve the hypothesis*, and it did |
| **7a-ii** — the rule programme and the fidelity ladder | **unmet-with-reason**, shipped as SPLIT | [059](059-7a-shipment.md)'s SPLIT decision. No rule that writes the wiring learns the cell; the ladder resumed with a *reading* learner instead. The diagnosis is [063](063-l4-eprop.md) |
| **7b** — cross-species transfer | **deferred-with-destination**: the phase after 7 | D14, 2026-09-15. Its named learner does not exist, and its scaffold puts a more artificial readout at the centre than the one that does the learning |

**The D10 primary is unmet and was unmeetable within the closing scope.** Its four cells require a
*plastic* wild-type and a *plastic* rewired null. Every learner in this phase that reaches competence
leaves `w_chem` frozen, so the two plastic cells were never populated by a competent policy. This is
not a null result on the hypothesis; it is the absence of an instrument that could have tested it, and
[063](063-l4-eprop.md) is why.

### Recommended (SHOULD)

| criterion | status | reasoning |
|---|---|---|
| 6a preprint submitted in the 7a-i window | **superseded-by-result** — recorded **cancelled** 2026-09-15 | Not deferred: D13's case for staking 6a early was a fast-moving field, and Phase 7's own results make a stronger combined package than 6a alone. The publication decision is relooked after this close |
| Panel 2, the Hebbian wiring contrast | **met** (verdict `inconclusive`) | [041](041-l4-panel2.md) |
| Imitation-warm-start arm | **met** (verdict `sanity_floor_fail` + `rule_destroys_clone`) | [043](043-l4-warm-start.md) |
| Dauer-connectome pathfinder condition | **deferred-with-destination**: the phase after 7, with 7b's comparative core | D14. Listed here on its own line rather than inherited from the block header |
| Homology-mapped weight-transplant transfer | **deferred-with-destination**: same | D14 |
| Species-appropriate third behaviour | **deferred-with-destination**: same | D14 |
| **Learnable-gap-junction ablation (D4)** | **deferred-with-destination, narrowed**: a PPO arm on the block-V cells | B.6's recorded status names two destinations. [063](063-l4-eprop.md) closes the first — every substrate-writing arm does worse, so no local rule makes a further plastic tensor promising — and [067](067-l4-feature-ablations.md) lowers its priority further, since *removing* gap junctions helped both wirings. **The second destination is untouched**: under PPO the wiring is learning-speed-relevant, gap junctions are frozen there, and making them plastic is unattempted |
| **Co-primary biological validation** (dopamine-gated forgetting; Leifer navigation re-weighting) | **unreachable-with-reason** | Both are predictions *about a plastic wiring*. They need a rule that writes the connectome to some benefit in order to produce a sign- or shape-level prediction, which [063](063-l4-eprop.md) established does not exist in this rule family and which D14 removed from the plausibility claim. Under a frozen substrate the model has no re-weighting to compare against the data. Calling this *deferred* would imply a schedule it has never had |

### Optional (MAY) — marked, not gates

| criterion | status |
|---|---|
| Biological-validation collaboration; ≥ 1 prediction tested against lab data | **unreachable-with-reason**, for the co-primary's reason |
| Journal submissions in flight | **deferred-with-destination**: the post-close publication decision |
| Spiking-STDP arm; neuromorphic deployment | **deferred-with-destination**: Future Directions, unscheduled |
| Reproducibility artefacts current | **met** — with a limit stated in § Reproducibility: every headline figure is re-derivable from the committed per-seed CSVs, while the pre-readout-width exports and V.3's tracked-experiment records are gone |

## The terminal reading of the pre-registered 2×2

Against the claim-discipline bars fixed before the phase ran:

- **The D2 primary is unmet**, and no result in the closing scope can convert it, because the learner
  that works does not write the wiring.
- **Every Phase 7 result is a performance claim**, not a dynamics or biology claim. None clears the
  ensemble-invariance and named-neuron-grounding bars, and none is offered as clearing them.
- **What the phase settled instead** is a different and narrower question than the one it registered:
  *is the wiring legible to a learner that reads it rather than writes it?* The answer is **yes, but
  only at a particular readout width and a particular learning rate** — which is a statement about the
  learner's operating point as much as about the wiring.

**The firmest part of the terminal reading is the null.** At the four-class pooled readout the rewired
null leads or levels at **both** learning rates tested — −0.0966 at 0.001, −0.0320 at 0.0001 — so
L.0's `wiring_is_inert_as_features` does not depend on the pin that reversed L.1. Where this phase has
a robust structural finding, it is negative, and it extends [034](034-connectome-structure-controls.md)'s
degree-statistics verdict to a third learning regime.

## The three shipped results, each with its standing condition

**1. The wiring is learning-speed-relevant under PPO.** +35.4% on a thermal cell, +23.5% on a hard
food-only cell, replicated on fresh rewirings at +55.3% and +40.1%, all four efficiency metrics at
q = 0.000, both learning gates full, and the untrained prior indistinguishable.
**Standing condition, stated in the same sentence as the claim: `rewire_seed` is unset in every one of
those panels, so each seed's rewired graph derives from its run seed and rewiring varies with
initialisation.** [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports the fly connectome's
apparent advantage dissolving under shared initialisation plus a degree-preserving null — the same
control, on the same kind of claim, in another organism. This is now the phase's most exposed result,
and the control is Phase 8's first act rather than a residual caveat.

**2. The wiring is not legible as fixed features through the four-class pool.** No advantage at the
registered ≥ 20% bar, with both wirings learning and the substrate verified frozen.
**Standing condition: the point estimates lean the other way** — the null competent on a median 405
episodes against 568 — recorded as a lean, post-hoc in direction, not as a result.

**3. Widening the readout makes the wiring legible at 0.001.** Interaction +0.2818, the wiring
effect's sign flipping between widths, no width main effect detected.
**Standing conditions: the rate, and the features.** At 0.0001 the interaction is −0.0657 and the
wild type leads at neither width; and the two feature ablations both read `carries_the_effect` for
reasons the word does not convey — removing gap junctions *helped both wirings* with the null catching
up, and grounding the signs reduced learnability on both at a rate where the null was already ahead.

## The negatives, with their causes

The phase's negative results are diagnoses, not absences, and this is the part most likely to be
misread:

- **The minimal three-factor rule never learned anything.** Its positive control — a task with a
  closed-form optimum — failed, twelfth in the sequence rather than first
  ([048](048-l4-rule-positive-control.md)). Seven registered panels had already run against it.
- **Node perturbation's feasibility was knowable in advance.** Learning speed scales as roughly 1/N in
  perturbed units; at 302 units with 2400-step episodes the substrate was near a worst case, and no
  perturbation set from 1208 down to 39 rescued it ([060](060-l4-perturbation-scale.md), [061](061-l4-reduced-perturbation.md)).
- **e-prop learns the cell, and the control says the wiring did not do it.** 17.570 foods of 20 at
  52.61% full clear, on the arm whose chemical matrix is **frozen**; letting the rule write the wiring
  costs 3.895 foods ([063](063-l4-eprop.md)).
- **A re-read confirmed what the negatives are evidence about.** Of 32 registered contrasts, ten are
  instrument findings, sixteen are substrate findings the instrument block does not reach, and five
  are about neither ([056](056-l4-ladder-reread.md)). No committed verdict changed.

## Limitations

**Three settings were never examined under the learner that works.** L.1b showed an inherited pin can
set the *sign* of a registered primary, which makes these conditions rather than footnotes:

| pin | value | examined where | why it could matter here |
|---|---|---|---|
| `forward_pass_depth` | 4 | **nowhere** — quoted as a fact, never varied | it defines the reservoir's features: at depth 4 only neurons within four hops of a sensory injection reach the motor pool, so it partly determines *which* wiring the learner can see |
| `trace_decay` | 0.9 | on the three-factor rule's control only | it decays the **readout's own** eligibility trace |
| `initial_log_std` | −1.0 | swept by hand **under the rule** (action-noise std 0.368–1.0, null) | [062](062-l4-frozen-readout.md) found the readout's *scale* costs more than its direction against a fixed action noise, and left "the readout's parameterisation or the fixed `initial_log_std` beside it" as the open question, calling it cheap to ask |

**The wide readout never had its own positive control.** The phase protocol requires one of a new
component, and the 78-parameter readout got gates instead: +16.2 and +16.0 foods over its own frozen
floors, and 80.3% / 87.3% full clear at the calibrated rate. That is strong de facto evidence and it
is not the registered article. It is defensible here because the component is a linear readout trained
by its own exact gradient, whose failure mode is not subtle — but it is recorded as a gap.

**One cell, one task, one organism.** Every result above is `hard350` or its thermal sibling, foraging
under chemical gradients, in *C. elegans*. Nothing here licenses a claim about other behaviours.

## Reproducibility

`campaigns/`, `exports/` and `experiments/` are all gitignored, so the durable record is what sits in
git. Stated plainly:

- **Re-derivable from git alone**: every headline figure in every logbook from [040](040-l4-panel.md)
  onward, from the committed per-seed CSVs under `supporting/`.
- **Needs the local campaign logs**: any re-derivation from raw. The logs still parse, and the
  analysis harnesses read them.
- **Gone**: the step-level exports for every campaign before the readout-width era, and the
  tracked-experiment records for V.3's registered panel. Four parsed fields now read as absent there.

**This already cost something measurable.** During L.1b's pilot, `peak_action_density` could not be
compared against L.0's rate-check run because that run's export had been deleted — recorded at the
time as uncompared rather than counted as matching.

*(Added 2026-09-19, after this record's first draft.)* **The surviving raw artefacts are now retained
outside the repository.** The maintainer has archived the campaign logs and the remaining exports
privately; they are **not public artefacts** and form no part of this repository's reproducibility
surface. They exist so a re-derivation from raw stays possible for the campaigns whose exports
survived. What is listed as *gone* above is unchanged: the pre-readout-width exports and V.3's
tracked-experiment records were already deleted before the backup was taken.

## What Phase 8 opens on

In dependency order, each item naming what it inherits:

1. **The init-vs-rewiring control** — block V's standing condition, and the one published critique
   aimed at this design. Needs its own design decision first: "the same initialisation" has no single
   meaning once the mask changes, since the init scale is `1/sqrt(chemical in-degree)` and rewiring
   preserves the degree sequence but not which neuron holds which degree.
2. **A calibration rung** over rate, readout width, `forward_pass_depth` and `initial_log_std`, before
   any new contrast. Inherited from L.1b, which showed one pin setting a primary's sign, and from
   [062](062-l4-frozen-readout.md), which already named the log-std question and called it cheap.
3. **The dynamics rung** — gap-junction coupling as a dynamical term rather than a fixed symmetric
   matrix. Inherited from [067](067-l4-feature-ablations.md), and it absorbs D4's surviving PPO
   destination. Sits behind (2).
4. **The placed-plasticity rung** — the rule at an anatomically identified site against a
   degree-stratified random subset of the same size. Its three confounds are already specified.
5. **L.2, intrinsic dynamics** — the carried SHOULD, now with (2) as a precondition.
6. **Methodology consolidation** — 42 requirements in `plasticity-evaluation`, most of them
   single-use rules from this phase, folded into the phase protocol so a rung designer reads twelve
   principles rather than 33 rules; and the byte-identity requirement renamed to parsed-field
   identity, which is what it checks. *(The rename landed early, 2026-09-19, ahead of the
   consolidation it was listed beside.)*
7. **The publication decision**, deferred to after this close by S.1's cancellation. The package is
   real: a replicated wiring advantage under gradient descent, a rule-programme negative with a
   diagnosed cause, and an operating-point finding about legibility. A referee reads Dhiman, so item 1
   is the difference between a paper and a rebuttal.

## Conclusions

1. **Phase 7's registered question could not be answered, and the reason is a result.** No rule in the
   plausible family writes this connectome to any benefit. That is a finding about the rule family and
   about node perturbation's scaling, established with a positive control that should have run first.
2. **The wiring is not inert, but what it is worth depends on how it is read.** Under gradient descent
   it buys learning speed. Under a small local readout it buys nothing through four pooled classes,
   robustly; something through 39 per-neuron weights, at one rate; and nothing a decade below that
   rate.
3. **The phase's most exposed result is its strongest one**, because rewiring and initialisation still
   vary together in every block V panel. Closing that is Phase 8's first act, not a footnote.
4. **The claim discipline held.** Every result is a performance claim, no dynamics claim was made, no
   committed verdict was rewritten, and three results this phase carry standing conditions in the same
   sentence as their claims rather than in a caveat section.

## Artefacts

- Logbooks [040](040-l4-panel.md)–[068](068-l1b-rate-calibration.md), each with its committed per-seed data under `supporting/`
- [`docs/research/phase-protocol.md`](../../research/phase-protocol.md) — the twelve principles this phase paid for
- `openspec/changes/archive/` — every registered change, with its pre-run design and its dated amendments
