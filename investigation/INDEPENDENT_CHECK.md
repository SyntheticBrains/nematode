# Independent check

Written 2026-11-14. A fresh agent that had not seen the brief, the maintainer's seeds, the decisions
log or the reviews was given RECOMMENDATION.md, COMPARISON.md and SPIKE_REPORT.md, with access to
the evidence they cite. It was asked for the strongest argument against the recommendation. Its
answer follows unedited. My response is in the separate section after it.

## The answer, unedited

**The case against the recommendation: stay, and replace H1 to H3 with one test that takes minutes**

The strongest argument is that the neural track's chain of reasoning, H2 then H3 then "the import lifts the readout", runs against the evidence this investigation collected itself. H2 is aimed at the one metric where a pass is cheap and says almost nothing about H3. I lead with this because it does not argue about cost estimates. It cuts the link that is supposed to give every milestone of the track value to the project.

**1. The track's payoff rests on an effect the field says is absent.** H1 is description and can never be an objective, so the only step that changes the project's model is H3: import a perturbation-fitted head coupling into C.2. That pays off only if a connectome-constrained fit captures causal coupling better than an unconstrained one. LANDSCAPE.md § 1.8 records the published answer:

- On perturbation responses, a fully connected model does as well as the masked one (Creamer).
- On whole held-out worms, connectome-constrained and dense models score within 0.01 to 0.02 of each other (Mi).
- The connectome shows clearly in only one place: predicting withheld neurons.

**2. H2 is aimed at exactly that one place, and its kill criteria barely bite.**

- Its "must beat the dense model" criterion is close to vacuous. Mi's dense model scored 0.08 on withheld neurons, against 0.32 for the connectome model.
- Its degree-null criterion asks whether a neuron's activity is better imputed from its anatomical partners than from random partners of the same degree. That is a fact about which neurons covary. It is not evidence that a fitted coupling gets causation right, and H3 depends on causation.
- So H2 can pass and leave H3's premise exactly where Creamer left it.

**3. H2's decision value for Phase 10 is close to zero.**

- If H2 passes, the project waits for licensed data that outside authors may never supply, then rents compute at a cost no one has measured.
- If it fails, the recommendation becomes "stay, plus H1".
- Either way, no Phase 10 step changes.

**4. The cost lands on the scarcest resource, the maintainer.**

- H2 is 5 to 8 sessions plus a fresh review.
- The spike needed 6 maintainer decision rounds and ran no fit.
- Phases 6a to 9 each closed without a positive on their primary question that held.
- C.2's stopped third pilot is waiting on the same maintainer, and W cannot run at its registered size.
- The rule "pause if a Phase 10 step slips two weeks" cannot measure attention diverted before the slip becomes visible.

**5. The "not stay" leg rests on a reading that cannot bear it.**

- C.2 scored 0.053 and its unfitted wiring 0.065, with overlapping 95% intervals (0.035 to 0.072, and 0.047 to 0.084). MA_C2_RESULT.md itself says the two "are not told apart here". That is absence of evidence, not evidence that behaviour fitting fails to buy neural fidelity.
- The probe runs four settling iterations from rest, with modulators at zero, a network with no slow dynamics, compared against responses integrated over 30 s.
- The measured arrays were built from the unlicensed, fuller record. The recommendation refuses that record for H3 but accepts it for H1, which is an inconsistent standard.

**6. H1 as specified repeats the trap the project has hit twice.**

- Its null is a joint relabelling of rows and columns. That destroys neuron identity entirely, so any model that roughly matches which neurons are strongly connected or strongly responsive will beat it.
- "The connectome's structure alone carries some signal" is exactly the claim that turned out to be degree statistics in Logbooks 034 and 103.
- H1 is to be applied to every arm of W, so a graded readout would be built on that unexamined floor.
- Its kill criterion, retirement if it fails to separate any model from chance, can never trigger: every model tested so far sits above chance.
- The failure that matters, being unable to tell models apart, has no kill criterion at all.

**7. What to do instead.** The project already has a degree-preserving rewiring, and C.2's encoder builds on every null wiring (AGENTS.md). Score the unfitted encoder on the true wiring against 8 to 16 degree-preserving rewirings, on the same measured test matrix. That costs minutes, not 5 to 8 sessions. It asks H2's question inside the project's own model, on the exact readout H1 would deploy.

- If the true wiring does not beat its rewirings, H1 is measuring degree statistics, and the neural track has nothing to offer at present. The answer is stay.
- If it does, H1 earns a place, with that null as its floor. There is still no case for H2 until there is evidence that connectome constraints help on perturbation coupling.

**The competing arguments, and why they are weaker.**

- **For a switch:** ORIENTATION.md argues that behaviour statistics under-determine the controller, because the benchmark kept finding non-worm ways to pass. Logbook 128 adds that wild-type behaviour does not pin the fit's perturbation responses. So W's wiring question, asked on behaviour alone, may be unanswerable, and only neural data could identify what the connectome contributes. This is serious. But ORIENTATION.md concedes that the same objection applies to neural data: the atlas agrees with itself at only 0.17, and dense models match masked ones. A switch is also blocked on licensed data today.
- **Against the hybrid on cost alone:** this is real, but the recommendation's kill rules and its use of rented capacity partly answer it.
- Neither competitor removes the causal link the hybrid depends on. My lead argument does.

**What evidence would settle it.**

1. **The rewiring test above,** on the box centre and on C.2's answer. Minutes of compute, read before anything else.
2. **Creamer's masked model against a degree-preserving null.** This could be checked from his released weights on the measured matrices, if the masks can be rewired. If his model does not beat the null on perturbation responses, H3's premise fails and H2 should not run.
3. **If H2 runs anyway,** register a bar that separates "partners covary" from "the coupling is causal". For example, require the held-out-neuron fit to also predict measured stimulation responses above a degree-null-fitted model. A pass on withheld-neuron imputation alone should not be allowed to open H3.

## My response (separate from the answer above)

I think the argument is right on its two strongest points, and I would change the recommendation
if the maintainer agrees. I have not changed it unilaterally. *(Later the same day: the maintainer
agreed, H0 was run and passed, and A32 revised RECOMMENDATION.md accordingly.)*

- **H2's link to H3 is weak.** Imputing withheld neurons from their partners shows that partners
  covary, not that a fitted coupling is causal. A pass on H2 alone should not open H3.
- **H1's relabelling null is too weak,** for the reason the project has met twice. The cheap test
  it proposes should come first: the unfitted encoder on the true wiring against 8 to 16
  degree-preserving rewirings, on the same measured matrix. It takes minutes, uses existing code,
  and decides whether any neural track is worth having.

Where I disagree:

- **Point 5.** The overlap between C.2 and its unfitted floor is what the recommendation reports. Its
  claim is that behaviour fitting did not move the head toward the data, not that it moved it away.
  Using Creamer's MIT-published arrays for a reading, while declining the unlicensed raw exports for
  a fit, is a licence distinction, not an inconsistency. The arrays are released under a licence and
  the exports are not.
- **Point 3.** H2's value is not only for Phase 10. A wiring bug in the field's second published fit
  matters to anyone who cites it. But that is a reason to report the bug, which the drafted note
  does, not a reason to spend 5 to 8 sessions.

A revised recommendation would read as follows:

1. **H0:** the rewiring test, before anything else.
2. **If the true wiring does not beat its rewirings:** stay, with no neural track.
3. **If it does:** H1 with the degree-preserving floor. H2 only with a causal bar, prediction of
   measured stimulation responses beside imputation. H3 gated as before.
