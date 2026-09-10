# Design: the variant through the clone assay

## What is being asked

Does a rule that learns from random weights **hold a policy that is already good**? The two are
different capabilities and the project has measured them coming apart: the three-factor rule found
better policies on some seeds and left them (the clone-destruction diagnostic), and all three
consolidation mechanisms braked the drift without holding the clone.

The assay is the registered one, unchanged, so the variant's result is directly comparable with
the three mechanisms that failed it:

- the wild-type plastic clone arm with the variant's rule keys and nothing else changed;
- started from the warm-start panel's plastic-set wild-type clone for each seed;
- seeds 1–8 paired, 2000 episodes, no extension;
- scored on plateau-tail full-clear success against the same seeds' committed `wt_clone_frozen`
  values (39.3, 44.0, 40.0, 21.3, 47.1, 33.3, 61.3, 23.3; mean 38.7);
- **holds** = mean within 5 points of the frozen clone's and ≥ 6 of 8 seeds no more than 10 points
  below their own; **improves** = mean above and ≥ 6 of 8 above their own; **pass** = either.

It remains a **screen, not a confirmatory test**: it reuses seeds Logbook 043 reported, so a pass
licenses running the registered panel and nothing more.

## The σ is the control's, not a new choice

`σ_node = 0.2` is the value that passed the positive control, and it is used here unchanged. The
assay is a gate, and a gate whose parameter is re-tuned against its own outcome is a search. If
the variant fails at that σ, the record says the variant fails the assay at the σ that cleared the
control — which is a fact about the variant, not an invitation to sweep.

Two consequences of σ carrying over are worth stating in advance, because they cut in opposite
directions and the result should be read with both in view. Perturbation is exploration: it will
degrade a competent policy's *immediate* behaviour, since a policy at its optimum can only be made
worse by jitter. But it is also what makes the rule able to hold anything at all, since without it
the rule drifts. The assay measures the net.

## The exploration tax, and how the record separates it from destruction

The pass rule cannot express the outcome this arm is most likely to produce. "Holds" needs the mean
within 5 points of 38.7 and at least 6 of 8 seeds within 10 of their own — but σ = 0.2 jitter on
activities bounded in (−1, 1) will cost a competent policy something *immediately*, before any
question of whether the rule retains it. A result that keeps most of the clone while paying a
visible exploration tax scores as a plain **fail**, indistinguishable in the record from the anchor
arm's collapse to 13.0. The three consolidation mechanisms paid no such tax, so a bare pass/fail
comparison against them is not like-for-like.

Two things are registered here to make the distinction, neither of which changes the pass rule:

- **A trajectory annotation.** Each arm's plateau tail over its **final** quarter against its
  **first** quarter, from the curves the harness already reads. A run that starts near the clone and
  stays there has held it and paid a tax; one that starts near the clone and ends near zero has been
  destroyed.
- **A frozen-perturbation control.** The same clone arm with `freeze_updates: true` at the same σ:
  perturbation applied, no weight ever written. It measures what the jitter alone costs a competent
  policy, so a fail can be attributed to the rule or to the exploration rather than left ambiguous.
  Eight runs, its own arm in the same harness.

Neither is a verdict. If the variant fails, it fails; these say *why*, as the consolidation
screen's rate-multiplier and cosine annotations did for their arms.

## What is not measured here

The endpoint cosine to the clone, which the consolidation screen reported, is reported here too —
a variant can hold the metric while having rewritten the policy underneath it, and the two are
different results. Nothing else is added: the harness's existing reporting is the registered one.

## What each outcome licenses

- **Pass** — the variant learns from random weights *and* holds a competent policy. The registered
  order's step 4 opens: a connectome **panel** arm becomes buildable, still behind I.2's statistic and metric
  before any panel is read from it. This would be the first time in Phase 7 that a rule cleared
  both gates.
- **Fail** — the variant learns but does not hold, which is the consolidation mechanisms' failure
  in a new place and a genuinely informative one: it would say the eligibility fixed credit
  assignment without fixing retention, and that block I needs a consolidation mechanism *on top of*
  the working eligibility rather than instead of it. The panel stays gated and I.4's re-read says
  so.

Both outcomes are named here so neither is reframed afterwards.

## Alternatives considered

- **Skipping straight to a connectome panel** — the clearance order was fixed before the control's
  result existed precisely so that a good result could not shorten it.
- **Sweeping σ on the assay** — makes the gate a search; rejected above.
- **Adding the assay's arms to a new harness** — the registered assay already has one, and using it
  unchanged is what makes the variant comparable with the three mechanisms that failed.
