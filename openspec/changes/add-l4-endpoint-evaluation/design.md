# Design: the endpoints, perturbation off

## What the assay could not see

Every number the clone assay has produced for a perturbing arm was measured with the perturbation
running. That is the right measurement of what the arm *does* during training, and it is the wrong
measurement of what it *learned*: on a task where an unperturbed competent policy clears 38.7% of
episodes, σ = 0.2 jitter alone drops a frozen policy to 8.9%, so a learning arm's 12.0 is the sum
of whatever it retained or gained and a perturbation tax that the frozen control shows is at least
30 points. Those two cannot be separated from the under-perturbation score alone.

The MLP control separates them cleanly, and the separation is large: 86% of the gap under σ = 0.2,
93% with it off. Nothing about the connectome guarantees the same, and one fact argues the other
way — the endpoints sit at cosine 0.70 to the clone, so the rule moved the weights substantially.
Whether it moved them toward a policy that is competent when it runs cleanly, or away from one, is
the question.

## The arm

The comparator's config with one key changed. `..._plastic_frozen_clone.yml` is itself a
single-key delta (`weights_path`) from the frozen plastic-set arm; this arm changes that same key
to the staged endpoints. So it runs exactly what the comparator ran — no eligibility mode, no
perturbation (the parent's defaults are `hebbian` and `0.0`), `freeze_updates: true` — on
different weights. With updates frozen the eligibility mode is inert, and with the scale at zero
the forward pass is the unperturbed one, so "evaluated with the perturbation off" is literal.

Budget 2000 episodes and the plateau-tail metric: the assay scored every arm that way against a
comparator scored at 600, and this arm joins that table on the same footing. For a frozen policy
the process is stationary, so the longer run only tightens the estimate.

## The endpoints

The I.1 clone-assay learning arm auto-saved its final weights per seed: `final.pt` is the
runner's end-of-run save, which also fires on an interrupted run, so completion is a stated
precondition — all eight source runs completed their 2000 episodes (verified from the logs before
this was written). They are copied to
`campaigns/l4-perturbation-clone/endpoints/nodeperturbation_wt_seed{seed}.pt`, the source export
of each recorded in the launch record, the way the clones were staged under `l4-warm-start/`. The
staged files are the assay's endpoints and nothing else: no re-training, no selection.

Two episodes of seed 1 were observed while confirming that an endpoint loads through the runner
(both `health_depleted`, 170 and 108 steps). Disclosed here; they carry no information at n = 2
and are not part of the record.

## A load-integrity check the harness already computes

For every arm the screen harness reports the cosine between the run's final weights and the clone
it started from. For a *frozen* run the final weights are exactly the weights that were loaded, so
this arm's cosine must reproduce the I.1 endpoints' own per-seed values — 0.649 to 0.748, committed
in the 050 record. That is the one check that separates the intended run from its only silent
failure mode: a cosine near 1.00 means the clone loaded rather than the endpoint, which would
produce a false "holds". **A seed whose cosine departs from its committed value by more than 0.01
voids that seed; any voided seed voids the verdict**, and the run is repeated after the cause is
found, not scored around.

## The rule, and what each outcome licenses — fixed before the run

The assay's registered pass rule, unchanged: **holds** if the mean sits within 5 points of the
comparator's 38.7 and at least 6 of 8 seeds sit within 10 points of their own clone; **improves**
if the mean exceeds 38.7 with at least 6 of 8 at or above their clone. Anything else **fails**.

- **Holds or improves.** The policy the rule left behind is competent once it runs without its
  exploration noise. The I.1 assay fail was an artefact of evaluating under perturbation, and the
  tension it exposed is not a property of the rule. *Licenses:* a registered amendment making
  "endpoint evaluated perturbation-off" the primary endpoint of the clone assay for any perturbing
  rule, with the under-perturbation score kept as the descriptive training-time reading; and the
  panel question reopening under I.2's statistic. *Does not license:* the low-σ programme, which
  is then answering a question that does not arise.
- **Fails.** The rule rewrote a competent policy into a worse one, and no amount of removing the
  noise recovers it. The I.1 fail stands as a genuine retention fail. *Licenses:* the low-σ
  programme, in the form the critical review of it fixed — a frozen dose-response on the connectome
  with a σ = 0 anchor in the same campaign, a σ-selection rule written in advance, the control's
  rate sweep as a new harness arm, and a clone assay with a pinned-rate learning arm as a rate
  control — authored as its own change.

A descriptive annotation beside the verdict, not a verdict input: each seed's endpoint score
against the same seed's under-perturbation score from the assay (mean 12.0), which is the
perturbation tax the rule was paying at the end of training.

## What this does not test

Whether the rule can learn on the connectome from scratch. The endpoints started from a competent
clone; a clean endpoint says the rule retained and possibly improved one, not that it can build
one. That is a later question and a different assay.

## Alternatives considered

- **Lower σ first** (the plan before review). Rejected as *first* step: it presupposes the
  tension is real, and eight frozen runs decide that for a fraction of the cost.
- **Evaluate at a small nonzero σ** rather than zero. Rejected: zero is the comparator's own
  condition and needs no new config semantics; a nonzero evaluation scale is a separate question
  about run-time tolerance, which the frozen dose-response answers if it is needed.
- **Re-run the learning arm to save endpoints under a new protocol.** Unnecessary; the endpoints
  exist and are the registered arm's.
