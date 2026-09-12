# Design: what the seven results are evidence about

## The number the re-read turns on

Logbook 043 ran PPO on the same substrate, same task, same seeds:

| arm | plateau tail |
|---|---|
| PPO from scratch, rewired null | **81.2%** |
| frozen full clone | 73.7% |
| PPO from scratch, wild type | **68.5%** |
| PPO warm-started from the clone | 34.5% |
| the local rule from the clone | **13.0%** |

Three things follow, and none of them is about the instrument:

1. **The task is solvable.** PPO reaches 68–81% from scratch. Nothing about this environment, its
   reward or its 2400-step episodes prevents learning.
2. **The substrate holds a competent policy.** A frozen clone sits at 73.7%. The 302-neuron
   recurrent graph with a fixed anatomical readout is not too weak to express one.
3. **The wild-type wiring is not advantaged.** Under low-noise PPO the rewired null beats it by
   12.6 points, on 0 of 8 seeds positive. Logbook 029 converges from a different regime — fifth of
   six architectures, *indistinguishable* from its null. The two findings differ (null ahead
   against no difference) and agree on what matters: neither shows a wild-type advantage.

## A correction carried on the record

An earlier reading of W6 held that PPO also destroys competent policies, so the binding constraint
might not be the rule at all. That over-read it. W6 compares warm-started PPO (34.5) with PPO from
scratch (68.5). Warm-started PPO *did* degrade the 73.7% clone, by 39 points — most plausibly a
stale value function and rollout buffer meeting a policy they were not fitted to. What the table
also shows is that PPO does not need the clone: from scratch it reaches 68.5, and a frozen clone
holds 73.7. So **both optimisers degrade a warm-started clone, the local rule by far more** — 61
points against 39 — and PPO solves the task without one. That is not evidence that the task destroys
policies in general; it is evidence that warm-starting is harmful under PPO here and that the rule
is worse still.

In the same warm-start regime the local rule reaches 13.0 against PPO's 34.5. The attribution to the
rule is therefore **cleaner** than that reading suggested, and the recommendation that followed from
it — diagnose the task before any rule family — is withdrawn here rather than left standing.

## Three categories, not two

I.4 was registered as "which survive as findings about the wiring and which are findings about the
instrument". Those are not exhaustive, and the PPO numbers create the missing category:

- **About the wiring** — the result measures a property of the connectivity and survives.
- **About the instrument** — the result measures a rule that could not have learned, whatever the
  wiring.
- **About neither: the premise was never established** — the result asked whether learning finds a
  wild-type advantage, and no optimiser has found one, so a null was the expected outcome under any
  instrument.

The third category is where most of the seven land, and collapsing it into "the instrument" would
overstate what repairing the rule could ever have delivered.

## How each result is classified

**By result type first**, because "premise" means something different for each kind of result,
and a single test applied across kinds files them wrongly in both directions:

- **No-learning results** — frozen priors and sign-grounding sweeps, such as 044's G1 (grounded
  wild-type frozen against the committed prior). No optimiser and no rule were involved, so neither
  the premise test nor the instrument test applies. These are **substrate findings and survive as
  they stand**, whatever block I found about the rule.
- **Wiring contrasts under learning** — wild type against the rewired null with a rule running
  (040–042, 044's G2 and G4, 047's S1 and S2). Their premise is that *learning finds a wild-type
  advantage*. No optimiser has established it: PPO from scratch puts the null ahead by 12.6 on 0 of
  8, and Logbook 029 found the two indistinguishable. → **premise**.
- **Clone assays** — does a mechanism hold a competent policy (043, 045, 046). Their premise is
  that *a competent policy can be held*, and 043's W1 established it: the frozen clone beats a
  random frozen substrate by +31.0 on 8 of 8. The premise held; the rule destroyed what frozen
  weights kept. → **instrument**.

Within a type, a result that also ran under the unrepaired rule carries the instrument note beside
its classification, but the type decides. Applying the optimiser test across all three would have
filed the clone assays as premise failures — because PPO shows the null ahead — when their own
premise was met, and would have reached the no-learning results only by elimination.

## What the re-read may not do

It may not change a committed verdict. Each stands as registered, in its own units, under the rule
it was registered with, and the re-read is a second reading placed beside it — the constraint I.2
registered for its own re-read, applied here. Where the two disagree, both are reported and the
committed one is the verdict.

It also may not convert a negative into a positive. Nothing here licenses a claim that the wiring
matters; the finding is about what the negatives are evidence *for*, which is a narrower and more
defensible statement than either "the wiring does not matter" or "the instrument was broken".

## What this leaves for B.8

Stated in the record so the shipment decision inherits a reading:

- the GO clause requires the receptor-gated neuromodulator stack (B.3), which was never built;
- the low-σ programme remains licensed by 052 and unrun, and the record says whether it is retired
  or deferred rather than leaving a live licence dangling;
- the citable result is a systematic negative with a validated instrument, which is what the risk
  table's "fails to beat its baselines" branch prescribed in advance.
