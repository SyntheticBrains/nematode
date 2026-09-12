# Re-read the ladder (7a-ii I.4)

## Why

Seven registered results asked whether the wild-type connectome's wiring is legible to a local
three-factor rule. All seven are negative. Block I then established that the rule those results ran
under was not a policy-gradient estimator at all (I.0), repaired it (I.1), and found the repair
fails on every multi-step task tried (I.1b, I.1c, I.3b) while a statistic matched to the outcome's
shape promotes nothing (I.2).

The obvious re-read is "the negatives are about a broken instrument". **The record does not support
that reading, and one number is why.** Logbook 043's warm-start panel ran PPO on the same substrate
and the same task: from scratch it reaches **68.5%** on the wild type and **81.2%** on the
degree-preserving rewired null, against a frozen clone that holds **73.7%**.

So the task is solvable, the substrate holds a competent policy, and **in the one regime where
learning demonstrably works the rewired null beats the wild type by 12.6 points on 0 of 8 seeds**.
Logbook 029 converges from a different regime, ranking the connectome fifth of six and
indistinguishable from its rewired null — a different finding from "null ahead", agreeing on the
point that matters: no wild-type advantage.

That changes what the seven results are evidence about. They were read as asking whether a local
rule could find a wild-type advantage. No optimiser has found one — including the one that solves
the task. The premise of the contrast, not only the instrument, is what most of the negatives are
about, and a record that attributed them to the instrument alone would be wrong in a way that
matters for B.8.

## What Changes

- **One record**, Logbook 056, stating for each of 040–047 — I.4 was registered for 040–046, and
  047 postdates it — whether it survives as a finding about
  the wiring, about the instrument, or about neither — with the committed verdict carried unchanged
  beside each re-read, as I.2's re-read requires.
- **The three-way split is the contribution.** "Instrument" and "wiring" are not exhaustive: a
  result can be uninformative because its premise was never established, which is the category the
  PPO numbers create and which the registered I.4 wording did not anticipate.
- **A correction on the record.** An earlier reading of 043's W6 — that PPO also destroys competent
  policies, so the constraint might not be the rule — over-read it. W6 shows *warm-starting* hurts
  PPO (34.5 against 68.5 from scratch), not that the task destroys policies. In the same warm-start
  regime the local rule reaches 13.0 against PPO's 34.5, so the attribution to the rule is cleaner,
  not muddier.
- **What each conclusion licenses for B.8**, stated so the shipment decision inherits a reading
  rather than a pile of verdicts.

Out of scope: the shipment decision itself (B.8), any new campaign, and any change to a committed
verdict. This runs nothing.

## Capabilities

**Modified**: `plasticity-evaluation` (what a re-read of a body of results must establish before it
attributes them).

## Impact

- New: Logbook 056 and its supporting directory. Edited: the experiments index, `CHANGELOG.md`,
  tracker, roadmap.
- No code, no campaigns, no committed number altered.
