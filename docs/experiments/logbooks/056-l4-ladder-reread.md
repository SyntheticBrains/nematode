# 056: Re-reading the Ladder — What Eight Negative Results Are Evidence About (7a-ii I.4 / Phase 7)

**Status**: completed — **no committed verdict changed**, and the obvious re-read is wrong. Block I
found that the three-factor rule Logbooks 040–047 ran under was not a policy-gradient estimator
([048](048-l4-rule-positive-control.md)), repaired it
([049](supporting/049-l4-node-perturbation/details.md)) and then failed to make the repair work on
any multi-step task ([051](supporting/051-l4-sigma-annealing/details.md),
[052](supporting/052-l4-endpoint-evaluation/details.md),
[055](supporting/055-l4-horizon-multistep/details.md)). The tempting conclusion is that the
phase's negatives are artefacts of a broken instrument. Classified one registered contrast at a
time — 32 of them across eight logbooks — **half the record is not about the instrument at all**:
16 contrasts ran either under no rule (frozen priors and frozen clones) or under the *reward-free
Hebbian* rule, which 048 never tested and which this substrate has been shown to drive to 78%,
64% and 67% on individual seeds with no reward at all. Ten contrasts are instrument findings, and
they survive as such, sharpened rather than voided: 043's `rule_destroys_clone` stands on a
premise its own W1 established (+31.0 on 8/8). **Five are about neither**, because their premise
was never established by any optimiser — Logbook 034 ran that contrast under PPO in Phase 6a and
found the wirings indistinguishable (−3.28, q = 0.770), and in 043's low-noise PPO arms the
degree-preserving rewired null beats the wild type by **12.6 points on 0 of 8 seeds** — so "learning
finds a wild-type advantage" has no demonstration to rest on and a working rule would not have
changed those cells.
One contrast, W6, is about PPO and carries a correction issued here: an earlier reading of it
during the programme is withdrawn on the record. The change registered an expectation that *most*
of the record would land in the premise category; it does not, and the reason is the fourth kind
of result the registration did not anticipate.

**Branch**: `feat/l4-ladder-reread`.

**Date**: 2026-09-12.

**OpenSpec change**: `add-l4-ladder-reread` (extends capability `plasticity-evaluation`; what a
re-read of a body of results must establish before it attributes them).

## Objective

I.4 as registered: with I.0–I.3b answered, state which of the registered negative results survive
as findings about the wiring and which are findings about the instrument, in one record, before the
7a shipment decision (B.8). Registered for 040–046; 047 postdates the registration and is included,
so the scope is 040–047.

This record runs nothing. Every number in it is quoted from a committed table.

## Background

Eight logbooks asked, in one form or another, whether the wild-type *C. elegans* wiring is
load-bearing under a local rule. All eight are negative or inconclusive. Block I then established
that the reward-modulated three-factor rule they were read against had a gradient alignment of
**+0.009 median** and ended *below* the cue-blind floor on a one-step task whose analytic reference
closes 99.9% of the available gap ([048](048-l4-rule-positive-control.md)); that an eligibility
carrying each unit's own perturbation passes that control at **+0.263** alignment on 8/8 seeds
([049](supporting/049-l4-node-perturbation/details.md)); and that the repair fails on every
multi-step task tried, most decisively on the MLP yardstick, where the learning arm scores 0.393,
0.148 and 0.144 foods against its own frozen control at **2.233** while its weights move 1.28–1.31
times their own norm ([055](supporting/055-l4-horizon-multistep/details.md)). A statistic matched
to the outcome's bimodal shape re-read seventeen committed contrasts and promoted none
([053](supporting/053-l4-mixture-statistic/details.md)).

So the instrument was broken, is repairable on a one-step task, and is still not a working learner
here. The question this record answers is what that does — and does not do — to the eight results
that preceded it.

## Method

No runs, no code. Every registered test in 040–047 is listed as its own row, with the committed
value, the committed result, **which rule was running when it was measured**, the kind of question
it asked, and its re-read. The table is
[`supporting/056-l4-ladder-reread/classification.csv`](supporting/056-l4-ladder-reread/classification.csv);
the reasoning per result is in
[`details.md`](supporting/056-l4-ladder-reread/details.md).

Two constraints, both registered: a committed verdict is carried unchanged beside its re-read and
is the verdict where the two differ; and a re-read may not convert a negative into a positive.

### Classification by kind, then premise before instrument

The registered scheme classified by result type first — a no-learning result, a wiring contrast
under learning, a retention assay — because "the premise" names a different claim in each, and one
test applied across all of them files results wrongly in both directions. Classifying every
contrast individually found a **fourth kind the registration did not anticipate**, and it is the
largest: contrasts measured under the **unmodulated, reward-free Hebbian** rule. 048 tested the
reward-modulated three-factor rule. The Hebbian arms carry no modulator, make no claim to be a
policy-gradient estimator, and 040 recorded their capability on this exact cell directly — the
reward-free wild-type arm settles at 78.3%, 64.4% and 67.3% on three of eight seeds. A mechanism
demonstrated to reach competent behaviour on the task is not an instrument that "could not have
learned", so 048 does not reach those contrasts and they are read as registered.

Each kind carries its own premise test:

| kind | premise | established? | contrasts |
|---|---|---|---|
| no rule running | — | not applicable | 4 |
| reward-free Hebbian | the rule reaches competent behaviour here | **yes** — 78.3 / 64.4 / 67.3 on 040's floor arm | 12 |
| three-factor, wiring contrast | learning finds a wild-type advantage | **no** — 034's PPO contrast (−3.28, q = 0.770) and 043's PPO arms (−12.6, 0/8) | 5 |
| three-factor, floor / retention / rule variant | a competent policy can be held | **yes** — 043's W1, +31.0 on 8/8 | 10 |
| PPO | — | not applicable | 1 |

## Results

### The 32 registered contrasts

| re-read | count | what it means |
|---|---|---|
| **substrate** | 16 | survives as registered; block I does not reach it |
| **instrument** | 10 | measures a rule now known not to learn; survives as a finding about that rule |
| **premise** | 5 | uninformative about both — no optimiser has shown the effect it sought |
| **neither** | 1 | 043's W6, a finding about PPO warm-starting |

### Per logbook

| logbook | committed verdict | verdict after the re-read | what survives |
|---|---|---|---|
| [040](040-l4-panel.md) | `sanity_floor_fail` | **unchanged** | the verdict itself is an instrument finding and is now corroborated: the rule beat none of its floors because it was not a learner. The fixed-point account (outcomes set by the random initial weights) and the descriptive Hebbian floor contrast (+16.5, 5/8) survive; T1 and T4, the wiring contrasts, are premise. |
| [041](041-l4-panel2.md) | `inconclusive` | **unchanged** | whole. All four tests ran under no rule or the reward-free Hebbian rule. The +14.1 contrast, the +3.3 prior and the count-scaling reversal (−15.6) stand exactly as committed. |
| [042](042-l4-panel3.md) | `inconclusive` | **unchanged** | whole, and it is the phase's cleanest surviving wiring measurement: +8.1 on 48 fresh seeds, q = 0.19, with the estimate shrinking on each fresh look (+16.2 → +11.9 → +8.1, pooled +9.6). |
| [043](043-l4-warm-start.md) | `sanity_floor_fail` + `rule_destroys_clone` | **unchanged** | split four ways. W1/W2 are frozen-substrate facts and survive. W4/W5 are the sharpest instrument evidence in the phase and their premise was met. W3 is premise. W6 is about PPO and carries the correction below. |
| [044](044-l4-atlas-signs.md) | `degree_statistics` | **unchanged** | whole. G1 is a frozen prior; G2–G4 are reward-free Hebbian. "Grounding the signs left the prior alone and made Hebbian learning worse" is untouched by anything block I found. |
| [045](045-l4-consolidation.md) | no mechanism passes | **unchanged** | as an instrument finding, with its scope narrowed: a 91% per-synapse rate cut applied exactly where reward was writing does not stop the drift — of a rule that was not estimating a gradient. It is not evidence about consolidation mechanisms in general. |
| [046](046-l4-decorrelation.md) | `no_recovery` | **unchanged** | whole, and its headline is the most durable claim in the block: the atlas grounds **214 of 3,709** synapses as inhibitory, so the anti-Hebbian arm's 0.057 share is everything it could reach. A transmitter-only atlas does not ground enough inhibition to build a brake from, whatever the rule. |
| [047](047-l4-structured-instruction.md) | `no_routing_effect` | **unchanged** | split. S1/S2 measure what a non-learning rule does under two routing regimes — which the logbook already said — and S3/S4 are premise. The bit-for-bit substrate reproduction across six panels survives as an engineering result. |

### The correction on W6

**The earlier reading, made during the programme:** W6 shows PPO also destroying a competent
policy, so the binding constraint might be the task rather than the rule, and the task should be
diagnosed before any rule family is chosen.

**What W6 shows:** warm-started PPO scores 34.5 against 68.5 from scratch, on a clone that frozen
weights hold at 73.7. Warm-starting *did* cost 39 points — most plausibly a stale value function
and rollout buffer meeting a policy they were not fitted to. What the same table also shows is that
PPO does not need the clone: 68.5 from random weights, and 81.2 on the rewired null.

**What no longer follows:** "the task destroys competent policies" is not supported — PPO solves the
task from scratch and frozen weights hold a clone at 73.7. Both optimisers degrade a warm-started
clone, and the two drops are **not directly comparable**: the rule started from the chemical-weights
clone (38.7 → 13.0, W4 −25.7 registered, interval entirely below zero) and warm-started PPO from the
full-parameter clone (73.7 → 34.5, descriptive; W6 −34.1 against PPO from scratch). No rule arm ran
from the full clone and no PPO arm from the chemical one. What *is* comparable is what each
optimiser does without a clone: **PPO reaches 68.5 from random weights and the rule reaches 17.8**
([040](040-l4-panel.md)'s `wt_plastic`), below its own unmodulated floor. So the attribution to the
rule is cleaner than that reading suggested — not because the rule loses more from a warm start, but
because PPO does not need one and the rule never reaches competence from any start, for a reason 048
names. The recommendation that followed from the earlier reading, to diagnose the task before
choosing a rule family, is **withdrawn here** rather than left standing.

## Analysis

1. **The instrument reading is the smaller half of the record.** Ten of 32 contrasts measure the
   broken three-factor rule. They survive as findings about it, and 048 makes them sharper: 040's
   `sanity_floor_fail` and 043's `rule_destroys_clone` were the correct readings of a rule whose
   updates were near-orthogonal to the policy gradient, and the phase recorded them long before it
   could explain them.
2. **The premise category is small, and it is where the wiring hypothesis actually died.** Five
   contrasts asked whether the wild type beats its rewired null under the three-factor rule. **The
   same contrast was already registered and run under PPO**, in Phase 6a: Logbook
   [034](034-connectome-structure-controls.md) puts wild minus rewired at **−3.28, CI[−8.56, +1.61],
   q = 0.770** over eight paired seeds, with no advantage in learning efficiency either (all
   q ≥ 0.36). Logbook 043's low-noise PPO arms, on Phase 7's own seeds, put the null **ahead by 12.6
   points on 0 of 8**. The two differ — indistinguishable against null-ahead — and agree on the point
   that matters. Repairing the rule would not have changed those five cells, and reporting them as
   instrument casualties would overstate what a repair could ever have delivered.
3. **The one wiring signal the phase produced is untouched by block I.** The reward-free Hebbian
   contrast — +16.5 descriptive on 040's floors, +14.1 registered on 041's sixteen seeds, +8.1 on
   042's forty-eight fresh ones, +9.6 pooled over 64 — never ran under the modulated rule. It is
   small, it shrank on each fresh look, it is not significant at any sample size run, and I.2's
   matched statistic promotes none of its contrasts (the level differences +25.0, +22.7 and +12.9
   sit below the combinatorial floor at two to five competent seeds an arm). But it is not an
   artefact of a broken instrument, and the record should not retire it as one.
4. **Two substrate facts outlast the whole block.** The prior over untrained policies is
   indistinguishable between wirings (+3.3 over 64 seeds, +0.77 after grounding), so any wiring
   advantage has to be created by a learning process rather than inherited from the graph; and the
   transmitter atlas grounds only 5.8% of synapses as inhibitory, which makes the receptor layer a
   prerequisite for the inhibitory-brake hypothesis rather than an item of fidelity work.
5. **What the registration expected, and got wrong.** The change predicted that most of the eight
   results would land in the premise category. They do not — 5 of 32 — because the registration's
   three kinds had no place for the reward-free Hebbian arms, which are the single largest group in
   the record. The prediction was right that the category exists and was missing from I.4's
   original wording, and wrong about its size.

## Conclusions

- **No committed verdict changed.** All eight stand as registered, in their own units, under the
  rule they were registered with. Every re-read in this record is placed beside its verdict, never
  in place of it.
- **The three-way split is the contribution**: about the substrate (16), about the instrument (10),
  about neither (5) — plus one result about PPO. "Instrument" and "wiring" were not exhaustive.
- **The citable claim after block I** is narrower and more defensible than either "the wiring does
  not matter" or "the instrument was broken": on this task, on this substrate, no optimiser tested —
  PPO included — finds an advantage for the wild-type wiring, and the local three-factor rule
  additionally fails a positive control that names exactly why.
- **A reading is withdrawn** (W6) with what no longer follows from it stated.

## What this leaves for B.8

- **GO is unreachable as written.** The clause requires 7a-ii to ground the panel in the
  receptor-gated neuromodulator stack — B.3 — which was never built and is queued behind block I,
  "paid for only if a rung after it can conclude". Nothing in this record makes a rung after it able
  to conclude.
- **7b's gate is not met and is not moved by this re-read.** It requires a registered result in
  which the wild type beats its rewired null under a local rule. The re-read's finding is stronger
  than "not yet": the contrast has no demonstration under *any* optimiser.
- **The STOP clause names this outcome in advance** — "the diagnosis itself is the Phase 7
  deliverable, and follow-on work picks up alternative rule families (e-prop, imitation-warm-start)".
  This record is that diagnosis. Between STOP and SPLIT, what separates them is whether 7a forms a
  self-contained citable result, which is B.8's decision and is not taken here.
- **The low-σ programme licensed by [052](supporting/052-l4-endpoint-evaluation/details.md) is
  deferred, not retired**, and the condition is recorded so the licence does not dangle: 052
  licensed it on a bimodal endpoint panel where two of eight seeds improved and seed 2 reached 73.4
  from a clone of 44.0. What has happened since is 055 — at σ = 0.2 on a multi-step task the rule
  writes 1.28–1.31 times its own weight norm in a direction that makes the policy worse. The
  programme's premise, that some σ both learns and leaves a runnable policy, has only ever been
  examined on the one-step control and the clone assay. It should be run behind a task the repaired
  rule can be shown to learn, and not before.

## Next Steps

- [ ] B.8: the 7a-ii logbook and the shipment decision, inheriting this reading.
- [ ] If work continues at the rule level, the task ladder before any further mechanism: the phase
  has never run the rule on a task between the one-step control and 2400-step episodes.

## Data References

- Classification table: `supporting/056-l4-ladder-reread/classification.csv`
- Per-result reasoning: `supporting/056-l4-ladder-reread/details.md`
- Sources, all committed: Logbooks [040](040-l4-panel.md), [041](041-l4-panel2.md),
  [042](042-l4-panel3.md), [043](043-l4-warm-start.md), [044](044-l4-atlas-signs.md),
  [045](045-l4-consolidation.md), [046](046-l4-decorrelation.md),
  [047](047-l4-structured-instruction.md), [048](048-l4-rule-positive-control.md),
  [029](029-continuous-architecture-ranking.md), [034](034-connectome-structure-controls.md), and
  the block-I records under
  `supporting/049-l4-node-perturbation/` through `supporting/055-l4-horizon-multistep/`.
- No sessions: this record ran nothing.
