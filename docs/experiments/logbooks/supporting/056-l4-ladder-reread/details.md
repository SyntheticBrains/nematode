# I.4 — the re-read, one registered contrast at a time

**Written 2026-09-12. Runs nothing. Changes no committed verdict.**

Every number below is quoted from a committed table. The classification is in
[`classification.csv`](classification.csv); this file is the reasoning.

## The four kinds, and the premise each one needs

The registered scheme named three kinds. Classifying all 32 contrasts found a fourth, and it holds
more of the record than any other.

### 1. No rule running (4 contrasts)

041's P4, 043's W1 and W2, 044's G1. Frozen weights on both sides; no rule, no optimiser, no
modulator. Whatever block I found about the three-factor rule cannot reach a comparison in which
nothing was written. These survive exactly as committed.

What they establish is load-bearing for the rest:

- **W1 +31.0, 8/8, q = .023** — the substrate holds a competent policy. This is the premise of every
  retention assay in the phase, and it is met.
- **P4 +3.3 over 64 seeds, G1 +0.77 over 64** — the prior over untrained policies is
  indistinguishable between the wild type and its degree-preserving scramble, before and after the
  signs are grounded. Any wiring advantage must be created by a learning process; none is inherited
  from the graph.
- **W2 −4.0, reverse** — the rewired null carries a cloned policy at least as well as the wild type.

### 2. The reward-free Hebbian rule (12 contrasts)

041's P1–P3, 042's R1–R2, 044's G2–G4, 046's D1–D4.

**Logbook 048 does not reach these.** It tested the reward-modulated three-factor rule: eligibility
`E ← λE + M∘(h_prev ⊗ h)`, scaled by a prediction error. The Hebbian arms run with the modulator
removed. They were never claimed to be policy-gradient estimators — they are the phase's
*unmodulated floor*, and what 040 measured about them is the opposite of a broken instrument: with
no reward at all, the wild-type Hebbian arm settles at **78.3%, 64.4% and 67.3%** on three of eight
seeds (arm mean 30.2 against the frozen floor's 8.2). A mechanism that reaches competent behaviour
on the task under test is not one that "could not have learned".

So these twelve contrasts are findings about the substrate under a demonstrated mechanism, and they
are read as committed. What bounds them is their own statistics, not block I:

- the wiring contrast shrank on every fresh look — **+16.2** (040's seeds 1–8, descriptive +16.5),
  **+14.1** (041, sixteen seeds, q = 0.28), **+8.1** (042, forty-eight fresh seeds, q = 0.19),
  pooled **+9.6** over 64;
- I.2's matched statistic promotes none of them: the level differences among each arm's own
  competent seeds are large and consistent (+25.0, +22.7, +12.9) and none is significant against a
  pooled-label null, because those contrasts carry two to five competent seeds an arm;
- grounding the synapse signs made the whole family worse (wild type 31.5 → 14.0, rewired 17.4 →
  9.1), and neither decorrelating term recovered it (D1 −0.90, D2 −2.60).

**046's headline is the most durable claim in the block and is not about any rule.** The atlas
grounds **214 of 3,709** chemical synapses as inhibitory — 5.8% — so the anti-Hebbian arm's
measured share of 0.057 is *everything it could reach*. A transmitter-only atlas does not ground
enough inhibition to build a brake from, which is a fact about the data, not about the instrument,
and it is what promotes the receptor layer (B.3) from fidelity work to a prerequisite.

### 3. The three-factor rule, wiring contrasts (5 contrasts) — premise

040's T1 and T4, 043's W3, 047's S3 and S4. Each asks whether the wild type beats its rewired null
with the three-factor rule running.

Their premise is that **learning finds a wild-type advantage**. It has no demonstration:

| regime | wild type | rewired null | Δ | seeds |
|---|---|---|---|---|
| PPO weight search, registered contrast (034) | — | — | **−3.28**, CI[−8.56, +1.61], q = 0.770 | 8 paired |
| low-noise PPO from scratch (043) | 68.5 | **81.2** | **−12.6** | 0/8 positive |
| full-parameter clone, frozen (043) | 73.7 | 74.7 | −1.0 | — |

**This contrast was registered and run under a working optimiser before Phase 7 began.** Logbook
[034](../../034-connectome-structure-controls.md), Phase 6a, ran the wild type against
degree-preserving rewired nulls under PPO weight search on the same continuous cell and found them
indistinguishable — nominally the null higher — with no wild-type advantage in learning efficiency
either (all q ≥ 0.36). Logbook 043 then ran low-noise PPO from scratch on Phase 7's own seeds and
put the null ahead by 12.6 on 0 of 8. The two differ (indistinguishable against null-ahead) and
agree on the point that matters. Logbook 029's fifth-of-six ranking is the context that motivated
034, not a wiring contrast itself.

**Neither PPO regime tested found a confirmed wild-type advantage** — 034's registered contrast is
indistinguishable, 043's puts the null ahead — so the premise these five contrasts rest on has no
demonstration behind it. They are therefore classified as **uninformative about both** the wiring
question and the instrument, and a working instrument would not have changed that classification:
the contrast would still be one whose sought effect no tested method has shown. Filing them as
instrument casualties would claim instead that a working rule might have found the advantage, which
no evidence supports.

This is a statement about what the nulls are evidence *for*, and it is bounded in both directions.
PPO is one optimiser family in two regimes, not every instrument; and the reward-free Hebbian
signal of §2 is a separate question, still unresolved, which nothing here closes.

### 4. The three-factor rule, floors / retention / rule variants (10 contrasts) — instrument

040's T2, T3 and the MLP band test; 043's W4 and W5; 045's three brakes; 047's S1 and S2.

Their premise — that a competent policy can be held on this substrate — **is met**, by W1. So the
null is attributed to the rule, and 048 explains it rather than voiding it:

- **040 T2/T3** the rule beats neither its frozen floor (+9.6, q = 0.50) nor its unmodulated
  Hebbian floor (−12.4, q = 0.73). Read now: a rule with +0.009 gradient alignment should not beat
  doing nothing, and it did not.
- **040's band test** the matched-rule MLP yardstick sits at **1.1%** — chance. This was the
  earliest instrument evidence in the phase, recorded in its first logbook and read as a property of
  the yardstick rather than of the rule. I.3b confirms it survives the repaired eligibility at every
  horizon.
- **043 W4/W5** from a 38.7% clone the rule falls to **13.0** (−25.7 against both its frozen and its
  Hebbian clone, both intervals entirely below zero). `rule_destroys_clone` is the correct reading
  and its premise was met.
- **045** three brakes on that drift. The anchor does not hold the policy at all (13.0), the
  per-synapse protective variable comes closest and fails one clause (29.4, 6/8 within hold, cosines
  0.67–0.79, at a uniform **91% rate cut**), the oracle gate fails through its own lag (28.9).
  Scope narrowed by the re-read: this is what braking a **non-estimator** does, not a result about
  consolidation mechanisms in general.
- **047 S1/S2** routing the third factor through the aminergic pathway made both wirings worse
  (−3.18, −7.30), with an instructed share of 0.92 confirming the intervention was real. The logbook
  already read this against 048 on the day: it measures what a non-learning rule does under two
  routing regimes.

### 5. PPO (1 contrast) — neither

043's W6, `wt_fullclone_ppo − wt_ppo` at −34.1. A finding about warm-starting PPO, not about the
connectome and not about the local rule. The correction it carries is in the logbook.

## Counts

| re-read | contrasts |
|---|---|
| substrate | 16 |
| instrument | 10 |
| premise | 5 |
| neither | 1 |
| **total** | **32** |

## What the re-read did not do

- It changed no committed verdict. All eight stand as registered.
- It converted no negative into a positive. The Hebbian wiring contrast survives as a small,
  shrinking, unresolved effect — which is what its own logbooks concluded — and not as evidence that
  the wiring matters.
- It licensed no work. The low-σ programme's status is stated in the logbook as deferred behind a
  task the repaired rule can be shown to learn; everything else in the block is closed.
