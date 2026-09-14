# 062: The Frozen Readout Is Part of the Limit, and Its Scale Does the Work (7a-ii R.1d / Phase 7)

**Status**: completed — **`readout_helps_but_not_enough`**. Under the rule the connectome may write
`w_chem` alone, so the sensory projection, the action-noise scale and the motor readout are all frozen;
[R.1c](061-l4-reduced-perturbation.md) eliminated the first two, and this tests the third. Substituting
the readout **more than doubles** what the rule reaches — 3.751 → 9.639 foods of 20 — and all three
substituted arms beat their own frozen control (q 0.016–0.026) where the anatomical one does not
(q 0.098). **So the readout is genuinely part of the limit.** But **none reaches competence** (best 5.75%
full clear against the 20% threshold, PPO's 18.945 foods on the same cell), so this is a result and not a
rescue: **R.1b stays blocked and R.2 stays live**. What does the work is the readout's **scale**, not its
direction — **+4.51 foods from the norm alone** with the anatomical direction preserved. And the control
added after seeing the harvested readouts earned itself: at matched norm, a **random** direction (9.639)
beats the **anatomical** one (8.265) which beats **PPO's** (6.123), so PPO's specific direction is the
worst of the three. Without that arm this would have read as "PPO found a better readout", which is the
opposite of what the ordering shows. Meanwhile **credited-synapse drift stays at 1.38–1.42×** across every
arm — the same 1.37–1.38 R.1c measured across every perturbation dimension — so the invariance now spans
**dimension and readout both**, which is the strongest remaining argument that what is wrong is credit
assignment. **This could not have satisfied D1 whatever it returned**: a rule taking its readout from a
gradient-trained run is not a biologically plausible local learner, and the harness carries that as a
field rather than as prose.

**Branch**: `feat/l4-frozen-readout`.

**Date**: 2026-09-14.

**OpenSpec change**: `add-l4-frozen-readout` (extends `plasticity-evaluation`: a diagnostic borrowing a
gradient-trained component states what it cannot establish, and a substituted component is tested against
a same-magnitude random control).

## Objective

Three tensors PPO trains are frozen under the rule, because the connectome's `plastic_weights` is
`w_chem` alone. R.1c eliminated two:

| frozen tensor | status before this record |
|---|---|
| `food_gains` (sensory projection) | **eliminated** — PPO moves it 0.177 relative, cosine 0.985 to init |
| `log_std` (action-noise scale) | **eliminated** — swept by hand across std 0.368–1.0, null |
| **`readout`** (2×4, motor-class means → action) | **untested**, largest relative norm change at 0.783 |

## The harvest, and why there were four arms rather than three

PPO on this cell at these eight seeds and at the arms' own action scale reaches **18.945 foods of 20 and
66.48% full clear**, tight across seeds (18.76–19.06). The stop clause passes, there are good readouts to
harvest, and **18.945** is the matched reference the minima use — 058's 19.31 came from 32 seeds at an
action std of 1.0 and is reported beside it.

**PPO does not refine the anatomical prior; it replaces it.** Norm **7.820** against the anatomical
default's **1.414** — 5.5× — at a cosine of **−0.178**. A representative harvested readout:
`[[2.22, 2.81, −0.84, −3.20], [3.01, −0.57, 4.65, 0.05]]`.

That was registered as a fourth arm **before any arm ran**. The readout multiplies the motor-class means
into the action mean while the action noise is **fixed** at std 0.368, so a 5.5× readout is a large shift
in commitment versus exploration — nothing to do with reading the motor classes better. Without
`anatomical_scaled`, "the direction matters" and "the scale matters" are not separable, and they imply
different follow-ups.

## The result

Four readouts at R.1c's best operating point — the `motor` mask, σ 0.1, `initial_log_std: -1.0` — seeds
1–8, 3000 episodes. 48 new runs, all succeeded. The `anatomical` pair is R.1c's committed arms, reused
under a load-path equivalence test rather than re-run.

| readout | norm | cos to anatomical | learning | frozen | shift | p | q | favouring | full clear | drift |
|---|---|---|---|---|---|---|---|---|---|---|
| `anatomical` | 1.414 | +1.00 | 3.751 | 3.150 | +0.60 | 0.098 | 0.098 | 7/8 | 0.02% | 1.38 |
| `anatomical_scaled` | 7.820 | +1.00 | 8.265 | 4.789 | **+3.48** | 0.020 | **0.026** | 6/8 | 1.97% | 1.38 |
| `rotated` | 7.820 | −0.03 | **9.639** | 6.805 | +2.83 | 0.004 | **0.016** | **8/8** | **5.75%** | 1.39 |
| `ppo` | 7.820 | −0.18 | 6.123 | 3.302 | +2.82 | 0.020 | **0.026** | 7/8 | 0.88% | 1.42 |

**All three substituted readouts beat their own frozen control by the registered minima; the anatomical
one does not.** The readout is part of the limit, and the effect is large — the learning level more than
doubles.

**Nothing reaches competence.** The best arm clears **5.75%** against the 20% threshold, and against
PPO's 18.945 foods on the same cell the best rule arm reaches 9.639. So the verdict is
`readout_helps_but_not_enough`: **R.1b stays blocked**, because block V's contrast is on time to
competence and no arm here becomes competent.

### The scale does the work

| comparison | gain | what it isolates |
|---|---|---|
| `anatomical` → `anatomical_scaled` | **+4.51 foods** | **scale**, direction held |
| `anatomical_scaled` → `ppo` | **−2.14 foods** | direction, norm held |

The norm alone accounts for the larger part of the improvement, with the anatomical direction untouched.
That points at the **readout-scale / action-noise interaction** rather than at reading the motor classes
better: a 5.5× readout against a fixed exploration noise is a policy that commits far harder, and both the
floor and the rule's contribution rise with it.

### The direction ordering, and why the fourth arm mattered

At matched norm: **`rotated` 9.639 > `anatomical_scaled` 8.265 > `ppo` 6.123.**

**PPO's own direction is the worst of the three.** That is the co-adaptation caveat this change registered
in advance, now measured: PPO's readout is tuned to PPO's `w_chem`, and the rule always starts from a
random one, so a readout fitted to another solution is actively unsuited to what the rule finds.

Without `anatomical_scaled` this would have read as `ppo` beating `anatomical` and been reported as "PPO
found a better readout" — **the opposite of what the ordering shows**. The arm was added on the strength of
the prepared checkpoints alone, before any arm ran.

**The registered ordering label does not fit this pattern and is not accepted.** The harness fired "any
change of direction helps at this scale", on the rule `max(rotated, ppo) − anatomical_scaled ≥ 1`. But
`ppo` sits *below* `anatomical_scaled`, so "any" is false: a **random** direction helps, and PPO's hurts.
The pattern is recorded as it is rather than as the branch named it.

### `ppo` is bimodal, which the mean hides

Per-seed learning foods:

| readout | per-seed |
|---|---|
| `anatomical` | 4.2, 3.3, 2.5, 3.4, 3.0, 2.2, 4.2, 7.1 |
| `anatomical_scaled` | 9.9, 9.7, 5.8, 8.3, 11.2, 8.2, 3.0, 9.9 |
| `rotated` | 12.8, 12.2, 11.4, 4.5, **17.3**, 2.8, 6.6, 9.5 |
| `ppo` | 10.6, 11.4, 13.4, **0.3**, 8.1, **0.4**, 4.5, **0.2** |

`ppo` either works or collapses: three seeds near zero against three above ten. That is what a
co-adapted readout looks like — it suits the rule's trajectory on some initialisations and fights it on
others — and it is why its mean of 6.123 is the least informative number in the table.

`rotated`'s best seed reaches **17.3 of 20**, within striking distance of PPO's 18.945. No arm's *mean*
comes close, but the substrate plainly can get there under this rule on a good initialisation.

### The drift invariance survives the readout

Credited-synapse drift: **1.38, 1.38, 1.39, 1.42** across the four arms, against R.1c's **1.37–1.38**
across every perturbation dimension from 302 units to 39.

So the rule writes about **1.4× its own weight norm** whatever readout it sits behind and whatever
dimension it is credited at. That invariance now spans **two independent structural axes**, and it is the
strongest remaining argument that the failure is in **credit assignment** rather than in the substrate's
input or output maps.

## What this licenses, and what it does not

**Not D1, whatever it returned.** Every substituted arm takes its readout from a gradient-trained run, so
none is a biologically plausible local learner. The harness carries
`satisfies_plausibility_deliverable: false` as a field. A positive result **locates a handicap**; it may
not be cited as "a local rule learns the connectome".

**Not R.1b.** No arm reaches competence, so block V's time-to-competence contrast is undefined for all of
them. **7b's gate is untouched.**

**What it does license** is a substrate-fidelity question with a measured motivation, and it is not the one
this change expected. The anatomical readout's **scale** — two unit-norm rows — costs more than its
direction does, against a fixed action noise. Whether that is a fault of the readout's parameterisation or
of the fixed `initial_log_std` beside it is now the open question, and it is cheap to ask.

### What this may not be cited as

- **Evidence that PPO found a good readout.** It found the worst of the three tested at its own norm.
- **A result about the readout's direction generally.** One norm, one mask, one cell, eight seeds.
- **A claim that scale is all that matters.** A random direction beat the anatomical one at matched norm by
  +1.37 foods, so direction is not inert — it is second to scale and does not favour PPO.
- **A rescue.** The best arm reaches 9.639 of 20 at 5.75% full clear against PPO's 18.945 and 66.48%.

## Conclusions

- **`readout_helps_but_not_enough`.** All three substituted readouts beat their own floor (q 0.016–0.026),
  the learning level more than doubles, and **none reaches competence**.
- **Scale does the work**: +4.51 foods from the norm alone, against −2.14 from PPO's direction.
- **PPO's direction is the worst of the three at matched norm**, which is the registered co-adaptation
  caveat measured rather than argued — and `ppo` is bimodal, working on three seeds and collapsing on
  three.
- **The fourth arm changed the conclusion**, and was added before any arm ran on the strength of the
  prepared checkpoints.
- **Drift stays at ~1.4× across dimension and readout both** — the invariance is now two-dimensional.
- **R.1b stays blocked, 7b's gate is untouched, and no committed verdict changed.**

## Next Steps

- [ ] **R.2 (e-prop)** is now the live path. All three frozen tensors are accounted for: two eliminated,
  the third shown to matter without being sufficient, and the drift invariance points at credit assignment.
- [ ] **The readout-scale / action-noise interaction** as a cheap substrate-fidelity question: the
  anatomical readout's unit-norm rows against a fixed `initial_log_std`, which this record shows costs more
  than the readout's direction does.
- [ ] **R.2b**, the matched dimension × horizon sweep, remains cheap and unaffected.

## Data References

- [`supporting/062-l4-frozen-readout/`](supporting/062-l4-frozen-readout/) —
  [`launch.md`](supporting/062-l4-frozen-readout/launch.md) (protocol, the harvest, the fourth arm's
  registration and the equivalence test), [`frozen_readout.json`](supporting/062-l4-frozen-readout/frozen_readout.json),
  [`per-seed.csv`](supporting/062-l4-frozen-readout/per-seed.csv).
- Campaign directories are gitignored: `campaigns/readout-harvest` (8 PPO harvests),
  `campaigns/readout-checkpoints` (the prepared checkpoints and their provenance sidecars),
  `campaigns/readout-substitution` (the 48 arm runs). The `anatomical` pair is R.1c's
  `campaigns/reduced-perturbation`.
