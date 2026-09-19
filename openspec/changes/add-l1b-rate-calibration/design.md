# Design: L.1's interaction re-read at the matched rate

## Two arms complete a 2×2 that is already three-quarters measured

L.4's rate-matched baseline ran `wt_wide` and `rn_wide` at 0.0001 on seeds 1–96 (Logbook 067). L.1's
four floors are inert to the rate under `freeze_updates` and are shared. So the 2×2 at 0.0001 is
missing exactly its two pooled learning cells, and those are the two arms this change runs:

| | pooled (8 params) | per-neuron (78) |
|---|---|---|
| wild type @ 0.0001 | **new** | 0.7013 (067) |
| rewired null @ 0.0001 | **new** | 0.7990 (067) |

The primary is L.1's primary, at one rate:

```text
I_1e-4 = (wt_wide − wt_pooled) − (rn_wide − rn_pooled)      per seed at 0.0001, then the paired test
```

Positive: widening still favours the wild type relative to the null at this rate. Zero: it does not.
Negative: it favours the shuffle.

**The sign-flip form of L.1 cannot reproduce here, and the design says so before the runs.** At 0.0001
the wide wiring effect is −0.0977 with the null ahead (067, 96 seeds). L.1's *registered* primary was
the interaction, not the sign flip; the sign flip was the size the interaction happened to have at
0.001. This change reads the registered primary. A positive here would mean the pool hides
wiring-specific structure at both rates, while the wild type leads the null at neither width at
0.0001 — a weaker claim than L.1's, stated as such.

## Why 0.0001 and not a sweep

Two decades either side of 0.001 have been looked at on this substrate. At the pooled width, L.0's
check on the wild type found 0.001 the optimum (seeds 101–104, tight at 0.001 only). *(Corrected
2026-09-19, before the campaign: L.0's published figures are **whole-run** mean foods — 17.417,
12.086, 12.810 — while every contrast here reads the **plateau-tail** mean, which on the same runs
is 18.085, 14.488 and 12.842. The ordering and the spread reading are unchanged; the citable margin
at the pooled width is 3.60 foods, not 5.33.)* At the per-neuron width, L.4's atlas check found 0.001 too
high for the grounded arms and 0.0001 clean, and the rate-matched baseline found 0.0001 better for
both wide wirings by a wide margin. So the rate already interacts with the width in the one wiring
that has been measured at both, and 0.0001 is the one alternative rate at which the wide cells exist
at 96 seeds. A third rate (0.0003) would show the shape of the reversal and is a question for the
dynamics rung's own calibration, not for the close: it would add 384 runs to answer something the
synthesis does not need.

## The minimum effect

L.4 registered a minimum as two-thirds of the removable effect. Here the quantity is L.1's own
interaction, +0.2818, and the question is whether it *survives*, so the minimum is **half of it,
0.141**: the pool-hiding claim survives if at least half of L.1's interaction is present at the
matched rate. Below that, a significant interaction is named *shrunk below half* and reads
rate-specific, because a claim that has lost more than half its size to a one-decade rate change is a
claim about the rate.

**Sensitivity, from the realised spreads.** The wide wiring effect at 0.0001 has sd 0.135 over the
96 seeds (067). The pooled cells at 0.0001 are unmeasured; at 0.001 they spread 0.218–0.228 per cell.
Treating the cells as independent (L.1 measured ρ ≈ 0.08 between widths):

| pooled-cell sd assumed | interaction sd | se at n = 96 | detectable at 80% | power at 0.141 |
|---|---|---|---|---|
| 0.22 (as at 0.001) | 0.339 | 0.0346 | 0.097 | **0.98** |
| 0.30 (pessimistic) | 0.445 | 0.0454 | 0.127 | **0.87** |

The realised sd and power are carried as fields in the record. The panel is 96 seeds because every
committed cell is on seeds 1–96 and a paired 2×2 needs the same seeds in every cell.

## Eight tests in one BH-FDR family

Per the L.1 and L.4 precedent: the interaction at 0.0001; the width and wiring main effects at 0.0001;
four learning gates, each 0.0001 arm against its own-width floor, one-sided; and the **three-way**,
`I_1e-3 − I_1e-4` per seed with L.1's committed learning arms supplying `I_1e-3` through the same
efficiency call. The priors are L.1's committed tests on the same floors and are not re-tested. The
wide wiring effect at 0.0001 is already committed in 067 and is reported as a reference outside the
family, as 067 did.

## The baseline is committed data, reused under the identity requirement

Three campaigns supply cells: this change's (pooled @ 0.0001), `campaigns/feature-ablations` (wide @
0.0001) and `campaigns/readout-width` (floors, and L.1's learning arms for the three-way). The wide
0.0001 arms ran under the current output controls on 2026-09-18 and nothing in `packages/`, the
runner or the configs has changed since; L.1's arms ran before the controls, and L.4's identity check
covered only `wt_wide` and `rn_wide` at 0.001 at seed 1. The spec's requirement is one seed per reused
arm under the current path, so **eight runs at seed 1** — the four floors, L.1's two pooled learning
arms, and the two wide 0.0001 arms — are compared to their committed logs on every field `read_log`
parses. L.1's wide learning arms were checked by L.4 and are not re-checked. Any field differing
re-runs that arm's cell in full for all 96 seeds; there is no partial reuse.

## The stop clauses

1. **Pilot on seeds 101–104**, eight runs: both pooled 0.0001 arms, before any registered seed. Each
   learns above L.1's pilot pooled floors (`campaigns/readout-width-pilot`), and each differs from
   the 0.001 pooled pilot arm of the same wiring. L.0's rate-check ran the wild-type arm on these seeds
   under the old flags; the pilot reproduces that arm under the controls, which is one more identity
   point. No reading at four pairs.
2. The identity check above passes, or the affected cell is re-run.
3. `launch.md` committed before anything runs.

## What the reading conditions

L.1's `pooling_hid_structure` is a committed verdict and is not rewritten. Each reading adds a dated
condition beside it in the tracker, Logbook 066 and the roadmap:

- `survives`: *the pool hid structure at 0.001 and at 0.0001; the wild type's lead at the per-neuron
  width is a 0.001 result.* The synthesis carries the interaction as the phase's positive and the
  sign flip as rate-specific.
- `rate_specific`: *the pool hid structure at 0.001 only.* The synthesis carries L.1 as a positive at
  one pinned rate that a decade lower does not reproduce, and the phase's positive for the wiring
  under a plausible learner is stated with that condition in the same sentence.
- `width_favours_the_shuffle_at_this_rate`: reported as the reverse direction, not explained, and the
  synthesis says the sign of the interaction itself depends on the rate.

None of them decides the rate. Both wirings learn better at 0.0001, so 0.001 is the worse operating
point for either; which rate a rung after 7 should run at is that rung's calibration to make.

## The honest prior

**`pool_effect_is_rate_specific`.** At the pooled width the wild type loses ~5 foods going to 0.0001
(L.0's check); at the per-neuron width both wirings gain, the null most. For the interaction to hold
at 0.141, the pooled null would have to lead the pooled wild type by at least 0.24 at 0.0001, against
0.097 at 0.001. That is not impossible — the null gained +0.44 from the rate at the wide width and
may gain at the pooled width too — but the wild type's known loss at the pooled width pushes the
other way. Registered before any arm runs.

## Read-only, and why

`scripts/analysis/l4_readout_width.py`, `scripts/analysis/l4_feature_ablations.py`,
`scripts/analysis/connectome_structure_efficiency.py` and `scripts/analysis/wiring_premise.py` are
unmodified and a test asserts it. Two of them produced the cells this 2×2 reuses; editing either
would let "the instrument changed" compete with the interaction. The helpers are imported from
`l4_readout_width.py`, as `l4_feature_ablations.py` did.
