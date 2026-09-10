# Structured-instruction test details

Analysis by `scripts/analysis/l4_structured_instruction.py` over the 64 runs in
`campaigns/l4-routing/logs` (4 arms × seeds 1–16 × 3000 episodes, with the one registered
extension applied). `panel.json` is the full output; `per-seed.csv` and `curves.csv` are the
per-run table and learning curves.

**Read this against the rule's positive control.** The control ran the day this panel finished and
the rule **failed** it: on a one-step association whose analytic reference reaches the optimum, the
three-factor rule ends below the cue-blind floor with updates near-orthogonal to the policy
gradient. This panel therefore measures **what a non-learning rule does under two routing
regimes**. It is not evidence about aminergic pathways in the animal, and the verdict below should
not be read as one.

## Verdict: `no_routing_effect`

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| S1 | wt routed − wt global | −3.18 | −9.74 … +2.83 | 0.975 | 8/16 | fail |
| S2 | rn routed − rn global | −7.30 | −11.77 … −3.01 | 0.975 | 6/16 | fail |
| S3 | wt − rn under routing | +0.87 | −3.21 … +5.02 | 0.975 | 8/16 | fail |
| S4 | wt − rn under the global scalar | −3.24 | −10.85 … +5.19 | 0.975 | 6/16 | fail |

Neither S1 nor S2 confirms, so the registered map gives `no_routing_effect`. **That name understates
what was observed, and the map's gap is worth stating.** All four tests are one-sided, but they ask different questions. **S1 and S2** ask whether routing
*helps*, each arm against its own broadcast control, so a large negative effect produces a high q
and reads as "no confirmed benefit", not as "no effect"; descriptively the effect was negative on
both wirings, and S2's interval (−11.77 … −3.01) lies entirely below zero. **S3 and S4** are the
wiring contrast in its registered direction — wild type over rewired null, under routing and under
the global scalar — and are not tests of routing at all; neither confirms, and S4's negative
estimate has the scramble above the animal. The verdict name is kept as registered rather than
renamed after seeing the data; the honest reading is **no confirmed benefit, with observed
degradation**, and a future map of this shape should carry an explicit harm branch. S3 and S4
annotate and do not decide.

| arm | mean | median | max | competent (≥20%) | instructed fraction | instructed share |
|---|---|---|---|---|---|---|
| wt_global | 10.8 | 3.2 | 70.5 | 3/16 | — | — |
| wt_pathway | 7.7 | 4.5 | 24.3 | 1/16 | 0.711 † | 0.924 † |
| rn_global | 14.1 | 6.0 | 52.9 | 4/16 | — | — |
| rn_pathway | 6.8 | 3.1 | 33.1 | 1/16 | 0.781 † | 0.904 † |

† Averaged over the runs whose telemetry exports were retained — **10 of 16** for `wt_pathway` and **7 of 16** for `rn_pathway`, reported per arm in `panel.json` as `n_read`. The instructed *fraction* is a deterministic property of each build, so a subset mean is a fair estimate of it; the *share* is a mean over those runs and not over the arm. The global arms have no pathway, so their fraction is null and their share is the whole update by definition rather than by measurement.

## Reading

- **Routing made both arms worse, and the harm is concentrated in the good seeds.** The means fall
  by 3.2 (wild type) and 7.3 (rewired null), but the medians barely move — 3.2 → 4.5 and 6.0 → 3.1.
  What changes is the top of the distribution: the best seed falls from 70.5 to 24.3 on the wild
  type and from 52.9 to 33.1 on the rewired null, and the count of competent seeds drops from 3 to
  1 and from 4 to 1. Routing did not lower a floor; it removed the ceiling.
- **The intervention was real, not marginal.** The instructed share of the update's magnitude is
  0.92 on the wild type against an instructed *fraction* of 0.71, so the routed synapses carry
  disproportionately more of the learning than their count suggests. A null here is not a null
  because nothing happened.
- **The rewired null derives a wider pathway than the animal** — 0.781 against 0.711, and 75.5% against 71.1% at seed 1 — because
  degree-preserving rewiring spreads the aminergic neurons' targets over more of the network. It
  therefore received *less* of the unmodulated Hebbian term, and lost more. That is the opposite
  of what the pre-registered informal prediction expected (below), and it is descriptive.
- **The wiring contrast points the wrong way and does not confirm**, at −3.24 under the global
  scalar: the rewired null scored above the wild type at n = 16. Across six panels this contrast
  has now been positive, null and negative without ever confirming, which is what one would expect
  of a measurement made with an instrument that does not learn.

## The prediction, and where it failed

Recorded before the runs: `no_routing_effect` at ~65%, `routing_helps_wild_type_only` at ~15%,
`routing_helps_both` at ~12%. **The verdict was right and the direction was wrong.** The reasoning
was that panel 1's committed table shows the unmodulated Hebbian floor beating the modulated arm
(30.2 against 17.8 on the wild type), so replacing the modulator with 1.0 on ~29% of synapses
should move the arm toward the better floor — a weighted-blend estimate of about +3.6. Routing
instead cost 3.2 points, and cost the rewired null 7.3.

The error was assuming outcomes combine linearly in the mixing fraction. They do not: running two
learning regimes in one network is not a weighted average of running each alone, and on a bimodal
outcome the arithmetic of means says almost nothing about what happens to the seeds that were
working. Recorded because a prediction that was committed in advance should be scored in public,
including when it is wrong.

## The substrate has not drifted in six panels

The registered consistency check compares the concurrently re-run global arms against panel 1's
committed values on the seeds they share:

| arm | panel 1 committed (seeds 1–8) | re-run here | difference |
|---|---|---|---|
| wt_global | 17.82778 | 17.82778 | 0 |
| rn_global | 11.2 | 11.2 | 0 |

**Bit-for-bit.** Everything added since panel 1 — the trace substrate, sign grounding, three
consolidation mechanisms, two decorrelating terms, the routing seam — is byte-identical on its
default path, as each change's tests asserted individually and as nothing had checked end to end
until now. Every committed comparator in Logbooks 040–046 rests on the same substrate it did when
it was written.

## Campaign facts, including what went wrong

- 64 runs at 3000 episodes; one registered extension applied (`wt_global` seed 3, non-converged at
  the budget, re-run at 4500 and replacing the shorter log as registered).
- **Seven runs were lost to operator error and re-run.** A branch switch during the campaign left
  the working tree without the routed arms' configs, so `rn_pathway` seeds 10–16 failed
  immediately — six on a missing config path and one on a torn tree (the brain from one branch, the
  rule from another). Runs already in flight were unaffected; the failures were discarded, not
  analysed, and the seven were re-run on the correct branch.
- One further log was destroyed by copying an empty file over it during the extension, and
  regenerated. The analysis refused the incomplete arm both times rather than scoring it, which is
  how both errors surfaced immediately.
- The panel as scored is 16 seeds per arm with no imputed values.
