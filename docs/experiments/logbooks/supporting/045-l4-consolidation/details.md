# Consolidation screen details

Analysis by `scripts/analysis/l4_consolidation_screen.py` over the 24 runs in
`campaigns/l4-consolidation/logs` (3 arms × seeds 1–8 × 2000 episodes, all exit 0, 50 minutes on
16 workers). `screen.json` is the full output and `per-seed.csv` the table. The comparator is the
warm-start panel's committed `wt_clone_frozen` values; no comparator arm was re-run.

**This is a screen, not a confirmatory test.** It reuses seeds already reported, declares no
multiple-comparisons family and assigns no verdict. A pass would license running the registered
panel and nothing more.

## Result: none of the three passes

| arm | mean | Δ vs frozen clone | within hold | at or above | cosine to clone | rate multiplier |
|---|---|---|---|---|---|---|
| anchor | 13.0 | −25.7 | 2/8 | 0/8 | 0.49 | 1.00 |
| rigidity | 29.4 | −9.4 | **6/8** | 2/8 | **0.74** | 0.09 |
| oracle | 28.9 | −9.8 | 4/8 | 3/8 | 0.55 | 0.49 |

The frozen clone's mean is 38.7. **Rigidity fails on the mean clause alone**: it meets the
per-seed clause at exactly the required 6 of 8 and misses the 5-point band by 4.4 points, and
both of the seeds it loses are catastrophic rather than marginal (seed 5: 2.4 against 47.1;
seed 7: 9.2 against 61.3). The anchor fails both clauses by a wide margin, as the pilot recorded
in advance that it would. The oracle fails on both.

## Per-seed

| seed | frozen clone | anchor | rigidity | oracle |
|---|---|---|---|---|
| 1 | 39.3 | 13.8 | 37.4 | 43.0 |
| 2 | 44.0 | 37.6 | 36.6 | 42.6 |
| 3 | 40.0 | 1.8 | 46.4 | 18.0 |
| 4 | 21.3 | 10.2 | **53.4** | **55.2** |
| 5 | 47.1 | 1.0 | 2.4 | 0.0 |
| 6 | 33.3 | 25.8 | 28.0 | **58.6** |
| 7 | 61.3 | 13.6 | 9.2 | 13.2 |
| 8 | 23.3 | 0.2 | 21.4 | 0.4 |

## Reading

- **Rigidity preserves the policy better than anything tried on this substrate, and it is still
  not enough.** Its endpoint cosines to the clone are 0.67–0.79 on every seed, against the 0.2–0.45
  the unbraked rules produced in the diagnostic, and its rate multiplier is a uniform 0.09: the
  brake engaged hard and held the weights near where they started. Six of eight seeds finish
  within the assay's per-seed band and two finish above their own clone. It still loses 9.4 points
  on the mean, and it loses them at two seeds it destroys outright. **A 91% cut in the effective
  rate, applied exactly where reward had been writing, does not stop the drift** — which is the
  same rate-insensitivity the diagnostic measured across three orders of magnitude, now with the
  cut targeted per synapse rather than applied globally.
- **The oracle's failure is the one the launch record predicted, and it is about lag, not about
  gating.** Its early curves say so: on seeds 1, 2, 3 and 8 the first hundred episodes score 2–4%
  against clones of 23–44%, so the policy is gone before a trailing estimate at an EMA rate of
  0.01 can respond to it. The gate then stays open, because success is low, and the arm runs
  ungated for the rest of the run — a reactive brake in a positive-feedback loop. Where the gate
  did close it worked: seeds 4 and 6 gate at multipliers 0.06 and 0.09 and finish at 55.2 and 58.6
  against clones of 21.3 and 33.3. **This arm therefore does not settle the question it was
  registered to settle.** It shows that a brake tied to a signal arriving a hundred episodes late
  cannot hold a policy this rule destroys in tens, not that a quality-gated brake could not.
- **The better the starting policy, the more it loses.** The rank correlation between a seed's
  frozen-clone level and its delta is −0.79 (rigidity), −0.74 (oracle) and −0.57 (anchor). The two
  seeds with the strongest clones (61.3 and 47.1) are destroyed in all three arms, and the weakest
  clone (21.3) improves by 32 points under two of them. Part of this is regression to the mean on
  a noisy metric and it is descriptive, not a registered contrast — but the pattern is consistent
  with the drift walking to fixed points of its own, from which a strong start simply has further
  to fall.
- **The anchor is the clearest negative.** A restoring force toward a slow moving average of the
  weights, at the stiffness the pilot pinned, does not hold the policy at all (13.0, 0 of 8 seeds
  at or above) and its cosine of 0.49 is barely better than the unbraked rules'. Opposing
  *departure* is not the same as preserving *behaviour*: the rule can walk a long way inside the
  region the anchor tolerates and arrive somewhere useless.

## What this licenses

Nothing runs under any of these mechanisms: the panel stays gated. The registered consequence of
all three failing is that the queue moves to structured, pathway-specific instruction. Two
qualifications belong beside it, and both were fixed before the runs:

1. The oracle did not test what it was registered to test, for the reason its own launch record
   named. A quality-gated brake with a signal fast enough to act — an episode's outcome applied
   at once rather than through a slow trailing mean, or a within-episode proxy — is untested.
2. Rigidity failed by 4.4 points on one clause while meeting the other, with the best policy
   preservation this substrate has shown. It is the closest a local rule has come to holding a
   competent policy here.

## Campaign facts

- Pilot: 20 runs (10 grid points × seeds 1–2 × 2000 episodes), 39 minutes, all exit 0. Grid,
  criterion and pins in `launch.md`, written before the screen.
- Screen: 24 runs (3 arms × seeds 1–8 × 2000 episodes), 50 minutes, all exit 0, no tracebacks.
- One defect surfaced and was fixed before the pilot: the sign-model load guard refused every
  weight file written before sign grounding existed, including the clones this assay starts from.
  A missing sign buffer is now read as the all-zero buffer it can only be, and a cross-sign-model
  load is still refused.
