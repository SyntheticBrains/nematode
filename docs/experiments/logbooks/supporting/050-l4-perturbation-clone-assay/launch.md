# Clone-assay launch record: the node-perturbation eligibility

Written and committed before the runs.

- **Date**: 2026-09-10
- **Pinned state**: `feat/l4-perturbation-clone-assay` at the commit adding the arm and its
  registry entry.
- **Step 3 of the clearance order** fixed in I.1's launch record before its result existed:
  positive control → gradient alignment → **clone assay** → connectome arm. Steps 1 and 2 passed
  ([records](../049-l4-node-perturbation/details.md)): at σ = 0.2 the variant covers 88% of the
  floor-to-optimum gap on 8 of 8 seeds with an alignment of +0.263.

## The question

Does a rule that learns from random weights **hold a policy that is already good**? The two are
different capabilities and this project has watched them come apart: the three-factor rule found
better policies on some seeds and left them, and all three consolidation mechanisms braked the
drift without holding the clone. This is also the variant's **first contact with the connectome** —
everything so far has been the MLP yardstick on a synthetic task.

## The assay, unchanged

The protocol registered with the clone-destruction diagnostic, in arms, comparator, budget, metric
and pass rule, so this result is directly comparable with the three mechanisms that failed it:

| element | value |
|---|---|
| arm | the wild-type plastic clone arm with the variant's rule keys and nothing else changed |
| start | `campaigns/l4-warm-start/clones/plastic_wt_seed{seed}.pt` |
| seeds | 1–8, paired |
| budget | 2000 episodes, no extension |
| metric | the committed plateau-tail full-clear success |
| comparator | the same seeds' `wt_clone_frozen` values (39.3, 44.0, 40.0, 21.3, 47.1, 33.3, 61.3, 23.3; mean **38.7**) |
| **holds** | mean within 5 points of the frozen clone's **and** ≥ 6 of 8 seeds no more than 10 points below their own |
| **improves** | mean above the frozen clone's **and** ≥ 6 of 8 above their own |
| **pass** | holds or improves |

A **screen, not a confirmatory test**: it reuses seeds Logbook 043 reported, so a pass licenses
running the registered panel and nothing more. The endpoint cosine to the clone is reported beside
the metric, since a variant can hold the metric while having rewritten the policy underneath it.

## σ = 0.2, and why it is not re-tuned

That is the value the positive control pinned. A gate whose parameter is chosen against its own
outcome is a search, not a gate. If the variant fails at this σ, the record says it failed the
assay at the σ that cleared the control.

**Perturbation cuts both ways here, and that is stated before the result.** It is exploration, so
it will degrade a competent policy's immediate behaviour — a policy at its optimum can only be made
worse by jitter. It is also what lets the rule hold anything at all, since without it the rule
drifts. The assay measures the net, and neither direction should be read as a surprise afterwards.

## What each outcome licenses

- **Pass** — the variant learns from random weights *and* holds a competent policy. Step 4 opens: a
  connectome arm becomes buildable, with any panel still behind I.2's statistic and metric. It
  would be the first time in Phase 7 that a rule cleared both gates.
- **Fail** — the variant learns but does not hold. That is the consolidation mechanisms' failure in
  a new place and genuinely informative: it would say the eligibility fixed credit assignment
  without fixing retention, and that block I needs a consolidation mechanism **on top of** the
  working eligibility rather than instead of it. The panel stays gated.

## Command

```bash
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_clone_nodeperturbation
uv run python scripts/run_campaign.py --config ${P}.yml --config ${P}_frozen.yml \
  --seeds 1-8 --runs 2000 \
  --output-dir campaigns/l4-perturbation-clone -- --theme headless --track-experiment
```

## Analysis

`uv run python scripts/analysis/l4_consolidation_screen.py --campaign-dir campaigns/l4-perturbation-clone --out docs/experiments/logbooks/supporting/050-l4-perturbation-clone-assay/screen.json --csv .../per-seed.csv`

## Results

Run 2026-09-10, 16 runs across both arms. **The variant fails the assay** (mean 12.0 against the
frozen clone's 38.7, 1/8 within hold) — and the frozen-perturbation control, registered by spec
review while the plastic arm was running, changes what that failure means. With its weights
untouched (cosine 1.00) it scores **8.9**: perturbation alone takes the clone from 38.7 to 8.9,
and it is already at 9.1 in its first quarter. The plastic arm scores **12.0, above** that
baseline, starting at 15.2 and settling at 12.0.

**The rule did not take the clone apart; the perturbation did, before the rule acted.** The assay
at this σ could not test retention because no competent policy survived to be retained. The
registered fail stands; the finding is the tension it exposes — the σ that makes the rule learn is
the σ that makes a competent policy unrunnable. Full reading in `details.md`.
