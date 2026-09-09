# Consolidation screen launch record

Written and committed before the pilot ran, per the registration.

- **Date**: 2026-09-09
- **Pinned state**: commit `HEAD` of `feat/l4-consolidation` (the three mechanisms, the three
  clone-arm configs, the screen harness). The commit carrying this record precedes the pilot.
- **What is screened**: three consolidation mechanisms, one arm each, every arm the wild-type
  plastic clone arm with the mechanism's rule keys and nothing else changed.
- **The assay**: the protocol registered with the clone-destruction diagnostic. Seeds 1–8 paired,
  2000 episodes, no extension, plateau-tail full-clear success, started from
  `campaigns/l4-warm-start/clones/plastic_wt_seed{seed}.pt`. Comparator: the warm-start panel's
  committed `wt_clone_frozen` values (39.3, 44.0, 40.0, 21.3, 47.1, 33.3, 61.3, 23.3; mean 38.7).
  **Holds** = mean within 5 points of the frozen clone's **and** ≥ 6 of 8 seeds no more than 10
  points below their own. **Improves** = mean above the frozen clone's **and** ≥ 6 of 8 above
  their own. **Pass** = holds or improves.
- **A screen, not a confirmatory test**: it reuses seeds the warm-start panel already reported.
  No multiple-comparisons family is declared and no verdict is assigned. A pass licenses running
  the registered panel and nothing more.

## The pilot, declared before it ran

Two mechanisms have hyperparameters with no value to inherit. Seeds 1–2 of the same clone arm at
the same 2000-episode budget, over this grid:

- **anchor**: stiffness `κ_a ∈ {0.01, 0.1, 1.0}` × anchor rate `ρ_a ∈ {0.0, 0.001}` — 6 runs/seed.
- **rigidity**: strength `κ_c ∈ {1, 10}` × growth `γ_c ∈ {0.01, 0.1}` at decay `λ_c = 0.001` —
  4 runs/seed.

**Criterion**: pin the combination with the highest mean plateau tail across the two seeds; ties
break toward the weaker constraint (the smaller `κ`). 20 pilot runs.

The **oracle** has no pilot: both its values are fixed by the comparator rather than chosen —
`s_ref = 0.4`, just above the frozen clone's 38.7% mean, and an EMA rate of `0.01`.

## The oracle is a bound, not a candidate

Declared before the run, per the registration. The oracle arm gates the plasticity rate on the
**environment's own episode-success flag**, a property of the task's scoring rather than of the
reward stream the rule observes. It is not a mechanism a nervous system could host and is not
offered as one. It exists to separate two readings of a negative screen: if the two shippable
mechanisms fail and the oracle passes, consolidation works on this substrate and the rule cannot
see when to apply it; if the oracle fails too, consolidation is not the missing piece.

Two structural facts follow from its pins and are recorded here so its result is read correctly:

1. With `s_ref` at the comparator's own level the arm can hold that level and **cannot register as
   improving** — the pass rule's "improves" branch is unavailable to it by construction.
2. Its trailing estimate starts at zero, so its **first ~100 episodes run near the full rate**
   while the estimate warms up. If that window is enough to take the clone apart, the arm will
   show it, and that is a result about the rule's speed rather than about gating.

## Commands (from the repository root)

```bash
# Pilot (grid above, seeds 1-2, 2000 episodes)
uv run python scripts/run_campaign.py --config <each pilot config> \
  --seeds 1-2 --runs 2000 --output-dir campaigns/l4-consolidation-pilot -- --theme headless --track-experiment

# Screen (seeds 1-8, 2000 episodes)
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_clone
uv run python scripts/run_campaign.py \
  --config ${P}_anchor.yml --config ${P}_rigidity.yml --config ${P}_oracle.yml \
  --seeds 1-8 --runs 2000 --output-dir campaigns/l4-consolidation -- --theme headless --track-experiment
```

## Analysis

`uv run python scripts/analysis/l4_consolidation_screen.py --campaign-dir campaigns/l4-consolidation --out docs/experiments/logbooks/supporting/045-l4-consolidation/screen.json --csv docs/experiments/logbooks/supporting/045-l4-consolidation/per-seed.csv`;
the pass rule, the comparator and the reporting are the ones fixed in that script.

## Pilot results

*(written here after the pilot and before the screen)*

## Screen results

*(written here after the screen)*
