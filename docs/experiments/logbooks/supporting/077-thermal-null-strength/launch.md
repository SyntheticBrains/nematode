# A.6t — the null-strength control and its split, on block V's thermal cell: registration and launch

**Registered 2026-10-04, before any scored run.** Change: `add-thermal-null-strength`.

A 15-episode smoke on seed 385 checked that all eight configs run; it exited cleanly on every arm,
its directory was deleted, and no statistic was read from it. No other seed in the band has run.

## The question

On hard350, about half of block V's `auc_success` lead over the degree-preserving null came from that
null's rewired gap junctions ([Logbook 074](../../074-null-strength-control.md),
[075](../../075-gap-only-split.md)). Block V's second cell, the thermal-plus-foraging cell at target
20, has only A.1's initialisation control behind it. Does the same hold there?

## What each null preserves, and what it does not

| property | current null (`rewired_degree_preserving`) | chemical-only null (`rewired_chemical_only`) | gap-held null (`rewired_gap_junctions_held`) |
|---|---|---|---|
| chemical in- and out-degree per neuron | preserved | preserved | preserved |
| which neurons connect chemically | rewired | rewired, a different random graph at each seed | **the current null's graph exactly** |
| gap-junction pairs and counts | rewired, strength moves | the wild type's | the wild type's |
| autapses (38 in the wild type) | lost | the wild type's | lost, as in the current null |
| one- and two-hop sensory-to-motor routes | manufactured | manufactured | manufactured |

## The design

Four wirings, each learning and frozen, under **PPO at block V's committed thermal point**: the
edge-order draw, pooled readout, depth 4, `target_foods_to_collect: 20`. **Seeds 385–512 (128)**,
fresh: A.6 and its split ended at 384. 3,000 episodes per run. 1,024 runs.

**Configs.** The wild-type and current-null arms are block V's committed thermal configs (`*_t20`),
unchanged. The four new configs differ from their current-null parents in `wiring` alone; a test
re-reads each through the real loader, and another builds the gap-held and current nulls at one seed
and checks they share the chemical mask while the gap-held null carries the wild type's gap junctions.

**Levels.** Three — `full`, `chemical` and `gap_held` — each gated against its own frozen floors. The
wild-type runs serve all three.

## Two primary interactions

```text
combined = gap(wild type vs chemical-only null) − gap(wild type vs current null)
split    = gap(wild type vs gap-held null)      − gap(wild type vs current null)
```

Both are paired by seed, and the wild type cancels, so each is the current null minus the narrower
null, seed by seed. Each is positive when the wild type stands better against the narrower null.

- **`combined`** holds gap placement, gap strength and autapses together, as A.6 did.
- **`split`** holds the gap junctions alone on the current null's exact chemical graph, as A.6's split
  did. It separates the gap junctions from the autapses, and no further: placement and strength move
  together.

## The metric

**`auc_success` is the primary**, as on hard350. `episodes_to_30pct_success` is reported beside it with
the censoring rule's own choice.

## The registered minimum, in both directions

**0.0553 `auc_success`: 2/3 of A.1's thermal effect at this point** (+0.0829, Logbook 070's thermal
`baseline_gap_mean`). Both interactions are read against it.

It is the thermal cell's own effect, for two reasons. Block V's magnitude on this cell has not matched
hard350's on any seed set: A.1 found it at 35% of its committed size on the episode metric and 51% on
`auc_success`. And A.6's hard350 minimum (0.0407) would be judged against a different effect.
Neither minimum is taken from this campaign's data.

**States** are B.1c's `classify`, imported unchanged: q from the two-sided folded Wilcoxon, BH-FDR
corrected across **the two primary interactions**, and an 80% bootstrap interval.

| state | condition |
|---|---|
| `move_wt` | q < 0.05, interval above zero, mean ≥ minimum |
| `move_null` | q < 0.05, interval below zero, mean ≤ −minimum |
| `below` | q < 0.05, interval excludes zero, absolute mean < minimum |
| `no_move` | q ≥ 0.05, interval inside (−minimum, +minimum) and including zero |
| `unresolved` | anything else |

## Sensitivity, from frozen committed data

The minimum detectable effect is `2.487 × sd / √n`. The spread is A.1's per-seed thermal interactions on
`auc_success`, the only committed thermal per-seed spread at this point.

| source of the spread | sd | n | MDE | MDE ÷ minimum |
|---|---|---|---|---|
| A.1 thermal, `dense_mask` | 0.2686 | 128 | 0.059 | 1.07 |
| A.1 thermal, `per_neuron_fanin` | 0.3008 | 128 | 0.066 | 1.19 |

**The proxy's expected error, stated.** On hard350, A.6's achieved interaction spread came in at 0.88
of its A.1 proxy (0.0614 against 0.0700). At that ratio the MDE here is about 0.053 to 0.058, near the
minimum. No direction is claimed: the wild type cancels in these interactions, which narrows the spread,
and two different null graphs per seed in `combined` widen it.

**What the panel can and cannot show.** If thermal behaves as hard350 did, the combined move is about
half the thermal effect, roughly 0.04, which is **below the minimum**. So the likeliest registered
outcome is `below` or `unresolved`, and the panel's main value is the reported gaps (below).

## The verdict maps

**`combined`** — A.6's map:

| state | verdict |
|---|---|
| `no_move` | **chemical**, if the gap against the chemical-only null excludes zero above; otherwise **no_gap_to_attribute** |
| `move_null` | **gap_or_autapse** |
| `move_wt` | **amplified** |
| `below` | **below_minimum** |
| `unresolved` | **unresolved** |

**`split`** — the gap-only split's map:

| state | verdict |
|---|---|
| `move_null` | **gap_junctions** |
| `below` | **partial** |
| `no_move` | **not_gap_junctions** |
| `move_wt` | **opposite** |
| `unresolved` | **unresolved** |

Block V's thermal claim is restated only from these verdicts and the gaps below, never from a share.

## Gates, read before the interactions

- **Floors.** Every learning arm must beat its own level's frozen floor. A level where either wiring
  fails, or both arms saturate at the 90% bar, is **unreadable**, and so is the panel.
- **Drift.** Recorded as PPO's positive control: PPO writes the chemical matrix, so it must move.

## Reported beside, not registered

- **The three gaps**, wild type against each null, with their intervals, on both metrics. The gap against
  the chemical-only null and against the gap-held null are what the combined paper's 8a half cites for
  thermal, as +0.025 `auc_success` against the gap-held null is cited for hard350.
- **The split's share of the combined move**, `split ÷ combined`, as on hard350 (about 87% under PPO).

## Launch

```bash
cfgs=(); for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import thermal_null_strength as t;print(' '.join(t.LEVELS_BY_STEM))"); do
  cfgs+=(--config configs/scenarios/thermal_foraging/$s.yml); done
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 385-512 --runs 3000 --workers 16 \
  --output-dir campaigns/a6t-thermal \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

uv run python scripts/analysis/thermal_null_strength.py --campaign campaigns/a6t-thermal \
  --out-dir build/a6t --out build/a6t/control.json --csv build/a6t/per-seed.csv
```

## Artefact retention (A.0)

- **Committed:** the per-seed CSV, the analysis JSON and this launch record.
- **Archived off-repo:** the raw campaign logs.
- **Deletion:** a campaign directory is removed only after its CSV is committed.

## Cost

A.1's thermal pilot ran 48 runs in 2,769 s at 14.8× parallelism, about 58 s of wall-clock per run.
1,024 runs is about **16.4 hours**.
