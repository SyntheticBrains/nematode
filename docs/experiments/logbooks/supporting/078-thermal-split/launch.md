# A.6t's follow-up — the gap-only split on block V's thermal cell at target 35: registration and launch

**Registered 2026-10-05, before any scored run.** Change: `add-thermal-split`. Seeds 513–640 have
not run.

## Why

[Logbook 077](../../077-thermal-null-strength.md) ran A.6 and its split at block V's thermal point
(target 20) and was unreadable: every level saturated, both learning arms at 94–96% against the 90%
bar. Described, not a verdict, block V's thermal lead over the current null (+0.089 `auc_success`) was
not visible against nulls holding the wild type's gap junctions. This panel asks the split's question at
a readable thermal point.

## How the point was chosen: a gate-only pilot

Six arms (wild type, current null and gap-held null, each learning and frozen), seeds **1001–1004**,
disjoint from every registered band, 3,000 episodes, at food targets 25, 30 and 40, then 35. Each target
was judged only by `scripts/campaigns/gate_preflight.py` (default 5-point margin). The rule, written down
before the target-35 pilot ran (`selection-rule.md`, quoted in `pilot.json`): **the lowest pilot target at
which every level is readable**.

| target | full | gap_held | wild type `auc_success` |
|---|---|---|---|
| 25 | saturated | saturated | 0.682 |
| 30 | near the bar | near the bar | 0.521 |
| **35** | **readable** | **readable** | **0.312** |
| 40 | readable | readable | 0.144 |

**Target 35 is chosen.** The pilot showed each arm's plateau to the person choosing, so the rule was fixed
before 35 ran and leaves no discretion. Every figure is in [pilot.json](pilot.json).

The preflight at target 35, through this panel's own module (`--panel thermal_split`):

| level | status | wild type plateau | null plateau | floors |
|---|---|---|---|---|
| full | readable | 71.5% | 64.0% | 0.0% / 0.0% |
| gap_held | readable | 71.5% | 72.9% | 0.0% / 0.0% |

## The design

**Three wirings**, each learning and frozen, under PPO at block V's committed point except the target:
edge-order draw, pooled readout, depth 4, **`target_foods_to_collect: 35`**. The wild type and the current
null; and the gap-held null, which has the current null's chemical graph exactly and the wild type's gap
junctions, so against the current null it differs in its gap junctions alone. **Seeds 513–640 (128)**,
3,000 episodes, **768 runs**.

**Configs.** Six target-35 configs, each its target-20 parent with that one key changed, through
`generate_thermal_target_configs.py`; a test re-reads each through the real loader, and another confirms
the gap-held null shares the current null's chemical mask. Their bodies are byte-identical to the
pilot's.

## Two registered readings, corrected together

```text
split = gap(wild type vs gap-held null) − gap(wild type vs current null)     paired by seed
lead  = gap(wild type vs gap-held null)
```

- **`split`** asks how much holding the gap junctions moves the wiring gap, as A.6's split did.
- **`lead`** asks whether the wild type leads a null with its own gap junctions at all. On hard350 it did
  (+0.025 `auc_success`); this is the figure the combined paper cites per cell.

**Primary metric `auc_success`**; `episodes_to_30pct_success` reported beside with the censoring rule's
choice. BH-FDR across the two readings per metric; states are B.1c's `classify`.

## The registered minimum

**0.0239 `auc_success`** = 2/3 × A.1's thermal effect at target 20 (+0.0829) × the wild type's own
`auc_success` ratio between the targets (0.312 at 35 on the pilot's 4 seeds ÷ 0.721 at 20 on A.1's 32).
The ratio uses the wild type alone, so no wiring gap informs the minimum. It is a scaled reference from
the nearest committed effect, not this point's own effect, which no committed data measures.

## Sensitivity

The proxy is A.6t's achieved per-seed spread at target 20 (committed in
[077's per-seed.csv](../077-thermal-null-strength/per-seed.csv)), at 128 seeds:

| reading | sd at target 20 | MDE unscaled | sd scaled by the auc ratio | MDE scaled | minimum |
|---|---|---|---|---|---|
| `split` | 0.154 | 0.034 | 0.067 | 0.015 | 0.024 |
| `lead` | 0.188 | 0.041 | 0.081 | 0.018 | 0.024 |

**No direction is claimed.** If the spread scales with the wild type's `auc_success`, both readings are
resolvable at the minimum; if it does not shrink at all, the detectable effects run 1.4–1.7× the minimum.
The achieved spread will be reported beside, and never used to re-read.

## Verdict maps

| state | `split` | `lead` |
|---|---|---|
| `move_null` | **gap_junctions** | **null_leads** |
| `below` | **partial** | **lead_below_minimum** |
| `no_move` | **not_gap_junctions** | **no_lead** |
| `move_wt` | **opposite** | **lead_remains** |
| `unresolved` | **unresolved** | **unresolved** |

`move_wt` for `lead` means the wild type leads the gap-held null by at least the minimum; `move_null`
means the null leads by at least the minimum.

## Gates, read before the readings

Both levels: each learning arm beats its own frozen floor, and the two learning arms do not both reach
the 90% bar. Either failing makes the panel **unreadable**. The pilot puts the arms 17–26 points below
the bar. Drift is recorded as PPO's positive control.

## Reported beside, not registered

The gap against the current null at target 35, which is block V's effect at this point.

## Launch

```bash
cfgs=(); for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import thermal_split as t;print(' '.join(t.LEVELS_BY_STEM))"); do
  cfgs+=(--config configs/scenarios/thermal_foraging/$s.yml); done
uv run python scripts/campaigns/gate_preflight.py --panel thermal_split --logs campaigns/a6t2-pilot/logs
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 513-640 --runs 3000 --workers 16 \
  --output-dir campaigns/a6t2-thermal-split \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

uv run python scripts/analysis/thermal_split.py --campaign campaigns/a6t2-thermal-split \
  --out-dir build/a6t2 --out build/a6t2/control.json --csv build/a6t2/per-seed.csv
```

## Artefact retention (A.0)

Committed: this record, `pilot.json`, the per-seed CSV and the analysis JSON. Archived off-repo: the
pilot's and the campaign's raw logs.

## Cost

From the matched target-35 pilot: learning runs 20.5 min, frozen 4.7 min (median). 384 of each at 16-way
parallelism is about **10.5 hours**; A.6t's learning runs ran about 15% slower under sustained full load,
so **10.5–12 hours**.
