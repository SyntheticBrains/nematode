# M.8 — plastic gap junctions on the settling substrate: registration and launch

**Registered 2026-10-07, before any scored run.** Change: `add-plastic-gap-junctions`. Optional M.8, run
alongside C.1's build from a separate worktree.

## The question

Logbooks 075 and 078 found that the degree-preserving null's rewired gap junctions carried about half of
block V's lead on hard350 and about 84% on the thermal cell at target 35. A rewiring moves gap
**placement** and gap **strength** together, because EM counts travel with their edges. **Is the wild
type's gap-junction advantage in where its junctions are, or in how strong they are?**

## The arms

The thermal cell at target 35, under PPO at block V's point otherwise: edge-order draw, pooled readout,
depth 4, Cook 2019, settling dynamics. The two wirings:

- **The wild type.**
- **The gap-only null** (`rewired_gap_junctions_only`). Its chemical graph, autapses included, is held
  exactly; its gap junctions are rewired by the degree-preserving swap with their counts. It differs
  from the wild type in gap placement and in each neuron's total gap strength, and in nothing else.

| level | wild type | gap-only null |
|---|---|---|
| `fixed` | learning; frozen (both reused from Logbook 078) | learning; frozen |
| `plastic` | learning with `plastic_gaps`; the same frozen floor | learning with `plastic_gaps`; the same frozen floor |

`plastic_gaps` gives every existing gap pair a learnable positive multiplier, symmetric and starting at
1, so each wiring starts from its own strengths and can tune them. A frozen arm learns nothing, so one
floor per wiring serves both levels.

**Seeds 513–576 (64)**, the first 64 of Logbook 078's band. That makes 192 new runs: the null's fixed
learning and frozen arms, and both wirings' plastic learning arms.

## The readings

Three readings, corrected together (BH-FDR on the primary). Each is positive when the wild type is ahead:

```text
base        = gap(fixed)
lead        = gap(plastic)
interaction = gap(plastic) − gap(fixed)
```

**Primary metric**: `auc_success`, with episodes reported beside.

**The minimum is 0.0577**, two-thirds of the 0.0865 that holding the null's gap junctions moved the
lead on this cell (Logbook 078's split). It is the cell's own committed gap effect.

| `base` | `interaction` | `lead` | verdict |
|---|---|---|---|
| excludes zero above | `move_null` | `no_move` or `below` | **strength** |
| excludes zero above | `no_move` | `move_wt` | **placement** |
| excludes zero above | `move_null` | `move_wt` | **partly_strength** |
| excludes zero above | `move_wt` | any | **placement_amplified** |
| excludes zero above | any other combination | | **mixed**, reported without attribution |
| does not exclude zero above | | | **no_gap_effect** |

Any `unresolved` reading makes the verdict `unresolved`. What each licenses:

- **strength**: the advantage lay in the wild type's EM-count strengths. A wiring that may tune its own
  strengths loses it, and gap-junction claims must be about strength.
- **placement**: the advantage survives tuned strengths, so it is in which neurons are coupled.
- **partly_strength**: both, with the split stated.
- **placement_amplified**: tuning helps the wild type more than the null.
- **no_gap_effect**: with the chemistry held, the gap-only null does not trail the wild type, and nothing
  is attributed.

**Gates.** Each learning arm beats its wiring's frozen floor, and the two learning arms of a level do not
both reach 90%.

**Sizing.** The proxy is Logbook 078's split, sd 0.1025 over 128 seeds. At 64 seeds the detectable effect
is about 0.032, below the 0.0577 minimum. An interaction can spread more than one gap, which would make
this proxy low. The achieved spread is reported beside and never used to re-read.

## Before launch

- **Identity check** ([identity.json](identity.json)): four learning and two frozen wild-type runs
  (seeds 513–516 and 513–514) were re-run on this change's code from the worktree. All six match
  Logbook 078's logs on every `Run:` line and the final `w_chem`, bit for bit, so the reuse is licensed.
  The first comparison failed only on the weights, because the worktree's experiment records and
  weights sat in the worktree. Once they were copied into the main repo, all six matched. The same copy
  precedes scoring the panel.

- **Pilot**, on seeds 1401–1404, all six arm types, 24 runs:

  - **Gate preflight** ([preflight.json](preflight.json)): both levels `readable`, with floors at 0.
  - **Plasticity check** ([plasticity.json](plasticity.json)): each plastic-gap learning run differs
    from its fixed-gap twin on every pilot seed, for both wirings, so the multipliers act.

- **Cost**, from the pilot's own run times at 16 workers:

  | arm | minutes |
  |---|---|
  | null, fixed, learning | 20 |
  | null, frozen | 5 |
  | wild type, plastic, learning | 21 |
  | null, plastic, learning | 18 |

  The 192 runs take **about 4.5 hours**.

- **Readiness review.** The gates were evaluated at the registered point on this cell. The cost comes
  from a pilot configured as the campaign: the same configs and 3,000 episodes. The pilot seeds are
  disjoint from the band. The verdict map names every branch.

## Launch

From the worktree at this commit, so C.1's development in the main checkout cannot touch the running
code or configs:

```bash
cd ../nematode-m8
T=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
uv run python scripts/run_campaign.py --config ${T}_rewired_gap_only_null_t35.yml \
  --config ${T}_rewired_gap_only_null_frozen_t35.yml --config ${T}_plastic_gaps_t35.yml \
  --config ${T}_rewired_gap_only_null_plastic_gaps_t35.yml --seeds 513-576 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/m8-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout: copy the worktree's records in, and score.
rsync -a --ignore-existing ../nematode-m8/experiments/ experiments/
rsync -a --ignore-existing ../nematode-m8/exports/ exports/
uv run python scripts/analysis/plastic_gaps.py score --logs campaigns/m8-panel/logs \
  --logs campaigns/a6t2-thermal-split/logs --out-dir build/m8 --out control.json --csv per-seed.csv
```

## Retention (A.0)

Committed: this record, `identity.json`, `preflight.json`, `plasticity.json`, the panel's `control.json`
and `per-seed.csv`. Archived off-repo: the identity, pilot and panel campaigns' raw logs and the
worktree's session records.
