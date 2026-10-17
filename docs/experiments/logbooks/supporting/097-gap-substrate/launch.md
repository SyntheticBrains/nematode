# G.0-A — the corrected gap model, chosen by learnability: registration and launch

**Registered 2026-10-17, before any pilot run.** Change: `add-ohmic-gap-coupling`. Tracker
`phase9-tracking` G.0-A; roadmap v4.5 § Phase 9.

## The question

G.0 read `artefact` ([Logbook 095](../../095-gap-operating-point.md)): on the thermal cell the wild type's
lead over the gap-only null rides on the model's additive, count-scaled gap coupling. The maintainer
chose (2026-10-17) to correct the model with **ohmic gap coupling** in the settling map, current in
proportion to the potential difference, selected against **log-compressed counts** in a gated pilot.
**Which corrected model is the better instrument for the readings block G must carry?** The pilot reads
learnability only; it never reads the gap between the wild type and the null.

## The models

- **Ohmic** (`gap_coupling: ohmic`, `gap_weight_scale: 0.03`): the injected sensors and each settling
  iteration's drive pass through `(I + L)⁻¹`, `L` the gap Laplacian; the coupling averages and adds no
  loop gain at any conductance. The scale is calibrated below.
- **Log** (`gap_weight_mode: log_count`): additive coupling with each count replaced by `log1p(count)`.
  Its wiring gap is already known as description: +0.012 on 16 seeds (G.0(e)).

## The probe and the calibration, before any run

On untrained brains, with `sign_scale_calibration.py`'s 64 fixed inputs and the pilot's 16 seeds, both
cells ([probe.json](probe.json)):

| brain | action spread across inputs | saturated fraction |
|---|---|---|
| additive counts | 0.238 | 0.058 |
| additive log counts | 0.041 | 0.000 |
| gap junctions off | 0.025 | 0.000 |
| ohmic × 1 (the built conductances) | 0.0003 | 0.000 |
| ohmic × 0.1 | 0.0078 | 0.000 |
| **ohmic × 0.03** | **0.0162** | 0.000 |
| ohmic × 0.01 | 0.0213 | 0.000 |

(Thermal shown; hard350 agrees within 0.001.)

**As first registered**, the pilot launched only if ohmic's spread was at least a quarter of the additive
count model's. **It stopped the pilot**: at the built conductances ohmic coupling averages the input away.
The probe also showed that yardstick was the artefact: the log-count and gap-off brains, which G.0 showed
learn as well as the count model (71% and 67% plateaus), fall far below a quarter of its spread (as did
unit weights, 70%, in a 4-seed pass of the same probe: 0.026), because that spread is its gap
amplification.

**Amended by the maintainer, 2026-10-17, before any pilot run**: the reference is the gap-off brain, a
known learner; the ohmic conductance scale is the largest on the grid ×1, 0.3, 0.1, 0.03, 0.01, 0.003,
0.001 whose spread is at least half the gap-off brain's on both cells. **It is ×0.03** (0.0162 against a
floor of 0.0125). At that scale the median neuron keeps 93% of its own drive and ALA, the most strongly
coupled, 29%.

## The arms

Both block-V cells on Emmons 2024, 3,000 episodes, seeds **9871–9886** (16): the thermal cell at target 35
and hard350. On each cell:

| arm | learning | frozen |
|---|---|---|
| ohmic, wild type | ✓ | ✓ |
| ohmic, gap-only null | ✓ | ✓ |
| log, wild type | ✓ | ✓ |
| log, gap-only null | ✓ | ✓ |
| additive counts, wild type (description) | ✓ | |
| MLP-PPO control | ✓ | ✓ (derived floor) |

2 cells × 11 arms × 16 seeds = **352 runs**. Configs from `generate_gap_substrate_configs.py`.

## The selection rule

1. **The cells must be readable**: the MLP control beats its frozen floor on both cells (the paired 80%
   interval of learning − frozen plateau above zero). If not: `unreadable`, nothing chosen.
2. **A model is eligible on a cell** when its wild type and its gap-only null each beat their own floors
   and the two do not both plateau at or above 90%: the readability gate the model's contrast will need.
3. **Ohmic is chosen** if eligible on both cells, unless on either cell its wild type is **shown slower**
   than log's by more than the cell's margin: the paired-by-seed 80% interval of
   `auc_success(ohmic) − auc_success(log)` lies entirely below −δ, with **δ = 0.0239** on the thermal
   cell and **δ = 0.0143** on hard350 (Logbook 080's margins for the same question).
4. Otherwise **log**, if eligible on both cells; otherwise **neither**, and the substrate decision
   returns to the maintainer.

**Operating characteristic** (Logbook 080's paired spreads, 16 seeds, normal approximation with a t
interval): ohmic is chosen with probability about 0.95 if the two learn equally fast, 0.76 if ohmic is
slower by 0.025, 0.38 by 0.05, 0.01 by 0.10. The instrument's percentile bootstrap interval is slightly
narrower, which shifts these to about 0.94, 0.73, 0.33 and 0.004.

**Never read**: the gap between the wild type and the gap-only null, on either model. The selection
output records each model's gate evidence (both wirings' plateaus and floors, per seed), because the
eligibility rule reads it; it is gate evidence only, never read or reported as a contrast. The
additive-count wild type is described beside (ohmic − additive and log − additive on `auc_success`).

**Failed runs**: a run that exits non-zero is re-run once on the same seed; `select` reads only a
complete pilot, every arm on every seed.

## Before launch

- **Smokes**, seeds 9891–9892, every pilot config at 20 episodes: 44 of 44 runs exited 0. Execution
  checks only, discarded.
- **Gates on committed runs, where they exist.** The MLP control's readability: Logbook 080's committed
  MLP runs pass on both cells (`080-across-step-state/mlp.json`: thermal plateaus 47–90%, hard350
  88–99%, every floor beaten). Log counts on the thermal cell: G.0's 16-seed level read readable (wild
  type 70.8%, null 67.6%, floors 0%, unsaturated; Logbook 095). The ohmic arms on both cells and the log
  arms on hard350 have no committed runs; the pilot is their gate.
- **Cost**: thermal from G.0's measured rates (22.9 minutes a learning run, 5.1 a frozen one, at 16
  workers); hard350 scaled by its 350-step episode cap. About **6 hours**, with the machine to itself.
- **Readiness review**: two spec reviews of the change (2026-10-17) and the probe above; findings fixed
  or recorded in the design's dated amendments.

## Launch

From a worktree at this commit:

```bash
git worktree add --detach ../nematode-g0a <this commit>
cd ../nematode-g0a
uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu
cfgs=($(uv run python scripts/analysis/gap_substrate_selection.py plan \
  | python3 -c "import json,sys; print(' '.join('--config '+c for c in json.load(sys.stdin)['configs']))"))
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 9871-9886 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/g0a-pilot \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the same worktree, so the registered harness reads the runs:
D=../nematode/docs/experiments/logbooks/supporting/097-gap-substrate
uv run python scripts/analysis/gap_substrate_selection.py select \
  --logs ../nematode/campaigns/g0a-pilot/logs --out-dir build/g0a --out $D/selection.json
```

## Retention (A.0)

Committed: this record, `probe.json`, `selection.json` (with every arm's per-seed plateau and floor, the
gate evidence the rule reads). Archived off-repo: the smoke and pilot
campaigns' raw logs and the worktree's session records.
