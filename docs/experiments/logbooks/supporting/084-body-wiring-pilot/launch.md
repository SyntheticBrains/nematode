# C.1e pilot — the wiring contrast through the body: registration and launch

**Registered 2026-10-09, before any scored run.** Change: `add-body-wiring-contrast` (design E). This
pilot sets the minimum and the size of C.1e's panel. Nothing committed measured a wiring effect on the
body cell, and the protocol requires the minimum to come from data on the same cell.

## The question it answers first

**How large, and how variable, is the wild type's advantage over the chemical-only null through the
body?** Its committed answer fixes the panel's minimum, its seed count and its gates.

## The arms

The cell: hard350's food layout at **500 steps**, through the frozen kinematic body (steering gain 2,
one-step reversals, wave floor 0.25), with the body-drive gain vector, Emmons 2024 with reversal on,
settling dynamics at depth 4, 3,000 episodes, `entropy_coef` 0.004. A new reference frame: nothing here
is a delta against 029, block V or any point-worm result.

| arm | config stem |
|---|---|
| wild type, PPO | `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_body500_wt_ppo` |
| wild type, frozen | `..._body500_wt_frozen` |
| chemical-only null, PPO | `..._body500_chemnull_ppo` |
| chemical-only null, frozen | `..._body500_chemnull_frozen` |

**Seeds 1701–1716 (16), 64 runs.** The configs come from `generate_body_wiring_configs.py --stage pilot` and are re-checked through the real loader by a test.

**The frozen-wiring learner is not here.** Its probe (seeds 1601–1604) read `fails_floor`: the wild type
plateaued at 0% on every seed and the chemical-only null at 0, 0, 25.3 and 0, against 0% floors. By the
change's rule it left before the pilot, and the reading-learner half of C.1e closes
unreachable-with-reason: through the body, the drive gains and sensor gains over a fixed wiring do not
carry foraging, within 3,000 episodes at these settings.

## What the pilot fixes

From [body_wiring.py](../../../../../scripts/analysis/body_wiring.py) `pilot`, before the panel registers:

- **The reference effect**: the paired wild-type-minus-null `auc_success` mean over the 16 seeds.
- **The minimum**: 2/3 of |reference|, floored at **0.0367**. The floor is a judgement carried from the
  point worm (B.1c's committed PPO minimum), only so that a near-zero pilot cannot make a trivial
  difference count as a move.
- **The panel's seed count**: the smallest n with `2.487 × sd / √n` at most the minimum, **never fewer
  than 16**, the pilot's own count, and capped at 64. A paired rank test on a handful of seeds fires on
  the consistency of the sign rather than the size, and a bimodal pilot's spread is unstable:
  resampling the probes' gaps moved n between 5 and 14. If n exceeds 32, the current null leaves the
  panel first.
- **The floor's size on this cell**: the 0.0367 floor as a share of the wild type's mean
  `auc_success`, reported beside the minimum, never used to re-read it. The floor comes from the point
  worm at 350 steps, and the body learns later, so an absolute difference is a larger share here.
- **The gates**: both learning arms beat their floors, and the level is not saturated (both learning
  arms at or above 90%). If PPO fails either, the panel does not run, and C.1e's deliverable is the
  diagnosis.

**Reported beside, never read as a verdict:** episodes to 30% success, and the frequency of competent
seeds (plateau ≥ 30%) per wiring, by an exact McNemar test on the seeds where the wirings disagree.

The pilot is a calibration, not a test: its wild-type-minus-null mean is reported with its interval, but
no claim about the wiring is made from it. The panel, on fresh seeds, makes that claim.

## Before launch

**The probes** ([probes.json](probes.json)), all on seeds 1601–1604 at this cell:

| probe | setting | wild type, PPO | chemical-only null, PPO |
|---|---|---|---|
| 1 | C.1 as merged | 0, 0, 0, 0 | 0, 0, 0, 0 |
| 2 | + drive gain vector | 0, 0, 0, 0 | 0, 0, 62.1, 52.7 |
| 3 | + entropy 0.004 (**this registration's point**) | 58.4, 76.1, 48.7, 83.5 | 27.7, 80.7, 18.8, 82.0 |
| 4 | + frozen wiring | 0, 0, 0, 0 | 0, 0, 25.3, 0 |

Probe 3's configs equal this pilot's, key for key, so it is evidence at the registered point.

**Gate preflight** ([probe-preflight.json](probe-preflight.json)) on probe 3's runs, mapped to the
registered stems, with probe 2's frozen floors (their configs differ only in `entropy_coef`, which a
frozen arm never uses): PPO **`readable`**, wild type 66.7% and null 52.3% against 0% floors, 23 points
below the saturation bar.

**Cost**, from probe 3: about 51 minutes per learning run and about 58 per frozen run at 12–16 workers.
64 runs at 16 workers are four rounds, **about 3.7 hours**.

**Readiness.** The gate was read at the registered point on this cell, from runs configured as the
pilot. The cost comes from those runs. The probe seeds (1601–1604) are disjoint from the pilot's
(1701–1716), and the panel will use fresh seeds from 1801.

## Launch

From a worktree at this commit, so the main checkout can change without touching the running code:

```bash
cd ../nematode-c1e
B=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_body500
uv run python scripts/run_campaign.py --config ${B}_wt_ppo.yml --config ${B}_wt_frozen.yml \
  --config ${B}_chemnull_ppo.yml --config ${B}_chemnull_frozen.yml --seeds 1701-1716 --runs 3000 \
  --workers 16 --output-dir ../nematode/campaigns/c1e-pilot \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-c1e/experiments/ experiments/
rsync -a --ignore-existing ../nematode-c1e/exports/ exports/
uv run python scripts/analysis/body_wiring.py pilot --logs campaigns/c1e-pilot/logs \
  --out-dir build/c1e --out pilot.json --csv per-seed.csv
```

## Retention (A.0)

Committed: this record, `probes.json`, `probe-preflight.json`, and the pilot's `pilot.json` and per-seed
CSV. Archived off-repo: the probe and pilot campaigns' raw logs and the worktree's session records.
