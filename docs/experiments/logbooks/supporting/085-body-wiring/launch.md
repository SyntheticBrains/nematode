# C.1e panel — the wiring contrast through the body: registration and launch

**Registered 2026-10-09, before any panel run.** Change: `add-body-wiring-contrast` (design F, revised
after the pilot). The pilot, [Logbook 084](../../084-body-wiring-pilot.md), fixed the minimum and the
size.

## The question

**Does the wild-type wiring help a connectome forage through the body, against a null that differs from
it only in its chemical wiring?** It is Phase 8b's central reading, in a new reference frame: no delta
against 029, block V or any point-worm result.

## The arms

The cell: hard350's food layout at 500 steps, through the frozen kinematic body (steering gain 2,
one-step reversals, wave floor 0.25), with the body-drive gain vector, Emmons 2024 with reversal on,
settling dynamics at depth 4, 3,000 episodes, `entropy_coef` 0.004.

| arm | config stem | role |
|---|---|---|
| wild type, PPO | `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_body500_wt_ppo` | |
| wild type, frozen | `..._body500_wt_frozen` | floor |
| chemical-only null, PPO | `..._body500_chemnull_ppo` | D21's primary null |
| chemical-only null, frozen | `..._body500_chemnull_frozen` | floor |
| MLP-PPO | `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body500` | beside |

The connectome configs are the pilot's own. **Seeds 1801–1864 (64), 320 runs.**

## The reading

**One registered reading**: wild type minus chemical-only null on `auc_success`, paired by seed, its 80%
bootstrap interval and its two-sided Wilcoxon q, classified at the **minimum of 0.0367** by
`mc.classify`.

| state | verdict | what it licenses |
|---|---|---|
| `move_wt` | **wild type ahead** | the wild type's advantage survives the body, at least the minimum; **the boundary stage runs** |
| `move_null` | **null ahead** | the null's wiring learns the body cell faster, by at least the minimum |
| `below` | **a difference below the minimum** | a significant difference smaller than 0.0367 |
| `no_move` | **no wiring effect at the minimum** | the interval sits inside ±0.0367 and includes zero |
| `unresolved` | **unresolved at this sensitivity** | none; the achieved interval is reported as the bound |

**Gates first.** Each learning arm must beat its frozen floor, paired by seed, and the two must not both
reach 90%. If either fails, the panel is **unreadable** and nothing is classified.

**Reported beside, never read as a verdict:**

- episodes to 30% success, the same contrast;
- the frequency of competent seeds (plateau ≥ 30%), by an exact McNemar test on discordant seeds;
- the MLP's plateaus;
- the panel's achieved spread and MDE, beside the registered minimum, never used to re-read it.

## The boundary stage, gated

**Only if the reading is `move_wt`**, the boundary-preserving null runs on the same seeds, PPO and frozen
(128 runs), and wild type minus boundary null is read at the same minimum against the panel's wild-type
runs. Its configs come from `generate_body_wiring_configs.py --stage boundary`. Under the body its
boundary holds 2,134 of the 3,709 chemical edges (58%), so it rewires 1,575 interior edges, and an
interior claim covers 42% of the chemical wiring. A `no_move` there is weaker evidence than under the
point worm's boundary. If the primary is not `move_wt`, the boundary stage is recorded as not run, with
its reason.

## Sizing

The pilot's per-seed sd of the gap is 0.161 ([pilot.json](../084-body-wiring-pilot/pilot.json)). The
smallest n whose MDE (`2.487 × sd / √n`) reaches 0.0367 exceeds the 64-seed cap, so the panel runs 64,
where the MDE is **0.050**. An effect between 0.0367 and 0.050 will read `unresolved`; that is stated
now, not discovered.

## Before launch

- **Gate preflight** ([preflight.json](preflight.json)) on the pilot's 64 runs, which are these
  configurations exactly: **`readable`**, wild type 57.3% and null 54.6% against 0% floors, 33 points
  below the saturation bar; `launch: true`.
- **Cost**, from the pilot's run times at 16 workers: about 70 minutes per connectome learning run, 60
  per frozen run, and about 45 per MLP run. 320 runs are about **20 hours**.
- **Readiness.** The gates were read at the registered point on this cell, from runs configured as the
  panel. The cost comes from those runs. The panel's seeds (1801–1864) are disjoint from the probes'
  (1601–1604) and the pilot's (1701–1716).

## Launch

From a worktree at this commit:

```bash
cd ../nematode-c1e
B=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_body500
M=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body500
uv run python scripts/run_campaign.py --config ${B}_wt_ppo.yml --config ${B}_chemnull_ppo.yml \
  --config ${B}_wt_frozen.yml --config ${B}_chemnull_frozen.yml --config ${M}.yml \
  --seeds 1801-1864 --runs 3000 --workers 16 --output-dir ../nematode/campaigns/c1e-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-c1e/experiments/ experiments/
rsync -a --ignore-existing ../nematode-c1e/exports/ exports/
uv run python scripts/analysis/body_wiring.py panel --logs campaigns/c1e-panel/logs \
  --out-dir build/c1e-panel --out panel.json --csv per-seed.csv
```

## Retention (A.0)

Committed: this record, `preflight.json`, and the panel's `panel.json` and `per-seed.csv`. Archived
off-repo: the panel campaign's raw logs and the worktree's session records.
