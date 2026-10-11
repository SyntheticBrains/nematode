# B.3 stage 0 — the connectome on the lawn cell: registration and launch

**Registered 2026-10-11, before any panel run.** Change: `add-b3-stage0`. The entropy pilot is
committed beside this record.

## The question

**Does the connectome, like D.1's MLP, take the dwelling state on lawns where its untrained policy does
not?**

This is the connectome's own positive control for D.1's roaming/dwelling readout. It is also the
baseline B.3's serotonin and PDF arms will be read against. B.3 reads its modulator knockouts only on
a readable result here.

## The cell

D.1's lawn cell unchanged, with the connectome in place of the MLP:

- **The learner:** the connectome PPO brain on Emmons 2024 (wild type, gap junctions, strict chemical
  mask), with reversal, klinotaxis sensing and settling to depth 4.
- **No modulators and no satiety input.** Satiety reaches the connectome through B.3's modulators.
- **The lawns:** 4 of 2.5 mm, with the worm starting on one.
- **Intake and movement:** intake 2% of a cell per step; a movement cost of 0.013 per mm; turns up to
  π per step.
- **Episodes:** 720 steps, 3,000 per run.
- **No state-shaping reward.**
- **`entropy_coef` 0.005,** chosen by the pilot.

| arm | config stem | role |
|---|---|---|
| learner | `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_connectome_e005` | |
| floor | `..._lawns_connectome_e005_frozen` | its untrained policy |

**Seeds 2001–2019 (19), 38 runs.** Seeds 2001–2016 are D.1's MLP panel, so the two learners pair by
seed. Each run is evaluated frozen for **30 held-out episodes** with D.1's instrument.

## The reading

**The learner's on-lawn dwelling share minus its floor's**, paired by seed, classified by
`mc.classify` at the **minimum of 0.278**, after the intake gate:

| state | verdict | what it licenses |
|---|---|---|
| `move_wt` | **dwells** | the connectome takes the dwelling state on lawns, at least 0.278 more of its on-lawn time than its untrained policy; B.3 reads its knockouts against this baseline |
| `move_null` | **dwells less than its floor** | the connectome dwells less than its untrained policy |
| `below` | **dwelling below the minimum** | a significant difference smaller than 0.278 |
| `no_move` | **no dwelling at the minimum** | the interval sits inside ±0.278 and includes zero |
| `unresolved` | **unresolved at this sensitivity** | none; the achieved interval is reported as the bound |

**Gate first.** The learner's intake plateau must beat its floor's, paired by seed, with the 80%
interval above zero. If not, the panel is unreadable.

**If the verdict is not `dwells`,** B.3 does not read modulator effects on this baseline. It stops for
a decision.

**Reported beside, never read as verdicts:**

- **The connectome against D.1's MLP:** dwelling share paired by seed over 2001–2016, from D.1's
  committed readings. Both learners ran at entropy 0.005 on the same cell, so this is a comparison of
  learners, not of settings.
- **How the spread falls:** the share of seeds that dwell, since the pilot split two and two.
- **Bouts, the depletion and dwelling-end statistics, and each run's learned action noise.**
- **The achieved spread and MDE,** beside the minimum, never used to re-read it.

## Sizing

The entropy pilot at 0.005 (seeds 9301–9304, [pilot-e005-states.json](pilot-e005-states.json)):

- **Per seed,** the learner's dwelling share minus its floor's is −0.006, +0.009, +0.934 and +0.728,
  a mean of **+0.417** with sd 0.486.
- **The minimum** is 2/3 of the reference, **0.278**.
- **The size.** The smallest n whose MDE (`2.487 × sd / √n`) reaches the minimum is **19**, with an
  MDE of 0.277. The spread is bimodal: seeds either dwell or roam throughout. A four-seed estimate of
  that spread is unstable, which is why the panel is sized above the 16-seed floor and why the achieved
  spread is reported.

## The entropy calibration

| entropy | intake, learner against floor | gate | dwelling minus floor, per seed |
|---|---|---|---|
| **0.005** | 3.15 against 1.71 | passes, +1.44 [+1.12, +1.77] | −0.01, +0.01, +0.93, +0.73 |
| 0.05 | 1.85 against 1.71 | passes, +0.14 [+0.04, +0.24] | 0.00, +0.01, −0.00, +0.00 |

By the rule fixed in the design, 0.005 is registered: both levels pass the gate, and it has the
larger dwelling over its floor. At 0.05 the connectome barely learns and never dwells, as D.1's MLP
did at that setting.

## Before launch

- **Gate preflight** on the 0.005 pilot, configured as here ([preflight.json](preflight.json)):
  learning +1.44 [+1.12, +1.77], passes.
- **Cost:** 16 runs (both levels) took about 20 minutes on 16 workers, so 38 runs take about **50
  minutes**, plus evaluation.
- **Seeds:** the panel's (2001–2019) are disjoint from the pilot's (9301–9304).

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
L=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_connectome_e005
uv run python scripts/run_campaign.py --config ${L}.yml --config ${L}_frozen.yml --seeds 2001-2019 \
  --runs 3000 --workers 16 --output-dir ../nematode/campaigns/b3s0-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/lawn_states.py --logs campaigns/b3s0-panel/logs \
  --arms connectome_e005 connectome_e005_frozen --learner-arm connectome_e005 \
  --floor-arm connectome_e005_frozen --minimum 0.278 --seeds $(seq 2001 2019) --episodes 30 \
  --compare-mlp docs/experiments/logbooks/supporting/087-patchy-lawns/panel.json \
  --out panel.json --csv per-run.csv
```

## Retention (A.0)

**Committed:** this record, `preflight.json`, the pilot's readings, and the panel's `panel.json` and
`per-run.csv`. **Archived off-repo:** the campaign directories and the worktree's session records.
