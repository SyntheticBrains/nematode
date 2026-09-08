# Warm-start panel launch record

Written and committed before any campaign command ran, per the registration.

- **Date**: 2026-09-08
- **Pinned state**: commit `374ac088` on `feat/l4-warm-start-panel` (the twelve configs, the campaign
  steps, the harness, the `{seed}` placeholder; every plastic-arm value is panel 1's registered pin
  and the PPO arms run the committed 029 recipe at low initial noise). The commit carrying this
  record precedes the launch.
- **Teacher**: `mlpppo_small_continuous2d_combined_klinotaxis.yml`, seeds 1–8, 6000 episodes; the
  seed with the highest committed plateau tail is the one teacher; its auto-saved weights are copied
  to `campaigns/l4-warm-start/teacher.pt`; recorded frozen (`freeze_updates: true`, the derived
  config written beside the results) for 300 episodes at seed 101 with `--record-rollouts`; the
  teacher's frozen plateau tail at that seed is the ceiling.
- **Clones**: for each seed 1–8 and each wiring, `plastic` (student: the plastic frozen arm) and
  `full` (student: the low-noise PPO arm) from the one rollout file, 300 epochs, lr 1e-3, batch 256,
  holdout 0.2; every fit in `clones.json`, weak if the held-out loss is not below half its initial.
- **Arms** (twelve, the registry in `scripts/analysis/l4_warm_start.py`), paired seeds 1–8:
  `wt_clone_frozen`, `rn_clone_frozen`, `wt_fullclone_frozen`, `rn_fullclone_frozen` at 600
  episodes; `wt_clone_hebbian`, `rn_clone_hebbian`, `wt_clone_plastic`, `rn_clone_plastic` at 2000;
  `wt_fullclone_ppo`, `rn_fullclone_ppo`, `wt_ppo`, `rn_ppo` at 3000. The single registered
  extension for a run the plateau detector marks non-converged is a fresh run at 1.5× replacing the
  shorter log.
- **No pilot.**
- **Commands** (from the repository root, in order):

```bash
C=configs/scenarios/foraging_predator_thermal
uv run python scripts/run_campaign.py --config $C/mlpppo_small_continuous2d_combined_klinotaxis.yml \
  --seeds 1-8 --runs 6000 --output-dir campaigns/l4-teacher -- --theme headless --track-experiment
uv run python scripts/campaigns/l4_warm_start_campaign.py teacher \
  --campaign-dir campaigns/l4-teacher --out-dir campaigns/l4-warm-start
uv run python scripts/campaigns/l4_warm_start_campaign.py clone --out-dir campaigns/l4-warm-start
S=$C/connectomeppo_small_continuous2d_combined_klinotaxis
uv run python scripts/run_campaign.py \
  --config ${S}_plastic_frozen_clone.yml --config ${S}_plastic_frozen_rewired_null_clone.yml \
  --config ${S}_plastic_frozen_fullclone.yml --config ${S}_plastic_frozen_rewired_null_fullclone.yml \
  --seeds 1-8 --runs 600 --output-dir campaigns/l4-warm-start-panel -- --theme headless --track-experiment
uv run python scripts/run_campaign.py \
  --config ${S}_plastic_hebbian_clone.yml --config ${S}_plastic_hebbian_rewired_null_clone.yml \
  --config ${S}_plastic_clone.yml --config ${S}_plastic_rewired_null_clone.yml \
  --seeds 1-8 --runs 2000 --output-dir campaigns/l4-warm-start-panel -- --theme headless --track-experiment
uv run python scripts/run_campaign.py \
  --config ${S}_lowstd_fullclone.yml --config ${S}_rewired_null_lowstd_fullclone.yml \
  --config ${S}_lowstd.yml --config ${S}_rewired_null_lowstd.yml \
  --seeds 1-8 --runs 3000 --output-dir campaigns/l4-warm-start-panel -- --theme headless --track-experiment
```

- **Analysis**:
  `uv run python scripts/analysis/l4_warm_start.py --campaign-dir campaigns/l4-warm-start-panel --teacher-json campaigns/l4-warm-start/teacher.json --clones-json campaigns/l4-warm-start/clones.json --out docs/experiments/logbooks/supporting/043-l4-warm-start/panel.json --csv docs/experiments/logbooks/supporting/043-l4-warm-start/per-seed.csv --curves docs/experiments/logbooks/supporting/043-l4-warm-start/curves.csv`;
  the confirmatory family and the verdict map are the ones fixed in that script and in the
  registration.
