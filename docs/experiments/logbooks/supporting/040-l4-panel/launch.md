# Panel launch record

Written and committed before the panel command ran, per the registration.

- **Date**: 2026-09-06
- **Pinned state**: commit `e9380fe4` on `feat/l4-panel-run` (arm configs carry both scaling
  switches, homeostasis, `initial_log_std −1.0`, `plasticity_rate 0.001`; the MLP arm `tanh` with
  `plastic_layers: hidden`). The commit carrying this record precedes the launch.
- **Arms** (seven): `wt_frozen`, `wt_hebbian`, `wt_plastic`, `rn_frozen`, `rn_hebbian`,
  `rn_plastic`, `mlp_plastic` — the config stems registered in `scripts/analysis/l4_panel.py`
- **Seeds**: 1–8, paired across every arm; `rewire_seed` unset so the rewired arms pair with the
  wild-type arms seed for seed
- **Budget**: 3000 episodes, uniform (pilot 3's registered budget rule); the single pre-registered
  extension for a seed still climbing at 3000 is a fresh run at 4500 replacing the shorter log
- **Recipe**: `plasticity_rate 0.001` (pilot 3's registered pooled rule), every other
  hyperparameter at the mixin defaults, on every arm
- **Command** (from the repository root):

```bash
uv run python scripts/run_campaign.py \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen.yml \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_hebbian.yml \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic.yml \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen_rewired_null.yml \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_hebbian_rewired_null.yml \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_rewired_null.yml \
  --config configs/scenarios/foraging_predator_thermal/mlpppo_small_continuous2d_combined_klinotaxis_plastic.yml \
  --seeds 1-8 --runs 3000 --output-dir campaigns/l4-panel -- --theme headless --track-experiment
```

- **Yardstick note**: probes 5 and 6 established that the MLP arm cannot hold a policy under the
  rule even with its readout frozen; it runs as registered and is read as a finding.
- **Analysis**: `uv run python scripts/analysis/l4_panel.py --campaign-dir campaigns/l4-panel --out docs/experiments/logbooks/supporting/040-l4-panel/panel.json --csv .../per-seed.csv --curves .../curves.csv`; the confirmatory family, the band test and the verdict map are the
  ones fixed in that script and in the registration.
- **Extensions applied**: one. `wt_plastic` seed 3 was still climbing at 3000 (no plateau detected) and
  received the single registered extension: a fresh run at 4500 episodes (`campaigns/l4-panel-ext`,
  its 3000-episode log kept beside it as `*.3000-episodes.log`), which converged with a 28.4%
  plateau tail against 29.3% on the shorter run; its log replaces the shorter one in the manifest.
  No other seed was extended. The verdict was not `robustness`, so no sensitivity pass was run.
