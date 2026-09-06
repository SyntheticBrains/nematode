# Panel 2 launch record

Written and committed before either campaign command ran, per the registration.

- **Date**: 2026-09-07
- **Pinned state**: commit `8ddc88fc` on `feat/l4-panel2` (the four count-initialised floors and
  `scripts/analysis/l4_panel2.py`; every other arm value is panel 1's registered pin, untouched).
  The commit carrying this record precedes the launch.
- **Arms** (eight): `wt_frozen`, `rn_frozen`, `wt_frozen_count`, `rn_frozen_count`, `wt_hebbian`,
  `rn_hebbian`, `wt_hebbian_count`, `rn_hebbian_count` — the config stems registered in
  `scripts/analysis/l4_panel2.py`
- **Hebbian panel**: the four Hebbian arms on seeds 1–16, paired; 1000 episodes, uniform. The
  single registered extension for a seed the plateau detector marks non-converged at 1000 is a
  fresh run at 1500 replacing the shorter log.
- **Prior sweep**: the four frozen arms on seeds 1–64, paired; 600 episodes. Seeds 1–16 double as
  the learning-gain floors.
- **No pilot**: every value the arms run with is panel 1's pin; the Hebbian rule does not use the
  modulator.
- **Reproduction check**: on seeds 1–8 the degree-scaled arms run the same configs as panel 1, so
  their per-episode outcomes over the first 1000 (Hebbian) and 600 (frozen) episodes are expected
  to equal panel 1's logs' prefix.
- **Commands** (from the repository root; the sweep first, then the Hebbian panel):

```bash
C=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic
uv run python scripts/run_campaign.py \
  --config ${C}_frozen.yml \
  --config ${C}_frozen_rewired_null.yml \
  --config ${C}_frozen_countinit.yml \
  --config ${C}_frozen_rewired_null_countinit.yml \
  --seeds 1-64 --runs 600 --output-dir campaigns/l4-panel2-sweep -- --theme headless --track-experiment
uv run python scripts/run_campaign.py \
  --config ${C}_hebbian.yml \
  --config ${C}_hebbian_rewired_null.yml \
  --config ${C}_hebbian_countinit.yml \
  --config ${C}_hebbian_rewired_null_countinit.yml \
  --seeds 1-16 --runs 1000 --output-dir campaigns/l4-panel2 -- --theme headless --track-experiment
```

- **Analysis**:
  `uv run python scripts/analysis/l4_panel2.py --campaign-dir campaigns/l4-panel2 --sweep-dir campaigns/l4-panel2-sweep --out docs/experiments/logbooks/supporting/041-l4-panel2/panel2.json --csv .../per-seed.csv --curves .../curves.csv`;
  the confirmatory family and the verdict map are the ones fixed in that script and in the
  registration.
