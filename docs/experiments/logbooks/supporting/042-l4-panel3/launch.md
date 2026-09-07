# Panel 3 launch record

Written and committed before the campaign command ran, per the registration.

- **Date**: 2026-09-07
- **Pinned state**: commit `f8af21da` on `feat/l4-panel3` (the harness; the two arm configs are
  panel 2's, untouched, and every value they run with is panel 1's registered pin). The commit
  carrying this record precedes the launch.
- **Arms** (two): `wt_hebbian` and `rn_hebbian` — panel 2's degree-scaled unmodulated-Hebbian
  stems, as registered in `scripts/analysis/l4_panel2.py` and reused by `l4_panel3.py`
- **Seeds**: 17–64, paired (`rewire_seed` derived from the run seed); no earlier panel used them
- **Budget**: 1000 episodes, uniform. The single registered extension for a seed the plateau
  detector marks non-converged at 1000 is a fresh run at 1500 replacing the shorter log.
- **Frozen floors**: panel 2's prior-sweep values for seeds 17–64 (`wt_frozen`, `rn_frozen`,
  600 episodes), read from the committed table
  `docs/experiments/logbooks/supporting/041-l4-panel2/per-seed.csv`; originally produced by the
  `campaigns/l4-panel2-sweep` campaign. Nothing is re-run.
- **Descriptive pooling**: seeds 1–16 from the same table, never confirmatory.
- **No pilot.**
- **Command** (from the repository root):

```bash
C=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic
uv run python scripts/run_campaign.py \
  --config ${C}_hebbian.yml \
  --config ${C}_hebbian_rewired_null.yml \
  --seeds 17-64 --runs 1000 --output-dir campaigns/l4-panel3 -- --theme headless --track-experiment
```

- **Analysis**:
  `uv run python scripts/analysis/l4_panel3.py --campaign-dir campaigns/l4-panel3 --out docs/experiments/logbooks/supporting/042-l4-panel3/panel3.json --csv docs/experiments/logbooks/supporting/042-l4-panel3/per-seed.csv --curves docs/experiments/logbooks/supporting/042-l4-panel3/curves.csv`;
  the confirmatory family and the verdict map are the ones fixed in that script and in the
  registration.
