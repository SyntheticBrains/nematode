# Sign-grounding launch record

Written and committed before either campaign command ran, per the registration.

- **Date**: 2026-09-09
- **Pinned state**: commit `1043f964` on `feat/l4-atlas-signs` (the vendored atlas and its
  provenance, the populated 302-entry transmitter table, `synapse_signs` and
  `enforce_synapse_signs`, the six arm configs, the harness). The commit carrying this record
  precedes the launch.
- **What is grounded**: 280 of 302 neurons carry a release identity from the atlas; the derived
  sign table (acetylcholine and glutamate excitatory, GABA inhibitory, monoamines and orphan or
  uptake-only identities unknown) grounds **3,176 of 3,709 chemical synapses** — 2,962
  excitatory, 214 inhibitory — leaving 533 on the sign they drew. Magnitudes, per-neuron incoming
  norms, the readout, the gains and the RNG stream are identical to the random-sign build; the
  network goes from 48% inhibitory to 13%.
- **Prior sweep**: `wt_frozen_atlas`, `rn_frozen_atlas` on seeds 1–64 at 600 episodes — panel
  2's protocol. Enforcement is irrelevant to a frozen arm and is not run. 128 runs.
- **Hebbian contrast**: `wt_hebbian_atlas`, `rn_hebbian_atlas`, `wt_hebbian_dale`,
  `rn_hebbian_dale` on seeds 1–16 at 1000 episodes — panel 2's protocol. 64 runs. The single
  registered extension for a run the plateau detector marks non-converged is a fresh run at 1.5×
  replacing the shorter log.
- **Comparators**: panel 2's committed per-seed table
  (`supporting/041-l4-panel2/per-seed.csv`). The random-sign arms are **not** re-run.
- **No pilot**: every value the arms run with is panel 1's registered pin, unchanged.
- **Commands** (from the repository root):

```bash
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic
uv run python scripts/run_campaign.py \
  --config ${P}_frozen_atlassigns.yml --config ${P}_frozen_rewired_null_atlassigns.yml \
  --seeds 1-64 --runs 600 --output-dir campaigns/l4-atlas-signs -- --theme headless --track-experiment
uv run python scripts/run_campaign.py \
  --config ${P}_hebbian_atlassigns.yml --config ${P}_hebbian_rewired_null_atlassigns.yml \
  --config ${P}_hebbian_atlassigns_dale.yml --config ${P}_hebbian_rewired_null_atlassigns_dale.yml \
  --seeds 1-16 --runs 1000 --output-dir campaigns/l4-atlas-signs -- --theme headless --track-experiment
```

- **Analysis**:
  `uv run python scripts/analysis/l4_atlas_signs.py --campaign-dir campaigns/l4-atlas-signs --out docs/experiments/logbooks/supporting/044-l4-atlas-signs/panel.json --csv docs/experiments/logbooks/supporting/044-l4-atlas-signs/per-seed.csv --curves docs/experiments/logbooks/supporting/044-l4-atlas-signs/curves.csv`;
  the confirmatory family, the substrate gate and the verdict map are the ones fixed in that
  script and in the registration.
