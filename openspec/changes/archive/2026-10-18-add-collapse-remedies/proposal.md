## Why

M.1(a) read **no cause indicated** ([Logbook 096](../../../docs/experiments/logbooks/096-collapse-diagnosis.md)):
Logbook 092's eight late body collapses leave no registered signature before they fall, and their
intact twins drift the same way. So M.1(b) takes D25's remedy list in D25's own order (roadmap v4.5
§ Phase 9, D25; tracker `phase9-tracking` M.1(b)). A remedy is a platform setting: it is chosen by
**rescue of the known collapsers**, never tuned per arm, held on the MLP positive control in M.1(c), then
applied identically to every arm and frozen for the phase. At most three rounds; if none rescues, the
collapse statistic ships alone and R.2 and G.5 run at 3,000 episodes.

## What Changes

- **PPO stabilisers and a placebo**, on the connectome PPO rule and the MLP-PPO brain alike, each off by
  default and byte-identical when off:
  - `kl_stop_factor` — **KL early stopping relative to the learner**: an update stops before a
    minibatch's step when its approximate KL exceeds this factor (3) times the median, over the learner's
    last 100 updates, of each update's largest minibatch KL.
  - `weight_clip` — **weight clipping relative to the initial scale**: after every optimiser step, each
    weight tensor is clipped to ±κ times its own largest magnitude at construction (κ = 2).
  - `adam_eps`, `adam_beta2` — **Adam's second-moment settings** (1e-4, 0.99).
  - `rollout_shuffle_stream` — **the placebo**: the minibatch permutations from a separate seeded stream,
    re-rolling a run's trajectory with no stabilising intent.
  - Running advantage normalisation stays registered as D25's fourth remedy and is not built here.
- **The rounds**: a placebo round, then up to three remedy rounds in D25's order on M.9's 16 seeds, each
  read **against the placebo, paired by seed**, on each run's drawdown from its peak (design Decision B),
  with a learning-speed guard; the MLP's retry seeds run beside as a preflight for M.1(c).
- **Analysis** (`scripts/analysis/collapse_remedies.py`): the paired drawdown and guard readings, collapse
  counts beside, the MLP preflight.
- **D25's selection rule amended** (the maintainer, 2026-10-17): from rescue counts against the original
  runs to a placebo-paired reading; recorded in the roadmap and the tracker.
- **Then** M.1(c), the confirmatory panel on fresh seeds (2801–2832): the MLP control, the wild type and
  the gap-only null, with and without the chosen remedy, against an equivalence margin; its connectome
  arms on the corrected gap model chosen by G.0-A. Its own change.

## Capabilities

**Modified**: `learning-rules` — PPO stabilisers on the connectome PPO rule and the MLP-PPO brain (ADDED
requirement; every existing scenario unchanged).

## Impact

- `learning_rules/ppo.py`, `brain/arch/connectome_ppo.py` (config keys, passed to the rule),
  `brain/arch/mlpppo.py` (the placebo's permutation stream is passed to the existing buffer),
  `brain/arch/_telemetry.py` (the KL stop's fields).
- Configs: each round's body and MLP configs as stated deltas of 092's.
- A remedy changes no existing reading: every stabiliser is off by default.
