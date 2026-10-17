## Why

M.1(a) read **no cause indicated** ([Logbook 096](../../../docs/experiments/logbooks/096-collapse-diagnosis.md)):
Logbook 092's eight late body collapses leave no registered signature before they fall, and their
intact twins drift the same way. So M.1(b) takes D25's remedy list in D25's own order (roadmap v4.5
§ Phase 9, D25; tracker `phase9-tracking` M.1(b)). A remedy is a platform setting: it is chosen by
**rescue of the known collapsers**, never tuned per arm, held on the MLP positive control in M.1(c), then
applied identically to every arm and frozen for the phase. At most three rounds; if none rescues, the
collapse statistic ships alone and R.2 and G.5 run at 3,000 episodes.

## What Changes

- **PPO stabilisers**, on the connectome PPO rule and the MLP-PPO brain alike, each off by default and
  byte-identical when off:
  - `target_kl` — **KL early stopping**: an update stops its remaining epochs once a minibatch's
    approximate KL exceeds 1.5 × the target.
  - `weight_clip` — **weight clipping relative to the initial scale**: after every optimiser step, each
    weight tensor is clipped to ±κ times its own largest initial magnitude.
  - `adam_eps`, `adam_beta2` — **Adam's second-moment settings**, currently the PyTorch defaults.
  - `advantage_norm: batch | running` — advantages normalised by an exponentially weighted running
    mean and standard deviation instead of each batch's own (registered for completeness of D25's list;
    beyond the three rounds unless an earlier round is skipped).
- **The rounds**, in D25's order: (1) KL early stopping at `target_kl` 0.01; (2) weight clipping at
  κ = 2; (3) Adam ε 1e-4 and β₂ 0.99. Each round re-runs M.9's 16 seeds, the eight collapsers and their
  eight intact twins, under the remedy, with telemetry, and stops at the first remedy that passes the
  rescue rule (design Decision B).
- **Analysis** (`scripts/analysis/collapse_remedies.py`): the discordant pairs, the rescue rule and the
  learning-speed guard, against the committed re-runs of M.1(a).
- **Then** M.1(c), the confirmatory panel on fresh seeds (2801–2832): the MLP control, the wild type and
  the gap-only null, with and without the chosen remedy, against an equivalence margin; its connectome
  arms on the corrected gap model chosen by G.0-A (the maintainer's sequencing decision, 2026-10-17). Its
  own change.

## Capabilities

**Modified**: `learning-rules` — PPO stabilisers on the connectome PPO rule and the MLP-PPO brain (ADDED
requirement; every existing scenario unchanged).

## Impact

- `learning_rules/ppo.py`, `brain/arch/connectome_ppo.py` (config keys, passed to the rule),
  `brain/arch/mlpppo.py` and `brain/arch/_ppo_buffer.py` (the running advantage statistics).
- Configs: each round's two body configs as stated deltas of 092's.
- A remedy changes no existing reading: every stabiliser is off by default.
