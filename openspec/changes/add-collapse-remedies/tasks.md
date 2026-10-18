# Tasks: M.1(b) — collapse remedies, placebo-paired

Phase 9 M.1(b) (roadmap v4.5 § Phase 9, D25 as amended 2026-10-17; tracker `phase9-tracking` M.1(b)),
after M.1(a) read no cause indicated ([Logbook 096](../../../docs/experiments/logbooks/096-collapse-diagnosis.md)).

- [x] 1. **Stabilisers and the placebo** on the connectome PPO rule and the MLP-PPO brain:
  `kl_stop_factor` (relative KL stop: factor 3, window 100), `weight_clip` (κ, `s` at construction as a
  non-persistent buffer, zero-`s` tensors exempt, before the mask and sign projection), `adam_eps` and
  `adam_beta2` (re-applied after an optimiser state is loaded), `rollout_shuffle_stream` (the placebo's
  permutation stream); each off by default, adding no key to the saved training state when off. The
  telemetry gains `kl_stopped` and `minibatches_stepped`, and an epoch with no recorded minibatch is
  handled. Tests: each against a hand case (a stop skips the crossing minibatch's step and the rest; no
  stop before 100 updates; a clipped tensor sits within ±κ·s; the optimiser carries and keeps the
  settings; the placebo changes the minibatch order and nothing else); off byte-identical (outputs,
  weights, optimiser state, saved keys, the next draw of every generator).
- [x] 2. **Configs**: the placebo and each round's body and MLP configs as stated deltas of 092's.
- [x] 3. **Analysis** `scripts/analysis/collapse_remedies.py`: drawdown per run, the placebo-paired
  drawdown and learning-speed readings, the pass rule, collapse counts and rescues/harms beside, the KL
  stop rate, the MLP preflight; failed runs read as registered. Tests on synthetic series covering every
  branch.
- [x] 4. **Registration**: the launch record (`docs/experiments/logbooks/supporting/098-collapse-remedies/launch.md`)
  with the rule, the predicted reach of each remedy, the cost and the retention line; a probe that these
  configs with every stabiliser key at its default reproduce M.1(a)'s runs; the roadmap's D25 and the
  tracker carry the dated amendments (the placebo-paired selection, the relative KL stop);
  `/nematode-review-spec`; no gate preflight applies (no round reads a floor or saturation gate).
- [x] 5. **Rounds** from a worktree, with the machine to itself: the placebo, then the remedies in order,
  stopping at the first pass. *(All three ran, 88 runs, every one exiting 0; none passed.)*
- [x] 6. **Readout**: Logbook 098; tracker M.1(b) with the chosen remedy or the fired stop rule; M.1(c)'s
  change registers next; `docs/usage.md` for the new config keys; CHANGELOG.
- [x] 7. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive (a new
  capability's spec given a real Purpose), PR.
