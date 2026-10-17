# Tasks: M.1(b) — collapse remedies, chosen by rescue of the known collapsers

Phase 9 M.1(b) (roadmap v4.5 § Phase 9, D25; tracker `phase9-tracking` M.1(b)), after M.1(a) read no
cause indicated ([Logbook 096](../../../docs/experiments/logbooks/096-collapse-diagnosis.md)).

- [ ] 1. **Stabilisers** on the connectome PPO rule and the MLP-PPO brain: `target_kl`, `weight_clip`,
  `adam_eps`/`adam_beta2`, `advantage_norm`; each off by default. Tests: each against a hand case (a KL
  stop skips the rest of the update; a clipped tensor sits within ±κ·s; the optimiser carries the
  settings; the running statistics update once per update); off byte-identical (outputs, weights,
  optimiser state, the next draw of every generator); the KL stops appear in the telemetry.
- [ ] 2. **Configs**: each round's wild-type and gap-only body configs as stated deltas of 092's.
- [ ] 3. **Analysis** `scripts/analysis/collapse_remedies.py`: rescues, harms, the McNemar test, the
  learning-speed guard and the pass rule against M.1(a)'s re-runs; tests on synthetic series covering
  every branch.
- [ ] 4. **Registration**: the launch record (`docs/experiments/logbooks/supporting/098-collapse-remedies/launch.md`),
  committed before any round; `/nematode-review-spec`.
- [ ] 5. **Rounds**, in order, each from a worktree, stopping at the first pass: KL early stopping, weight
  clipping, Adam's second moment.
- [ ] 6. **Readout**: Logbook 098; tracker M.1(b) and the chosen remedy or the fired stop rule; M.1(c)'s
  change registers next.
- [ ] 7. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR.
