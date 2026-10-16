# Tasks: M.1(a) — the collapse instrument and the diagnosis

Phase 9 block M (roadmap v4.5 § Phase 9, D25, D26; tracker `phase9-tracking` M.1(a)). Registered in
`docs/experiments/logbooks/supporting/096-collapse-diagnosis/launch.md` before any log is read.

- [ ] 1. **Telemetry module and hooks**: the per-update statistics on the connectome PPO rule and the MLP
  brain, behind a flag the brains read; the replay returns the second-to-last settling iterate when on.
  Tests: each statistic against a hand computation on a tiny batch; telemetry on and off identical on
  every output, the final weights and the next value of every generator.
- [ ] 2. **CLI**: `--update-telemetry` and `--checkpoint-every N` in `run_simulation.py` (`run_campaign.py`
  already passes extra flags through to each run, so it needs no change); the JSONL path and the checkpoint names; refusals (`N` ≤ 0). Tests: the files
  appear with the expected rows and names; off writes nothing new.
- [ ] 3. **`collapse.py`**: D25's definition, incidence with Wilson intervals, Kaplan–Meier
  time-to-collapse, the sensitivity grid, 092's rule beside. Tests on synthetic series, including a
  recovery, a late onset censored at the horizon and a never-learning run.
- [ ] 4. **`collapse_diagnosis.py`**: the identity check, the event-aligned stacks, the envelope (the eight
  intact twins aligned on their collapsed twins' onsets, mean ± 2 sd, leave-one-out for the intact side),
  the departure rule and the cause readout of Decision D, the MLP incidence. Tests on synthetic telemetry.
- [ ] 5. **Registration**: the launch record (the 16 paired runs and the 8 MLP runs, the identity rule,
  the cause set and its patterns, the indicated/not-indicated rule, the retention line), committed;
  `/nematode-review-spec`.
- [ ] 6. **Campaign**: 24 runs at 6,000 episodes with `--update-telemetry --checkpoint-every 250` and the
  output controls, from a separate worktree.
- [ ] 7. **Readout**: Logbook 096; tracker M.1(a) ticked with the causes indicated; M.1(b)'s remedy order
  recorded for its change; CHANGELOG; `docs/usage.md`.
- [ ] 8. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR.
