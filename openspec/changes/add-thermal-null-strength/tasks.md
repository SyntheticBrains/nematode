# Tasks: A.6t — the null-strength control and its split on block V's thermal cell

Phase 8b carried control A.6t. Registered in
`docs/experiments/logbooks/supporting/077-thermal-null-strength/launch.md` before any scored run.

- [x] 1. **Scorer**: `score_level` takes the block-V cell, defaulting to its own.
- [x] 2. **Configs**: the four thermal narrower-null configs, one key each, through the generator; tests
  through the real loader, and the gap-held null's chemical mask checked against the current null's.
- [x] 3. **Analysis**: `thermal_null_strength.py`, with tests for the registration constants, the verdict
  maps, the gates and the manifest.
- [x] 4. **Registration**: the launch record, before any scored run.
- [ ] 5. **Campaign**: seeds 385–512, 3,000 episodes, 1,024 runs, with the output controls.
- [ ] 6. **Readout**: the analysis JSON and per-seed CSV committed; Logbook 077; the experiments index.
- [ ] 7. **Records**: the tracker's A.6t; the roadmap's block V conditions and A.5's fallback package;
  the README status line if it names block V's thermal cell.
- [ ] 8. **Close-out**: the full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
