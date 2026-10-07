# Tasks: M.8 — plastic gap junctions on the settling substrate

Phase 8 optional M.8, run alongside C.1's build. Registered in
`docs/experiments/logbooks/supporting/082-plastic-gaps/launch.md` before any scored run.

- [ ] 1. **The gap-only null**: `rewire_chemical` in the rewiring; `rewired_gap_junctions_only` in the
  brain. Tests: the chemical graph held exactly, gap degrees held, the pinned default null.
- [ ] 2. **Plastic gaps**: `plastic_gaps`, the multiplier parameter and its use in both forward paths,
  the refusal under leaky. Tests: symmetry, positivity, no created pairs, a non-zero gradient, off
  byte-identical.
- [ ] 3. **Configs**: the thermal t35 arms through a generator, with loader tests.
- [ ] 4. **Analysis**: `plastic_gaps.py` (the identity check, the readings, the verdict map, the
  plasticity check, the per-seed CSV) in the gate preflight's shape, with tests.
- [ ] 5. **Identity check and pilot**: 6 re-runs; the pilot on 1401–1404; preflight; plasticity
  check; cost.
- [ ] 6. **Registration**: the launch record, then `/nematode-review-spec`; the tracker's M.8 and M.7
  decisions.
- [ ] 7. **Campaign**: from a separate worktree, seeds 513–576, 192 runs.
- [ ] 8. **Readout**: Logbook 082; tracker M.8.
- [ ] 9. **Close-out**: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
