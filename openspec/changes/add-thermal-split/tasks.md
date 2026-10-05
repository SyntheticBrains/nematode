# Tasks: the gap-only split on block V's thermal cell at target 35

A.6t's follow-up. Registered in `docs/experiments/logbooks/supporting/078-thermal-split/launch.md`
before any scored run.

- [x] 1. **Pilot**: seeds 1001–1004 at targets 25, 30, 40, then 35 under the rule fixed before it;
  `pilot.json` and `selection-rule.md` committed.
- [x] 2. **Configs**: six target-35 configs through `generate_thermal_target_configs.py`; tests through
  the real loader and on the gap-held null's chemical mask.
- [x] 3. **Analysis**: `thermal_split.py`, with tests for the constants against the pilot record, the
  verdict maps and the manifest.
- [x] 4. **Registration**: the launch record, with the gate preflight's output.
- [ ] 5. **Review**: `/nematode-review-spec`, including its campaign-readiness checks, before launch.
- [ ] 6. **Campaign**: seeds 513–640, 3,000 episodes, 768 runs, with the output controls.
- [ ] 7. **Readout**: Logbook 078 with the analysis JSON and per-seed CSV; the experiments index.
- [ ] 8. **Records**: the tracker's A.6t; block V's thermal condition in the roadmap.
- [ ] 9. **Close-out**: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by exit
  code; `openspec validate --strict`; archive and PR.
