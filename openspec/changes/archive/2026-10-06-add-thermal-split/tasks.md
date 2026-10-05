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
- [x] 5. **Review** — **done: no blocking findings; the minimum's 4-seed sampling range (0.019–0.029) and the dry run and rule-timing evidence added to the launch record.** Original scope: `/nematode-review-spec`, including its campaign-readiness checks, before launch.
- [x] 6. **Campaign** — **done: 768/768, 13 h.** Original scope: seeds 513–640, 3,000 episodes, 768 runs, with the output controls.
- [x] 7. **Readout** — **done: Logbook 078; `gap_junctions` and `lead_below_minimum`.** Original scope: Logbook 078 with the analysis JSON and per-seed CSV; the experiments index.
- [x] 8. **Records** — **done: tracker A.6t met; roadmap and README carry block V's thermal half.** Original scope: the tracker's A.6t; block V's thermal condition in the roadmap.
- [x] 9. **Close-out** — **done: full suite 7,051 passed; hooks pass with everything staged, judged by exit code; validated; archived.** Original scope: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by exit
  code; `openspec validate --strict`; archive and PR.
