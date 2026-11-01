## 1. The window restriction (before anything else edits the simulation path)

- [x] 1.1 Give `speedup_identity_check.py` `--build` and `--out`, then record a baseline with `baseline --build build/c1b-fit/identity` on `main`'s code, before any edit to the simulation path.
- [x] 1.2 `to_behaviour_steps(animal, start, stop)`, `measurement.measure_window`, and `benchmark_grade.simulate_v2` using them, with `analysis_steps`'s length check kept. Tests: a windowed animal equals the whole episode windowed, field by field and type by type.
- [x] 1.3 `speedup_identity_check.py check --build build/c1b-fit/identity --out docs/experiments/logbooks/supporting/119-c1b-fit/identity.json`: every job identical.

## 2. The fit

- [x] 2.1 `scripts/analysis/c1b_fit.py`: `BandObjective`, `defined_stop`, the seed scheme, `pilot`, `run`, `select`, `grade`, `read` and `smoke`, guarded by the launch record (Part A for `pilot`, Part B for the rest) with `113-c1-fit/encodings.md` bound. Tests: the objective's scenario, the guard, the selection rules and ties, disjoint populations.
- [x] 2.2 The smoke (CMA 5999, animals 99,050,000, grading 99,060,000).

## 3. Registration and the pilot

- [x] 3.1 Launch record 119, Part A, committed.
- [x] 3.2 `pilot`: G_p, D at G_p, at ⌈1.5·G_p⌉ and at 300, the cap, the projection, and reachability on 5 discarded populations. Return to the maintainer if it passes fewer than 3, or if the projection is over 20 h.
- [x] 3.3 Part B: the cap, the projection and the pilot's reachability, committed.

## 4. The fits and the verdict

- [x] 4.1 `run` (8 restarts), `select` (3 populations per plateaued restart), `grade` (5 registered populations per restart, all 8), `read`: `fit.json` with the verdict.

## 5. Close-out

- [x] 5.1 Logbook 119, the tracker (C.1 on v2, C.4, the seed table), the index row, AGENTS.md and the CHANGELOG.
- [x] 5.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
