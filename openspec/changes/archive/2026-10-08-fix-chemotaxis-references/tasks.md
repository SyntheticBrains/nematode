# Tasks: M.7 — a verified chemotaxis reference set, and no literature verdict on the simulated index

- [x] 1. **The reference file**: rebuild `literature_ci_values.json` from the verified entries, with the
  dropped entries and their reasons recorded; update the loader to the rebuilt schema. Tests.
- [x] 2. **The tracker**: stop computing the literature range, typical value, citation and verdict;
  keep the index and its level; describe the level as a banding. Older records load. Tests.
- [x] 3. **Retire the comparison API**: remove `ChemotaxisValidationBenchmark` and the built-in fallback
  dataset; the summary printout drops the literature lines. Tests.
- [x] 4. **Records**: tracker M.7, CHANGELOG.
- [x] 5. **Close-out** — **done 2026-10-08: full suite 7404 passed, 34 skipped; hooks; validate; archived.** Original scope: full suite; hooks by exit code; `openspec validate --strict`; archive; PR.
