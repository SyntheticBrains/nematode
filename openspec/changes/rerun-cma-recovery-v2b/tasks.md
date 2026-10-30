## 1. The objective

- [x] 1.1 `ranking_score` (1,000 × u + D_c) and `Objective(ranking=True)`. Tests: undefined never ahead of defined (the review's case); near the truth within 1.5%; the capped D unchanged; u counts only one-sided undefinedness.
- [x] 1.2 The observations' digest: `candidates_pinned()`, and a refusal in `probe-b`, `run-b` and `read-b`; a test.

## 2. The probe

- [x] 2.1 `probe-b`: 1164 and 6985 to the guarded plateau (cap 200), 9105 for 80 generations, base 300 M, CMA 5995–5997; the answers' gains on the capped D. Return to the maintainer if any is below 0.80. (Returned: 1164 0.787, 6985 0.981, 9105 1.000.)
- [x] 2.2 `diagnose-b` and, by the maintainer's decision, `probe-b2`: 1164's second restart (fit 3, CMA 5998), read as the gate selects (0.523; the selected restart stays at 0.787).
- [x] 2.3 The maintainer's decision: 1164 accepted below the bar (`PROBE_B_ACCEPTED`), and 4 restarts per truth (`RESTARTS_B`). `_run_fits` and `_read_fits` take the restart count, and attempt 1's `recovery.json` re-reads byte-identical. Tests: the accepted truth, best of four.
- [x] 2.4 The spec review of the amendment: `read-b` reports the better of the first two restarts as description; `probe_diagnosis.json` is bound; the spec removes attempt 1's uncapped-ranking requirement; the fail branch sends the per-restart diagnosis to the maintainer.

## 3. Registration and the fits

- [ ] 3.1 `supporting/117-cma-recovery-v2b/launch.md` Part A: the ranking score, the reused truths and the observations' SHA-256, the cap's provenance, both probe rounds and the maintainer's decision disclosed; committed.
- [ ] 3.2 `run-b`: 16 fits (4 truths × 4 restarts), base 230 M, CMA 5430–5445, cap 272; `read-b`: `recovery.json`.

## 4. Close-out

- [ ] 4.1 Logbook 117; the tracker (F.0; C.1b fits with the same score, its plan handles presence and selects among restarts on the registered grade, and a reachability pilot on the real bands precedes its registration); AGENTS.md, naming the stages `probe-b`, `diagnose-b`, `probe-b2`, `run-b` and `read-b`.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
