## 1. The objective

- [ ] 1.1 `predictive_distance_v2(..., compressed=True)` and `Objective(compressed=True)`; tests (the order of far against undefined; near the truth the uncapped order; the capped D unchanged).

## 2. The probe

- [ ] 2.1 `probe-b`: 1164 and 6985 to the guarded plateau (cap 200), 9105 for 80 generations, base 300 M, CMA 5995–5997; the answers' gains on the capped D. Return to the maintainer if any is below 0.75.

## 3. Registration and the fits

- [ ] 3.1 `supporting/117-cma-recovery-v2b/launch.md` Part A, with the probe disclosed; committed.
- [ ] 3.2 `run-b`: 8 fits, base 230 M, CMA 5430–5437, cap 272; `read-b`: `recovery.json`.

## 4. Close-out

- [ ] 4.1 Logbook 117; the tracker (F.0; C.1b fits with the same objective); AGENTS.md.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
