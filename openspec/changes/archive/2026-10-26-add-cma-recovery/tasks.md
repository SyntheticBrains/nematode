# Tasks: F.0b-i — sep-CMA recovers C.1's model near the wild type

Phase 10, block F, tracker item F.0. The maintainer's decisions of 2026-10-25:

- C.1 is fitted by sep-CMA (D38 as amended);
- F.0b split, C.1's model first;
- 4 truths × 2 restarts;
- a pilot-set cap with a plateau stop;
- each truth's best restart, with a median gain of at least 0.75;
- the start at the box centre, σ₀ 0.2.

## 1. Tooling

- [x] 1.1 `scripts/analysis/cma_recovery.py`, reusing `npe_recovery`'s simulator, distance and pool and `npe_region_check`'s gain fraction. Its stages:
  - `preflight`;
  - `pilot`;
  - `run`: resumable each generation;
  - `read`: guarded.
- [x] 1.2 Tests:
  - the plateau rule: it fires on a flat series and not on a descending one, and never before generation 39;
  - the cap from G_p;
  - the truth-selection rule: flips are skipped, and so is a truth with no reachable gain;
  - the best-restart choice by S;
  - the answer is `xfavorite` and lies in the box when the mean has left it;
  - the cap falls back to 300 when the pilot's gain rises after ⌈1.5·G_p⌉;
  - the seed layout's disjointness;
  - `read` refuses when a record is uncommitted or changed.
- [x] 1.3 A smoke of the fit loop on a tiny budget (CMA seeds 5991–5999, animals at `99_000_000 + 200·k + i`, discarded).

## 2. Registration

- [x] 2.1 `supporting/112-cma-recovery/launch.md` Part A, committed before 2.2.
- [x] 2.2 `preflight` (stop if a check fails), then `pilot` (stop if its gain at 300 is below 0.75; the maintainer decides if the projection exceeds 20 h).
- [x] 2.3 Part B, committed with `preflight.json` and `pilot.json` before 3.1.

## 3. The run

- [x] 3.1 `run`: the 8 fits, then each fit's mean re-simulated.
- [x] 3.2 `read`: the gate and the reported measures, into `recovery.json`.

## 4. Documentation and tracker

- [x] 4.1 Logbook 112: the truths, the preflight, the pilot, the gate, the spread, the parameter errors and the cost.
- [x] 4.2 The tracker: F.0's C.1 half; the binding's dated amendment (F.0 clears per model); the seed table (this change's schemes, the smoke's and the planning probe's).
- [x] 4.3 AGENTS.md and the CHANGELOG.

## 5. Close-out

- [x] 5.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [x] 5.2 `openspec validate add-cma-recovery --strict`.
- [x] 5.3 Archive, with the spec's changes merged.
- [x] 5.4 PR.
