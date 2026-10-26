# Tasks: C.1 — the mechanistic baseline fitted to the wild type, and the mutant encoding table

Phase 10, block C, tracker item C.1 (and D37's table for C.4). The maintainer's decisions of 2026-10-26:

- the full encoding table in C.1's Part A;
- structural knockouts only;
- tph-1's leaving left out and reported beside;
- 8 restarts.

## 1. Tooling

- [ ] 1.1 `cma_recovery.run_fit` gains an optional `objective`, with the default unchanged.

- [ ] 1.2 `scripts/analysis/c1_fit.py`, with stages:

  - `run`: 8 restarts, resumable;
  - `grade`: every answer on the five registered populations, plus 10 re-simulations each;
  - `read`: the chosen restart, the verdict and the reported measures;
  - `smoke`.

  `run` is guarded on the launch record and the encoding table.

- [ ] 1.3 Tests:

  - the default objective is unchanged, and a custom objective is used;
  - capped restarts are not chosen;
  - no plateau, no verdict;
  - the wild-type verdict from family passes, 3 of 5;
  - the `unidentified` rule on toy verdicts;
  - the seed layout's disjointness;
  - the guard binds the encoding table;
  - `cma_recovery`'s guard is unchanged at its default path.

- [ ] 1.4 A smoke of `run` on a tiny budget (CMA seed 5992, animals at `99_100_000`, discarded).

## 2. Registration

- [ ] 2.1 `supporting/113-c1-fit/launch.md` and `encodings.md`, committed before 3.1.

## 3. The run

- [ ] 3.1 `run`: the 8 restarts.
- [ ] 3.2 `grade` and `read`, into `fit.json`.

## 4. Documentation and tracker

- [ ] 4.1 Logbook 113: the verdict, every restart, the spread, the parameters and the cost.
- [ ] 4.2 The tracker:
  - C.1;
  - C.4's encoding table and the `unidentified` rule;
  - W.2's note (the restart spread on real data);
  - the observation-size note, resolved;
  - the seed table.
- [ ] 4.3 AGENTS.md and the CHANGELOG.

## 5. Close-out

- [ ] 5.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [ ] 5.2 `openspec validate add-c1-fit --strict`.
- [ ] 5.3 Archive, with the new spec's Purpose written.
- [ ] 5.4 PR.
