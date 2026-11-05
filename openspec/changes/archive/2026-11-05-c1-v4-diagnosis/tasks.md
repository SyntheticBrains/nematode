## 1. The option

- [x] 1.1 `BatchedMechanistic`: `r_dwell_on_food`, off by default; the coupling and intermittent dwelling removed after their probes (kept at `886e7bfc`). Tests for each scenario.
- [x] 1.2 `speedup_identity_check.py check --build build/c1-v4/identity --out supporting/125-c1-v4-fit/identity.json`: all 59 jobs byte-identical.

## 2. The records

- [x] 2.1 Every probe's record and script in `supporting/125-c1-v4-fit/probes/`; the drafted fit design parked in `parked_fit_design.md`.
- [x] 2.2 Logbook 125; the tracker (C.1, C.4, the seed table); AGENTS.md and the CHANGELOG.

## 3. Close-out

- [x] 3.1 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
