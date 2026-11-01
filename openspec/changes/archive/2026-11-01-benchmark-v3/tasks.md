## 1. The statistics, behind v3's own tuples

- [x] 1.1 Identity baseline (`speedup_identity_check.py baseline --build build/benchmark-v3/identity`) on the code before any edit.
- [x] 1.2 `chemotaxis`: the onset rate and the direction-flip fraction from fragment samples; `grading`: v3's tuples, the log early/late, `z_star` at 21; the reported reversal share. Tests for each scenario, and for v1 and v2 unchanged.
- [x] 1.3 `speedup_identity_check.py check`: v1 and v2 byte-identical.

## 2. Bands and grading

- [x] 2.1 `benchmark_bands.py --version v3` (refuses until Part A is committed): v2's lawn, uniform and kinematics bands byte-identical, with their rule checks re-run at v3's bar; local search's γ and log early/late at m = 21; chemotaxis's five at m = 21; the leave-one-out allowance.
- [x] 2.2 `benchmark_grade.py --bands v3`, `grade_populations_v3`, `v3_is_valid`, the default chain; tests.

## 3. Registration, the oracle and the controls

- [x] 3.0 Preflights on discarded seeds: the adversary gate (99,950,000) and one oracle restart for 60 generations (99,960,000, CMA 9423). Return to the maintainer if an adversary escapes in 3 of 5, or the oracle cannot bring a new statistic within the bar. Commit `probe.json`, `preflight.json` and the brief's summary in `supporting/120-benchmark-v3/`.
- [x] 3.1 Launch record 120, Part A (rules, seeds, the validity map, the disclosure), committed.
- [x] 3.2 Bands built; the round trip; the oracle's fit (CMA 9421–9422, cap 100).
- [x] 3.3 Part B: the oracle's answer, committed. The oracle's 40 populations; held-out López-Cruz; Chen leave-one-out; adversaries; negatives; W's minimum and the 3-of-5 probability; `controls_v3.json`.

## 4. Close-out

- [x] 4.1 Logbook 120; the tracker (B v3; C.1 refit next, on v3); `data/benchmark/IMPORT.md`; AGENTS.md; the CHANGELOG.
- [x] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
