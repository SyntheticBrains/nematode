## 1. The statistics, behind v3's own tuples

- [ ] 1.1 Identity baseline (`speedup_identity_check.py baseline --build build/benchmark-v3/identity`) on the code before any edit.
- [ ] 1.2 `chemotaxis`: the onset rate and the direction-flip fraction from fragment samples; `grading`: v3's tuples, the log early/late, `z_star` at 21; the reported reversal share. Tests for each scenario, and for v1 and v2 unchanged.
- [ ] 1.3 `speedup_identity_check.py check`: v1 and v2 byte-identical.

## 2. Bands and grading

- [ ] 2.1 `benchmark_bands.py --version v3` (refuses until Part A is committed): v2's lawn, uniform and kinematics bands; local search's log early/late; chemotaxis's five; the leave-one-out allowance.
- [ ] 2.2 `benchmark_grade.py --bands v3`, `grade_populations_v3`, `v3_is_valid`, the default chain; tests.

## 3. Registration, the oracle and the controls

- [ ] 3.1 Launch record 120, Part A (rules, seeds, the validity map, the disclosure), committed.
- [ ] 3.2 Bands built; the round trip; the oracle's fit (CMA 9421–9422, cap 100).
- [ ] 3.3 Part B: the oracle's answer, committed. The oracle's 40 populations; held-out López-Cruz; Chen leave-one-out; adversaries; negatives; W's minimum and the 3-of-5 probability; `controls_v3.json`.

## 4. Close-out

- [ ] 4.1 Logbook 120; the tracker (B v3; C.1 refit next, on v3); AGENTS.md; the CHANGELOG.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
