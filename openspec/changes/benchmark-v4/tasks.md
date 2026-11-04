## 1. The statistics, behind v4's own tuples

- [ ] 1.1 Add v3 grading jobs to `speedup_identity_check.py` (v3's grade with and without presence, on its stored populations), then record its baseline (`--build build/benchmark-v4/identity`) on the code before any statistic edit.

- [ ] 1.2 `grading`:

  - the activity tally (decision 1) and statistics (decision 2);
  - `ACTIVITY_STATISTICS_V4` and v4's tuples, with `z_star` at 24 and 23;
  - v1–v3 untouched.

  Tests for each scenario. The pause floor is derived from the noise library in code, and a test asserts 0.02 mm.

- [ ] 1.3 `speedup_identity_check.py check`: v1–v3 byte-identical.

## 2. Bands and grading

- [ ] 2.1 `benchmark_bands.py --version v4`, refusing until Part A is committed:
  - v3's bands byte-identical;
  - the activity bands at n = 200 and 64, with κ (`default_rng(2045)`);
  - the rule check (`default_rng(2046)`);
  - N2's activity bands (`default_rng([2047, n])`);
  - `activity_graded: null`, and `--onset graded|reported` to set it after Part B; every v4 grade refuses while it is null.
- [ ] 2.2 `benchmark_grade.py --bands v4`, `grade_populations_v4` (the onset rate graded or reported, per Part B), `v4_is_valid` and the default chain; `track_import` carries the activity tally; tests.

## 3. Registration, the oracle and the controls

- [ ] 3.0 Preflights on discarded seeds:

  - the adversary gate (606,000,000): C.2's stage-2 answer on 5 populations, against provisional bands built in process;
  - one oracle restart for 60 generations (607,000,000, CMA 9433).

  Return to the maintainer if the answer is caught in fewer than 3 of 5, or if the oracle cannot bring the pause share and the flip fraction within the bar. Commit `preflight.json` in `supporting/124-benchmark-v4/`.

- [ ] 3.1 Launch record 124, Part A (rules, seeds, the validity map, the disclosure), committed.

- [ ] 3.2 Bands built; the rule check; the round trip; the oracle's fit (CMA 9431–9432, cap 100).

- [ ] 3.3 Part B: the oracle's answer and the onset rate's status, committed; then `benchmark_bands.py --version v4 --onset …` sets `activity_graded`.

- [ ] 3.4 Then, recorded in `controls_v4.json`:

  - the oracle's 40 populations;
  - fresh N2;
  - the adversary;
  - the negatives;
  - C.1's v3 answer and PD1074's held-out cohorts, reported;
  - W's minimum and the 3-of-5 probability.

## 4. Close-out

- [ ] 4.1 Logbook 124. The tracker (B v4; C.1's refit if v4 fails it; C.2's next change against v4; the seed-band table's v4 rows: 600–607 M, generators 2045–2047, CMA 9431–9433), `data/benchmark/IMPORT.md`, AGENTS.md and the CHANGELOG.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
