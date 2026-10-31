## Why

C.1's v1 answer fails benchmark v2 on all five populations (D 6.1–6.2; [Logbook 115](../../../docs/experiments/logbooks/115-benchmark-v2.md)), so C.1 must be refitted on v2 before it can enter C.4. F.0b-i on v2 now clears sep-CMA for C.1's model on v2's objective ([Logbook 117](../../../docs/experiments/logbooks/117-cma-recovery-v2b.md); median gain 0.873). That recovery left four requirements for C.1b's plan:

- **the objective:** fit with the ranking score (definedness first, then 10·asinh(z/10)), and grade on the capped D;
- **the plateau guard:** two of its 16 restarts stopped at generations 47 and 52 while their window still held medians with an undefined statistic;
- **presence:** the ranking score leaves it out, but the wild-type pass needs it;
- **reachability first:** both of truth 1164's probe misses were on the kinematics speed medians, and B v2's own reachability oracle missed once by 0.07.

The maintainer's decisions of 2026-10-31:

- **restarts are chosen on separate populations**, graded with presence, and the verdict is read on fresh registered ones;
- **the reachability pilot** is one restart run to 300 generations on discarded seeds, which also re-checks the cap on real data;
- **the record conversion and measurement are restricted to each family's analysis window** (exact, about 7% per simulation).

## What Changes

- **`scripts/analysis/c1b_fit.py`**, C.1 refitted on v2:
  - the objective: the ranking score against the real bands at n = 200;
  - sep-CMA from the box centre, 8 restarts;
  - a plateau stop guarded by σ and by a window free of undefined medians;
  - a pilot, a run, selection, grading, a reading and a smoke.
- **The reachability pilot** (discarded seeds), before registration Part B. It sets the cap by F.0b-i's rule, projects the cost, and grades its answer on 5 discarded populations. It returns to the maintainer if that answer does not pass 3 of 5.
- **Selection:** every plateaued restart is graded with presence on 3 selection populations. The restart passing most is chosen, with ties broken by the lower median D.
- **The verdict:** the chosen restart passes the wild type on at least 3 of 5 fresh registered populations, graded with presence. v2's 3-of-5 oracle for a faithful controller is 0.955. Every restart, plateaued or not, is graded on those populations too, for C.4's `unidentified` rule, as C.1's v1 fit graded every restart.
- **The window restriction** (`BatchedTrajectories.to_behaviour_steps(animal, start, stop)`, `measurement.measure_window`, used by `benchmark_grade.simulate_v2`), byte-identical, and checked with `speedup_identity_check.py` against a baseline from `main`.
- **Launch record 119** (Part A before the pilot, Part B after it), binding the registered mutant encoding table (`113-c1-fit/encodings.md`) that C.4 grades C.1b's predictions under, and **Logbook 119**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `controller-fit`: C.1 is refitted on benchmark v2, with selection on separate populations, a doubly guarded plateau and a reachability pilot.
- `benchmark-grading`: a v2 simulation builds and measures only each family's analysis window, with identical output.

## Impact

- `scripts/analysis/c1b_fit.py` (new); `scripts/analysis/benchmark_grade.py` (`simulate_v2`); `quantumnematode/evaluation/batched.py` (`to_behaviour_steps`); `quantumnematode/benchmark/measurement.py` (`measure_window`); `scripts/analysis/speedup_identity_check.py` (`--build` and `--out`, so each change keeps its own baseline and record).
- `docs/experiments/logbooks/supporting/119-c1b-fit/`; Logbook 119; the tracker (C.1, C.4, the seed table).
- **Cost:**
  - the pilot: about 2.4 h at about 29 s per generation;
  - the fits: about 7–13 h, and at most about 18 h at a cap of 272;
  - grading: under an hour.
