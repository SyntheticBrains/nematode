## Why

C.1b's fits on benchmark v2 found **reversal-dominated locomotion** ([Logbook 119](../../../docs/experiments/logbooks/119-c1b-fit.md), addendum). Its answers reverse 42–56% of the time off food, against real worms' 17.3% (Roberts' tracker, off food) and 12–16% (Scheer's flags, on lawns). The benchmark let that through:

- **local search** grades only the curve's shape, and its linear early/late band (4.21 ± 1.62) accepts any ratio from −0.66 to 9.09, so a worm reorienting 9.5 times per 2-min bin, flat, passes;
- **backward motion is invisible from midbody positions**: a worm backing continuously draws the path of one going forward;
- **no statistic grades how often a worm reorients**.

Every later controller (C.2, C.3, W's nulls) could use the same gap. So the benchmark is revised before C.1 is refitted or C.2 begins.

**A discriminability probe** (description, discarded seeds; `build/benchmark-v3/probe.json`) scored candidates on C.1b's 8 restarts and the off-food κ probe against held-out real data. No single position-based statistic caught all four reversal-dominated answers; three together caught every answer, and the held-out real worms passed all three (|z| ≤ 1.23).

The maintainer's decisions of 2026-11-01:

- **grade three position-based statistics:**
  - local search's early/late ratio on a log scale;
  - the onset rate on Chen's chemotaxis tracks;
  - the direction-flip fraction on those tracks;
- **report the share of time reversing, not graded,** where a simulator supplies primitive labels. Grading it would need a head–tail measurement model, a D36 amendment and a second tier for imported tracks. It is the escalation if a fit is caught by the reported share but passes the three;
- **the new statistics sit in the chemotaxis family,** which grows from 3 to 5, every family keeping its equal weight;
- **a reachability oracle** for the new statistics is part of v3's controls.

## What Changes

- **Benchmark v3** (`data/benchmark/bands_v3.json`), 21 graded statistics at z\* 3.0381:
  - the small lawn, the uniform well and kinematics unchanged from v2;
  - local search: γ, and **the log early/late ratio** in place of the linear one;
  - chemotaxis: drift, the log quintile ratio, the weathervane slope, **the onset rate per minute** and **the direction-flip fraction** (consecutive 1-s steps turning more than 135°, both at least 0.03 mm).
- **Reported beside the grade:** the share of seconds reversing in local search and chemotaxis, from primitive labels, against Roberts' real 17.3% off food.
- **v3's controls** (`scripts/analysis/benchmark_v3_controls.py`), registered before any reading:
  - held-out López-Cruz batches pass;
  - Chen's leave-one-out passes, with an allowance;
  - B's negatives, v2's blind worm and C.1's v1 answer fail;
  - a new adversary check (C.1b's reversal-dominated answers each fail a new statistic);
  - a reachability oracle;
  - W's minimum and the 3-of-5 probability recomputed.
- **`benchmark_grade.py --bands v1|v2|v3`**, with v3 the default once VALID (`controls_v3.json`, its bands' SHA-256). v1 and v2 stay byte-identical.
- **Preflights before Part A,** on discarded seeds: the adversary gate on 5 populations per answer, and a 60-generation oracle restart.
- **`data/benchmark/IMPORT.md`:** imported local search is graded on log early/late, and the new chemotaxis statistics, like chemotaxis itself, cannot be imported.
- **Launch record 120** (Part A before any reading, Part B after the oracle's fit) and **Logbook 120**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `benchmark-grading`: v3's statistics, bands and reported reversal share.
- `benchmark-controls`: v3's validity map.

## Impact

- `quantumnematode/benchmark/{grading,chemotaxis}.py`: v3's statistic tuples and the onset rate and flip fraction; v2's tuples untouched.
- `scripts/analysis/{benchmark_bands,benchmark_grade}.py`: `--version v3`, `grade_populations_v3`, `v3_is_valid`; `scripts/analysis/benchmark_v3_controls.py` (new).
- `data/benchmark/bands_v3.json`; `docs/experiments/logbooks/supporting/120-benchmark-v3/`; Logbook 120; the tracker; AGENTS.md; the CHANGELOG.
- **Cost:** bands in minutes; the oracle's fit about 1–2 h; the controls' reading under an hour. C.1's refit on v3 follows in its own change (about 12–18 h).
