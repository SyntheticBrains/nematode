## Why

F.0b-i ([Logbook 112](../../../docs/experiments/logbooks/112-cma-recovery.md)) showed that sep-CMA recovers C.1's 20-parameter model near the wild type (median gain 0.991), and it cleared F.0 for C.1. But it was read on benchmark v1, whose 11 statistics see exact simulated tracks. B v2 ([Logbook 115](../../../docs/experiments/logbooks/115-benchmark-v2.md)) is now the benchmark:

- 19 statistics in five families, adding kinematics and chemotaxis;
- simulated tracks passed through a measurement model;
- C.1's v1 answer fails it.

C.1b refits C.1 on v2. That requires the method to recover known parameters under the objective C.1b will fit, so B v2's registration names this re-run, in full, as the next change.

The maintainer's decisions of 2026-10-28:

- **the truths** are re-selected on v2 by F.0b-i's own rule;
- **the cap** is set again by a 300-generation pilot;
- **the objective** is the distance over v2's 19 graded statistics, five families, as C.1b will fit;
- **v2's simulation is made faster first**, without changing any number. A v2 simulation costs 3.25× v1's (36 s against 11 s), so a re-run as is would take about 45 h.

## What Changes

- **A faster v2 simulation, number for number:**

  - a fit's objective reads statistics only, so its grading skips the presence resamples (2,000 per population, about 14 s);
  - the noise library's stretch offsets are computed once, not at every splice;
  - measured steps are built without a per-step dataclass copy.

  Parity tests show the statistics, D and the measured positions unchanged. Registered gradings keep the presence tests.

- **`cma_recovery_v2.py`** re-runs F.0b-i on v2, on F.0b-i's rules except where v2 requires otherwise:

  - the v2 simulator and its objective, the distance over v2's 19 statistics on its n = 200 band scales and five-family weights;
  - truths re-selected on v2 from the 43 F.0a prior simulations F.0b-i's rule admits, with the gates' oracle read on discarded seeds before registration;
  - the pilot, the cap rule, the gates and the read, with new seeds;
  - the plateau stop as C.1b will use it, with the σ ≤ σ₀ guard B v2 adopted.

- **`cma_recovery.run_fit`** takes an optional simulator and plateau guard. Its defaults keep v1's run reproducible.

- **Launch record 116**, in two parts, and **Logbook 116**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `fit-recovery`: F.0b-i's recovery re-read on benchmark v2.
- `benchmark-grading`: a fit's objective may grade without presence. Registered gradings may not.

## Impact

- `scripts/analysis/cma_recovery_v2.py` (new); `scripts/analysis/cma_recovery.py` (`run_fit` options, defaults unchanged); `scripts/analysis/benchmark_grade.py` (`grade_populations_v2(presence=...)`); `packages/quantum-nematode/quantumnematode/benchmark/measurement.py` (the speed-ups).
- `docs/experiments/logbooks/supporting/116-cma-recovery-v2/` and Logbook 116; the tracker's F.0 entry.
- **Cost:** about 20–25 h of compute after the speed-up. The preflight is about 0.5 h, the pilot about 4–5 h, and the registered fits up to about 16–19 h at a 210-generation cap. The pilot measures it.
