# 118: Phase 10 — A v2 Simulation 1.54× Faster on One Core, Every Output Identical

**Status**: completed. One v2 simulation of C.1's model now takes 17.8 s on one core, against 27.4 s (**1.54×**), and 32 simulations on 16 workers run **1.35×** faster. Every graded statistic, distance and presence reading is byte-identical to before: all 55 of the identity check's jobs match on both benchmarks. No registered result changes.

> | Measure | Before | After | Speed-up |
> |---|---|---|---|
> | One simulation, one core | 27.4 s | 17.7–17.9 s | 1.54× |
> | 32 simulations, 16 workers | 75.6–79.4 s | 56.6–57.9 s | 1.35× |
> | Identity check | | 55 of 55 jobs identical | |

**Date**: 2026-10-31.

**OpenSpec change**: `speed-up-v2-simulation`.

**Records** (in [`supporting/118-v2-speedup/`](supporting/118-v2-speedup/)):

- `identity.json`: every job's comparison;
- `timing.json`: the timings, old and new code alternating in one session.

## Objective

Every remaining Phase 10 fit runs v2 simulations: C.1b, F.0b-ii, C.2 and W. A profile of one simulation (26 s, one core) put more than half its time outside the simulation itself:

| Part | Time |
|---|---|
| converting the evaluator's arrays into per-step records | 5.1 s |
| the measurement model | 6.3 s |
| drawing C.1's uniforms one animal at a time | 3.2 s |

The maintainer chose to speed up all three, with every output byte-identical, so every registered result, floor and stored record stays valid.

## What changed

- **The record conversion** (`BatchedTrajectories.to_behaviour_steps`) reads each column once with `.tolist()`, which gives the same Python floats as `float` element by element. It computes the signed speed with `math`'s cosine and sine on those floats, as before, and builds each step in one call.
- **The measurement model** (`measure`) adds the noise to the positions as arrays, the same sums element for element, and rebuilds each step from one attribute read. **The noise splice** (`NoiseLibrary._splice`) lays each candidate's usable offsets end to end in the order drawn, instead of on a padded grid, so the first minimum is the same one, with the same draws.
- **C.1's uniforms** (`BatchedMechanistic`) are drawn in blocks of 4,096 seconds per animal and taken three per second, in order. One draw of 3k uniforms from a torch generator gives the numbers k draws of three would, and leaves the generator in the same state (checked on torch 2.14.1, and pinned by a test). The block is tied to the run's generators, which the evaluator builds fresh every run and only the controller draws from.

## Results

### Identity (`identity.json`)

The check ran in two stages:

- **`baseline`** ran on the code before any edit, on discarded seeds (base 99.7 M);
- **`check`** ran after the change.

It covers every path the changed code serves:

| Jobs | Path | Identical |
|---|---|---|
| 3 baseline vectors (the box centre and two F.0a vectors) | v2, without presence | 3 of 3 |
| the same 3 | v2, with presence | 3 of 3 |
| the same 3 | v1 (`npe_recovery.simulate`) | 3 of 3 |
| attempt 2's probe answers ([Logbook 117](117-cma-recovery-v2b.md)) | v2, without presence | 40 of 40 |
| observations from `candidates.jsonl`, with their presence readings | v2, with presence | 3 of 3 |
| F.0a's stored simulations ([Logbook 111](111-npe-recovery.md)) | v1 | 3 of 3 |

The baseline is recorded once and never overwritten. On its jobs, NaN, +inf and −inf stay distinct. The check fails if any input differs from what the baseline recorded: its jobs, the 40 probe answers, the 3 observations or the 3 F.0a records. The figures above are from the baseline re-recorded on the code before the change under these rules, after the PR's review.

The check also ran once before any edit, so that a stored record which no longer reproduced would not be blamed on the change. All 55 reproduced.

The suite compares each new path, field by field and type by type, with a reference copy of the code it replaced:

- the conversion, with and without the gradient fields and phases;
- the splice, on the vendored library and on one with ties and one-offset stretches;
- the noise draws against the old draw loop;
- the controller's actions across a block's end, across two runs, and across a reset.

### Timing (`timing.json`)

Old and new code were timed back to back in one session, in High Power mode, alternating the order, and every bench record was identical between them.

- **On one core,** one simulation went from 27.4 s to 17.8 s: **1.54×**.
- **On 16 workers,** 32 simulations went from 77.5 s to 57.3 s on average: **1.35×**.
- **Both are under the change's estimate of 1.6–1.8×.** The 16-worker gain is the smaller because the machine already delivers only about 6 cores' worth of work at 16 workers, so less per-simulation work shows less there.

Combined with High Power mode (1.35×), a 16-worker generation of 16 simulations now takes about 29 s, against about 54 s at attempt 2's probe.

## What this establishes, and what it does not

**It establishes** that C.1's v2 and v1 simulations are 1.35–1.54× faster with identical output, on every path the stored records and the identity check exercise.

**It does not** speed up the simulation kernels (sensing, odour, movement, about 10 s of the 17.8 s) or the connectome controller's draws. The kernels' cost is per-step Python overhead over 200 animals, which the compute plan's whole-generation batching addresses before C.2's cost pilot.

## Artefacts

- `quantumnematode/evaluation/batched.py`, `quantumnematode/evaluation/controllers.py`, `quantumnematode/benchmark/measurement.py`.
- `scripts/analysis/speedup_identity_check.py baseline|check`.
- Tests:
  - `tests/quantumnematode_tests/evaluation/test_speedup_identity.py`;
  - the splice, draw and type checks in `tests/quantumnematode_tests/benchmark/test_measurement.py`.
- **The compute plan,** recorded in the Phase 10 tracker under W:
  - whole-generation batching before C.2's cost pilot;
  - sequential stopping of W's null draws in W's registration;
  - surrogate-assisted CMA dropped.
