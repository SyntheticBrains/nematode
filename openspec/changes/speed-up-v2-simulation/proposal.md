## Why

Every remaining Phase 10 fit runs v2 simulations: C.1b, F.0b-ii, C.2 and W. One v2 simulation of C.1's model takes about 26 s on one core. A profile (2026-10-31) puts more than half of that outside the simulation itself:

- **converting the evaluator's arrays into per-step records** (`BatchedTrajectories.to_behaviour_steps`): 5.1 s;
- **the measurement model** (`benchmark.measurement.measure`, with its noise splices): 6.3 s;
- **drawing C.1's uniforms one animal at a time** (`controllers._uniforms`, 3.4 M `torch.rand` calls): 3.2 s.

The simulation kernels (sensing, odour, movement) take about 10 s, and their cost is per-step Python overhead over 200 animals, which batching a whole generation per call addresses later (the compute plan below).

The maintainer's decisions of 2026-10-31:

- speed up all three parts;
- every graded statistic and D stays byte-identical, so every registered result, floor and stored record stays valid;
- High Power mode is already on (1.35×); the compute plan's other levers are recorded in the tracker.

## What Changes

- **The record conversion** reads each array column once as Python floats and builds each step positionally. The signed speed keeps `math.cos` and `math.sin` on the same floats, so every field is the same value.
- **The measurement model** builds measured steps from columns rather than field by field. Each noise splice keeps its random-number draws and its first-minimum tie rule, with only its per-call overhead removed.
- **C.1's uniforms** are drawn in blocks per animal and consumed in order. One `torch.rand(k)` call yields the same numbers, and leaves the generator in the same state, as k one-at-a-time draws (checked on torch 2.14.1), and a test pins it.
- **Byte-identity checks:**
  - tests comparing each new path with the old one on fixed inputs;
  - a one-off check that re-simulates stored records (attempt 2's discarded probe simulations and the throughput bench's 32 records) and compares every statistic and D exactly.
- **The compute plan** goes into the Phase 10 tracker: whole-generation batching before C.2's cost pilot; sequential stopping of W's null draws in W's design; surrogate-assisted CMA dropped.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `batched-evaluation`: the record conversion and C.1's uniform draws are faster, with identical output.
- `benchmark-grading`: the measurement model is faster, with identical output.

## Impact

- `quantumnematode/evaluation/batched.py` (`to_behaviour_steps`), `quantumnematode/evaluation/controllers.py` (`BatchedMechanistic`'s draws), `quantumnematode/benchmark/measurement.py` (`measure`, `NoiseLibrary.draw` and `_splice`).
- Tests beside each, and `scripts/analysis/speedup_identity_check.py` for the one-off check.
- **No registered result changes.** Expected about 26 s → 14–16 s per simulation on one core (1.6–1.8×), re-timed with the same 32-simulation bench.
