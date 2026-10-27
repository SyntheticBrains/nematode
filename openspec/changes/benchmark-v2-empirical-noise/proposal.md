## Why

B v2's measurement model gives every simulated 1-s track independent Gaussian position noise. Its registered checks stopped the run (launch record 115, amendment 1, `oracle_amendment1.json`). The loss model and Chen's round trip pass, but the jitter check fails. On the authors' 13,227 paused stretches (199,047 s of band-side PD1074), the real tracker reads a still worm differently from the model:

- **false reversal onsets:** 2.14 per paused minute, against the model's 3.87 (z +12.1 on the reversal band at n = 200);
- **10-s window speed:** median 0.0038 mm/s against 0.0016 (z −3.1), and 90th percentile 0.0072 against 0.0026 (z −4.0);
- **1-s steps:** smaller at the median (0.0054 against 0.0078 mm) and larger at the 90th percentile (0.0158 against 0.0142).

The real noise is correlated from second to second, heavy-tailed, and drifts over 10 s; some of the drift may be real micro-motion of a paused worm. A parametric Gaussian cannot reproduce that, and two of the kinematics statistics (the on-lawn reversal rate and the dwelling speeds) depend on it.

By the registered rule, the noise model is revised in a new change before B v2 continues. The maintainer chose (2026-10-27) **an empirical noise model**: real noise, resampled.

## What Changes

- **A noise library** of the real tracker's readings of a still worm: every paused stretch's offsets from its own mean, from half of PD1074's band dates (the library half, split by date and registered). It is vendored as `data/benchmark/scheer_tracker_noise.npz`.
- **The lawn tracker model** draws its noise from the library instead of a Gaussian. For each simulated animal, whole stretches are drawn from its own stream and joined end to end into a noise track as long as the animal's. The joining rule is chosen by a pilot on the library half. Losses are unchanged, and Chen's rig keeps its Gaussian jitter, which passed its round trip.
- **The jitter check is re-registered and read on the other half** of the band dates (the check half), which the library never saw. The model's counterpart of each real still stretch is cut at a random point from a full-length noise track, so blocks are joined at the rate a simulated track sees. Its gate is amendment 1's.
- **B v2 then continues as registered:** reachability, Part B, the N2 read, the negatives and the adversaries.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `benchmark-grading`: the lawn rig's position noise is empirical. This adds a requirement beside B v2's measurement model.

## Impact

- `packages/quantum-nematode/quantumnematode/benchmark/measurement.py`: a noise library and its draw. `TrackerModel` gains an optional library, and the Gaussian stays for Chen's rig and for unit tests.
- `scripts/analysis/benchmark_v2_controls.py`: a `noise` stage builds the library and runs the pilot, and the `oracle` stage reads the re-registered jitter check.
- `scripts/analysis/benchmark_bands.py`: the lawn model's entry in `bands_v2.json` names the library and its checksum. The bands themselves are unchanged.
- `data/benchmark/scheer_tracker_noise.npz` with its `PROVENANCE.md` entry.
- The 115 launch record gains amendment 2, committed before the check half is read.
- **Cost:** about a day of code. The library, the pilot and the check take minutes.
