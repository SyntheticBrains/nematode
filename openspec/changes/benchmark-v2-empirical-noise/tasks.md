## 1. The library and the draw

- [x] 1.1 `noise` stage: split PD1074's band dates (seed 2043, 21/20), collect the library half's paused stretches, and write `data/benchmark/scheer_tracker_noise.npz` (offsets in µm, lengths) with its `PROVENANCE.md` entry.
- [x] 1.2 `measurement.py`: a `NoiseLibrary` (load, `draw(n, rng)`, splicing at the nearest offset, as revised), and `TrackerModel` with an optional library; the Gaussian path unchanged.
- [x] 1.3 Tests: a drawn track's length and per-animal stream; the splice (the offset after the nearest match); the library and check halves disjoint; Gaussian models unchanged.

## 2. The pilot and the registration

- [x] 2.1 The draw in-sample on the library half (model side 157 M); stop if any gated reading leaves its permutation range. (The first pilot stopped the end-to-end joins.)
- [x] 2.1a The revision: splicing at the nearest offset, and the on-lawn reversal floor at 0.03 mm by its rule; the kinematics bands, κ and rule check rebuilt (Part A's record kept as `bands_v2_record_partA.json`).
- [x] 2.1b The second pilot: the floor rule recomputed (0.03 mm), the gate, the draw in-sample (passes), then the reachability fit, which stops the change: both restarts plateau at an RMS z of about 2.6, and the answer's median on 5 fresh populations is 2.53 (reversals −4.25, roaming p90 −3.57, dwelling median +2.58).
- [ ] 2.2 The permutation gate: 1,000 date halvings (seed 2044), the central 95% of each reading's real-half difference; reported beside the Gaussian model's misses.
- [x] 2.2a `BatchedStateHazards` gains an optional per-state speed spread (a log SD, drawn per window), off by default.
- [x] 2.3 The reachability worm (stopped again: with the spread every speed is reached, |z| \<= 1.3, but reversals read 1.31 against 2.28 per minute, z -4.17, median RMS z 2.00): sep-CMA over the 10 per-state parameters (log box [θ/4, 4θ] around B.4's, spreads on [0, 1.5]), 2 restarts (CMA 9411–9412), animals 158 M, the σ guard and a cap of 100; re-simulated on 5 discarded populations (158.9 M); stop if the median RMS z exceeds 1.
- [x] 2.3a The reversal coupling (11th parameter, β on [0, 3]); refit; if the median RMS z exceeds 1, drop the on-lawn reversal rate (z\* for 19) and register reachability on the four speeds.
- [x] 2.3b The fallback applied (median RMS z 1.59): the reversal rate reported, 19 statistics; the onset reading reported; the lawn model on Scheer's families only.
- [x] 2.4 Tests: the permutation gate on toy readings; the reachability worm's parameter map and objective.
- [x] 2.5 `bands_v2.json`'s lawn model names the library and its checksum (rebuild; the bands and their record unchanged).
- [x] 2.6 115's launch record, amendment 2: the split, the library, the draw, the re-registered check and its gate, the reachability worm's parameters, the seeds (156–158 M, 2043, 2044, 9411–9412); the pilot and the review's readings as description. Committed before the check half is read directly.

## 3. The check and B v2

- [ ] 3.1 `oracle` as amended: the losses, the jitter check on the check half, Chen's round trip, then reachability with the fitted worm, into `oracle_amendment2.json`.
- [ ] 3.2 On a pass, B v2 continues from its Part B (`revise-benchmark-v2` tasks 3.2 onward), on the library model.

## 4. Close-out

- [ ] 4.1 Full suite, pre-commit on all files, strict validation; archive with B v2.
