## 1. The library and the draw

- [ ] 1.1 `noise` stage: split PD1074's band dates (seed 2043, 21/20), collect the library half's paused stretches, and write `data/benchmark/scheer_tracker_noise.npz` (offsets in µm, lengths) with its `PROVENANCE.md` entry.
- [ ] 1.2 `measurement.py`: a `NoiseLibrary` (load, `draw(n, rng)`, end-to-end joins), and `TrackerModel` with an optional library; the Gaussian path unchanged.
- [ ] 1.3 Tests: a drawn track's length and per-animal stream; the joins (offsets as stored, the last stretch cut); the library and check halves disjoint; Gaussian models unchanged.

## 2. The pilot and the registration

- [ ] 2.1 The draw in-sample on the library half (model side 157 M); stop if any gated reading leaves its permutation range.
- [ ] 2.2 The permutation gate: 1,000 date halvings (seed 2044), the central 95% of each reading's real-half difference; reported beside the Gaussian model's misses.
- [ ] 2.3 The reachability worm: sep-CMA over the 8 per-state parameters (log box [θ/4, 4θ] around B.4's), 2 restarts (CMA 9411–9412), animals 158 M, the σ guard and a cap of 100; re-simulated on 5 discarded populations (158.9 M); stop if the median RMS z exceeds 1.
- [ ] 2.4 Tests: the permutation gate on toy readings; the reachability worm's parameter map and objective.
- [ ] 2.5 `bands_v2.json`'s lawn model names the library and its checksum (rebuild; the bands and their record unchanged).
- [ ] 2.6 115's launch record, amendment 2: the split, the library, the draw, the re-registered check and its gate, the reachability worm's parameters, the seeds (156–158 M, 2043, 2044, 9411–9412); the pilot and the review's readings as description. Committed before the check half is read directly.

## 3. The check and B v2

- [ ] 3.1 `oracle` as amended: the losses, the jitter check on the check half, Chen's round trip, then reachability with the fitted worm, into `oracle_amendment2.json`.
- [ ] 3.2 On a pass, B v2 continues from its Part B (`revise-benchmark-v2` tasks 3.2 onward), on the library model.

## 4. Close-out

- [ ] 4.1 Full suite, pre-commit on all files, strict validation; archive with B v2.
