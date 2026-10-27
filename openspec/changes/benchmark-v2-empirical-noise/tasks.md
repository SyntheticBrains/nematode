## 1. The library and the draw

- [ ] 1.1 `noise` stage: split PD1074's band dates (seed 2043, 21/20), collect the library half's paused stretches, and write `data/benchmark/scheer_tracker_noise.npz` (offsets in µm, lengths) with its `PROVENANCE.md` entry.
- [ ] 1.2 `measurement.py`: a `NoiseLibrary` (load, `draw(n, rng, rule)`), and `TrackerModel` with an optional library; the Gaussian path unchanged.
- [ ] 1.3 Tests: a drawn track's length and per-animal stream; both joining rules (the plain rule's jumps, the continued rule's continuity); the library and check halves disjoint; Gaussian models unchanged.

## 2. The pilot and the registration

- [ ] 2.1 The pilot on the library half: both joining rules through the jitter check, the matched worm's shift on one discarded population (199 M); choose the rule, or stop.
- [ ] 2.2 `bands_v2.json`'s lawn model names the library, its checksum and the chosen rule (rebuild; the bands are unchanged).
- [ ] 2.3 115's launch record, amendment 2: the split, the library, the rule, the re-registered check and its seeds; the pilot as description. Committed before the check half is read.

## 3. The check and B v2

- [ ] 3.1 `oracle` as amended: the losses, the jitter check on the check half, Chen's round trip, then reachability, into `oracle_amendment2.json`.
- [ ] 3.2 On a pass, B v2 continues from its Part B (`revise-benchmark-v2` tasks 3.2 onward), on the library model.

## 4. Close-out

- [ ] 4.1 Full suite, pre-commit on all files, strict validation; archive with B v2.
