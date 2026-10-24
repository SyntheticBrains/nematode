## Why

The benchmark (D34) grades statistics read from trajectories, never from primitive labels (D36). B.1
vendored the real tracks. B.2 builds the readouts that turn a track, real or simulated, into the
statistics B.3 bands, and validates each on real worms before anything is graded:

- **A reorientation detector.** None exists: the klinokinesis harness reads the controller's internal
  heading, which a primitive reversal does not change, and refuses 1-s captures. López-Cruz 2019
  counts only reorientations that begin with a reversal ("time-dependent"), so the detector must find
  reversal onsets from positions alone.
- **The roaming/dwelling classifier on 1-s tracks downsampled to 5 s.** It already works: B.1's store
  reproduces D.1's windows. B.2 records it as the benchmark's path.
- **The local-search curve:** reversal-initiated reorientations per 2-min bin after removal from food,
  and its decay fitted as αe^(−γt)+β. López-Cruz deposits the bins but no fit, and its time zero and
  fit start were owed to this item (the tracker's 2026-10-24 addition).
- **The repaired pirouette quintile ratio** (D31, the old K.1): the reorientation rate across the
  worm's own dC/dt quintiles, a unit-free statistic. The reference curve is digitised from
  Pierce-Shimomura 1999 and committed beside it.

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **`quantumnematode/benchmark/readouts.py`:**

  - **`reversal_onsets(x, y)`:** reversal-initiated reorientations from positions at 1 s. A step turns
    back against the track's recent heading by more than a threshold angle and travels at least a
    minimum distance. That distance is relative to the track's own median step, with an absolute floor
    where units are mm, so it reads pixels and millimetres alike.
  - **`local_search_curve(onsets, bin_s=120)`** and **`fit_local_search(curve, start_bin)`.**
  - **`quintile_ratio(onsets, dcdt)`:** the reorientation rate in the worm's lowest dC/dt quintile over
    its highest.

- **The detector tuned on Roberts (decided).** Its thresholds are chosen on three of Roberts's five
  wild-type cohorts against the tracker's reversal onsets, picked by a seeded rule fixed here. It is
  validated without retuning:

  - on the other two cohorts;
  - on Calhoun's pirouette onsets, for rate and timing only (Calhoun is in pixels);
  - on Scheer's held-out reversal flags;
  - on Roberts's dead worms, as a noise floor;
  - on simulated primitives-driven tracks, where the bout labels serve only to validate it.

  Each validation has a gate fixed here, before the detector is tuned.

- **López-Cruz's time zero (decided).** The deposited bins are recording time, which started 4 minutes
  after removal, so time since removal is the bin plus 4 minutes. The simulated assay, timed from
  removal, is compared on that alignment.

- **The fit's start (decided).** The fit starts at the wild type's peak bin, recording 4–6 min, which
  is 8–10 min after removal. It is read from the deposited wild type now and applied identically to
  every curve. Starts at the first plotted bin and one bin later are reported beside it. γ from the
  deposited wild type is recorded as the reference.

- **The quintile ratio's reference (decided):** a digitisation of Pierce-Shimomura 1999's pirouette rate
  against dC/dt, committed with its method and its units mapping. Lee 2026's 0.533 → 0.056 s⁻¹ is
  cited beside it.

## Capabilities

### New Capabilities

- `benchmark-readouts`: the trajectory readouts, their real-worm validation and the López-Cruz
  conventions.

## Impact

- **Code:**
  - `quantumnematode/benchmark/readouts.py`;
  - `scripts/analysis/reorientation_detector_calibration.py`;
  - `scripts/analysis/local_search_reference.py`.
- **Data:**
  - `data/benchmark/lopezcruz2019_reference.json`: the reference γ, the fit window and time zero;
  - `data/chemotaxis/pierce_shimomura1999_pirouettes.json`: the digitisation.
- **Default behaviour:** none.
- **Cost:** about 4–6 engineering days. No campaign.
- **Seeds:**
  - the detector's simulated validation uses a new band, 7601–7632;
  - the Roberts cohort split's seed belongs to its record.
- **Downstream:** B.3 bands and grades these statistics; B.4's controls read them on real and
  simulated tracks.
