# 107: Phase 10 — The Benchmark's Readouts: A Reversal Detector for Off-Food Tracks, López-Cruz's Decay, and the Classifier at 1 s

**Status**: completed. **B.2 ships, with the reversal detector scoped to off-food tracks.**

- **The reversal-onset detector** reads 1-s positions alone. It stopped at its registered preflight, was retried at bars set from the instrument (the maintainer's decision, registered before any held-out data were read), and then passed every off-food gate.
- **It failed on food,** against Scheer's reversal flags, so the maintainer scoped it to off-food readouts.
- **López-Cruz's local-search decay** is γ = 0.114 per minute [0.101, 0.127], on a registered window and time zero.
- **The roaming/dwelling classifier** reads 1-s captures through its 5-s path unchanged.
- **The quintile ratio's reference**, digitised from Pierce-Shimomura 1999's Figure 8, is a ratio of 2.26–2.58.

> | Gate | Result | Bar |
> |---|---|---|
> | Roberts, held-out cohorts | recall 0.658, precision 0.716 | ≥ 0.55 / ≥ 0.60 (retried; first registered 0.70) |
> | Roberts, dead worms | 0 onsets in 4.2 min | ≤ 0.1 per min |
> | Simulated primitives | recall 0.871, precision 0.920 | ≥ 0.80 / ≥ 0.80 |
> | Calhoun | recall 0.532, rate ρ 0.665 | ≥ 0.50 / ≥ 0.5 |
> | **Scheer held-out, on food** | **recall 0.426, precision 0.455** | ≥ 0.50 / ≥ 0.50 (**fails**) |

**Date**: 2026-10-24.

**OpenSpec change**: `add-benchmark-readouts`.

**Records**:

- [`supporting/107-benchmark-readouts/detector_attempt1_preflight.json`](supporting/107-benchmark-readouts/detector_attempt1_preflight.json): the first run, stopped at its preflight;
- [`supporting/107-benchmark-readouts/detector.json`](supporting/107-benchmark-readouts/detector.json): the retry, every gate;
- `data/benchmark/lopezcruz2019_reference.json`: the local-search reference.

## Objective

The benchmark (D34) grades statistics read from trajectories, never from primitive labels (D36). B.2 was to build those readouts and validate each on real worms before anything is graded:

- a reorientation detector;
- the classifier on 1-s tracks;
- the local-search curve, with López-Cruz's time zero, fit window and γ;
- the repaired pirouette quintile ratio (D31's K.1).

These were the maintainer's decisions of 2026-10-24:

- detect reversal onsets, tuned on Roberts;
- López-Cruz's bins are recording time;
- the fit starts at the wild type's peak bin;
- the quintile ratio's reference is a digitised Pierce-Shimomura 1999 figure.

## What was built

**`quantumnematode.benchmark.readouts`**, reading real and simulated tracks through the same functions:

- **`reversal_onsets`**, on positions sampled each second. A step turns back against the track's recent heading by more than θ and travels at least a minimum distance. That distance is relative to the track's own median step, with a 0.01 mm floor where the units are mm, so pixels and mm read alike.
  - **A reversal is a bout:** it lasts while the worm keeps moving against its old heading, and the refractory runs from its end.
  - **An optional retrace test** requires the step to point back toward where the worm was W seconds earlier, within 45°.
  - **A step touching a missing position** is never a candidate.
- **`reference_onsets`:** one bout rule for every source's reverse label. Gaps of up to 1/3 s are merged, and runs of at least 2/3 s are kept: B.1's two frames at 3 frames/s.
- **`match_onsets`,** **`local_search_curve`** (onsets per 2-min bin, counting a bin only where the animal is present 90% of it), **`fit_local_search`** (αe^(−γ(t − t₀)) + β, with no SciPy dependency) and **`quintile_ratio`.**

**Scripts:**

- `scripts/analysis/reorientation_detector_calibration.py`;
- `scripts/analysis/local_search_reference.py`.

**Two amendments in implementation**, both found by the unit tests before any data were read:

1. **The first retrace form passed sharp forward turns.** That form only required the step to shorten the distance to the window's start, and a 150° turn running off also does that. The retrace test now requires the step to point back toward that start.
2. **A reversal's end counted as a second onset,** because resuming the old heading also turns back against the reversal's heading. Reversals are now bouts, as described above.

## The detector

### The first run stopped at its preflight

The Roberts cohorts were split by `default_rng(2028)`: C15, C162 and C57 calibrate; C186 and C21 validate. The best of 270 settings on the calibration cohorts reached recall 0.62 and precision 0.68, below the registered 0.70. So the script stopped before reading any held-out data, as the preflight was registered to do.

**Why** (measured on the calibration cohorts only):

- **Roberts's reversals are short.** The median lasts 1.7 s, and 30% last under 1 s.
- **Recall rises with duration but stays low:** 0.56 for bouts under 1 s, 0.59 for 1–2 s, 0.63 for 2–4 s, 0.76 over 4 s.
- **The misses are real misses.** Only 15% have a detection within 2 s.
- **Detections far from reversals:** 32% sit more than 1 s from any reversal, and 27% more than 2 s.

This is the limit of 1-s sampling against a 30-Hz tracker, not a tuning failure.

### The retry

The maintainer set the Roberts held-out bars from the instrument, recall ≥ 0.55 and precision ≥ 0.60. The amendment was committed before the retry ran (`e47abbee`). Every other gate stayed as registered.

The chosen detector is θ = 135°, W = 3 s, κ = 0.25 and R = 5 s, with retrace **off**: the retrace test did not raise F1 on Roberts.

| Gate | Recall | Precision | Other | Passes |
|---|---|---|---|---|
| Roberts held-out (C186, C21) | 0.658 | 0.716 | 1,482 reference onsets | yes |
| Roberts dead worms | — | — | 0 onsets in 4.2 min | yes |
| Simulated constant hazards on López-Cruz's assay (seeds 7601–7632) | 0.871 | 0.920 | 2,820 reversal starts | yes |
| Scheer held-out wild type, on food (573 animals) | 0.426 | 0.455 | 95,225 reference onsets | **no** |
| Calhoun, within 2 s, after the first 2 min | 0.532 | 0.900 | rate ρ 0.665 | yes |

**Forward turns, described:**

- On Roberts's held-out cohorts, 8% of detections fall while the tracker reads forward with no reverse within 2 s (0.27 per worm-minute).
- On the simulated tracks, 6.6% of the omega and delta bouts that no reversal started drew a detection.

**Scheer on food.** Scheer's `MovingReverse` flag gives about 4.2 reversal onsets per animal-minute on food, and the detector finds about 3.9. Under half of them coincide within 1 s. On a lawn, reversals are short and slow (B.1: 2.05 s at 0.07 mm/s), and the flag is a speed-sign threshold at 3 frames/s. Calhoun's gate is the weakest of the passing ones: its pirouettes have no published definition.

### Scoped to off-food readouts

This was the maintainer's decision. The detector is registered for the local-search curve and for the quintile ratio on off-food captures. **B.3 grades on-lawn behaviour through the roaming/dwelling classifier and lawn statistics, never through on-food reversal counts.**

## López-Cruz's local-search reference

- **Time zero:** food removal. The deposited bins are recording time, which began 4 min after removal, so bin \[a, a + 2) min is \[a + 4, a + 6) min after removal.
- **The window:** from the wild type's peak bin, recording 4–6 min, to recording 38–40 min. That is 8–44 min after removal, 18 bins, the last bin a 45-min simulated assay timed from removal also covers.

| Curve | γ (per min) | α | β |
|---|---|---|---|
| **Fig. 1C wild type (n = 413), the reference** | **0.114 [0.101, 0.127]** | 2.54 | 0.40 |
| the same, starting one bin earlier | 0.087 | | |
| the same, starting one bin later | 0.138 | | |
| pdfr-1 (row 91) | 0.143 | | |
| its same-dates wild type (row 90) | 0.184 | | |

The interval comes from a parametric bootstrap over the bins' mean ± SEM. The reference matches the value the tracker recorded on 2026-10-24 (0.115 ± 0.011 from the 4–6 min bin). López-Cruz's detection thresholds are not available, so simulated curves are compared in decay, not in absolute level.

## The classifier at 1 s

A simulated 1-s capture classified at its 1-s step gives the same states as the same capture taken at every fifth step at 5 s (Flavell's uniform lawn, where the worm is always on food). B.1's store test already showed real 1-s tracks reproduce D.1's states on 99.96% of windows. The benchmark classifies every 1-s track through that path.

## The quintile ratio

**The statistic** (`quintile_ratio`) is the onset rate in the worm's lowest dC/dt quintile over its highest. Any monotone rescaling of dC/dt leaves it unchanged.

**The reference** is Pierce-Shimomura et al. 1999's Figure 8: pirouette initiation rate against dC/dt 4 s earlier, in biotin (A) and NH4Cl (B) gradients.

- **The source.** The maintainer downloaded the PDF by browser; it is the journal's copyright and is not vendored.
- **The digitisation.** Page 8 was rendered at 500 dpi and the axes calibrated from the tick marks. The authors' smoothed black curves were then read stroke by stroke, setting aside their dashed spontaneous-rate line and their point-count histogram (`scripts/analysis/pierce_shimomura_digitise.py`). Reading error is about ±0.0007 s⁻¹.
- **The values** are in `data/chemotaxis/pierce_shimomura1999_pirouettes.json`.
- **A check on the calibration.** The dashed line reads 0.029–0.0297 s⁻¹, and the paper's text gives the spontaneous rate as 0.0296 ± 0.0021 s⁻¹.

| Curve | Rate at most negative dC/dt | Rate at most positive dC/dt | Ratio |
|---|---|---|---|
| biotin (A) | 0.0505 at −0.18 mM/s | 0.0196 at +0.18 | **2.58** |
| NH4Cl (B) | 0.0378 at −0.11 | 0.0167 at +0.12 | **2.26** |

**The reference ratio is 2.26–2.58**: the figure's extreme bins stand for the worm's extreme quintiles, ranks only. Lee 2026's 0.533 → 0.056 (9.5) is about four times higher. Its units and measure were not checked, so it is cited beside the reference, not used as it.

## Conclusions

1. **B.2 ships.** The readouts read real and simulated tracks alike, and each was validated by gates fixed before it was tuned.
2. **From 1-s positions, reversals are found off food but not on it.** The registered preflight caught the limit before any held-out data were spent. The retry, at bars set from the instrument, passes every off-food gate. On a lawn the detector misses most reversals, so the benchmark does not grade on-lawn reversal counts.
3. **The local-search reference is fixed:** γ = 0.114 per minute on 8–44 min after removal, with time zero at removal.
4. **The pirouette ratio's reference is 2.26–2.58,** read from the paper the old klinokinesis band cited, not the 9.5 a secondary source gives.

## Next steps

- [ ] B.3: bands and grading rules from the `bands` cohorts, on-lawn through the classifier and lawn statistics; the D36 amendment's form; the NeuroSparkSNT check.
