# 108: Phase 10 — The D36 Amendment: Reversals on a Lawn Are Short and Slow

**Status**: completed. **The primitives' reversals now depend on food.** A reversal that starts on a lawn draws the on-food kinematics measured on real worms. One off every lawn keeps the published off-food values.

> | Reversals | Duration (s) | Speed (mm/s) | Distance (mm) |
> |---|---|---|---|
> | on a lawn (measured, Scheer 2023, held-out cohorts) | gamma, 2.047 ± 0.899 | 0.069 | about 0.14 |
> | off every lawn (published, Zhao 2003) | gamma, 2.69 ± 1.75 | 0.14 | about 0.38 |
>
> - `continuous.reversal_kinematics: off_food` reproduces every primitives run made before the amendment.
> - On Flavell's uniform lawn a slow constant-hazard worm (0.02 mm/s) reads **23.3% roaming with off-food reversals and 0.0% with on-food ones**. Fast controllers move by at most about four points.

**Date**: 2026-10-24.

**OpenSpec change**: `amend-food-dependent-kinematics`.

**Records** (in [`supporting/108-food-dependent-kinematics/`](supporting/108-food-dependent-kinematics/)):

- `pilot.json`: the pilot below;
- [Logbook 105](105-reference-extraction.md)'s `kinematics.json`, now with each stratum's SD.

## Objective

D36 fixes the primitives' kinematics from data. E.2 took them from published sources, all off food. [Logbook 105](105-reference-extraction.md) checked them against Scheer & Bargmann 2023's on-food flags, on the OD 2 wild type's held-out cohorts, and found a disagreement: on the lawn, reversals last 2.05 s [2.01, 2.08], below the published span of 2.10–3.71 s.

The maintainer's decision of 2026-10-24: **food-dependent kinematics**, as its own change before B.4. B.4's negatives run on the primitives, so the primitives must be final before any negative is read.

## What changed

- **`data/locomotion/primitives.json`** gains `on_food_reversal_duration_s` (gamma, 2.047 s, SD 0.899 s) and `on_food_reversal_speed_mm_per_s` (0.069).
  - Both are under a new status, `measured`: computed from deposited tracks.
  - Each names its script (`scripts/analysis/primitive_kinematics_check.py`) and record (105's `kinematics.json`).
  - The loader refuses a `measured` entry unless both exist.
- **The check records SDs.** `primitive_kinematics_check.py` now writes each mean quantity's pooled SD beside its value. Re-run, every earlier value in `kinematics.json` is unchanged.
- **The primitives.**
  - A reversal takes the on-food values when the worm is on a lawn at the start of the step it begins in. Its kinematics are fixed for the whole bout.
  - "On a lawn" is the lawn's geometry, the same test the capture and the classifier use. A depleted cell still counts, and a phase without food has no lawn.
  - Omega and delta bouts are unchanged.
- **`continuous.reversal_kinematics`**, `food_dependent` or `off_food`:
  - Unset means food-dependent under primitives.
  - Setting it with free locomotion is refused.
  - The batched evaluator reads it, and its parity suite passes on the new default.

**Why the default changed.** D36 says every arm and every negative runs on the same primitives. If the generated configs had to opt in, one config could drop the amendment. Free locomotion, the default outside Phase 10, is untouched. The earlier records stay reproducible under `off_food`.

## The fit

The on-lawn distribution is right-skewed, with most mass at 1.3–2.3 s. Against a gamma of the same mean and SD, discretised to thirds of a second like the flag:

| | 10th | 25th | 50th | 75th | 90th | 99th percentile |
|---|---|---|---|---|---|---|
| Scheer, on the lawn (s) | 1.33 | 1.33 | 1.67 | 2.33 | 3.33 | 5.33 |
| gamma(2.047, 0.899) (s) | 1.33 | 1.67 | 2.00 | 2.67 | 3.33 | 5.00 |

The gamma is a third of a second long in the body and close in the tail. At the 1-s step, bouts are whole steps anyway.

**The speed leans long.** It is path length over a bout's whole seconds, so the 7,258 of 24,890 bouts that span no whole second do not enter it. Tracking noise of a few µm per sample also lengthens a path at 0.07 mm/s. Neither changes the speed's order of magnitude: a reversal on a lawn still moves about a third as far as an off-food one.

## The pilot

Seeds 9001–9064, disjoint from every registered band. Each controller ran on Flavell's uniform lawn and Scheer's small lawn under both settings. Each animal's on-lawn windows were classified with the calibrated model, over the protocol's analysis window. The pilot is description.

| Controller | Flavell, off food → on food (roaming) | Scheer, off food → on food (roaming) |
|---|---|---|
| constant, 0.01 mm/s, reversal 0.03/s | 0.006 → 0.000 | 0.001 → 0.000 |
| constant, 0.02 mm/s, reversal 0.05/s | **0.233 → 0.000** | 0.084 → 0.000 |
| constant, 0.04 mm/s, reversal 0.03/s | 0.984 → 0.967 | 0.908 → 0.871 |
| constant, 0.15 mm/s (the parity suite's) | 0.988 → 0.989 | 0.941 → 0.955 |
| pirouette, 0.15 mm/s, β = 3 | 0.981 → 0.980 | 0.991 → 0.993 |

The off-food reversals carried a slow worm far enough to be read as roaming. That bias would have entered every fit of a slow, dwelling-like controller. Fast controllers are roaming already, and barely move. Scheer's small-lawn readings rest on fewer animals with on-lawn windows (24–61 of 64), since these non-learning controllers rarely stay on a 3 mm lawn.

The pilot's first run, before the SD was measured, drew on-food durations with the off-food CV (SD 1.33 s). Its 0.02 mm/s worm also read 23.3% against 0%. No on-food reading moved by more than three points between the two runs, and none changed direction.

## What stays under `off_food`

These records were made before the amendment. Each reproduces under `reversal_kinematics: off_food`:

- [Logbook 104](104-batched-evaluator.md)'s cost pilot and statistic-level calibration;
- [Logbook 107](107-benchmark-readouts.md)'s simulated detector validation (seeds 7601–7632). It reads only López-Cruz's off-food phase, where the kinematics are unchanged, so it is not re-run;
- E.2's positive control. It reads the first step on which the worm reaches the lawn, so it cannot move.

**A seed's later draws shift.** A gamma draw's rejection sampler uses a shape-dependent share of the generator's stream. So once a reversal starts on a lawn, the same seed's later draws differ from an `off_food` run's. The two runners still make the same calls in the same order, so their parity holds.

## Conclusions

1. **The primitives' reversals now match real worms' on and off food.** The values come from cohorts held out of the benchmark's bands.
2. **The amendment matters where it was feared to.** A slow controller was read as roaming because its reversals carried it too far: 23.3% against 0%.
3. **Every earlier record is reproducible** with one setting.

## Next steps

- [ ] B.3: bands and the grading rule (`add-benchmark-grading`). Its preflight runs on the amended primitives.
