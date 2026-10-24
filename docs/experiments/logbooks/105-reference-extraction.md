# 105: Phase 10 — The Benchmark's Real Worms: Scheer at 1 s with Cohorts, the Search Tracks, and the Primitives Checked on Food

**Status**: completed. **B.1 ships, with one disagreement owed to the maintainer.** Four public datasets are vendored with their licences, units and provenance, and the Scheer store reproduces D.1's windows. The cohort split is fixed. On food, real reversal bouts are shorter than the primitives' published off-food values, by the rule registered before the data were read. Off food, the same worms match those values.

> - **Scheer & Bargmann 2023:** 2,847 animals in eight arms, at 1 s, with the authors' per-frame flags and every animal's genotype, recording date and cohort side. 32.5 MB.
> - **The extraction's positive control passes.** Re-deriving D.1's 5-s windows from the new store reproduces every mask and label. The classifier's decoded states agree on 99.96% of windows.
> - **The split:** recording dates, a third held out, drawn from counts alone. 24 of the OD 2 wild type's 65 dates are held out.
> - **The kinematics check:** inside the lawn, reversal bouts last 2.05 s [2.01, 2.08], outside the published span of 2.10–3.71 s. Off the lawn they last 3.30 s [2.99, 3.67], inside it. The form of the D36 amendment is the maintainer's choice, before B.3 registers.

**Date**: 2026-10-24.

**OpenSpec change**: `add-reference-extraction`.

**Records** (in [`supporting/105-reference-extraction/`](supporting/105-reference-extraction/)):

- `scheer_extraction.json`: per arm, the animal and date counts, the lawn-geometry check and the Fig. 1 match;
- `kinematics.json`: the check, its rule, every stratum, and the readings beside it.

The vendored data and their provenance are in `data/benchmark/` (`PROVENANCE.md`).

## Objective

The benchmark (block B, D34) is valid only if real tracks pass it:

- its positive control holds out **whole cohorts**;
- its mutant control holds out real mutant cohorts (D37).

Neither was possible with what the repository held. D.1's `scheer2023_windows.npz` kept the wild type as 5-s window measures, with no positions, cohorts or mutants. No search tracks were vendored.

E.2 also owed a check. Its primitives' kinematics are published off-food values, and they were to be checked against Scheer's on-food flags on held-out cohorts before any fit.

These were the maintainer's decisions of 2026-10-24:

- vendor 1-s tracks;
- the D37 mutants plus the food-density arms;
- recording dates as cohorts, with a third held out;
- the kinematics check on the held-out cohorts.

During extraction, the maintainer also chose to add the N2 wild type as tph-1's control (below).

## What was built

- **`scripts/analysis/extract_benchmark_references.py`**, one subcommand per source and one for the split.
- **`scripts/analysis/_scheer_pickle.py`**, the restricted unpickler, moved out of D.1's script so both use it.
- **`scripts/analysis/primitive_kinematics_check.py`.**
- **`quantumnematode.benchmark.references`**, which reads every set back as per-animal tracks with their units and cohort fields:
  - `load_scheer_tracks()`;
  - `scheer_split()`;
  - `load_search_tracks()`;
  - `load_lopezcruz()`.
- **A minimal Igor binary-wave reader** for Roberts's deposit. There is no new dependency.

## The four datasets

| Set | Deposit, licence | What is vendored |
|---|---|---|
| Scheer & Bargmann 2023 | Zenodo 8310289 (Dryad mirror), CC0 | 2,847 animals, 40 min each: midbody positions at 1 s (µm from the lawn centre), the per-frame reversal, omega, pause, head-poke, in-lawn and missing flags, the 10-s labels and run masks, 64 boundary radii, and the cohort fields |
| Calhoun et al. 2014 | Dryad 10.5061/dryad.3j2j9, CC0 | 47 worms off food, 30 min at 3 frames/s: positions at 1 s in **pixels** (no scale is published), and the per-frame pirouette and omega onsets |
| Roberts et al. 2016 | Dryad 10.5061/dryad.35qv6, CC0 | the five intact wild-type cohorts (122 worms, about 9 min each) and four dead worms: positions at 1 s in mm, and per frame at 29.97 frames/s, the tracker's forward/reverse/pause state and the authors' four-state Viterbi decoding |
| López-Cruz et al. 2019 | Mendeley 10.17632/prdtw5jgcd.1, CC BY 4.0 | all 184 rows of reorientations per 2-min bin after food removal, each mutant linked to its same-dates wild type |

**Scheer's arms:**

| Arm | Animals | Dates |
|---|---|---|
| PD1074 wild type, OD 2 | 1,586 | 65 |
| PD1074, OD 1 | 45 | 1 |
| PD1074, OD 4 | 83 | 3 |
| PD1074, uniform lawn | 100 | 3 |
| N2 wild type, OD 2 | 1,380 | 70 |
| tph-1 | 139 | 5 |
| pdfr-1 | 395 | 16 |
| tax-4 | 499 | 19 |

**What the deposits held that the design did not expect.**

- **tph-1 has no PD1074 control.** tph-1 is an N2-background strain, and it was recorded only beside the N2 wild type: all five of its dates have N2 animals, none has PD1074. The maintainer added `N2_od2_Data` as an eighth arm, kept apart from PD1074. pdfr-1 shares 11 of its 16 dates with PD1074, and tax-4 17 of 19.
- **The lawn is re-detected every background segment.** The centre and boundary are stored once per background slot, and each animal uses about five slots of 20. Each animal's centre and radii are the median over the slots its frames use.
- **The authors' `Lawn_Boundary_Dist` is the head's distance.** It agrees with their head-in-lawn flag on every frame. The store's midbody distance was therefore checked against their centroid distance. The two differ by a median of 0.022–0.025 mm by arm, and the sign agrees with their midbody-in-lawn flag on 99.0–99.8% of samples.
- **The genotype file was pickled with pandas 1.x.** Its numeric index classes, which pandas 2 removed, are read as `pandas.Index`, as pandas' own `read_pickle` reads them.
- **Calhoun's spreadsheet** has no pixel scale. Its `Non-omega turns` sheet is all zeros, its pirouette sheet is six columns too long, and 27 cells hold malformed text. Aligned at the start, all 709 omega onsets fall on pirouette flags, and at any other offset none does, so the surplus columns are dropped. The malformed cells are counted and treated as missing.
- **Roberts's units and codes** come from the authors' Igor program in the same archive. Positions are in mm at 29.97 frames/s. Tracker states are 2 forward, 1 reverse and 0 paused. Viterbi states are 0 X, 1 R, 2 F and 3 Y.
- **López-Cruz has pdfr-1 but neither tax-4 nor tph-1.**

## The extraction's positive control

The 1-s store must reproduce what D.1 vendored. All 1,586 Fig. 1 animals match `PD1074_od2_Data` by video and source file, in order, with equal positions. Re-deriving their 5-s windows from the store gives:

| Condition | Required | Result |
|---|---|---|
| on-food and off-lawn masks, labels, missing windows | identical | identical on every animal |
| speed | within the rounding's bound, 0.28 µm/s | worst 0.25 µm/s |
| the calibrated model's likelier emission state | ≥ 99.5% of on-food windows | 99.92% |
| decoded states | ≥ 99% of on-food windows | 99.96% |

The test runs in the default tier in about 2 s.

## The cohort split

A cohort is a recording date, and everything recorded on one date sits on one side. The split was drawn from each animal's date and arm alone, before any behavioural statistic was computed. It was drawn at seed 2027, by the rule registered in the design: feasibility checked first, then at most 10,000 draws, stopping the extraction if none passed. Draw k = 16 passed, and no arm was share-infeasible.

| Arm | Held-out dates | Held-out animals |
|---|---|---|
| PD1074, OD 2 | 24 of 65 | 573 (36%) |
| N2, OD 2 | 28 of 70 | 537 (39%) |
| tph-1 | 2 of 5 | 35 (25%) |
| pdfr-1 | 6 of 16 | 138 (35%) |
| tax-4 | 7 of 19 | 192 (38%) |
| PD1074, OD 4 | 1 of 3 | 18 (22%) |
| PD1074, uniform lawn | 1 of 3 | 31 (31%) |
| PD1074, OD 1 | none: one date | — |

**Held out means held out of the bands, not unseen by the instrument.** D.1 calibrated the classifier on an animal-level split across every wild-type date, and its leaving reference used every animal. B.3 decides whether to re-calibrate on band cohorts only, and B.4 reports whichever applies.

## The kinematics check

The check uses the held-out OD 2 wild type: 573 animals, 24 cohorts. It reads only flags and positions, never labels, run masks or roaming fractions. Intervals are 95% bootstraps over dates. Bouts are flag runs with one-frame gaps merged and runs shorter than two frames dropped.

| Quantity | Inside the lawn (> 0.5 mm) | Within 0.5 mm of the edge | Outside (> 0.5 mm) | Published, off food |
|---|---|---|---|---|
| reversal duration (s) | **2.05 [2.01, 2.08]** | 2.00 [1.98, 2.03] | 3.30 [2.99, 3.67] | span of means ± 2 SEM: 2.10–3.71 (Zhao 2003) |
| reversal speed (mm/s) | 0.069 [0.067, 0.071] | 0.076 [0.074, 0.078] | 0.174 [0.148, 0.205] | 0.14 (Zhao), mean only |
| omega duration (s) | 4.04 [3.66, 4.57], n = 219 | 4.92 [4.43, 5.49] | 2.36 [2.02, 2.70], n = 45 | 7 ± 3, escape omegas (Broekmans 2016) |
| forward speed, 95th percentile (mm/s) | 0.085 | 0.078 | 0.285 [0.25, 0.31] | `v_max` 0.2 |
| turning rate, 95th percentile (°/s) | 175 | 177 | 53 [47, 60] | `omega_max` 50 |
| pause fraction | 0.51 [0.48, 0.54] | 0.50 | 0.07 [0.03, 0.11] | reported only |

**The registered rule.** Only reversal duration is acted on: it is the one quantity whose sources state a mean, an SD and an n. Its inside-lawn interval lies wholly below the span, so **this is a disagreement**. The rule sends it to the maintainer; `primitives.json` is unchanged. The other rows are description.

**What the strata show.**

- **On the lawn, reversals are short and slow.**
- **Off the lawn, the same animals reverse for 3.3 s at 0.17 mm/s.** That is inside the published off-food span, and near Zhao's speed.

So the published values are right for the condition they came from. The point worm's primitives use one set of values on and off food. The two foreseen forms of the amendment are:

- **food-dependent kinematics**, keyed on the existing food-contact sense;
- **the off-food values kept**, with the on-food gap stated as a limit of the on-lawn grading.

The outside stratum is small (169 reversals, 45 omegas). It is made of excursions just past the edge, and forward speed there exceeds `v_max`. Those numbers bear on the amendment, but they are not a reading of their own.

**Missing frames.** The registered rule keeps a bout whether or not it touches a frame the tracker lost. A first run of the script dropped such bouts, against the rule's text. That removed every omega, because the omega flag always coincides with lost frames while the worm coils. Under that run, inside-lawn reversals were 2.05 s [2.01, 2.08], so the disagreement does not depend on the choice. Both readings are in `kinematics.json`.

**Two descriptive rows mean little.**

- **The turning-rate percentile on food (175°/s) is tracking noise.** A pixel is about 8.6 µm, and a dwelling worm moves a few tens of µm a second, so its 1-s heading is near random. It says nothing about `omega_max`.
- **Omega durations (2–5 s) are the span of the authors' omega flag.** That flag comes from posture tracking. They are not Broekmans's escape omegas.

## What this means for B.2–B.4

- **B.2** validates its reorientation detector on Calhoun's flags and Roberts's tracker states, with the dead worms as its noise floor. Calhoun's statistics must be unit-free.
- **B.3** bands on the `bands` cohorts. Each mutant is graded against the wild type that shares its dates: N2 for tph-1, mostly PD1074 for pdfr-1 and tax-4.
- **B.4** holds out the `held_out` cohorts. tph-1 has two held-out dates; OD 1 has none.
- **Before B.3 registers**, the maintainer chooses the form of the D36 amendment.

## Conclusions

1. **B.1 ships.** The benchmark's real worms are in the repository with cohorts, a fixed split, and a store that reproduces what the project already calibrated on.

2. **The primitives are off-food animals.** On food, real reversals are shorter (2.0 s against 2.1–3.7) and slower. Off food, the same animals match the published values. The registered rule sends this to the maintainer before any fit.

3. **The deposits needed reading, not just loading.** Missing these would have broken B silently:

   - tph-1's control was a different wild-type strain;
   - the lawn moves between segments;
   - the boundary distance was the head's;
   - Calhoun's surplus columns.

   Each is now tested or recorded.

## Next steps

- [ ] The maintainer: the form of the D36 amendment (food-dependent kinematics, or the off-food values kept with the on-food gap stated), before B.3 registers.
- [ ] B.2: the reorientation detector, validated on Calhoun and Roberts; López-Cruz's readout, time zero and γ.
- [ ] B.3: bands and grading on the `bands` cohorts; whether to re-calibrate the classifier on them alone; the NeuroSparkSNT check owed at B's registration.
- [ ] N, in parallel with B.
