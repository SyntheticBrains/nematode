## Context

From the B.2 survey of 2026-10-24:

- **No trajectory-based reorientation detector exists.**
  - `validation/behavioural_curves.py` (`kinematics`) counts turns from the controller's internal
    heading. Under primitives a reversal keeps the heading (`env/primitives.py`), so it is invisible
    there.
  - `scripts/analysis/behavioural_chemotaxis_validation.py` refuses any capture whose step is not 5 s.
  - `BehaviourStep.primitive` is documented as diagnosis only, and no grading code reads it.
- **The reference data** (B.1, `data/benchmark/`):
  - **Roberts 2016:** five wild-type cohorts, 122 worms of about 9 min each. Positions in mm at 1 s,
    the tracker's per-frame state at 29.97 frames/s (2 forward, 1 reverse, 0 paused), and the
    authors' Viterbi decoding. Roberts calls the reverse–pause–forward transitions the
    reorientations. There are no omega flags. There are four dead-worm tracks.
  - **Calhoun 2014:** 47 worms, 30 min at 3 frames/s, in pixels. One-frame pirouette and omega onsets;
    every omega coincides with a pirouette. No cohorts.
  - **Scheer 2023:** per-frame `MovingReverse` and `Omega` on food at 3 frames/s, with B.1's split.
  - **López-Cruz 2019:** reorientations per animal in 2-min bins, 0–44 min. The wild type of Fig. 1C
    (n = 413) rises to 2.74 at 4–6 min, then decays.
- **López-Cruz's methods** (PMC9161785):
  - "reorientations that start with a reversal were increased during local search (time-dependent…),
    but reorientations resulting from turning during forward movement occurred at similar frequencies
    during local and global search"; "In all figures except Fig. S1, we quantified the number of
    time-dependent reorientations only."
  - Frames at 3 fps. Behaviour "recorded for 45 min, starting four minutes after the initial food
    removal".
  - Fig. 1C's legend: "The first two minutes are not plotted as these are affected by transfer."
  - There is no exponential fit. The detection thresholds are in Fig. S1 and Pokala 2014's
    supplement, neither available.
- **The classifier at 5 s.** `validation/roaming_dwelling.py` (`reference_stride`,
  `downsample_to_reference`) and `scripts/analysis/lawn_states.py` (`episode_windows(step_seconds=…)`)
  already classify 1-s captures at 5 s. B.1's store test reproduces D.1's windows from 1-s tracks.

The maintainer's decisions of 2026-10-24:

- reversal onsets, tuned on Roberts;
- the bins are recording time;
- the fit starts at the wild type's peak bin;
- the quintile ratio's reference is a digitised Pierce-Shimomura 1999 figure.

## Goals / Non-Goals

**Goals**

- Readouts that read real and simulated tracks through the same functions, from positions alone.
- Each readout validated on real worms, by gates fixed before it is tuned.
- López-Cruz's time zero, fit window and reference γ, registered.

**Non-goals**

- Bands and grading rules: B.3.
- The classifier itself, which D.1 calibrated: this change only records its 1-s path.
- Forward-turn ("time-independent") reorientations. López-Cruz does not grade them.

## Decisions

### Decision A: The reversal-onset detector

The input is positions sampled each second. Steps are `v_t = p_{t+1} − p_t`, with lengths `ℓ_t`.

- **The minimum step** is `d_min = max(κ · median(ℓ over the track's steps above the floor), floor)`.
  The floor is 0.01 mm where the units are mm, and none in pixels, so the threshold is relative to
  the track's own movement and reads both units.
- **The recent heading** `h_t` is the vector sum of the steps of length ≥ d_min in the previous W
  seconds.
- **An onset at t:** `ℓ_t ≥ d_min`, `h_t` defined, and the angle between `v_t` and `h_t` greater than θ.
  After an onset, none is counted for R seconds.
- **Retrace**, tuned on or off: the onset's step must also bring the worm closer to where it was W
  seconds earlier, `|p_{t+1} − p_{t−W}| < |p_t − p_{t−W}|`. A reversal moves back along the worm's own
  path; after an omega or delta turn the worm runs off in a new direction, and from 1-s positions
  alone the two otherwise look alike.
- **Missing positions:** a step touching a missing position is never a candidate and leaves the
  heading history, and a retrace test reaching back to one fails. Each animal's onset rate is per
  second of valid steps.
- **The tuned parameters** are θ ∈ {90°, 105°, 120°, 135°, 150°}, W ∈ {2, 3, 5} s, κ ∈ {0.1, 0.25, 0.5},
  R ∈ {2, 3, 5} s and retrace ∈ {off, on}. 270 combinations are searched exhaustively on the
  calibration cohorts, maximising the F1 of matched onsets.

**Reference onsets**, the same rule in seconds for every source:

- a reversal bout is a run of the source's reverse label, after gaps of at most 0.34 s are merged;
- runs shorter than 0.67 s are dropped;
- its onset is the run's first frame;
- the reverse label is the tracker state 1 for Roberts and `MovingReverse` for Scheer. At 3 frames/s
  this is B.1's bout rule.

**Matching:** a detected onset at second t matches a reference onset within ±1 s, one to one, earliest
first. Recall is matched reference onsets over all reference onsets; precision is matched detections
over all detections.

### Decision B: Calibration and validation, with fixed gates

**Calibration:** the Roberts cohorts are split by `default_rng(2028).permutation` of
(C15, C21, C57, C162, C186). The first three calibrate and the last two validate. The split is
recorded, and no other data touch the tuning.

**The preflight.** Before any held-out data are read, the best in-sample F1 on the calibration cohorts
is compared with the held-out bar. If its recall or precision is below 0.70, the script stops and
reports, and the held-out cohorts stay unread. A detector that cannot reach the bar on the cohorts it
was tuned on cannot pass on new ones, and reading them would spend them for nothing.

**Validation, without retuning**, each with its gate:

| Validation | Gate |
|---|---|
| Roberts, held-out cohorts | recall ≥ 0.70 and precision ≥ 0.70 |
| Roberts, dead worms | ≤ 0.1 detected onsets per minute |
| Simulated primitives (ConstantHazards on López-Cruz's assay, seeds 7601–7632), against the primitives' reversal starts | recall ≥ 0.80 and precision ≥ 0.80 |
| Scheer, held-out wild-type cohorts, on food, against filtered `MovingReverse` onsets | recall ≥ 0.50 and precision ≥ 0.50 |
| Calhoun, against pirouette onsets, within ±2 s | recall ≥ 0.50, and a Spearman ρ ≥ 0.5 between each worm's detected and flagged rates |

**Why the bars differ:**

- **Scheer** is on food, where reversals are short and slow (B.1: 2.05 s at 0.07 mm/s).
- **Calhoun** marks pirouettes, which include more than reversals, at an unstated threshold.
- **The simulated tracks** have no noise, so their bar is the highest. The bout labels are read only
  here, to validate the detector, never to grade (D36).

**Forward turns, described.** Two false-positive rates are reported beside the gates:

- on Roberts's held-out cohorts, detections while the tracker reads forward with no reverse state
  within ±2 s;
- on the simulated tracks, detections within 2 s of the end of an omega or delta bout that no reversal
  started.

They measure how often a forward turn is taken for a reversal.

**Every gate is read once, after calibration.** A failed gate is reported with the detector's numbers.
The benchmark does not register on a failed detector: the maintainer decides between a registered
retry and a revised readout. Nothing is silently retuned.

### Decision C: The local-search curve and its fit

- **The curve:** per animal, detected onsets in 2-min bins of time since removal, counted only in bins
  where at least 90% of the animal's positions are present. The mean and SEM are taken over animals,
  per bin.
- **López-Cruz's alignment (decided):** a deposited bin labelled `[a, a + 2)` min covers `[a + 4, a + 6)`
  min after removal.
- **The fit (decided):** αe^(−γ(t − t₀)) + β by least squares on bin means, unweighted.
  - It starts at the deposited wild type's peak bin, recording 4–6 min (8–10 min after removal; t₀ is
    that bin's centre).
  - It runs to the last bin both López-Cruz and the 45-min simulated assay cover: recording 38–40 min,
    42–44 min after removal. That is 18 bins.
- **The reference:** the deposited Fig. 1C wild type, fitted the same way. Its γ, α and β are recorded
  in `data/benchmark/lopezcruz2019_reference.json`, with the window and time zero. The interval comes
  from a parametric bootstrap over the bins' mean ± SEM at its n.
  - **Described beside it:** fits from recording 2–4 min and from 6–8 min, and pdfr-1 (row 91) against
    its same-dates wild type (row 90).
- **Simulated curves** are fitted identically, with bootstrap intervals over animals.

### Decision D: Roaming and dwelling on 1-s tracks

The benchmark classifies every 1-s track, real or simulated, by downsampling to the 5-s step
(`downsample_to_reference`, `episode_windows(step_seconds=1.0)`), with the calibrated model unchanged.
B.1's store test is the positive control: 1-s Scheer tracks reproduce D.1's decoded states on 99.96%
of windows. This change adds only a pointer and a test that a simulated 1-s capture goes through the
same path.

### Decision E: The pirouette quintile ratio (D31's K.1)

- **The statistic:** each second's dC/dt as the worm sensed it, taken from the capture. Within each
  worm, seconds are binned by its own dC/dt quintiles. The ratio is the detected-onset rate in the
  lowest quintile (most negative dC/dt) over the rate in the highest, pooled over worms per quintile.
  The rates' units cancel, and so do dC/dt's.
- **The reference (decided):**
  - Pierce-Shimomura et al. 1999's figure of pirouette frequency against dC/dt, digitised from the
    published figure image. The axis calibration points, each digitised point and the method are
    recorded in `data/chemotaxis/pierce_shimomura1999_pirouettes.json`.
  - The reference ratio is the curve's rate at its most negative dC/dt bin over its most positive.
  - **The units mapping, registered:** the figure's dC/dt axis is in its own units. The simulated
    worm's is not mapped onto it; only the ranks are, the extreme bins against the extreme quintiles.
    That limit is stated.
  - Lee 2026's 0.533 → 0.056 s⁻¹ (a ratio of 9.5) is cited beside it. Its units are checked against Lee 2026 at implementation; only the ratio is used.
  - If the figure cannot be obtained, the reference is Lee 2026's ratio alone, and that is recorded.
- **Where it applies:** it is computed on any capture carrying dC/dt. Which assays grade it is B.3's
  choice.

## Risks / Trade-offs

- **The detector sees displacement, not posture.** A reversal shorter than about 1 s, or slower than
  d_min, is missed. The Scheer gate and the simulated gate measure how much that costs on food and in
  the model.
- **Roberts's states are by eye, at collection** (README; the program calls them the tracker's). They
  are a reference, not ground truth, and Calhoun and Scheer are independent checks.
- **López-Cruz's own thresholds are unavailable.** The detector cannot be matched to them, so the curve
  is compared in shape and decay rate, not in absolute count. B.3 decides whether the absolute level
  is graded.
- **Digitising a figure adds a reading error.** It is recorded per point, and the reference ratio's
  interval carries it.
