## Context

B v2 (change `revise-benchmark-v2`, launch record 115) adds a measurement model so simulated and real tracks are read through the same tracker. Its registered checks read:

- **The loss model and Chen's round trip pass.**
- **The Gaussian jitter fails** against the real tracker's readings of a still worm. The independent noise gives too many false reversal onsets (3.87 against 2.14 per paused minute) and too little 10-s drift (window speed median 0.0016 against 0.0038 mm/s).

**Facts the design rests on** (the spec review's readings on library-half dates and discarded seeds; description):

- **The paused stretches** last 13 s at the median (90th percentile 22 s, at least 10 s by construction). Their offsets from their own mean have an SD of 0.0149 mm, about 3× the median 1-s step, because they carry drift.

- **The tracker's false-onset rate varies strongly by recording date.** Over the library half's dates it ranges from 1.08 to 3.26 per paused minute (SD 0.52). The library half reads 2.28 and the check half 2.01, a gap of 1.9 n = 200 band scales. The check half's figure is implied by amendment 1's committed all-date total. A gate of one band scale is therefore failed by real noise from other dates.

- **B.4's matched open-loop worm misses the kinematics at n = 200 with no noise at all** (discarded seeds, 199.6 M):

  - dwelling speed median 0.0182 (z +7.3);
  - roaming speed median 0.0437 (z +6.7);
  - reversals 3.22 per minute (z −5.9).

  Its parameters were matched on per-state mean speeds and hazards, not on the classifier's 10-s windows, so it cannot be the reachability oracle's known answer.

## Goals / Non-Goals

**Goals:**

- a lawn noise model that reproduces the real tracker's readings of a still worm, judged on dates the model never saw, against the tracker's own date-to-date variation;
- a reachability oracle whose known answer can meet its bar, shown before registration;
- no free parameter of the noise model fitted to the check.

**Non-Goals:**

- a mechanistic model of the tracker;
- separating real micro-motion of a paused worm from tracker noise. Both are in what the tracker reports for a still worm, so both are carried over, and a still simulated worm then reads like a still real worm;
- **the tracker's noise while the worm moves.** The library is the still worm's noise, applied at every second, as the Gaussian model's was. Noise during motion (for instance from posture changing the centroid) is not measured, because no clean truth exists for a moving real worm. It is stated as an assumption in the logbook;
- changing Chen's rig, whose Gaussian jitter passed its round trip;
- changing any statistic, band or threshold of B v2. Part A's reachability gate is kept, and only its worm is replaced.

## Decisions

1. **The split.** PD1074's 41 band dates are split into a library half and a check half by `default_rng(2043).permutation`. The library half gets the first 21 dates and the check half the other 20. Both halves are band cohorts, so no held-out PD1074 cohort is read.

2. **The library.**

   - **The stretches:** the library half's paused stretches, as amendment 1 defines them: runs of at least 10 s, each second with all three frames flagged `pause` and its position present. There are 6,213.
   - **What is stored:** each stretch's offsets from its own mean position (x and y), in micrometres as int32, as the Scheer store does, stored consecutively with their lengths.
   - **Where:** `data/benchmark/scheer_tracker_noise.npz`. Its `PROVENANCE.md` entry gives the source, the split and the checksum.

3. **The draw.**

   - For a simulated animal of n seconds with seed s, `rng = default_rng([s, 2033])` (the lawn stream).
   - Stretches are drawn uniformly with replacement and joined end to end, offsets as stored, until the track reaches n seconds. The last stretch is cut to fit.
   - Where two stretches meet there is a jump: the difference between one's last offset and the next one's first.
   - The noise is added to the animal's positions, and then the losses are drawn from the same stream, as now.
   - **There is one rule and no other freedom:** no rotation, scaling, reflection or continuity shift, and nothing tuned to the check. A continuity shift was considered and dropped: it makes the track wander about 0.3 mm RMS over 45 min, which nothing can check.

4. **The pilot** (on the library half and on discarded seeds; description, not a verdict):

   - **The draw in-sample:** decision 05's check, with the library half as both the library and the real side, and the model side at 157 M. If any gated reading falls outside its permutation range in-sample, the joins themselves distort the noise. The change then stops before registration and goes back to the maintainer.
   - **The gate's oracle:** decision 05's permutation ranges are reported, beside the Gaussian model's amendment-1 misses, to show the gate keeps its power.
   - **The reachability worm's fit** (decision 06), before registration.

5. **The re-registered jitter check** (115's launch record, amendment 2, committed before the check half is read directly):

   - **The real side:** the check half's paused stretches.

   - **The model side:** for check stretch k of length L, a noise track of 2,700 s (the small-lawn assay's length) is drawn at `default_rng([156_000_000 + k, 2033])`, and L seconds are cut at a uniform random point from the same stream, so stretches are joined at the rate a simulated track sees.

   - **The readings:**

     - onsets per paused minute;
     - the 10-s window speed's median and 90th percentile;
     - the 1-s step quantiles, reported only.

   - **The gate, calibrated on the tracker's own date variation:**

     - the band dates are halved 1,000 times at random by date (`default_rng(2044)`, 21 against 20), and each reading's real difference, half A minus half B, is recorded;
     - each gated reading's model-minus-real difference must lie within that distribution's central 95% (the 2.5th to 97.5th percentiles).

     Real noise from other dates passes by construction. The Gaussian model's misses (onsets +1.73 per minute, window speed median −0.0022 mm/s) lie far outside a spread of about 0.3 and 0.0002.

   - **Disclosed:** the check half's real onset rate (2.01) is implied by committed numbers. Its window speeds are not exactly.

   - **On failure:** the run stops and returns to the maintainer. No further noise model is tried in this change.

6. **The reachability worm** (it replaces B.4's matched worm in the reachability oracle; Part A's gate is unchanged):

   - **The structure:** `BatchedStateHazards`, with B.4's two states, transitions and window.
   - **The free parameters, 8:** each state's speed, curvature SD, reversal hazard and omega hazard. Each is log-uniform on [θ/4, 4θ] around B.4's matched value θ (`negatives_params.json`), and the unit box's centre is B.4's worm.
   - **The objective:** the RMS z of the five gated statistics (the four speeds and the reversal rate) at n = 200, through the lawn model with the library, for 200 fresh animals per evaluation (`158_000_000 + 200·k + i`).
   - **The fit:** sep-CMA from the centre, σ₀ 0.2, population 16, 2 restarts (CMA seeds 9411 and 9412). The plateau rule stops a restart only once σ ≤ σ₀, and the cap is 100 generations. The answer is the best restart's `xfavorite`.
   - **Its own oracle, before registration:** the answer is re-simulated on 5 discarded populations (`158_900_000 + 200·g + i`). The median RMS z must be at most 1.
   - **If it is not:** no open-loop two-state point worm reaches the statistics through the tracker. The statistics it misses are the diagnosis, and the change stops before registration and goes back to the maintainer.
   - **Registered:** the answer's parameters go into amendment 2. The 40 oracle populations (`150_000_000 + 200·p + i`) are read with it under Part A's gate: mean z in [−1, 1], and |z| ≤ z\* in at least 90% of populations, for each of the five statistics.

7. **Then B v2 continues**, writing `oracle_amendment2.json`:

   - the losses check (it applies no noise, so it is re-run and must still pass);
   - the jitter check;
   - Chen's round trip (unchanged);
   - the reachability oracle with the reachability worm;
   - then Part B, the N2 read, the negatives, the blind worm and C.1's answer, all on the library model.

## Risks / Trade-offs

- **[Joining artefacts]** → the gap between one stretch's last offset and the next one's first is about 0.02 mm, against a median 1-s step of 0.005 mm, and it comes about every 13 s. The in-sample pilot shows whether the joins distort the readings, and the check reads them at the rate a simulated track sees.
- **[Micro-motion as noise]** → a paused real worm's slight motion is carried over as noise. That is the intent: the benchmark compares what the tracker reports. It raises every simulated dwelling window's speed slightly, as it raises real ones.
- **[A two-state worm cannot reach the statistics]** → for instance, real dwelling's p90-to-median window-speed ratio (2.2) may need speed variability within a state, which this worm lacks. The fit shows it before registration. If it fails, the change stops there, and the diagnosis says which statistic no open-loop point worm reaches.
- **[A wider gate]** → the permutation gate admits model noise that sits within the tracker's own date variation. That is the comparison the bands make too: their scales are between-date.
- **[The bands file changes]** → `bands_v2.json` is rebuilt to name the library in its lawn model. The bands, the rule checks and the leave-one-out are deterministic and unchanged, and their record must come back byte-identical.
- **[Cost]** → the reachability fit takes about 4 s per evaluation, 16 per generation on 16 workers: about 5 s per generation, at most about 20 min for both restarts at the cap.
