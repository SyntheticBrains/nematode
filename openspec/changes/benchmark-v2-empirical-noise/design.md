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

3. **The draw** (revised 2026-10-28, after the first pilot; see "The first pilot" below).

   - For a simulated animal of n seconds with seed s, `rng = default_rng([s, 2033])` (the lawn stream).
   - The track starts at a random stretch's first offset and runs along it.
   - **Where a stretch ends, the track is spliced.** Of 50 stretches drawn uniformly with replacement, the track continues from the one holding the offset nearest its last one, at the offset after that match. A stretch's own last offset never matches.
   - So the track never jumps further than the nearest match, and every step after a splice is a real step. The last stretch is cut to fit.
   - The noise is added to the animal's positions, and then the losses are drawn from the same stream, as now.
   - **No other freedom:** no rotation, scaling or reflection, and nothing tuned to the check. The candidate count, 50, is the largest the first pilot tried.

3b. **The on-lawn reversal floor** (decided 2026-10-28, after the first pilot).

- The kinematics family's `reversals_per_on_lawn_min` uses the off-food detector with its step floor raised from 0.01 to **0.03 mm**.
- **The rule:** the smallest floor on a 0.01-mm grid at which the real tracker's still-worm onset rate, on the library half, is under 1% of the real on-lawn rate, both read at that floor. The pilot recomputes it and stops if it does not give 0.03.
- **Why:** at 0.01 mm the tracker's noise exceeds the floor often (median step 0.0054 mm, 90th percentile 0.0158), and about 45% of the real on-lawn "reversals" are noise or sub-0.03-mm wobble (4.11 per minute at 0.01 against 2.27 at 0.03).
- **With the floor**, a still worm reads almost no reversals under any noise model: real 0.013, resampled 0.039, Gaussian 0.000 per paused minute. C.1's 0.15 per minute is still far from the real worms' 2.27.
- The jitter check's onset reading uses the same floor, so it measures what is graded.
- The kinematics bands, κ and the rule check are rebuilt at the floor. The other families, and every off-food use of the detector, are unchanged.

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

   - **The free parameters, 10** (revised 2026-10-28, after the 8-parameter fit; see "The second pilot" below):

     - each state's speed, curvature SD, reversal hazard and omega hazard, log-uniform on [θ/4, 4θ] around B.4's matched value θ (`negatives_params.json`);
     - each state's **speed spread**, a log SD in \[0, 1.5\]: at every 10-s window boundary each animal draws z, and runs that window at its state's speed × exp(spread · z). The window is the classifier's scale, so the state's speed is the median. This is `BatchedStateHazards`'s optional `speed_log_sd`, off by default, so B.4's negatives are unchanged.

   - **The objective:** the RMS z of the five gated statistics (the four speeds and the reversal rate) at n = 200, through the lawn model with the library, for 200 fresh animals per evaluation (`158_000_000 + 200·k + i`).

   - **The fit:** sep-CMA from the centre, σ₀ 0.2, population 16, 2 restarts (CMA seeds 9411 and 9412). The plateau rule stops a restart only once σ ≤ σ₀, and the cap is 100 generations. The answer is the best restart's `xfavorite`.

   - **The reversal coupling, 11th parameter** (decided 2026-10-28, after the 10-parameter fit, with its fallback fixed before the refit runs). Each window's reversal hazard is the state's hazard × exp(β · z), where z is the window's speed draw and β is uniform on [0, 3]. Reversals then come more often out of faster movement, as real reversals interrupt forward runs. This is `BatchedStateHazards`'s optional `reversal_speed_coupling`, off by default.

   - **The fallback, registered now:** if the 11-parameter worm's median RMS z on its 5 fresh populations still exceeds 1, the on-lawn reversal rate leaves the kinematics family:

     - five kinematics statistics remain, and z\* is recomputed for 19 statistics;
     - the bands are rebuilt without it, and reachability is registered on the four speeds, with the 11-parameter worm as fitted;
     - no further structure is tried.

     Roaming speed still catches C.1's main exploit (2.6× the real speed at the classifier's scale).

   - **Its own oracle, before registration:** the answer is re-simulated on 5 discarded populations (`158_900_000 + 200·g + i`). The median RMS z must be at most 1.

   - **If it is not:** no open-loop two-state point worm reaches the statistics through the tracker. The statistics it misses are the diagnosis, and the change stops before registration and goes back to the maintainer.

   - **Registered:** the answer's parameters go into amendment 2. The 40 oracle populations (`150_000_000 + 200·p + i`) are read with it under Part A's gate: mean z in [−1, 1], and |z| ≤ z\* in at least 90% of populations, for each of the five statistics.

7. **Then B v2 continues**, writing `oracle_amendment2.json`:

   - the losses check (it applies no noise, so it is re-run and must still pass);
   - the jitter check;
   - Chen's round trip (unchanged);
   - the reachability oracle with the reachability worm;
   - then Part B, the N2 read, the negatives, the blind worm and C.1's answer, all on the library model.

### The first pilot (2026-10-28; description)

The first draw joined stretches end to end, and the pilot's in-sample check stopped the change.

**The joins added false onsets.** With every candidate draw, the onset rate came out above the real 2.28 per paused minute:

| Draw tried (library half, 0.01 floor) | Onsets above real (per paused min) |
|---|---|
| joined end to end | +1.19 |
| an autoregressive model (orders 1–8) on the library's own residuals | +0.44 to +0.58 |
| nearest-offset splicing with 50 candidates | +0.43 |

The gate allowed ±0.25.

**The window speeds:** splicing passed both (median +0.00015, p90 +0.00011 mm/s, against ranges of about ±0.0004 and ±0.0003). The plain joins failed the p90 (+0.0010), and the autoregressive model failed both.

**The cause:** every resampled track has joins and real stretches have none, so the detector's 0.01-mm floor sits inside the noise. Decision 3b removes the dependence, and decision 3's splicing keeps the speeds. Both were chosen after these in-sample readings. Amendment 2 discloses this, and the check half stays unread.

### The second pilot (2026-10-28; description)

- **The floor and splicing passed in-sample.** The floor rule gave 0.03 mm. The draw read onsets +0.0034 per paused minute (range ±0.009), window speed median +0.00013 (±0.0004) and p90 +0.00006 (±0.0003).

- **The 8-parameter reachability worm did not reach.** Both restarts plateaued near an RMS z of 2.6, and the answer's median on 5 fresh populations was 2.53:

  | Statistic | z |
  |---|---|
  | reversals | −4.25 |
  | roaming speed p90 | −3.57 |
  | dwelling speed median | +2.58 |

  The roaming reversal hazard and the dwelling curvature sat at their box edges.

- **Why:** at one constant speed per state, a dwelling worm (about 0.017 mm/s) never makes the 0.03 mm steps the detector needs, and roaming has no spread to give the real p90. Real worms' speed varies within a state.

- **The fix:** the maintainer chose to add the speed spread, as decision 6 says. The stop rule is unchanged.

## Risks / Trade-offs

- **[Joining artefacts]** → end-to-end joins jump about 0.02 mm every 13 s, and they failed the first pilot. Splicing at the nearest offset replaces them; the second pilot reads it in-sample, and the check reads it at the rate a simulated track sees.
- **[Micro-motion as noise]** → a paused real worm's slight motion is carried over as noise. That is the intent: the benchmark compares what the tracker reports. It raises every simulated dwelling window's speed slightly, as it raises real ones.
- **[A two-state worm cannot reach the statistics]** → without within-state speed spread it could not (the second pilot). With the spread, the refit shows whether it can before registration; if not, the change stops there, and the diagnosis says which statistic no open-loop point worm reaches.
- **[A wider gate]** → the permutation gate admits model noise that sits within the tracker's own date variation. That is the comparison the bands make too: their scales are between-date.
- **[The bands file changes]** → `bands_v2.json` is rebuilt to name the library in its lawn model. The bands, the rule checks and the leave-one-out are deterministic and unchanged, and their record must come back byte-identical.
- **[Cost]** → the reachability fit takes about 10 s per generation of 16 on 16 workers, so at most about 35 min for both restarts at the cap.
