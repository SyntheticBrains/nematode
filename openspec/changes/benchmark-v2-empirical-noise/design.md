## Context

B v2 (change `revise-benchmark-v2`, launch record 115) adds a measurement model so simulated and real tracks are read through the same tracker. Its registered checks read:

- **The loss model and Chen's round trip pass.**
- **The Gaussian jitter fails** against the real tracker's readings of a still worm. The independent noise gives too many false reversal onsets (3.87 against 2.14 per paused minute) and too little 10-s drift (window speed median 0.0016 against 0.0038 mm/s).

Two facts about the paused stretches bear on resampling them:

- they last 13 s at the median (90th percentile 22 s, at least 10 s by construction);
- their offsets from their own mean have an SD of 0.0149 mm, about 3× the median 1-s step, because the offsets carry drift.

## Goals / Non-Goals

**Goals:**

- a lawn noise model that reproduces the real tracker's readings of a still worm, judged on data the model never saw;
- no new free parameters fitted to the check.

**Non-Goals:**

- a mechanistic model of the tracker;
- separating real micro-motion of a paused worm from tracker noise. Both are in what the tracker reports for a still worm, so both are carried over, and a still simulated worm then reads like a still real worm;
- changing Chen's rig, whose Gaussian jitter passed its round trip;
- changing any statistic, band, threshold or gate of B v2.

## Decisions

1. **The split.** PD1074's 41 band dates are split into a library half and a check half by `default_rng(2043).permutation`. The library half gets the first 21 dates and the check half the other 20. Both halves are band cohorts, so no held-out PD1074 cohort is read.

2. **The library.**

   - **The stretches:** the library half's paused stretches, as amendment 1 defines them: runs of at least 10 s, each second with all three frames flagged `pause` and its position present.
   - **What is stored:** each stretch's offsets from its own mean position (x and y), in micrometres as int32, as the Scheer store does. Stretches are stored consecutively, with their lengths.
   - **Where:** `data/benchmark/scheer_tracker_noise.npz`. Its `PROVENANCE.md` entry gives the source, the split and the checksum.

3. **The draw.**

   - **The noise track:** for a simulated animal of n seconds with seed s, `rng = default_rng([s, 2033])`, the lawn stream. Stretches are drawn uniformly with replacement and joined end to end until the track reaches n seconds, and the last one is cut to fit. The noise is added to the animal's positions, and then the losses are drawn from the same stream, as now.
   - **The joining rule** is one of two, chosen by the pilot (decision 04):
     - **(a) plain:** each stretch's offsets are used as stored. Where two stretches meet there is a jump the size of the difference between one's last offset and the next one's first;
     - **(b) continued:** each new stretch is shifted so that its first offset equals the previous stretch's last offset, plus a 1-s step drawn from the library's own within-stretch steps. The track stays continuous, but it drifts as a random walk of the stretches' end-to-start changes.
   - **No other freedom:** no rotation, scaling or reflection of the offsets, and no tuning to the check.

4. **The pilot** (on the library half only; description, not a verdict).

   - Both joining rules run amendment 1's jitter check with the library half as both the library and the real side.
   - The chosen rule has the smaller largest |z| over the three gated readings.
   - Also reported, for the chosen rule: the reversal-rate and dwelling-speed shift that the noise adds to B.4's matched open-loop worm on one discarded population. This is the reachability oracle's direction, on seeds disjoint from its registered populations (199 M).
   - **If neither rule brings every gated reading within one scale in-sample**, the empirical model cannot pass out of sample either. The change stops before registration and goes back to the maintainer.

5. **The re-registered check** (115's launch record, amendment 2, committed before the check half is read):

   - **The real side:** the check half's paused stretches.
   - **The model side:** for check stretch k of length L, a noise track of 2,700 s (the small-lawn assay's length) is drawn at `default_rng([156_000_000 + k, 2033])` by the chosen rule, and L seconds are cut at a uniform random point from the same stream. So stretches are joined at the rate a simulated track sees.
   - **The gate (amendment 1's):** onsets per paused minute and the 10-s window speed's median and 90th percentile must each differ by at most one n = 200 band scale, of the reversal rate and of the two dwelling speeds respectively. The 1-s step quantiles are reported.
   - **On failure:** the run stops and returns to the maintainer. No further noise model is tried in this change.

6. **Then B v2 continues** with the losses check (it applies no noise, so the library cannot change it; it is re-run and must still pass), Chen's round trip (unchanged), and the reachability oracle as registered, writing `oracle_amendment2.json`. After that come Part B, the N2 read, the negatives, the blind worm and C.1's answer. The measurement model they use is the one with the library.

## Risks / Trade-offs

- **[Joining artefacts]** → the gap between one stretch's last offset and the next one's first is about 0.02 mm, against a median 1-s step of 0.005 mm. Rule (a) adds a jump about every 13 s, and rule (b) adds a slow drift. The pilot chooses between them, and the check reads the joins at the rate a simulated track sees.
- **[Micro-motion as noise]** → a paused real worm's slight motion is carried over as noise. That is the intent: the benchmark compares what the tracker reports. It is stated, and it raises every simulated dwelling window's speed slightly, as it raises real ones.
- **[The reachability oracle]** → the matched open-loop worm's parameters were measured from raw tracks, which already carry the tracker's noise, so the noise is counted twice for it. Its dwelling speed through the Gaussian model already read 0.0182 against the real 0.0128 on a smoke population. The oracle may fail on the matched worm, which is that oracle's registered branch ("the failure belongs to the matched worm or to the statistic"). The pilot reports the shift beforehand.
- **[Half the data]** → the library uses half the paused stretches, about 6,600, which is ample for resampling.
- **[The bands file changes]** → `bands_v2.json` is rebuilt to name the library in its lawn model. The bands, the rule checks and the leave-one-out are deterministic and unchanged, and their record must come back byte-identical.
