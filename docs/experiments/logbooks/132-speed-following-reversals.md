# 132: Phase 10 — The D36 Amendment: An On-Food Reversal That Follows the Worm's Own Speed

**Status**: completed. **The primitives gain an option for on-food reversals.** Under it, a reversal that starts on a lawn backs at a speed that follows the worm's run speed over its last 3 s of runs, as a real worm's does. The speed is interpolated between knots measured on real worms. It is off by default, and byte-identical when off.

> | Speed in the last 3 run seconds, at the knot (mm/s) | 0.005 | 0.018 | 0.040 | 0.052 | 0.069 | 0.104 |
> |---|---|---|---|---|---|---|
> | real reversal speed, and the option's (mm/s, Scheer 2023, held-out cohorts) | 0.048 | 0.057 | 0.075 | 0.080 | 0.086 | 0.096 |
> | a line clipped to the deciles' range, as first designed | 0.049 | 0.057 | 0.071 | 0.078 | 0.084 | 0.084 |
>
> - **Unset or `pooled`,** every on-food reversal backs at 0.069 mm/s as before, and C.1's batched path is byte-identical to before the change (59 of 59 jobs).
> - **Logbook 131's refit answer under the option** passes all 24 wild-type statistics and every mutant in 10 of 10 fresh populations, as under the probe's patch on the same animals.

**Date**: 2026-11-10.

**OpenSpec change**: `add-speed-following-reversals`.

**Records** (in [`supporting/132-speed-following-reversals/`](supporting/132-speed-following-reversals/)):

- `coupling.json`: the measurement, from `scripts/analysis/reversal_speed_coupling.py`;
- `confirmation.json`: the confirmation, beside its script (`confirm.py.txt`);
- `identity.json`: the identity check of C.1's batched path;
- `mutants.log.txt`: the parity test against five deliberately broken windows, beside its script (`mutants.py.txt`).

## Objective

D36 fixes the primitives' kinematics from data. Since [Logbook 108](108-food-dependent-kinematics.md), a reversal that starts on a lawn backs at one speed, 0.069 mm/s, pooled over every state.

[Logbook 131](131-c2-clean-start.md) found that real on-lawn reversals follow the speed just before them. A dwelling worm's run slower than a roaming worm's, so with one pooled speed a simulated dwelling reversal carries the worm about 20% too far. C.2's fit then traded the on-lawn onset rate against the dwelling windows' speed, and passed the wild type in 8 or 9 of 15 populations. With the coupling patched in scripts and the network refitted, it passed in 15 of 15.

The maintainer's order of 2026-11-10: this change, then C.2's registered fit.

## The measurement

`reversal_speed_coupling.py` reads Scheer & Bargmann 2023's OD 2 wild type. Bouts follow the kinematics check's rule. The rule is the one the primitives apply:

- each reversal bout starting with the midbody inside the lawn's boundary counts;
- it is read against the worm's last 3 run seconds before it, wherever they fell;
- a run second has no reversal or omega frame and a finite 1-s displacement.

On the held-out cohorts there were 95,435 on-lawn starts. The record counts each exclusion: 116 lacked 3 earlier run seconds, 11,615 spanned no whole second (the shortest reversals, which Logbook 108's speed also leaves out), and 7 had a path with a lost sample. That leaves 83,697.

| Set | Bouts (dates) | Line, 95% bootstrap over recording dates | Reversal speed, deciles 1 and 10 | Duration, deciles 1 and 10 |
|---|---|---|---|---|
| **held out, every on-lawn start** | 83,697 (24) | 0.0464 [0.0451, 0.0478] + 0.606 [0.561, 0.655] × v | 0.0484, 0.0839 | 2.00, 2.25 s |
| held out, more than 0.5 mm inside | 21,940 (24) | 0.0458 + 0.574 × v | 0.0466, 0.0770 | 2.03, 2.30 s |
| band, every on-lawn start | 148,557 (41) | 0.0473 + 0.578 × v | 0.0493, 0.0842 | 1.98, 2.26 s |

- **The held-out cohorts are the source,** as for Logbook 108's values: the simulated worm's body and the bands it is graded against come from disjoint recording dates. The band cohorts' line lies inside the held-out line's intervals.
- **Every on-lawn start, not the inside stratum.** The primitives give on-food kinematics to every reversal that starts on a lawn, the edge included. The inside stratum agrees at the slow end and runs lower at the fast end, where roaming worms near the edge reverse fastest.
- **The relation is concave at the fast end,** so the primitives take knots, not the line. The knots are the mean speed before and the mean reversal speed in deciles 1 to 9 of the speed before, and in the top decile split at its 95th and 99th percentiles: 12 knots, from 0.0484 mm/s at 0.0046 to 0.0961 at 0.1042. The line runs 0.002 to 0.004 mm/s low in the 80th to 95th percentiles, and overshoots past them. Clipped at the top decile's mean, as first designed, it capped the fastest 5% of reversals at 0.084, where real worms reverse at 0.086 and 0.096.
- **Every bin is half-open but the last,** so each bout falls in one.

**No tracker floor.** The knots are measured on tracked speeds and applied to true ones, so reading the true speed through the tracker's noise looked natural. The record rules it out:

- The slowest decile's speed before, 0.0046 mm/s, is below the still-worm library's paused-worm mean, 0.0075 mm, so that library overstates the noise in these tracks.
- Through it, a still simulated worm would reverse at the speed of real worms moving at 0.0075 mm/s, about 0.050, faster than the slowest two deciles (0.0484 and 0.0501).
- Without it, the first knot holds a still worm at 0.0484, the slowest decile's real value.

## What changed

- **`data/locomotion/primitives.json`** gains `on_food_reversal_speed_coupling`: the knots and the window, condition `on food`, status `measured`. It names the script and its record.
  - The loader refuses a file without the entry by name.
  - It checks that the knots are at least two, equal in length and finite, the speeds before non-negative and strictly increasing, the reversal speeds positive, and the window positive.
- **The primitives.**
  - A worm's state keeps the commanded speeds of its last N run steps, N = round(3 s ÷ step), oldest first.
  - Bout steps neither enter the window nor clear it, and it persists across bouts and assay phases.
  - A reversal that starts on a lawn backs at the knots' interpolated speed at the window's mean once the window is full, and at the pooled 0.069 mm/s until then.
  - The duration keeps the pooled on-food gamma, and nothing off food changes.
- **`continuous.on_food_reversal_speed`,** `pooled` or `follows_run_speed`:
  - Unset means `pooled`.
  - It is refused without primitives, and `follows_run_speed` is refused with `reversal_kinematics: off_food`.
  - Without lawns it cannot act, and cannot be set either: primitives need a step of at most one second, which is accepted only on lawns.
- **Both runners** keep the window the same way, as Python lists, and take its mean and the interpolation through the same two functions, in pure Python. Their run speeds are computed identically, so each reversal's speed matches bit for bit.

## Checks

- **Tests** (`test_speed_following_reversals.py`, 24):

  - the pinned knots and their refusals, including a missing entry and infinite values;
  - the interpolation at, between and beyond the knots;
  - the pooled speed before the window fills;
  - an omega that leaves the window as it was;
  - the 6-step window at a 0.5-s step;
  - the environment on a lawn, on a depleted cell, off the lawn and in a phase without food;
  - the refusals, the config without lawns that cannot load, and the default.

- **No draw changes.** On Flavell's lawn, with a worm that never leaves it, the option moves the worm differently, but its RNG state and every step's primitive match the pooled run's. Unset and `pooled` runs are identical.

- **Parity** (`test_batched_parity.py`): with the option on, the two runners make the same bouts over 300 steps, and the same paths within 1e-3 over 150.

  - It covers three controllers, on Scheer's small lawn and Flavell's uniform lawn: the MLP, a constant worm at 0.02 mm/s, and a worm whose run speed cycles through seven values inside the knots, one each step.
  - Under the cycling worm, every bout's kind, food, window mean, speed and duration match exactly, animal by animal.
  - An evaluator that ignores the window fails.

- **The parity test against broken windows** (`mutants.log.txt`). Five deliberate bugs in the batched runner's window each fail the exact comparison on both lawns:

  | Deliberate bug | Caught by the exact comparison | Caught by the path check |
  |---|---|---|
  | bout steps pushed into the window | both lawns | Flavell's lawn |
  | a 2-step window | both lawns | both lawns |
  | a 5-step window | both lawns | both lawns |
  | only the newest speed | both lawns | both lawns |
  | the window cleared at each bout start | both lawns | Flavell's lawn |

  The first form of the test missed four of the five entirely, and caught the fifth in one case of four. Its MLP ran fast enough to put every on-lawn reversal on the line's ceiling, and its constant worm ran at one speed, so the window's mean never changed what a reversal did.

- **Identity** (`identity.json`): with the option unset, C.1's batched path matches the baseline taken before the change in 59 of 59 jobs: benchmark v1, v2 and v3, with and without presence, and the stored records.

- **Against main.** In the review, positions, headings and primitives of per-step and batched runs on all four lawn assays were byte-identical between the branch, option unset, and main. That ran before the knots, which change only the option's own path and the loader.

## The confirmation

`confirm.py` grades Logbook 131's refit answer on the same 10 fresh populations of 200 (880,000,000 + 200·g + i) twice: under the package's option, and under Logbook 131's patch. The patch differs from the option in three ways:

- it used the band cohorts' line;
- it read realized displacements;
- it clipped at 0.049 and 0.084.

The rule was fixed before the grade: the option carries the probe's result if the wild type passes in at least 8 of 10.

| Body | Wild type | tph-1 | pdfr-1 | tax-4 |
|---|---|---|---|---|
| **the package's option** | **10 of 10** | 10 of 10 | 10 of 10 | 10 of 10 |
| Logbook 131's patch | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |

**The rule is met.** The largest paired shift between the two bodies is 0.47 z, on pre-exit enrichment (+0.34 under the patch, −0.13 under the option), the noisiest statistic (its SD between populations is about 1.2). Every shift is well inside the bars.

## The review of the PR

An independent review of the branch before merging found no defect in the code, and these points, all acted on:

- the parity test could not see the window (above);
- knots in place of the line and its clip, whose ceiling understated the fastest reversals;
- the record's exclusions counted, and its bins made half-open (one bout had been counted in two deciles);
- the loader's missing-entry error and finiteness checks;
- the no-lawn case found to be refused already, and pinned by a test instead of an unreachable check;
- one wrong sentence in this logbook (the floor's speed is faster than the slowest two deciles, not every decile), and the design's description of the batched window.

The identity check and the confirmation were re-run on the knots.

## Known differences from real worms

- **Delta turns.** Scheer's data has no delta flag, so a real worm's deep dorsal coils count as run seconds in the measurement. The simulator's delta bouts stay out of the window, as every bout does. A fitted C.2 makes about 3 delta turns a minute on the lawn, so a simulated worm's window may read a little faster than a real worm's would.
- **Duration.** It rises from 2.00 to 2.25 s across the deciles, and stays pooled. Logbook 131's refit passed without it, and keeping it leaves every draw where it is.
- **Tracked and true speeds.** Tracking noise lifts the slowest tracked speeds; the first knot holds the slow end at the slowest decile's real reversal speed.

## What this leaves

- **C.2's registered fit and C.1's refit before C.4 set the option on,** fixed in the fit's registration.
- **Earlier C.1 and C.2 records name the pooled setting,** which reproduces them.

## Seeds

- **Tests:** 9601–9605. The parity tests run on the parity seeds, 5001–5004.
- **The confirmation:** 880,000,000 + 200·g + i, discarded.
- **The identity check:** its fixed jobs on 99,700,000, discarded.
