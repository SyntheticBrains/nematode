# 132: Phase 10 — The D36 Amendment: An On-Food Reversal That Follows the Worm's Own Speed

**Status**: completed. **The primitives gain an option for on-food reversals.** Under it, a reversal that starts on a lawn backs at a speed that follows the worm's run speed over its last 3 s of runs, as a real worm's does. It is off by default, and byte-identical when off.

> | Speed in the last 3 run seconds, decile (mm/s) | 0.005 | 0.011 | 0.018 | 0.026 | 0.064 |
> |---|---|---|---|---|---|
> | real reversal speed (mm/s, Scheer 2023, held-out cohorts) | 0.048 | 0.052 | 0.057 | 0.063 | 0.084 |
> | the option: clip(0.0464 + 0.606 × v, 0.0484, 0.0839) | 0.049 | 0.053 | 0.057 | 0.062 | 0.084 |
>
> - **Unset or `pooled`,** every on-food reversal backs at 0.069 mm/s as before, and C.1's batched path is byte-identical to before the change (59 of 59 jobs).
> - **Logbook 131's refit answer under the option** passes all 24 wild-type statistics and every mutant in 10 of 10 fresh populations, as under the probe's patch on the same animals.

**Date**: 2026-11-10.

**OpenSpec change**: `add-speed-following-reversals`.

**Records** (in [`supporting/132-speed-following-reversals/`](supporting/132-speed-following-reversals/)):

- `coupling.json`: the measurement, from `scripts/analysis/reversal_speed_coupling.py`;
- `confirmation.json`: the confirmation, beside its script (`confirm.py.txt`);
- `identity.json`: the identity check of C.1's batched path.

## Objective

D36 fixes the primitives' kinematics from data. Since [Logbook 108](108-food-dependent-kinematics.md), a reversal that starts on a lawn backs at one speed, 0.069 mm/s, pooled over every state.

[Logbook 131](131-c2-clean-start.md) found that real on-lawn reversals follow the speed just before them. A dwelling worm's run slower than a roaming worm's, so with one pooled speed a simulated dwelling reversal carries the worm about 20% too far. C.2's fit then traded the on-lawn onset rate against the dwelling windows' speed, and passed the wild type in 8 or 9 of 15 populations. With the coupling patched in scripts and the network refitted, it passed in 15 of 15.

The maintainer's order of 2026-11-10: this change, then C.2's registered fit.

## The measurement

`reversal_speed_coupling.py` reads Scheer & Bargmann 2023's OD 2 wild type. Bouts follow the kinematics check's rule. The rule is the one the primitives apply:

- each reversal bout starting with the midbody inside the lawn's boundary counts;
- it is read against the worm's last 3 run seconds before it, wherever they fell;
- a run second has no reversal or omega frame and a finite 1-s displacement.

| Set | Bouts (dates) | Line, 95% bootstrap over recording dates | Reversal speed, deciles 1 and 10 | Duration, deciles 1 and 10 |
|---|---|---|---|---|
| **held out, every on-lawn start** | 83,697 (24) | **0.0464 [0.0451, 0.0478] + 0.606 [0.561, 0.655] × v** | 0.0484, 0.0839 | 2.00, 2.25 s |
| held out, more than 0.5 mm inside | 21,940 (24) | 0.0458 + 0.574 × v | 0.0466, 0.0770 | 2.03, 2.30 s |
| band, every on-lawn start | 148,557 (41) | 0.0473 + 0.578 × v | 0.0493, 0.0842 | 1.98, 2.26 s |

- **The held-out cohorts are the source,** as for Logbook 108's values: the simulated worm's body and the bands it is graded against come from disjoint recording dates. The band cohorts' line lies inside the held-out line's intervals.
- **Every on-lawn start, not the inside stratum.** The primitives give on-food kinematics to every reversal that starts on a lawn, the edge included. The inside stratum agrees at the slow end and runs lower at the fast end, where roaming worms near the edge reverse fastest.
- **Only 104 bouts lacked 3 earlier run seconds.**

**No tracker floor.** The line was fitted to tracked speeds and is applied to true ones, so reading the true speed through the tracker's noise looked natural. The record rules it out. The slowest decile's speed before, 0.0046 mm/s, is below the still-worm library's paused-worm mean, 0.0075 mm, so that library overstates the noise in these tracks. Through it, a still simulated worm would reverse at 0.0509 mm/s, faster than any real decile. Instead the clip sets the slow end: the line at a still worm (0.0464) falls below the slowest decile's mean, so a still worm backs at 0.0484, that decile's real value.

## What changed

- **`data/locomotion/primitives.json`** gains `on_food_reversal_speed_coupling`: the intercept, slope, low, high and window, condition `on food`, status `measured`. It names the script and its record. The loader checks intercept > 0, slope ≥ 0, 0 < low < high and window > 0.
- **The primitives.**
  - A worm's state keeps the commanded speeds of its last N run steps, N = round(3 s ÷ step), oldest first.
  - Bout steps neither enter the window nor clear it, and it persists across bouts and assay phases.
  - A reversal that starts on a lawn backs at the line on the window's mean, clipped, once the window is full, and at the pooled 0.069 mm/s until then.
  - The duration keeps the pooled on-food gamma, and nothing off food changes.
- **`continuous.on_food_reversal_speed`,** `pooled` or `follows_run_speed`:
  - Unset means `pooled`.
  - It is refused without primitives, and `follows_run_speed` is refused with `reversal_kinematics: off_food`.
- **Both runners** keep the window the same way, and take its mean and the line through the same two functions. Their run speeds are computed identically, so each reversal's speed matches bit for bit.

## Checks

- **Tests** (`test_speed_following_reversals.py`, 18):
  - the pinned entry and its refusals;
  - the line and its clip;
  - the pooled speed before the window fills;
  - an omega that leaves the window as it was;
  - the 6-step window at a 0.5-s step;
  - the environment on a lawn, on a depleted cell, off the lawn and in a phase without food;
  - the refusals and the default.
- **No draw changes.** On Flavell's lawn, with a worm that never leaves it, the option moves the worm differently, but its RNG state and every step's primitive match the pooled run's. Unset and `pooled` runs are identical.
- **Parity** (`test_batched_parity.py`): with the option on, the per-step runner and the batched evaluator make the same bouts over 300 steps, and the same paths within 1e-3 over 150. That holds for the MLP and for a constant worm at 0.02 mm/s, on Scheer's small lawn and Flavell's uniform lawn. An evaluator that ignores the window fails.
- **Identity** (`identity.json`): with the option unset, C.1's batched path matches the baseline taken before the change in 59 of 59 jobs: benchmark v1, v2 and v3, with and without presence, and the stored records.

## The confirmation

`confirm.py` grades Logbook 131's refit answer on the same 10 fresh populations of 200 (880,000,000 + 200·g + i) twice: under the package's option, and under Logbook 131's patch. The patch differs from the option in three ways:

- it used the band line;
- it read realized displacements;
- it clipped at 0.049.

The rule was fixed before the grade: the option carries the probe's result if the wild type passes in at least 8 of 10.

| Body | Wild type | tph-1 | pdfr-1 | tax-4 |
|---|---|---|---|---|
| **the package's option** | **10 of 10** | 10 of 10 | 10 of 10 | 10 of 10 |
| Logbook 131's patch | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |

**The rule is met.** The largest paired shift between the two bodies is 0.27 z, on roaming survival at 5 minutes (+1.16 to +1.42), all well inside the bars.

## Known differences from real worms

- **Delta turns.** Scheer's data has no delta flag, so a real worm's deep dorsal coils count as run seconds in the measurement. The simulator's delta bouts stay out of the window, as every bout does. A fitted C.2 makes about 3 delta turns a minute on the lawn, so a simulated worm's window may read a little faster than a real worm's would.
- **Duration.** It rises from 2.00 to 2.25 s across the deciles, and stays pooled. Logbook 131's refit passed without it, and keeping it leaves every draw where it is.
- **Tracked and true speeds.** Tracking noise lifts the slowest tracked speeds; the clip holds the slow end at the slowest decile's real reversal speed.

## What this leaves

- **C.2's registered fit and C.1's refit before C.4 set the option on,** fixed in the fit's registration.
- **Earlier C.1 and C.2 records name the pooled setting,** which reproduces them.

## Seeds

- **Tests:** 9601–9605.
- **The confirmation:** 880,000,000 + 200·g + i, discarded.
- **The identity check:** its fixed jobs on 99,700,000, discarded.
