## Why

C.2's staged fit found a worm that does not run ([Logbook 123](../../../docs/experiments/logbooks/123-c2-speed-noise.md)). Its speed output is pinned at zero and its curvature at the maximum, so it moves only by reversals and delta turns. C.1's v3 answer errs the other way: it reverses about 3% of the time, against real worms' 12–17% ([Logbook 121](../../../docs/experiments/logbooks/121-c1-v3-fit.md)).

Benchmark v3 sees neither error. On the lawn it grades states, bouts and 10-s window speeds, but never what the worm does second to second. Every later controller, and every W arm, could use the same gap.

On Scheer's lawn, real worms (wild type at OD 2, 1,586 animals, the authors' per-frame flags) are paused 50% of the time, reversing 14% and moving forward about 35%. C.2's stage-2 answer is paused about 50%, reversing 30–47% and moving forward 0%.

**A planning probe** (`docs/experiments/logbooks/supporting/124-benchmark-v4/probe/activity_probe.json`; description, seeds 600,000,000+) read three statistics from 1-s positions alone. Each is taken over a worm's on-lawn seconds, on Scheer's band cohorts per date (41 dates, 1,013 animals) and on simulated tracks after Scheer's rig model. In brackets, how far each controller sits from the real value, in between-date SDs:

| Statistic | Real (date SD) | Held-out real | C.2 stage 2 | C.1 v3 | Constant hazards | Pirouette |
|---|---|---|---|---|---|---|
| Pause share at the instrument floor, 0.02 mm | 0.482 (0.071) | 0.489 | 0.598 (+1.6) | 0.573 (+1.3) | 0.127 (−5.0) | 0.029 (−6.4) |
| Direction-flip fraction on the lawn | 0.132 (0.026) | 0.124 | 0.005 (−4.8) | 0.096 (−1.4) | 0.032 (−3.8) | 0.009 (−4.7) |
| Reversal onsets per on-lawn minute | 2.28 (0.48) | 2.20 | 0.09 (−4.6) | 0.81 (−3.1) | 0.95 (−2.8) | 0.31 (−4.1) |

The flip fraction catches C.2's worm, whose reversals start from standstill. The onset rate catches C.1's rare reversals. The pause share, at a floor set by the instrument, agrees with the authors' own pause flag: 0.482 against 0.498.

v2 left the on-lawn onset rate reported, not graded, because no open-loop point worm reached it through the tracker (Logbook 115).

The maintainer's decisions (2026-11-05):

- **grade the pause share and the on-lawn flip fraction,** and grade the onset rate only if a reachability oracle reaches it, with the fallback registered first;
- **a new activity family in D_fit,** so the fits see it;
- **Scheer's small lawn only.**

## What Changes

- **Benchmark v4** (`data/benchmark/bands_v4.json`): v3's 21 statistics with their bands byte-identical, and a sixth family, **activity**, on Scheer's small lawn. Each of its statistics is counted over a worm's on-lawn seconds (both samples present and on the lawn):
  - **the pause share:** 1-s displacements shorter than 0.02 mm. That floor is the instrument's: 95% of the real tracker's 1-s readings of a still worm (`scheer_tracker_noise.npz`) fall below it;
  - **the on-lawn direction-flip fraction:** v3's rule (pairs of 1-s displacements, both at least 0.03 mm, turning more than 135°) on the on-lawn track;
  - **the on-lawn onset rate:** `grading.on_lawn_reversals` at the 0.03-mm floor, per on-lawn minute. It is graded only if the oracle reaches it; otherwise it is reported, as in v2 and v3.
- **The bar:** 24 graded statistics at z\* = 3.0781, or 23 at 3.0654 if the onset rate is reported only. D_fit becomes the small lawn, the uniform well, kinematics and activity.
- **Bands** from the lawn method: a cluster bootstrap over Scheer's band dates, with a per-statistic κ measured by half-splits and a rule check at v4's bar.
- **v4's controls** (`scripts/analysis/benchmark_v4_controls.py`), registered before any reading:
  - the activity family's rule check;
  - the track-import round trip;
  - a reachability oracle (C.1's 25-parameter model fitted to the activity statistics alone);
  - a fresh-N2 positive control on N2's unread activity statistics;
  - the adversary, C.2's stage-2 answer;
  - v3's negatives at v4's bar;
  - W's minimum and the 3-of-5 probability recomputed.
- **`benchmark_grade.py --bands v1|v2|v3|v4`**, with v4 the default once VALID; v1–v3 stay byte-identical.
- **Reported beside it:** C.1's v3 answer graded on v4, which decides whether C.1 is refitted.
- **Launch record 124** (Part A before any reading; Part B after the oracle's fit; Part C, the onset decision, after its check) and **Logbook 124**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `benchmark-grading`: v4's activity family, its bands and its conditional onset rate.
- `benchmark-controls`: v4's validity map.

## Impact

- `quantumnematode/benchmark/grading.py`: the activity tally and statistics and v4's tuples; v1–v3's untouched.
- `scripts/analysis/benchmark_bands.py --version v4`, `scripts/analysis/benchmark_grade.py` (`grade_populations_v4`, `v4_is_valid`, the default chain), and `scripts/analysis/benchmark_v4_controls.py` (new).
- `data/benchmark/bands_v4.json`, `data/benchmark/IMPORT.md` (imported tracks carry the activity family), `docs/experiments/logbooks/supporting/124-benchmark-v4/`, Logbook 124, the tracker, AGENTS.md and the CHANGELOG.
- **Cost:** bands in minutes; the oracle's fit about 1–2 h, on the small lawn alone; the controls' reading about an hour.
- **Not in this change:** C.2's readout box and saturation report, and its readiness restarts. Those are C.2's next change, fitted against v4.
