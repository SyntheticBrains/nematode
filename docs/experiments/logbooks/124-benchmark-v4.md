# 124: Phase 10 — B v4: On-Lawn Activity Graded from Positions, VALID, and C.1's v3 Answer Fails It on the Onset Rate

**Status**: completed. **B v4 is VALID**, and `benchmark_grade.py` now grades on it by default. Every registered check holds:

- the activity family's rule check at v4's bar;
- the track-import round trip;
- the reachability oracle: C.1's model, fitted to the activity statistics alone, reaches all three on 40 fresh populations, so the onset rate is graded;
- fresh N2, whose activity statistics no reading had computed;
- the adversary, C.2's stage-2 answer, caught in 5 of 5 populations;
- B.4's six negatives, v2's blind worm and C.1's v1 answer, each failing the wild type in 5 of 5.

**What v4 adds.** v3 graded the lawn's states, bouts and 10-s window speeds, but never what a worm does second to second. It let through a worm that never runs forward and one that rarely reverses. v4 adds a sixth family, **activity**, on Scheer's small lawn, read from 1-s positions alone: the pause share, the direction-flip fraction and the reversal-onset rate. That gives 24 statistics at z\* 3.0781.

**C.1's v3 answer fails v4.** Its onset rate is 0.78–0.85 per on-lawn minute against real worms' 2.28 (z −6.4 to −6.7), so it fails the activity family in all 5 of its registered populations. It passed v3 in 3 of the same 5, on the same animals. C.1 is refitted on v4 before the fidelity table.

> | Registered check | Result |
> |---|---|
> | Activity rule check at 3.0781 (gate 0.90) | 0.985 |
> | Round trip, 1,013 band animals, CSV and WCON | every tally equal |
> | Reachability: the oracle's answer on 40 populations | pass (mean z −0.10, +0.04, +0.10; 100% within the bar) |
> | Fresh N2, 2 units, at most 1 may fail | 0 fail |
> | C.2's stage-2 answer, caught in 3 of 5 | caught in 5 of 5 |
> | B.4's six, v2's blind worm, C.1's v1 answer, fail at 3 of 5 | each fails 5 of 5 |

**Date**: 2026-11-05.

**OpenSpec change**: `benchmark-v4`.

**Records**: [`supporting/124-benchmark-v4/`](supporting/124-benchmark-v4/):

- `launch.md` (Parts A, B and C);
- `probe/` (the planning probe and its script as run), `preflight.json` (disclosed);
- `identity.json` (v1–v3 unchanged);
- `bands_v4_record.json`, `round_trip.json`, `oracle_fit.json`, `oracle_check.json`, `read.json`, `controls_v4.json`.

The bands are [`data/benchmark/bands_v4.json`](../../../data/benchmark/bands_v4.json) (SHA-256 `355177a6…`).

## Objective

Two controllers exposed a gap in v3 on the lawn:

- **C.2's stage-2 answer does not run** ([Logbook 123](123-c2-speed-noise.md)). Its speed output is pinned at zero and its curvature at the maximum, so it moves only by reversals and delta turns.
- **C.1's v3 answer rarely reverses** ([Logbook 121](121-c1-v3-fit.md)): about 3% of its time, against real worms' 12–17%.

v3 passes both on the lawn. On Scheer's lawn, the authors' per-frame flags put real worms (PD1074 at OD 2) paused 49.8% of the time and reversing 14.3%.

The maintainer's decisions (2026-11-05):

- grade the pause share and the on-lawn flip fraction, and the onset rate only if a reachability oracle reaches it, with the fallback registered first;
- a new activity family in D_fit, so the fits see it;
- Scheer's small lawn only.

v1–v3 stay frozen. All 59 jobs of the identity check reproduce every v1–v3 statistic, D and presence reading byte for byte with v4's code in place (`identity.json`).

## The statistics

Each is counted over a worm's **on-lawn seconds**, those whose two samples are both present and on the lawn, and pooled over a population.

| Statistic | What it reads | Band at n = 200 |
|---|---|---|
| pause share | 1-s displacements shorter than 0.02 mm | 0.4819 ± 0.0322 |
| flip fraction | pairs of 1-s displacements, both at least 0.03 mm, turning more than 135° | 0.1321 ± 0.0125 |
| onset rate (per min) | the on-lawn detector's reversal onsets at its 0.03 mm floor | 2.284 ± 0.225 |

- **The pause floor is the instrument's.** It is the smallest 0.0025-mm step below which at least 95% of the real tracker's 1-s readings of a still worm fall: 0.02 mm, with 95.4% of 87,035 still-worm seconds below it (`scheer_tracker_noise.npz`). It is derived in code, and a test asserts it. At this floor the pause share, 0.482, agrees with the authors' own pause flag, 0.498.
- **The bands** come from the 41 band dates of PD1074 at OD 2 (1,013 animals) by the lawn method: a cluster bootstrap over dates, scaled by κ (1.221, 1.207 and 1.216, from half-splits). Every v3 band is copied byte-identical.
- **D v4** has 24 statistics in six equal families, so z\* = 3.0781 (v3: 21 and 3.0381). Activity's three statistics weigh 1/18 each in D. D_fit is now the small lawn, the uniform well, kinematics and activity, at 1/12 each for activity.
- **Simulated tracks** are tallied after Scheer's rig model, over the small lawn's analysis window, as kinematics are. Imported small-lawn tracks carry the family.

The planning probe (`probe/activity_probe.json`, description) chose the three. In between-date SDs from the real value:

| Statistic | Real (date SD) | C.2 stage 2 | C.1 v3 | Constant hazards | Pirouette |
|---|---|---|---|---|---|
| pause share | 0.482 (0.071) | +1.6 | +1.3 | −5.0 | −6.4 |
| flip fraction | 0.132 (0.026) | −4.8 | −1.4 | −3.8 | −4.7 |
| onset rate | 2.28 (0.48) | −4.6 | −3.1 | −2.8 | −4.1 |

The flip fraction catches C.2's worm, whose reversals start from standstill. The onset rate catches C.1's rare reversals.

## The validity map

1. **The rule checks** pass at both bars v4 could have had (`bands_v4_record.json`). The activity family reads 0.985 at each: the pause share fails in 1 split of 200, the flip fraction in 2 and the onset rate in none. Re-read at v4's bar, the small lawn reads 0.95, kinematics 0.98 and local search 0.992.

2. **The round trip** (`round_trip.json`): all 1,013 band animals, written as CSV and as WCON and imported again, give activity tallies equal to the native path's, animal by animal.

3. **The reachability oracle.**

   - **The fit** (`oracle_fit.json`): C.1's 25-parameter model fitted by sep-CMA to the three activity statistics alone, on the small lawn. Both restarts ran to the cap of 100 generations, their median objectives still falling. Restart 1 was taken (final objective 0.570, against restart 0's 0.831).

   - **The check** (`oracle_check.json`, 40 populations at 602 M):

     | Statistic | Mean z | Share within z\* | z SD |
     |---|---|---|---|
     | onset rate | +0.105 | 100% | 0.31 |
     | pause share | −0.095 | 100% | 0.21 |
     | flip fraction | +0.043 | 100% | 0.32 |

   - **The onset rate is reached,** so it is graded. v2 had left the on-lawn onset rate reported, because no open-loop point worm reached it through the tracker (Logbook 115). C.1's closed-loop model does.

   - **The answer says nothing about the other families.** Fitted to activity alone, it dwells faster than it roams, and its reversal and turn hazards are 3–35 times C.1's v3 answer's.

   - The populations' z's spread by an SD of 0.21–0.32, well under 1. The bands' scales include the spread between recording dates, which one parameter set's populations do not have.

4. **Fresh N2** (`read.json`): v2's two halves of N2's 26 unread held-out dates, each graded against N2's own activity bands from its 42 band dates.

   | Unit | Animals | pause share z | flip fraction z | onset rate z |
   |---|---|---|---|---|
   | half 0 | 294 | −0.10 | −0.92 | −0.26 |
   | half 1 | 207 | −1.11 | +1.10 | +1.43 |

   Neither fails. Both would fail by chance with probability 0.000225.

5. **The adversary,** C.2's stage-2 answer on the wiring it was fitted on, is caught in 5 of 5 populations by all three statistics. Its flip fraction is 0.005 (z −10, the cap), its onset rate 0.086 per minute (z −9.7) and its pause share 0.60 (z +3.5 to +3.7).

6. **The negatives** each fail the wild type in 5 of 5 populations, and chemotaxis in 5 of 5:

   | Negative | D | pause share z | flip fraction z | onset rate z |
   |---|---|---|---|---|
   | `hmm` | 4.5–5.4 | −8.4 to −8.3 | −2.5 to −2.0 | −4.2 to −3.9 |
   | `poisson` | 6.5–7.2 | −10 | +1.0 to +1.2 | −1.6 to −1.5 |
   | `random_walk` | 7.4–8.0 | −10 | −10 | −10.0 to −9.9 |
   | `correlated_walk` | 7.4–8.0 | −10 | −10 | −9.4 to −9.2 |
   | `untrained_mlp` | 5.3–6.0 | +1.5 to +2.4 | +1.5 to +2.1 | −3.9 to −2.9 |
   | `untrained_connectome` | 4.2–4.4 | +4.8 to +5.0 | +1.7 to +2.2 | −4.1 to −3.7 |
   | v2's blind worm | 3.8–4.7 | +2.6 to +3.9 | +3.9 to +5.4 | −4.4 to −2.9 |
   | C.1's v1 answer | 6.3–6.5 | +3.5 to +4.5 | −9.0 to −8.8 | −9.1 to −8.9 |

   The untrained MLP passes the activity family in 1 of its 5 populations. It fails the wild type in all 5 on other families.

## Reported beside the verdict

- **C.1's v3 answer** on its own registered populations (525 M), graded in full on v4. Its v3 statistics on v4 equal its stored v3 grade exactly, so the two grades are like for like.

  | | Pops 0–1 | Pops 2–4 |
  |---|---|---|
  | v3 grade | fail (small lawn) | pass |
  | v4 grade | fail (small lawn, activity) | fail (activity) |

  - **The onset rate** fails in every population: 0.78–0.85 per minute (z −6.4 to −6.7).
  - **The pause share** sits at +2.6 to +3.0 and **the flip fraction** at −2.8, both inside the bar but near it.
  - D is 2.00–2.08 and D_fit 2.38–2.47.

  It **fails the wild type on v4, at 0 of 5.** The answer moves too little and reverses too rarely, the error the planning probe read.

- **PD1074's held-out cohorts** (573 animals on 24 dates), as description, since the probe read them: the pause share 0.489 (z +0.31), the flip fraction 0.124 (−0.91) and the onset rate 2.20 per minute (−0.55). All pass.

- **W's minimum:**

  - on the fitted families, now the small lawn, kinematics and activity, it is **2.597** (v3: 2.967). The activity family's halves differ little, so adding it lowers the weighted RMS;
  - on the held-out families it is 2.047, v3's.

  W's registration cites the minimum for the family set it reads its arms on.

- **The 3-of-5 probability:** a faithful population passes all six families with probability 0.95 × 0.95 × 0.992 × 0.98 × 0.988 × 0.985 = 0.854, so a faithful controller passes the wild type at 3 of 5 with probability **0.975** (v3: 0.977).

## Changes to the registration before Part A

The design's order did not close, and its seeds overlapped, so two things were fixed before Part A was committed:

- **The onset decision has its own part, C.** The design had Part B record the onset rate's status before the oracle's 40 populations, which it also said decide it. Part B now records the fitted answer. The `oracle-check` stage reads the populations, and `--onset` refuses any choice the check does not support. Part C then records the decision with the decided bands, before the read.
- **The oracle's fit moved from 601 M to 610 M.** Its index spans 2,000,000 animal seeds per restart, so at 601 M restart 1 would have reused the adversary's 603 M block, the overlap v3's erratum records.

After the read, `verdict` stopped on a type error before writing anything. The bands record keeps each bar and its statistic count beside the family checks. The fix reads only the family checks, and a test now covers `verdict`. No rule changed.

## Seen before registration

Disclosed in Part A:

- **The planning probe** read PD1074's band and held-out cohorts and the four controllers.
- **The pause floor's derivation** read the still-worm library.
- **The activity bands,** built in process for the preflights by the registered functions, which the guarded build reproduced exactly.
- **An implementation smoke** on 8 animals (discarded seeds). The N2 and W code ran on PD1074, so N2's activity statistics were not computed.
- **The preflights** (`preflight.json`, discarded seeds):
  - the adversary, caught in 5 of 5, and in 5 of 5 without the onset rate;
  - a 60-generation oracle restart whose answer passed all three statistics in 5 of 5.

No rule was changed after any of these. The registered readings used fresh seeds (602–604 M and 610–612.32 M) and C.1's own populations (525 M).

## Cost

From `preflight.json`, on 16 workers:

- the activity bands, 17 s; the guarded build with its rule checks, a few minutes;
- the oracle, 4.9 s per generation on the small lawn, so its 200 registered generations took about 16 min;
- the adversary's 5 populations, 9 s;
- the read, a few minutes.

## Implications

- **C.1 is refitted on v4,** in its own change, before the fidelity table. Its fits must now reverse at the real rate on the lawn. The oracle shows C.1's model can reach the activity statistics. Its registration carries the 3-of-5 probability 0.975, and the design costs it at about 12–18 h.
- **C.2's next change fits against v4.** With activity in D_fit, a worm that does not run forward fails its own objective. The readout box and saturation report, the staged procedure and the slow-noise control's re-run on a responsive fit belong to it.
- **W** reads its fitted arms on v4's family set, with the minimum 2.597 on it.
- **Imported small-lawn tracks** carry the activity family (`data/benchmark/IMPORT.md`).
