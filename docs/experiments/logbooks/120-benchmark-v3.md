# 120: Phase 10 — B v3: Reorientation Graded from Positions, VALID, and C.1b's Reversal-Dominated Answers Fail It

**Status**: completed. **B v3 is VALID**, and `benchmark_grade.py` now grades on it by default. Every registered check holds:

- the bands' rule checks at v3's bar;
- Chen's round trip for the two new statistics;
- the reachability oracle: C.1's model, fitted to the new statistics alone, reaches them on 40 fresh populations;
- all 10 held-out López-Cruz batches;
- Chen's leave-one-recording-out validation;
- C.1b's four reversal-dominated answers, each caught in 5 of 5 populations;
- B.4's six negatives, v2's blind worm and C.1's v1 answer, each failing the wild type (and the open-loop four chemotaxis) in 5 of 5.

**What v3 adds.** v2 let a worm that reverses half the time off food pass local search. v3 grades how often a worm reorients, from positions alone, through three statistics: local search's early/late ratio on a log scale, and chemotaxis's onset rate and direction-flip fraction on Chen's tracks. No one of them catches every reversal-dominated answer, but together they catch all four. The share of time reversing is reported, not graded.

> | Registered check | Result |
> |---|---|
> | Rule checks at v3's bar (gate 0.90) | small lawn 0.945, kinematics 0.975, local search 0.991 |
> | Chen's round trip, onset rate and flip fraction | pass (z +0.11, +0.03) |
> | Reachability: the oracle's answer on 40 populations | pass (mean z +0.01, −0.03, −0.01; 100% within the bar) |
> | Held-out López-Cruz, log early/late, 10 batches | all pass (z −0.81 to +1.14) |
> | Chen leave-one-out, 6 recordings, allowance 1 | 0 fail, all present |
> | C.1b's restarts 2, 4, 6 and the κ probe, caught in 3 of 5 | each caught in 5 of 5 |
> | B.4's six negatives, v2's blind worm, C.1's v1 answer fail at 3 of 5 | each fails 5 of 5; the open-loop four fail chemotaxis 5 of 5 |
> | The lawn N2 control | carried from v2 (passed) |

**Date**: 2026-11-01.

**OpenSpec change**: `benchmark-v3`.

**Records**: [`supporting/120-benchmark-v3/`](supporting/120-benchmark-v3/):

- `launch.md` (Part A, Part B);
- `brief.md`, `probe.json`, `preflight.json` and `preflight_controls.json` (the planning readings and preflights, disclosed), with the probe's and preflight's scripts as run (`probe.py.txt`, `preflight.py.txt`);
- `identity.json`, `rebuild_check.txt` (v1 and v2 unchanged);
- `bands_v3_record.json`, `round_trip.json`, `oracle_fit.json`, `read.json`, `controls_v3.json`.

The bands are [`data/benchmark/bands_v3.json`](../../../data/benchmark/bands_v3.json) (SHA-256 `8af38429…`).

## Objective

C.1b's fits on v2 failed the wild type, and the addendum to [Logbook 119](119-c1b-fit.md) traced the failure to reversal-dominated locomotion. Off food the answers reverse 42–56% of the time, against real worms' 17.3% (Roberts' tracker). v2 let that through:

- **Local search graded only the curve's shape.** Its linear early/late band (4.21 ± 1.62) accepts any ratio from −0.66 to 9.09, so a worm reorienting 9.5 times per 2-min bin, flat, passed.
- **Backward travel is invisible from midbody positions.** A worm backing continuously draws the path of one going forward.
- **Nothing graded how often a worm reorients.**

Every later controller could use the same gap, so the benchmark is revised before C.1 is refitted or C.2 begins. The maintainer's decisions (2026-11-01):

- grade three position-based statistics, chosen by a discriminability probe;
- report the share of time reversing, without grading it;
- put the new statistics in the chemotaxis family;
- include a reachability oracle for them.

v1 and v2 stay frozen.

## The statistics

| Statistic | Family | What it reads | Band |
|---|---|---|---|
| log early/late | local search | log(mean reorientations 8–14 min after removal / 34–44 min), in place of the linear ratio | 1.393 ± 0.375, from López-Cruz's 17 band batches |
| onset rate (per min) | chemotaxis | the off-food detector's onsets over the kept seconds | 1.115 ± 0.943, from Chen's 6 standard recordings |
| flip fraction | chemotaxis | consecutive 1-s displacement pairs, both at least 0.03 mm, turning more than 135° | 0.0234 ± 0.0204, from the same recordings |

- **The flip floor** is v2's own (the on-lawn reversal rate's), above the Chen rig's per-axis jitter of 0.0087 mm.
- **D v3** has 21 statistics in five equal families, so z\* = 3.0381 (v2: 19 and 3.0078). Chemotaxis's five statistics weigh 1/25 each. D_fit is unchanged.
- **The bands.** Local search and chemotaxis are rebuilt at v3's prediction factor (k = 17 batches: 1.2205; k = 6 recordings: 2.0146). γ's scale grows 2.1%, since v2 kept v1's factor. Everything else is v2's, byte-identical.
- **The reported reversal share:** where steps carry primitive labels, the share of seconds in a reversal bout over the local-search and chemotaxis windows, beside Roberts' 17.34% of tracked frames in reverse. Imported tracks report none.

## Why these three

The discriminability probe (`probe.json`, discarded seeds) scored candidates on C.1b's 8 restarts and the κ probe against held-out real data. The registered adversaries (`read.json`, 5 populations each) show the same pattern:

| Adversary | Reversal share | log early/late z | onset rate z | flip fraction z | Caught by |
|---|---|---|---|---|---|
| C.1b restart 2 | 52–53% | −3.40 to −3.46 | +3.67 to +3.74 | +8.5 to +8.7 | all three |
| C.1b restart 4 | 50–51% | **−3.53 to −3.75** | +2.1 to +2.3 | +1.1 to +1.2 | log early/late alone |
| C.1b restart 6 | 36% | −2.6 to −2.7 | +2.9 | **+5.9 to +6.0** | flips alone |
| the κ probe | 52% | −3.35 to −3.38 | +3.8 | +8.7 to +8.9 | all three |

The bar is 3.038. Restart 4 backs up continuously: its onsets and flips stay within the bar, and only its flat search curve gives it away. Restart 6 decays enough to pass local search but turns back too often. Every adversary also fails the wild type in all 5 populations.

## The validity map

1. **The rule checks** pass at v3's bar: the small lawn 0.945 and kinematics 0.975 (v2's splits, against 0.94 and 0.96 at v2's bar), and local search 0.991.

2. **Chen's round trip** (`round_trip.json`): the authors' smoothed fragments with the model's jitter read an onset rate of 1.307 per minute against 1.201 raw (z +0.11), and a flip fraction of 0.0252 against 0.0247 (z +0.03).

3. **The reachability oracle.**

   - **The fit** (`oracle_fit.json`): C.1's 20-parameter model fitted by sep-CMA to the three new statistics alone. Restart 0 plateaued at generation 80 (final objective 0.127); restart 1 ran to the cap of 100 (0.090) and was taken.

   - **The check** (`read.json`, 40 populations at 471 M):

     | Statistic | Mean z | Share within z\* | z SD |
     |---|---|---|---|
     | log early/late | +0.008 | 100% | 0.17 |
     | onset rate | −0.027 | 100% | 0.015 |
     | flip fraction | −0.014 | 100% | 0.016 |

   - **Its reversal share** is 5.0–5.2%, so the new statistics can be met by a worm that is not reversal-dominated. The answer is fitted to these three statistics only: it dwells faster than it roams and steers away from the odour, so it says nothing about the other families.

   - **The detector-bias risk** of the design (a faithful worm's ratio sitting about 0.73× low through our detector, z ≈ −0.8) did not bind: C.1's model reaches the band's centre through the detector (mean z +0.01).

4. **Held-out López-Cruz:** all 10 held-out batches pass log early/late (z −0.81 to +1.14). γ reads −0.36 to +1.17, reported.

5. **Chen's leave-one-out:** every standard recording passes on all five statistics and is present (largest |z| 1.85), 0 failures against an allowance of 1. The allowance is recomputed at the nominal rate 1 − (1 − 0.05/21)⁵ = 0.0118.

6. **The adversaries** are each caught in 5 of 5 populations (the table above).

7. **The negatives** each fail the wild type in 5 of 5 populations, and the open-loop four fail chemotaxis in 5 of 5:

   | Negative | D | Onset rate z | Flip fraction z |
   |---|---|---|---|
   | `hmm` | 5.2 | +2.6 to +2.7 | +5.1 to +5.2 |
   | `poisson` | 6.1–7.5 | +2.5 | +6.2 to +6.3 |
   | `random_walk` | 7.4–7.6 | −0.9 | −0.9 |
   | `correlated_walk` | 7.4–7.5 | −0.7 | −0.7 |
   | `untrained_mlp` | 5.7–6.5 | +3.7 to +3.8 | +7.2 to +7.4 |
   | `untrained_connectome` | 4.4–5.4 | +4.6 | +6.6 to +6.8 |
   | v2's blind worm | 4.8–4.9 | +3.9 to +4.0 | +7.0 to +7.3 |
   | C.1's v1 answer | 6.1–6.2 | +5.5 to +5.6 | +4.9 to +5.7 |

8. **The lawn N2 control** is carried from v2 (passed), since the lawn families and kinematics are unchanged and v3's bar is more lenient.

**Recomputed and reported:**

- **W's minimum:**

  - on the fitted families it is 2.967, identical to v2's;
  - on the held-out families it is 2.047 (v2: 2.015). Local search alone gives 2.326 and chemotaxis alone 2.140.

  W's registration must read its arms on the family set whose minimum it cites.

- **The 3-of-5 probability:** a faithful population passes all five families with probability 0.945 × 0.95 × 0.991 × 0.975 × 0.988 = 0.857. A faithful controller therefore passes the wild type at 3 of 5 with probability **0.977** (v2: 0.955). C.1's refit on v3 carries this.

## Seen before registration

Everything below was disclosed in Part A:

- **The planning brief** read held-out data:
  - López-Cruz's 10 held-out batches;
  - Roberts' five cohorts;
  - Calhoun's 47 worms;
  - Chen's 8 recordings.
- **The probe** read C.1b's 9 answers.
- **The preflights,** on discarded seeds:
  - the adversary gate (each adversary caught in 5 of 5);
  - a 60-generation oracle restart (its answer passed 5 of 5);
  - the controls preflight: the round trip; one oracle generation, timed; the negatives on 2 populations each, all failing.
- **A dry build** of the bands printed the rule checks and the leave-one-out.

No rule was changed after any of these. The registered readings used fresh seeds (470–475 M).

## Cost

- **The bands:** built in about 1.5 min.
- **The round trip:** a few minutes.
- **The oracle's fit:** 180 generations at about 24 s each on 16 workers (local search and chemotaxis only), 73 min.
- **The read:** 7 min.

## Implications

- **C.1 is refitted on v3,** in its own change. Its fits can no longer pass by reversing half the time. Its registration carries the 3-of-5 probability 0.977, and the oracle shows the new statistics are reachable by C.1's model.
- **The reported share is the watch.** If a later fit is caught by it (well above 17%) but passes the three statistics, grading it is proposed in a new change, as Part A registers.
- **C.2 and C.3** fit D_fit, which v3 leaves unchanged. Whether C.2's fit also reads the new statistics is C.2's decision.
- **Imported tracks** are graded on log early/late for local search. The chemotaxis family, and so both new chemotaxis statistics, still cannot be imported (`data/benchmark/IMPORT.md`).
