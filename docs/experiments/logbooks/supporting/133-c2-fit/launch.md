# Launch record: C.2 fitted to the wild type and its mutants by a staged procedure

**Part A registered 2026-11-10**, change `fit-c2`. It is committed before the readiness pilot, F.0b-ii and the registered fit, and fixes every rule, threshold and seed in the change's design. Part B, added after the pilot and before F.0b-ii's preflight, records the pilot. Part C, added after F.0b-ii and before the registered fit, records the recovery. Neither changes a rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

## Part A: the rules

### The controller and its arm

- **The controller:** the encoded connectome under the procedure's options (`c2_encoder_probe.PROCEDURE_SETTINGS`):
  - per-ligand tying, 137 parameters;
  - the soft-capped readout with pool centres;
  - the graded box with the inhibition floor at 0.09;
  - PDFR-1's and SER-5's cited signs;
  - slow (OU) speed noise.
- **The arm:** `benchmark_grade.ARMS["connectome_c2_following"]`. Its configs are the encoded arm's with `continuous.on_food_reversal_speed: follows_run_speed` as their only difference ([Logbook 132](../../132-speed-following-reversals.md)). The body is in the config.
- **Every stage, reading and control runs on it,** at 200 animals a candidate and population 48.

### The objective

For a candidate's record against a truth T (the bands' centres; in F.0b-ii, the truth's observation):

- **the ranking score** on the stage's families: 1,000 × the statistics undefined in one record + the weighted RMS of 10·asinh(z/10), z on the bands' scales at n = 200;
- **the mutants**, where the stage fits them:
  - each mutant's term is the RMS of 10·asinh(z/10) over the statistics the encoding table grades it on, z its difference from the wild type on the same animals against T's, capped at 10, undefined at the cap;
  - with n_f families and n_m mutants, each mutant weighs n_f / n_m, so the mutants together are half;
- **the saturation guard:** + 500 when over half the motor neurons sit at abs(h) > 0.99 in over half the steps;
- **the graded responsiveness guard:** + 300·(1 − s/0.5) when s < 0.5, s the smaller of the speed and reversal outputs' 5th-to-95th-percentile spreads over 8 recorded animals. Both guards read the wild type's simulation only;
- **the failure probability,** in the final stage only: + 10·Σ Φ((abs(z) − bar) / SD) over every defined wild-type statistic and every graded mutant statistic, with the bands' bars (3.07809 and 2.24140) and SD from the noise step.

**The graded statistics** are the encoding table's (`113-c1-fit/encodings.md`, bound):

| Mutant | Graded | Reported only |
|---|---|---|
| tph-1 | the roaming fraction, direction more | leaving (exits per on-lawn hour) |
| pdfr-1 | the roaming fraction and leaving, by magnitude | |
| tax-4 | the roaming fraction, direction less; leaving, by magnitude | |

### The procedure

sep-CMA (`CMA_diagonal`), bounds [0, 1]. Each stage's answer is its mean (`xfavorite`) clipped to the box.

| Stage | Fits | Families | Mutants | Start | Generations | CMA seed |
|---|---|---|---|---|---|---|
| **1** | the 77 parameters of the wiring, sensory, readout and noise blocks with the pool centres; the other 60 at 0.5 | D_fit's four | none | the graded box's centre, 0.5 everywhere; σ₀ 0.2 | 150 | run's first |
| **2a** | all 137 | D_fit's four | none | stage 1's answer, the speed noise's timescale set to 1 s; σ₀ 0.1 | 300 | second |
| **2b** | all 137 | all six | the run's, at half | 2a's search state | 200 | third |
| **noise** | none: 2b's answer on 30 populations | all six | the run's | | | |
| **final** | all 137 | all six | the run's, at half, plus the failure probability | 2b's search state | 50 | fourth |

- **The search state passes** from 2a to 2b and from 2b to the final stage: a new strategy at the previous answer, with the previous σ and per-parameter steps (`CMA_stds` = stds/σ).
- **Fixed budgets:** every stage runs its count. The plateau rule (`cma_recovery.plateau_reached`) is read on the final stage and reported, never a stop.

### The noise step

- 2b's answer is graded on 30 populations of 200 without presence.
- Each statistic's SD of z between them (ddof 1, undefined populations dropped) is floored at 0.05. A statistic defined in fewer than 2 populations takes the floor.

### Readouts and state

- **Every 10 generations, at each stage's generation 0 and at its last,** the stage's mean is graded on the same 200 readout animals, on all six families and the three mutants, without presence.
- **Recorded:** every statistic's value and z, the families passing, time on the lawn, the output spreads, each mutant's reading, and the distance to the run's truth.
- **State is saved atomically after every generation.** A resumed run refuses a saved state whose settings differ from its own, and a stored candidate is refused at another vector.

### The readiness pilot

- **The registered procedure,** run once on discarded seeds before any registered stage, configured exactly as the fit.
- **Its go rule:** its answer, graded with presence on 5 populations, passes the wild type in at least 3, and tph-1 and tax-4 each in at least 3.
- **Its kill rules,** each on the mean of the stage's last three readouts. The pilot stops and returns to the maintainer if:
  - stage 1's smaller output spread is below 0.5, or it holds the lawn less than 75% of the time;
  - 2a's D_fit is above 2.5.
- **In the registered runs these checkpoints are reported, not stops.**
- **If the go rule fails,** the procedure is diagnosed and this Part A is amended before anything registered runs.
- **Reported:** 10 more populations' pass rates, the cost per stage, the final stage's plateau reading.

### F.0b-ii, C.2's recovery control

- **The truth:** the pilot's answer.

- **The observation:** the truth on all six assays and the three mutants at 200 animals (q = 0). It replaces the bands' and the mutant bands' centres in the objective, with the scales unchanged.

- **The recovery distance** D_rec: the registered capped D on the six families, with each mutant's RMS capped z over its graded statistics, against the truth's differences, together half.

- **The preflight** (`recovery-preflight`, run once). Gain = (prior D − D) / (prior D − floor):

  | Quantity | From |
  |---|---|
  | floor | the median D_rec from the observation to re-simulations 1–10 |
  | oracle | the median over re-simulations 11–20 |
  | prior D | the median over 200 uniform draws of the box, `default_rng(5805)` |
  | box centre | the median over 10 re-simulations of 0.5 everywhere |
  | wild-type reference | a record at the bands' and the mutant bands' centres, reported |

  It passes only if all four hold. Otherwise the recovery returns to the maintainer.

  - prior D − floor ≥ **1.0**;
  - the oracle's gain ≥ **0.85**;
  - the box centre's gain ≤ **0.25**;
  - **stability:** no statistic flips between defined and undefined across re-simulations 1–10.

- **The recovery fit:** the registered procedure from the box centre against the observation, every stage, budget, guard and noise step identical.

- **The gate:** the answer re-simulated 10 times. **The gain at its median D_rec must be at least 0.75.** If it falls short, no registered fit runs until the maintainer decides.

- **The limit, stated in advance:** the truth sits beside the wild type, so the gate cannot tell recovering it from fitting the wild type. F.0b-ii tests whether the procedure reaches the region from the box centre, and the spread between runs. It does not test identifying the truth.

- **Reported:** each stage's readouts against the truth, the answer's distance from the truth by block, and the answer's own wild-type grade against the bands.

### The registered fit and its verdict

- **The answer** is the final stage's mean at generation 50, clipped to the box.
- **The registered grade:**
  - the answer on 5 registered populations of 200, every family with presence;
  - each mutant on the small lawn on its population's own animals, graded on its table statistics at the mutant bar in its resolved direction.
- **The verdict: C.2 passes when the wild type passes in at least 3 of the 5, and tph-1 and tax-4 each in at least 3 of the 5.**
  - A population passes the wild type when every family passes, with presence for local search and chemotaxis.
  - pdfr-1 is graded and reported. It is not part of the verdict.
  - **The verdict is in-sample:** every graded statistic and both resolved mutants are fitted.
- **The verdict map:**
  - **pass:** C.2 enters C.4, and W's arms run this procedure, with W's compute decided from the measured cost before W registers;
  - **fail:** the failing statistics and each stage's readouts are the diagnosis. The roadmap's C.2 risk row goes to the maintainer, and no W arm registers on this procedure.
- **Reported beside, never re-read as the verdict:**
  - 10 more populations' pass rates;
  - the held-out reading;
  - every readout;
  - the plateau reading;
  - the reversal share against Roberts' 17.3%;
  - the cost per stage;
  - **the restart spread:** the median over g < 15 of the capped D between the pilot's population g and the registered answer's population g, on W's fitted families and on all six.
- **Description-only outcomes, named in advance:**
  - **a pass with a poor held-out reading** (the answer's median distance to a held-out half above the band reference's 95th percentile): C.2 passes, and C.4 states the held-out distance as a limit;
  - **a pass with the leave-one-out's tax-4 failing:** C.2 passes, and C.4's tax-4 row reads as fitted, not predicted.

### The held-out reading

W's rule (`benchmark_v4_controls.w_minimum_v4`):

- 1,000 splits of a date set into halves (seed 2034);
- PD1074's band scales at the half's size rounded to 25;
- the small lawn, kinematics and activity's graded statistics.

For each split, the distance to each half from three things:

- the other real half, which is W's minimum;
- each of the answer's 15 graded populations;
- each of 15 populations of 200 animals drawn without replacement from PD1074's band cohorts (`default_rng(2048)`), what a faithful simulator would match.

**The sets:**

- **N2's 26 unread held-out dates,** W's minimum's own (its real-against-real 95th percentile reproduces 2.597, which a test checks);
- **PD1074's 24 held-out dates,** beside it. These cohorts are the source of the primitives' on-food reversal values and the speed coupling's knots, and B.4 read them.

**Reported** as each distance's median and 95th percentile, as description. No new simulation. The uniform well's single held-out date, chemotaxis (no held-out recording) and López-Cruz's held-out batches are not read.

### The tax-4 leave-one-out

- **After the registered fit,** the same procedure on its own seeds with tax-4 out of the objective. tph-1 and pdfr-1 are then together half, and the failure probability omits tax-4.
- **tax-4 is read at every readout,** and its answer's tax-4 on 15 populations is reported beside the registered fit's.
- **It gates nothing.**

### Seeds

Each run of the procedure has 20,000,000 animal seeds from its base B (animals `B + offset + i`):

| Within a block | Use |
|---|---|
| B + 4,000,000·(s − 1) + 200·(48·(n − 1) + m) + i | stage s = 1–4, generation n, member m; 2a's 14,400 candidates reach 2,880,000 |
| B + 16,000,000 + i | the readout animals, the same at every readout |
| B + 17,000,000 + 200·p + i | the noise populations, p < 30 |
| B + 18,000,000 + 200·g + i | the grade: g < 5 the verdict (or the go rule), 5 ≤ g < 15 the pass rate; in the recovery, the answer's re-simulations, g < 10 |

| Run | Base | CMA | Status |
|---|---|---|---|
| the readiness pilot | 900,000,000 | 9504–9507 | discarded |
| F.0b-ii's recovery fit | 920,000,000 | 5801–5804 | registered |
| the truth's observation (q = 0) and re-simulations (1–20) | 940,000,000 + 200·q + i | | registered |
| the prior draws | 941,000,000 + 200·j + i, j < 200 | `default_rng(5805)` | registered |
| the box centre | 942,000,000 + 200·q + i, q < 10 | | registered |
| the registered fit | 960,000,000 | 5811–5814 | registered |
| the tax-4 leave-one-out | 980,000,000 | 5821–5824 | registered, description |
| slow noise's control | 888,000,000 + i, reference episode 888,000,200 | | the control's reading |
| the stand-in preflight | 885,000,000–887,100,999 | `default_rng(9512)` | discarded |
| the smoke | 99,880,000–99,889,999: 8 animals, population 6, two generations a stage | 9508–9511 | discarded |

The tests check that every block is disjoint from the others and from the seed table's earlier blocks, which reach 880,001,999.

### Disclosures

Seen before this record was committed, all on discarded seeds:

- **Logbook 131's probes and curriculum** (`131-c2-clean-start/probes/`, each script as run): the route this procedure fixes, adapted after each reading. The stage-1 subset, the 1-s timescale, the guards, the failure probability, its weight and floor, and every stage's budget come from it. Procedure B differs from the route as the design's table states:

  - stage 1 at 200 animals and population 48;
  - 250 generations after 2a, not 500;
  - the mutants from 2b;
  - the search state carried from 2a;
  - the package's speed-following knots throughout;
  - one noise measurement of 30 populations;
  - tph-1 on the roaming fraction alone.

- **Logbook 132's confirmation:** Logbook 131's answer passes 10 of 10 under the arm's body.

- **The stand-in preflight** (`probes/standin_preflight.py.txt` and `.json`, `probes/standin_gate_reach.py.txt` and `.json`), Logbook 131's answer standing in for the truth:

  | Quantity | Stand-in | Bar |
  |---|---|---|
  | floor | 0.36 | |
  | the oracle's gain | 0.996 | ≥ 0.85 |
  | prior D (20 draws) | 6.41 (4.37–8.59) | separation 6.05 ≥ 1.0 |
  | the box centre's gain | 0.074 | ≤ 0.25; fails only for a prior D above 7.47 |
  | the wild-type reference's gain | 0.872 | reported |
  | definedness flips | none in 8 | none |

  B02's answer, from another lineage, gains 0.80 and 0.81. The truth's own lineage gains 0.96–0.97. The bars were set before these were read, and are unchanged by them.

- **Slow noise's positive control** (`speed_noise_control.json`), run under its registered rule before this record. Its result is below.

- **The spec review** (independent reviewer and author, 2026-11-10). It found tph-1's leaving graded against the table, and the control owed. Both are fixed here.

- **The script's checks:**

  - 27 tests cover every spec scenario, with 18 deliberate bugs in `c2_fit.py` each caught by its test (`probes/mutants.py.txt`, `probes/mutants.log.txt`);
  - the held-out reading's real halves reproduce W's minimum to 1e-12;
  - the smoke ran every stage, the noise step, the grade and the reading on discarded seeds.

- **The maintainer's decisions (2026-11-10):**

  - procedure B;
  - a readiness pilot first;
  - F.0b-ii at one truth before the fit, its limit stated;
  - symmetric fit splits;
  - a full tax-4 leave-one-out.

- **Not seen:** every registered simulation of F.0b-ii (920–942 M), the registered fit (960 M) and the leave-one-out (980 M).

### Slow noise's positive control

`speed_noise_control.json` (run 2026-11-10 at HEAD `52212e43`; seeds 888,000,000 + i, reference episode 888,000,200): **passes.**

Logbook 131's answer under the procedure settings on the speed-following arm, at each timescale and without the OU noise, on the same 200 animals, under the registered rule (`c2_encoder_probe.read_speed_noise_control`):

| Check | Result | Bar |
|---|---|---|
| the speed output responds: its median over one reference episode, before the soft cap | −0.92 (capped, −0.86) | within ±2 |
| the roaming 90th percentile's rise above the shortest timescale | +1.50 at 2 s, +3.45 at 8.7 s, **+4.90 at 30 s**, +4.65 at 100 s, +4.71 at 300 s (band scales) | ≥ 3.0 at some timescale |
| at the shortest timescale, every kinematics statistic against independent noise | 0.04–0.62 band scales | ≤ 2.0 |

Slow noise reaches the window statistics it was built for, on an answer whose speed output responds. Logbook 123's failure was the worm's, which did not run. The rule's responsiveness bound reads the output before the soft cap. Here both readings sit well inside it.

### Cost

At the measured rates on 16 workers (77.7 s a generation on two assays, 309.5 s on six with the mutants, readouts included):

| Stage | Estimate |
|---|---|
| a run of the procedure | about 31.5 h: stage 1 3.2 h, 2a 6.5 h, 2b 17.2 h, the final stage 4.3 h, the noise step and the grade under 0.5 h |
| the pilot, F.0b-ii's fit, the registered fit | one run each |
| `recovery-preflight` | 231 simulations, about 30 min |
| the leave-one-out | about 28.5 h |
| in all | about 124 h |

### The guards

- **The manifest** (`manifest.json`, bound): the SHA-256 of `c2_fit.py` and every analysis script it imports, the arm's four assay configs, `bands_v4.json`, `primitives.json`, the neuromodulator data files, the encoding table, and one digest over the package's source.
  - Every guarded stage refuses unless every digest holds and v4's bands are the validated ones.
  - Every record states HEAD and the manifest's digest.
- **`pilot`** refuses unless this Part A is committed and unchanged, with the manifest, the encoding table and the control's record bound and the control passing.
- **`recovery-preflight`** needs Part B, with `pilot.json` bound and its go rule passing. It runs once.
- **`recovery`** needs the preflight's record committed, unchanged and passing.
- **`run`, `grade`, `read` and `loo`** need Part C, with both recovery records bound and the gate passing.
- **`smoke`, `manifest` and `speed-noise-control`** are unguarded. The control and the smoke use discarded or the control's own seeds, and the smoke reads band-cohort dates only.
