# Launch record: C.1, the mechanistic baseline fitted to the wild type

**Registered 2026-10-26**, change `add-c1-fit`. It is committed, with [the mutant encoding table](encodings.md), before any fit simulation runs. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

## The model and the objective

- **The model:** `BatchedMechanistic`, 20 parameters, on F.0a's box (`npe_recovery.PRIOR`; no range uses search data).

- **The objective** for z is the benchmark's own D (`npe_recovery.simulate(z)["D"]`):

  - over the 11 wild-type statistics, with families weighted equally and z capped at 10;
  - against the band cohorts at n = 200;
  - for 200 fresh animals on `scheer_lawn`, `uniform_lawn_scheer` and `food_removal_lopezcruz` per evaluation.

  Local search is fitted, since D37 holds it out only for C.2 and C.3.

- **The observation's size:** the bands' n = 200 scales come from a bootstrap over recording dates at 200 animals. A 200-animal fit against them already carries the real worms' between-date spread, so no noise term is added. That resolves C.1's note in the tracker.

## The fit

These are F.0b-i's validated settings ([Logbook 112](../../112-cma-recovery.md)), through `cma_recovery.run_fit`:

- **sep-CMA:** start at the box centre, σ₀ 0.2, population 16, bounds [0, 1]. The answer is `es.result.xfavorite`, and the loop never consults `es.stop()`.
- **Restarts:** 8 (r = 0–7), on CMA seeds 5741 + r.
- **The plateau rule:** generations are counted from 1. A restart stops at the first n ≥ 40 with S_n > S\_{n−20} − 0.05, where S_n is the mean median objective over generations n − 19 to n.
- **The cap:** 210 generations, set by F.0b-i's pilot on a synthetic near-wild-type truth. On real data the plateau criterion is the safeguard.
- **The plateau criterion (D38):** a restart stopped by the cap has not met it, so it cannot be chosen. **The chosen restart** is the plateaued restart with the lowest S over its last 20 generations. If none plateaus, no verdict is read, and the run returns to the maintainer.

## Grading and the verdict

- **The registered grade:** every restart's answer is graded on five registered populations of 200 animals, g = 0–4 (animals `126_000_000 + 200·g + i`), on the three assays through `grade_populations`.
- **A population passes** when every family passes: each statistic at |z| ≤ 2.84, and local search only where its presence test holds.
- **The verdict (registered):** **the chosen restart passes the wild type when at least 3 of its 5 populations pass.**
- **The verdict's oracle:**
  - B.4's registered failure rates for real wild-type data are 0.075 on the small lawn, 0.05 on the uniform well and 0.038 for López-Cruz;
  - a faithful population therefore passes all three families with probability 0.925 × 0.95 × 0.962 = 0.845;
  - on a single population, a perfect C.1 would fail 15.5% of the time;
  - with 3 of 5 it passes with probability 0.971.
- **The verdict map:**
  - **pass:** C.1 enters C.4;
  - **fail:** the failing statistics are the diagnosis, and the roadmap's risk row ("the primitives or the benchmark are wrong; fixed before C.2") goes to the maintainer.

**Reported, not gated:**

- every restart's stop, final objective, verdict and per-population D and D_fit;
- each answer's median D over 10 fresh re-simulations;
- the restart spread: the median D between plateaued restarts' answers' re-simulations, paired by repeat, over every pair;
- the chosen answer's parameters, with weathervane steering and the search time marked as the two F.0b-i recovered.

## Seeds

- **Fit animals:** `110_000_000 + 200·k + i`:
  - evaluations at k = 10,000·r + 16·(n − 1) + p;
  - an answer's re-simulations at k = 10,000·r + 5,000 + q (q < 10);
  - the largest k is 75,009.
- **Grading:** the five populations at `126_000_000 + 200·g + i`. C.1's band (5501–5756) stays reserved for its posterior-predictive checks.
- **The smoke:** CMA seed 5992, animals at `99_100_000`, discarded.

All of these are disjoint from every band and every earlier scheme.

## Readiness

- **The verdict** is the benchmark's own rule. B.4 showed it readable: held-out real worms pass under its allowance, and six negatives fail. Its oracle is computed above from committed rates.
- **There is no positive control to preflight.** F.0b-i was the method's control, and a real-data fit has no oracle answer.
- **Cost:** F.0b-i's registered fits are this configuration with another target: 30–34 s per generation of 16, plateauing at 83–184 generations. That gives about 11 h, and at most about 16 h at the cap.

## The guard

`scripts/analysis/c1_fit.py`'s `run`, `grade` and `read` refuse unless this record is committed and unchanged, with `encodings.md` byte-identical to its version in the commit that last touched this record.
