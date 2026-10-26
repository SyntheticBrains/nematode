## Context

- **C.1's model** is `BatchedMechanistic`, with 20 parameters on F.0a's box (`npe_recovery.PRIOR`). No range uses search data.
- **The simulator:** `npe_recovery.simulate` runs 200 animals on the three wild-type assays and grades them with `benchmark_grade.grade_populations`. Its `D` is the benchmark's distance to the bands at each family's size, n = 200 here.
- **F.0b-i** ([Logbook 112](../../../docs/experiments/logbooks/112-cma-recovery.md)):
  - sep-CMA from the box centre (σ₀ 0.2, population 16) with the plateau rule and a cap of 210 recovers near-wild-type worms (median gain 0.991);
  - fits ran at 30–34 s per generation and plateaued at 83–184 generations;
  - only weathervane steering and the search time are recovered as parameters;
  - two restarts on the same data differ by a median D of 1.12.
- **The bands** (`data/benchmark/bands.json`):
  - wild-type centres and scales at n = 200, from band cohorts only;
  - mutants are graded on Scheer's small lawn by their roaming-fraction and leaving differences from their own wild type, at |z| ≤ 2.24;
  - only tph-1's higher roaming and tax-4's lower roaming have registered directions.
- **The connectome's data layer** knocks out genes by name (`modulator_knockout_genes`). A producer gene, mapped through `neuromodulator_layer.MONOAMINE_GENES` (tph-1 → serotonin), holds its ligand at zero. A receptor gene (pdfr-1) removes its couples.
- **tph-1's leaving:** its band dates show Δ +0.07 per on-lawn hour, and B.4's single read of its held-out dates showed 4.0× more ([Logbook 110](../../../docs/experiments/logbooks/110-benchmark-controls.md)).

## Goals / Non-Goals

**Goals**

- C.1 fitted to the wild type by the method F.0b-i validated, with a registered verdict.
- The mutant encoding table D37 requires, registered before any fit, for every controller.
- The restart spread on real data, for W.

**Non-goals**

- Grading the mutants, which C.4 does under this table.
- C.2's encoder, which is F.0b-ii.
- Reading any held-out data.

## Decisions

01. **The objective.** f(z) = `simulate(z)["D"]`: the benchmark's D over the 11 wild-type statistics (families weighted equally, capped at 10), against the band cohorts at n = 200, for 200 fresh animals per evaluation. Local search is fitted, since D37 holds it out only for C.2 and C.3.

02. **sep-CMA, as F.0b-i validated it.**

    - **Settings:** start at the box centre with σ₀ 0.2, population 16, bounds [0, 1]. The answer is `xfavorite`, and the loop never consults `es.stop()`.
    - **The plateau rule:** generations are counted from 1; a fit stops at the first n ≥ 40 with S_n > S\_{n−20} − 0.05.
    - **The cap:** 210 generations.
    - **Restarts:** 8 (r = 0–7), on CMA seeds 5741 + r.

03. **The plateau criterion (D38).** A restart stopped by the cap has not met it, so it cannot be chosen. **The chosen restart** is the plateaued restart with the lowest S over its last 20 generations. If no restart plateaus, the run returns to the maintainer before any verdict is read.

04. **Grading.**

    - **The registered grade:** every restart's answer is graded on five registered populations of 200 animals each, g = 0–4, at animals `126_000_000 + 200·g + i`, on the three assays, through `grade_populations`. Common animals across restarts make their verdicts comparable.
    - **Re-simulation:** each answer is also re-simulated 10 times with fresh animals, for its median D to the bands and the restart spread.

05. **The wild-type verdict (registered).** A population passes when every family passes on it:

    - each statistic at |z| ≤ 2.84;
    - local search only where its presence test holds.

    The chosen restart **passes the wild type** when **at least 3 of its 5 registered populations pass**.

    **The verdict's oracle**, from B.4's registered failure rates for real wild-type data (0.075 on the small lawn, 0.05 on the uniform well, 0.038 for López-Cruz): a faithful population passes all three families with probability 0.925 × 0.95 × 0.962 = 0.845, so on a single population a perfect C.1 would fail 15.5% of the time. With 3 of 5, it passes with probability 0.971 (protocol principle 4).

    **The verdict map:**

    - **pass:** C.1 enters C.4;
    - **fail:** the failing statistics are the diagnosis, and the roadmap's risk row ("the primitives or the benchmark are wrong; fixed before C.2") goes to the maintainer.

    **Reported beside:**

    - D and D_fit;
    - each statistic's z;
    - every restart's verdict, stop and D;
    - the restart spread, the median pairwise D between the answers' re-simulations, paired by repeat;
    - every parameter, with weathervane steering and the search time marked as the ones F.0b-i recovered.

06. **The encoding table (registered, `encodings.md`), for every controller.**

    **The mutants are graded, at C.4, against the controller's own wild type on Scheer's small lawn:**

    - **tph-1:** roaming fraction (direction: more), at the bar of 2.24. Its leaving is **left out** and reported beside, against the band dates' reference and B.4's recorded held-out value, with no held-out data read again. The bar is unchanged.
    - **pdfr-1:** roaming fraction and leaving, by magnitude. Its search is reported, not graded.
    - **tax-4:** roaming fraction (direction: less) and leaving, by magnitude.

    **C.1's encodings** are structural knockouts, with no free magnitude:

    - **tph-1:** a_intake = b_intake = 0. Switching loses its dependence on ingestion, which NSM's serotonin reports (Rhoades 2019; Flavell 2013). Satiety's dependence is kept, since it runs through other signals.
    - **pdfr-1:** **not encodable.** C.1 has no PDF signal, so it makes no pdfr-1 prediction, a stated limit of the mechanism.
    - **tax-4:** κ_w = β_r = β_o = 0. The head-sweep and dC/dt responses are removed (tax-4 is the cGMP channel of the chemosensory neurons; Komatsu 1996). The edge response to falling food contact is kept, since food contact is the lawn density under the worm, not a gradient.

    **C.2's and C.3's encodings**, the same architecture:

    - **tph-1:** `modulator_knockout_genes: [tph-1]`, so no serotonin is made;
    - **pdfr-1:** `modulator_knockout_genes: [pdfr-1]`, so PDFR-1's couples are removed;
    - **tax-4:** the encoder's input to every tax-4-expressing sensory class is zeroed. These are AWC, AWB, ASE, ASG, ASI, ASJ, ASK and AFD (Komatsu 1996), and URX, AQR, PQR and BAG (Gray 2004; Hallem 2011), which receive no input in these assays. The row binds whatever encoder F.0b-ii builds.

    **The `unidentified` rule:**

    - for a controller, take its restarts whose answers pass the wild type (3 of 5);
    - if a mutant's verdict differs among them, the prediction is `unidentified`;
    - otherwise it is their shared verdict;
    - with one passing restart, its verdict;
    - with none, the chosen restart's verdict, reported with the wild-type failure stated.

07. **The launch record and its guard.** `launch.md` (decisions 1–6) and `encodings.md` are committed before any fit simulation. `c1_fit.py run` refuses unless `launch.md` is committed and unchanged, and `encodings.md` is byte-identical to its version in that commit (B.4's guard). `cma_recovery`'s guard is generalised to take the launch record's path, its default unchanged, so both scripts share one implementation.

08. **Readiness.**

    - **The verdict is the benchmark's own rule,** which B.4 showed readable: held-out real worms pass under its allowance, and six negatives fail.
    - **No positive control is involved.** There is no oracle for a real-data fit, and F.0b-i was the method's control.
    - **The cap of 210** comes from F.0b-i's pilot on a synthetic near-wild-type truth. On real data the plateau criterion (decision 3) is the safeguard.
    - **Cost** comes from F.0b-i's registered fits, which are this configuration with a different target (protocol principle 9). Typically 8 × about 160 generations × 32 s ≈ 11 h, and at most 8 × 210 × 34 s ≈ 16 h.
    - **The spec review gates the launch,** since the run is over two hours.

09. **Seeds.**

    - Fit animals are `110_000_000 + 200·k + i`, with k = 10,000·r + 16·(n − 1) + p for evaluations, and 10,000·r + 5,000 + q (q < 10) for an answer's re-simulations. The largest k is 75,009, so the animals stay below 125.1 million.
    - The five grading populations are at `126_000_000 + 200·g + i` (g < 5). C.1's band, 5501–5756, stays reserved for its posterior-predictive checks.
    - The smoke uses CMA seed 5992 at `99_100_000`, discarded.
    - All of these are disjoint from every band and earlier scheme.

10. **The fit loop.** `cma_recovery.run_fit` gains `objective: Callable[[dict], float] | None`. With `None`, it keeps its F.0b-i behaviour: the distance to `observation`. C.1 passes `lambda record: record["D"]`. Checkpoints, markers and resumption are unchanged.

## Risks / Trade-offs

- **[C.1 fails the wild type]** → the registered diagnosis and the roadmap's pivot. That is a result, not a defect: the fit has passed its recovery control.
- **[A restart sticks, as truth 2324's did]** → 8 restarts. Only plateaued restarts are chosen, and their spread is reported.
- **[The cap carried from F.0b-i is short for real data]** → the plateau criterion decides. A capped restart cannot be chosen, and if none plateaus, the maintainer decides before any verdict.
- **[tph-1's encoding acts on parameters F.0b-i did not recover]** → a_intake and b_intake (median errors 0.34 and 0.18) can differ between restarts, so zeroing them can predict anything from no change to a large one. The `unidentified` rule catches a prediction that depends on the restart. An `unidentified` tph-1 is an expected outcome, not a surprise.
- **[tax-4's encoding leaves C.1's roaming unchanged]** → a structural prediction that may fail tax-4's lower roaming. It is registered before the fit and graded as it falls.
- **[The tax-4 class list rests on the literature, not vendored data]** → cited in the table. The O₂ and CO₂ classes are inert in these assays either way.
