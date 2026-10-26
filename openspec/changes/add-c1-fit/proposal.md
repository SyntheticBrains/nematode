## Why

F.0 has cleared for C.1 ([Logbook 112](../../../docs/experiments/logbooks/112-cma-recovery.md)). sep-CMA, configured as C.1's real fit, reproduces worms near the wild type to within their own noise. C.1 is the phase's first real fit, the mechanistic baseline every later controller is compared with. The roadmap's risk row names its failure: if C.1 cannot pass the wild type, the primitives or the benchmark are wrong, and that is fixed before C.2.

D37 also requires the mutants to be encoded per controller, in a table registered before any fit. C.1's fit is the first fit, so the table registers now.

The maintainer's decisions of 2026-10-26:

- **the full table in C.1's Part A**, with rows for C.1, C.2 and C.3. C.2's and C.3's rows are written at gene and neuron level, independent of C.2's encoder, which F.0b-ii builds;
- **C.1's mutants as structural knockouts only:** an encoding removes a mechanism and has no free magnitude. A mutant with no counterpart in C.1's model is recorded as not encodable, with no prediction;
- **tph-1's leaving left out of the graded set:** its three band dates disagree with its held-out dates ([Logbook 110](../../../docs/experiments/logbooks/110-benchmark-controls.md)). It is reported beside the graded set;
- **8 restarts**, the best chosen by its own objective.

## What Changes

- **The encoding table** (`supporting/113-c1-fit/encodings.md`), committed in Part A:

  | Mutant | Graded (Scheer's small lawn, against the controller's own wild type) | C.1 | C.2 and C.3 |
  |---|---|---|---|
  | tph-1 | roaming fraction (more); leaving reported beside | switching loses its intake dependence: a_intake = b_intake = 0 | `modulator_knockout_genes: [tph-1]`: no serotonin is made |
  | pdfr-1 | roaming fraction and leaving, by magnitude | not encodable: C.1 has no PDF signal | `modulator_knockout_genes: [pdfr-1]`: PDFR-1's couples removed |
  | tax-4 | roaming fraction (less); leaving by magnitude | chemosensory gradient responses removed: κ_w = β_r = β_o = 0 | the encoder's input to tax-4's sensory classes is zeroed |

  It also registers the **`unidentified` rule**. Among the restarts whose answers pass the wild type, a mutant prediction whose verdict differs between restarts is `unidentified`, not passed or failed. C.4 grades the table, and this change records what C.4 needs.

- **C.1's fit** (`scripts/analysis/c1_fit.py`, reusing `cma_recovery`'s fit loop):

  - **objective:** the benchmark's own D on the 11 wild-type statistics, against the band cohorts at n = 200, for a fresh 200-animal population each evaluation;
  - **settings:** 8 restarts of sep-CMA from the box centre (σ₀ 0.2, population 16), with F.0b-i's plateau rule and cap of 210;
  - **the answer:** each restart's answer is re-simulated 10 times, and graded on one registered 200-animal population (animals 5501–5700).

- **The wild-type verdict:** the benchmark's own rule. The answer of the restart with the lowest final objective **passes the wild type when every family passes** on the registered population, local search with its presence test.

  - **On a pass,** C.1 enters C.4.
  - **On a fail,** the failing statistics are the diagnosis, and the risk row's pivot goes to the maintainer before C.2.

- **Reported beside the verdict:**

  - every restart's D and its own verdict;
  - the restart spread on real data, for W;
  - the parameters, read only where F.0b-i recovered them (weathervane steering and the search time).

- **C.1's observation-size note is resolved.** The bands' n = 200 scales come from a bootstrap over recording dates at 200 animals. A 200-animal fit against them therefore already carries the real worms' between-date spread, and a point fit needs no extra noise term.

- **`cma_recovery.run_fit`** gains an optional objective. Its default, the distance to a truth's observation, is unchanged.

## Capabilities

### New Capabilities

- `controller-fit`: fitting a controller to the wild type, its verdict, and the mutant encoding table it registers.

## Impact

- **Code:** `scripts/analysis/c1_fit.py`, and the optional objective in `cma_recovery.run_fit`, with tests.
- **Records:** `docs/experiments/logbooks/supporting/113-c1-fit/`:
  - `launch.md` and `encodings.md`, committed before any fit;
  - `fit.json`;
  - Logbook 113.
- **Cost:** F.0b-i's registered fits are this configuration with a different target: 30–34 s per generation of 16, plateauing at 83–184 generations. So the 8 restarts take about 11 h, and at most about 16 h at the cap. Grading and re-simulation add minutes.
- **Seeds:**
  - CMA seeds 5741–5748, and grading animals 5501–5700, both in C.1's band (5501–5756);
  - fit animals at `110_000_000 + 200·k + i`;
  - a smoke on CMA seed 5992 at `99_100_000`, discarded.
