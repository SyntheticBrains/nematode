# Launch record: C.1b, the mechanistic baseline refitted to the wild type on benchmark v2

**Part A, registered 2026-10-31**, change `fit-c1-v2`. It is committed before the pilot and before any registered simulation this record scores. Part B, appended after the pilot and before any fit, records the pilot's cap, projection and reachability and changes no rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

The template is C.1's registration on v1 ([launch record 113](../113-c1-fit/launch.md)). The mutant encoding table there (`113-c1-fit/encodings.md`) is unchanged, and is bound by this record's guard: C.4 grades C.1b's mutant predictions under it.

## Part A

### Why C.1b

- **C.1's v1 answer fails benchmark v2** on all five populations, with D 6.1–6.2 ([Logbook 115](../../115-benchmark-v2.md)).
- **F.0b-i on v2 clears sep-CMA for C.1's model** on v2's objective ([Logbook 117](../../117-cma-recovery-v2b.md), median gain 0.873). It ranked definedness first, then a compressed distance.

### The model and the objective

- **The model:** `BatchedMechanistic`, 20 parameters, on F.0a's box (`npe_recovery.PRIOR`). No range uses search data.
- **A simulation:** `cma_recovery_v2.simulate_v2`: 200 fresh animals on v2's four assays through its measurement models, graded without presence.
- **The objective** (`c1b_fit.BandObjective`) is the ranking score against the real bands at n = 200: **1,000 × u + D_c**.
  - **u** counts the 19 graded statistics the candidate leaves undefined.
  - **D_c** is the weighted RMS, the five families equal and split equally within each, of 10·asinh(z/10), for z = (value − centre) / scale on each statistic's band.
  - The registered D, capped, is reported beside every answer.

### sep-CMA and the stop rule

- **Settings:** F.0b-i's: the box centre, σ₀ 0.2, popsize 16, bounds [0, 1]. `es.stop()` is never consulted, and the answer is `xfavorite`.

- **Restarts:** 8, r = 0–7, on CMA seeds **5749 + r** (5749–5756, the next eight of C.1's band).

- **The plateau rule, guarded twice** (`c1b_fit.defined_stop`). A restart stops at the first generation n ≥ 40 where all three hold:

  - S_n > S\_{n−20} − 0.05;
  - σ ≤ 0.2;
  - **no generation from n − 39 to n has a median of 1,000 or more.**

  Otherwise it stops at the cap.

- **The cap:** set by the pilot (below) and recorded in Part B.

- **Choosability:** a restart stopped by the cap has not met the plateau criterion and cannot be chosen.

### The reachability pilot (before Part B; discarded seeds)

- **The run:** one restart, CMA seed **5740**, animals at 430,000,000, for 300 generations whatever the rule says.
- **What it records:**
  - **G_p**, the first generation at which the doubly guarded rule holds;
  - the registered D, the median over 10 re-simulations, at G_p, at ⌈1.5·G_p⌉ and at 300.
- **The cap:** min(300, ⌈1.5·G_p⌉). It is 300 if the rule never holds, and 300 if D at ⌈1.5·G_p⌉ exceeds D at 300 by more than **0.27**. That tolerance is F.0b-i's 0.05 of gain times the median prior − floor of F.0b-i on v2's four truths, 5.42 (`116-cma-recovery-v2/preflight.json`).
- **Reachability:** the answer at 300 is graded with presence on 5 discarded populations (animals at 432,000,000). If fewer than 3 pass, the change returns to the maintainer with the failing statistics before Part B, and `run` refuses unless the maintainer's decision is registered. This is not a verdict.
- **The projection:** 8 × (cap + 1) × the median seconds per generation over generations 101–300. Over **20 h**, the maintainer decides before launch.

### Selection (apart from the verdict)

- **Grading:** every plateaued restart's answer is graded with presence on **3 selection populations** (animals at 421,000,000).
- **The choice:** the restart passing the most is chosen. A tie goes to the lower median D over the three, then to the lower r.
- **No plateau:** if no restart plateaus, no verdict is read, and the run returns to the maintainer.

### The verdict

- **Grading:** every restart, plateaued or not, is graded with presence on **5 registered populations** (animals at 420,000,000).
- **A population passes** when every family passes: each statistic at |z| ≤ z\* = 3.0078, with local search and chemotaxis passing only where present.
- **The verdict (registered):** **the chosen restart passes the wild type when at least 3 of its 5 registered populations pass.**
- **The verdict's oracle** (Logbook 115, Part B):
  - a faithful population passes all five families with probability 0.818;
  - so a faithful controller passes 3 of 5 with probability 0.955.
- **The verdict map:**
  - **pass:** C.1 (v2) enters C.4;
  - **fail:** the failing statistics are the diagnosis, and the roadmap's risk row ("the primitives or the benchmark are wrong; fixed before C.2") goes to the maintainer.

**Reported, not gated:**

- every restart's stop, final score, selection passes, registered grades and median D over 10 re-simulations;
- the restart spread over plateaued pairs;
- the chosen answer's parameters, with weathervane steering and the search time marked as the two F.0b-i recovered;
- every restart's registered grades, which C.4's `unidentified` rule reads.

### Seeds

Animals are `base + 200·k + i`, for i < 200.

| Base | k | Use | Range |
|---|---|---|---|
| 400,000,000 | 10,000·r + 16·(n − 1) + p; answers 10,000·r + 5,000 … 5,009 | restart r's generation n, member p; its answer, re-simulated | 400.0–415.0 M |
| 420,000,000 | g < 5 | the registered populations | 420.0–420.001 M |
| 421,000,000 | 3·r + s, s < 3 | restart r's selection populations | 421.0–421.005 M |
| 430,000,000 | 16·(n − 1) + p; 5,000, 5,100, 5,200 (+ 0–9) | the pilot and its answers, discarded | 430.0–431.04 M |
| 432,000,000 | g < 5 | the pilot's reachability populations, discarded | 432.0–432.001 M |

These ranges are disjoint from each other and from every earlier scheme (the seed table in the Phase 10 tracker's design.md).

**Discarded:** the smoke, CMA seed 5999, animals at 99,050,000, with its grade at 99,060,000.

### Disclosures

Seen before this record was committed:

- **The window restriction** (change `fit-c1-v2`, task 1). `benchmark_grade.simulate_v2` now builds and measures only each family's analysis window. Against a baseline recorded on `main`'s code before the edit, all 55 of the identity check's jobs are byte-identical (`identity.json`), so no registered quantity changes.
- **The smoke** (discarded seeds): two generations ran, with medians of 1,008.7 (one undefined statistic) and then 17.9, and the 2-generation answer was graded once. Nothing about a real-data fit's outcome was read.

### The guard

`scripts/analysis/c1b_fit.py`:

- `pilot` refuses unless this record is committed and unchanged, with `113-c1-fit/encodings.md` byte-identical;
- `run`, `select`, `grade` and `read` also require Part B, with `pilot.json` byte-identical to its version in the commit that last touched this record.
