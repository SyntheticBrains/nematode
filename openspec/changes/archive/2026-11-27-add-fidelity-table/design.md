## Context

C.4 is the fidelity table: every controller of Phase 10 and the benchmark's negatives, graded on the current benchmark (v4) and on the mutant set under the per-controller encoding table registered before any fit. C.2 against C.1 is a secondary contrast. The maintainer decided on 2026-11-25 that C.2 enters as a failing in-sample row and C.3 does not enter, and on 2026-11-27 that C.1 enters through its v3 answer.

## Goals / Non-Goals

**Goals:**

- Grade C.1 (v3)'s predicted mutants once, under rules fixed before the run.
- Assemble the table so every cell traces to a committed record.

**Non-Goals:**

- No new fit, and no re-reading of a registered verdict. C.1 (v3)'s wild type on v4 and C.2's verdict are reported as they were read.
- No held-out read.
- No ranking of the two controllers.

## Decisions

1. **C.1's row is its v3 answer, on the arm it was fitted on.** That is restart 6 of `supporting/121-c1-v3-fit/fit.json` on `benchmark_grade.ARMS["mlp"]`, the pooled on-food reversal, the body C.1 (v3) was fitted and graded on.

   - Moving it to the speed-following arm would grade a controller that was never fitted there.
   - The bodies differ from C.2's, and the contrast states it.

2. **The mutants are graded on the registered populations, on the same animals as the wild type.** Each mutant's difference is from the controller's own wild type (D34).

   - Restarts 0, 4 and 6 each run on 525,000,000 + 200·g + i, g < 5, which are Logbook 121's registered populations.
   - The wild type is simulated on every v4 assay, and tph-1 and tax-4 on the small lawn, on the same animals.
   - The differences are graded on v4's mutant bands at n = 200. They are v2's, carried byte-identical, so the version does not change a mutant's reading.

3. **The wild type must reproduce before any mutant is simulated.** The grade runs the wild type alone first. For each restart and population, the v3 grade's D and wild-type pass must equal the committed `population_D` and `population_passes`, and restart 6's v4 D must equal Logbook 124's.

   - D must match to within 10⁻⁹, and the pass exactly.
   - If any differs, `grade` records the mismatch and simulates no mutant, and `table` refuses.
   - Otherwise every population runs again with the mutants. Its registered wild type must equal the first pass's exactly.
   - This shows the code still simulates the registered answer.

4. **The rule is the encoding table's.**

   - A population passes a mutant when all of the mutant's graded statistics pass in it. A restart passes when at least 3 of its 5 registered populations pass.
   - The restarts that pass the wild type are 0, 4 and 6, read on v3, the benchmark the answer was fitted and verdicted on. The tracker recorded them on 2026-11-03.
   - `c1_fit.mutant_verdict` gives the prediction: `pass`, `fail` or `unidentified`.
   - The verdict is stated with C.1 (v3)'s v4 wild-type failure (0 of 5, Logbook 124) beside it. pdfr-1 is not encodable.
   - Which restarts pass v4's wild type is not known: only the chosen one has been graded on v4, and it fails. Steps 1–4 of the rule use the restarts the fit registered as passing, so the table uses those. The grade also records restarts 0 and 4 on v4, as description.

5. **Description beside the rule:**

   - **More populations:** restart 6 on 10 more populations (555,000,000 + 200·g + i, g < 10), for pass rates over 15, as C.2's.
   - **tph-1's leaving,** reported against both reference values: the band dates' +0.07 per on-lawn hour and B.4's single held-out read of 4.0×.
   - **C.1's v4 pilot answer,** from `pilot.json` (discarded seeds).

6. **The held-out cell.** C.2's held-out reading is reported as read in its fit. C.1's cell is "not read", by the maintainer's decision of 2026-11-27.

   - The benchmark-controls exception covers a registered fit's `read` stage, and C.4 is not a fit.
   - C.1 (v3) already fails v4 on band-cohort populations.

7. **C.2 against C.1 is description.** The table sets them side by side and names the conditions:

   - C.2's recovery control (F.0b-ii) failed;
   - the bodies differ;
   - C.2's mutants are fitted and C.1's predicted;
   - each is graded on its own registered populations, so the populations are not paired.

   No verdict is read on which is closer, by the maintainer's decision of 2026-11-27.

8. **Every row of the table comes from a committed record,** bound by its SHA-256 in the launch record: C.1 (v3)'s fit record, C.2's fit record, B v4's read record, Logbook 136's pilot record and this change's grade record. `table` recomputes nothing that a record holds.

9. **Guards:**

   - `grade` refuses unless launch record 137's Part A is committed, its bound files are unchanged, and the script and the package match Part A's commit.
   - `grade` and `table` each write their record once.
   - The smoke runs on 99,870,000 + 200·g + i with one restart and two populations, and reads no record.

## Risks / Trade-offs

- **Seen before the run:** Logbook 136's separation probe ran restart 6 on the speed-following arm with `r_dwell_on_food` at 0.0224, and both mutants failed in 10 of 10. That is a different body and one more parameter, but it makes a failing prediction likely. Part A discloses it.
- **A reproduction failure** would leave C.1's mutant cells unread. The guard reports it rather than reading predictions from a controller it cannot show it simulates.
- **Five populations per restart:** a mutant's verdict at the 3-of-5 threshold is noisy. The 10 more populations for restart 6 show its pass rate beside it.
