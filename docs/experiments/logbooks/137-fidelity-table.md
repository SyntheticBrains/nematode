# 137: Phase 10 — The Fidelity Table: No Controller Reproduces the Worm With Its Mutants; C.1 (v3)'s Two Predicted Mutants Fail, and C.2's Passing Mutants Are All Fitted

**Status**: `completed` — **C.4 is read, and the fidelity table is published.** No controller of Phase 10 reproduces both the wild type and its mutants on benchmark v4.

- **C.1 (v3)'s predicted mutants:** the one new registered reading. Both fail at every restart that passed v3's wild type, 0 of 5 populations each.
  - **tph-1** goes the wrong way: removing serotonin's intake signal lowers roaming, where real worms roam more.
  - **tax-4** leaves the small lawn 5.7 to 15.8 more times an hour than its wild type, where real worms leave 0.31 more.
- **C.2:** passes the wild type in 5 of 5 registered populations, and tax-4 and pdfr-1 in 5 of 5. Every one of those mutants was fitted, its tph-1 fails, and its recovery control failed ([Logbook 133](133-c2-fit.md)).
- **The rest of the table:** C.1 (v3)'s wild type fails v4 in all 5 populations at each of the three restarts, on the on-lawn activity family. Every negative fails, and the real-worm controls pass.
- **Unresolved:** with C.2's control failed and the two controllers on different bodies, the table cannot say which controller is closer to the worm. By the maintainer's decision, that contrast is description only.

> | | C.1 (v3) | C.2 |
> |---|---|---|
> | Wild type on v4 | **0 of 5** | **5 of 5** |
> | tph-1 | predicted: **fail** | fitted: **fail** (0 of 5) |
> | pdfr-1 | not encodable | fitted: pass (5 of 5) |
> | tax-4 | predicted: **fail** | fitted: pass (5 of 5) |
> | Recovery control | passed (0.985) | **failed** (0.635) |

**Date**: 2026-11-27.

**OpenSpec change**: `add-fidelity-table`. It adds requirements to `controller-fit`: C.1 (v3)'s predicted mutants, graded once under the encoding table, and the table assembled from committed records.

**Pre-registration**: [supporting/137-fidelity-table/launch.md](supporting/137-fidelity-table/launch.md), Part A, committed (`a584fdad`) before `grade` ran.

## Objective

C.4 is the fidelity table: Phase 10's controllers and the benchmark's negatives graded on the current benchmark and on the mutant set, under the per-controller encoding table registered before any fit ([encodings.md](supporting/113-c1-fit/encodings.md)). C.2 against C.1 is the secondary contrast.

The maintainer had already decided who enters:

- **C.2:** as a failing in-sample row ([Logbook 133](133-c2-fit.md)).
- **C.3:** does not enter.
- **C.1:** through its v3 answer ([Logbook 136](136-c1-v4-fit.md)).

Almost every cell was already in a committed record. One was not. C.1 (v3) was fitted to the wild type alone ([Logbook 121](121-c1-v3-fit.md)), so its tph-1 and tax-4 rows are its only predictions, and they had never been graded. **The question: do C.1 (v3)'s predicted mutants pass, and what does the table show?**

## Method

Launch record 137 holds the full design. In brief:

- **The new reading:** C.1 (v3)'s restarts 0, 4 and 6, the three that pass v3's wild type. Each ran on its own registered populations (525,000,000 + 200·g + i, g < 5) on the body it was fitted on (`benchmark_grade.ARMS["mlp"]`, the pooled on-food reversal).
- **Two passes:**
  - First the wild type alone. It had to reproduce its committed v3 distances and passes, and restart 6 its committed v4 grade, before any mutant was simulated.
  - Then the wild type on every v4 assay and each mutant on Scheer's small lawn, on the same animals.
- **The grading:** each mutant's differences from its own wild type, graded on v4's mutant bands at n = 200, under the registered encodings:
  - tph-1 zeroes the switch's dependence on intake.
  - tax-4 zeroes both weathervane gains and the dC/dt reversal and omega gains.
- **The rule:** a population passes a mutant when all its graded statistics pass together, and a restart passes at 3 of 5. The prediction is the encoding table's `unidentified` rule across the three restarts.
- **Description:** restart 6 also ran 10 more populations (555,000,000).
- **The table:** every other row is copied from its committed record, and each count the table can recompute is checked against its record.
  - C.1's held-out cell is "not read".
  - C.2 against C.1 is description, by the maintainer's decisions of 2026-11-27.

## Results

### The reproduction check: passes

All 15 registered wild-type populations reproduce their committed v3 D within 10⁻⁹ and their passes exactly, and restart 6 its v4 D and passes (0 mismatches). The second pass's wild type equals the first's ([grade.json](supporting/137-fidelity-table/grade.json), `reproduction`). So the code still simulates the registered answers.

### C.1 (v3)'s predicted mutants: both fail

| Restart | tph-1 Δ roaming (z) | tph-1 passes | tax-4 Δ roaming (z) | tax-4 Δ exits per on-lawn hour (z) | tax-4 passes |
|---|---|---|---|---|---|
| 0 | −0.200 to −0.186 (−9.18 to −8.92) | 0 of 5 | −0.077 to −0.051 (+0.89 to +1.46) | +5.74 to +6.27 (10, capped) | 0 of 5 |
| 4 | −0.141 to −0.118 (−8.03 to −7.58) | 0 of 5 | −0.045 to −0.014 (+1.60 to +2.29) | +14.85 to +15.82 (10) | 0 of 5 |
| 6 | −0.200 to −0.174 (−9.18 to −8.68) | 0 of 5 | −0.053 to −0.019 (+1.42 to +2.18) | +13.24 to +14.13 (10) | 0 of 5 |

The bands' centres are +0.27 for tph-1's roaming difference, and −0.12 and +0.31 for tax-4's two. The bar is |z| ≤ 2.24.

- **tph-1's prediction is `fail`.** All three restarts agree.
  - At every restart, removing the intake signal makes the worm roam less, by 0.12 to 0.20. Real tph-1 worms roam more.
- **tax-4's prediction is `fail`.** All three restarts agree.
  - Its roaming difference passes in 13 of 15 populations across the three restarts. Its lawn exits fail in all 15.
  - This is the mechanism [Logbook 136](136-c1-v4-fit.md)'s probe found at restart 6 on the speed-following body: C.1's worm holds the lawn by weathervane steering, which tax-4's encoding removes.
- **Restart 6 on 10 more populations** (description): the wild type, tph-1 and tax-4 each pass in 0 of 10, so 0 of 15 overall ([grade.json](supporting/137-fidelity-table/grade.json), `more_populations`).
- **tph-1's leaving** (reported, not graded): restart 6's Δ exits per on-lawn hour is −0.29 to −0.44. That is against the band centre of +0.07 and B.4's single held-out reading of +4.04.
- **Restarts 0 and 4 on v4** (description): both fail v4's wild type in all 5 populations. Like restart 6, each fails the on-lawn activity family in every population, though restart 4 passed v3 in 5 of 5.

### The fidelity table

The table is [fidelity_table.md](supporting/137-fidelity-table/fidelity_table.md), and every figure is from [fidelity_table.json](supporting/137-fidelity-table/fidelity_table.json):

| | C.1 (v3) | C.2 |
|---|---|---|
| Body | the pooled on-food reversal | the speed-following on-food reversal |
| Fitted | the wild type on v3's five families; no mutant | all six families with tph-1, pdfr-1 and tax-4 at half |
| Recovery control | its reduced recovery: gain 0.985, passed | F.0b-ii: gain 0.635, failed |
| Wild type on v4 | 0 of 5 | 5 of 5 (15 of 15) |
| … small lawn | 3 of 5 | 5 of 5 |
| … kinematics | 5 of 5 | 5 of 5 |
| … activity | **0 of 5** | 5 of 5 |
| … uniform well | 5 of 5 | 5 of 5 |
| … local search | 5 of 5 | 5 of 5 |
| … chemotaxis | 5 of 5 | 5 of 5 |
| Median D (D_fit) | 2.06 (2.45) | 1.47 (1.63) |
| Median reversal share (real worms 17.3%) | 3.4% | 5.5% |
| tph-1 | predicted: fail | fitted: fail, 0 of 5 |
| pdfr-1 | not encodable | fitted: pass, 5 of 5 |
| tax-4 | predicted: fail | fitted: pass, 5 of 5 |
| tph-1's leaving, Δ exits per on-lawn hour (reported) | median −0.40 | median +0.03 |
| Held-out dates | not read | N2: median 2.71, against real halves' 95th percentile 2.60; PD1074: 2.00, against 1.37 |

- **C.3:** not run in Phase 10.
- **B v4's eight negatives:** each fails the wild type in 5 of 5. Their median D runs from 4.30 (the untrained connectome) to 7.72 (the correlated random walk).
- **The real-worm controls pass:** N2's 26 unread dates and PD1074's 24 held-out dates, both on activity.
- **W's minimum:** the 95th percentile of the distance between two halves of real dates is 2.60 on the fitted families, and 2.05 on local search and chemotaxis.
- **C.1 (v4)'s pilot answer** is in the table's description rows: it passes tph-1 in 5 of 5 on discarded seeds, and the wild type and tax-4 in 0 of 5 ([Logbook 136](136-c1-v4-fit.md)).

### C.2 against C.1 (description)

C.2 passes v4's wild type and C.1 (v3) does not, and C.2's median D is lower: 1.47 against 2.06, on populations that are not paired. The table names four conditions on any comparison:

- C.2's recovery control failed, so its fit's reach is unvalidated.
- The bodies differ. C.2's on-food reversal follows its speed ([Logbook 132](132-speed-following-reversals.md)), and C.1 (v3)'s is pooled.
- C.2's mutants are fitted and in-sample, and C.1's predicted.
- C.1 (v3) was fitted before v4 existed, so v4's activity family, which it fails, was never in its objective.

On the mutants the two fail differently. C.2 cannot move tph-1 even when fitted to it, and C.1 (v3) predicts tph-1 the wrong way round. C.2, fitted to tax-4, holds the lawn. C.1's tax-4 leaves it at all three answers here and at every point Logbook 136 probed.

## Corrections made in the open

- **An independent review before Part A** found real problems. All were fixed before the commit:
  - **Blocking:** the restart rule could be read per statistic or per population. It now says all of a mutant's graded statistics must pass in the same population.
  - **A failed reproduction would still have simulated and recorded the mutants.** The wild type now runs alone first.
  - **Weak tests:** 21 of 40 planted bugs survived the first tests. They clustered in the simulation path, the reproduction check's indexing, the git guards and the copied cells.
    - Re-planted against the revised tests, every one is caught except one. That one is equivalent on the committed record: N2's failure list is empty and its pass is true.
  - **Smaller fixes:**
    - N2's line now says its scope is activity;
    - the reversal share is read as a value;
    - the table's recounts are checked against their records;
    - the pin is the last change to Part A's text;
    - two data files are bound;
    - the preflight's output is committed.
- **The first smoke printed mutant outcomes** on its two discarded populations, both failing. Part A discloses it. The revised smoke prints only whether the wild type repeats.
- **After the PR review** (no figure changed):
  - **The Markdown table omitted the description row.** `fidelity_table.json` holds C.1 (v4)'s pilot answer, but `markdown()` did not render it. It now does, and `fidelity_table.md` was re-rendered from the unchanged JSON. The re-render adds only that section.
  - **The two guards used different pins.** The bound-record check pinned to the last commit touching `launch.md`, and the code check to the last commit changing Part A's text. Both now use Part A's commit. When `grade` and `table` ran, `a584fdad` was the only commit touching `launch.md`, so the two pins were the same commit.
  - **This logbook and the index said the wild type "fails v4 in 0 of 5",** which inverts the reading: it passes in 0 of 5, failing all 5 populations at each restart. The wording is corrected here, in the index, the tracker and [Logbook 136](136-c1-v4-fit.md).
- **A correction to the roadmap:** the D37 note of 2026-11-27 said that C.1's tax-4 row "carries no prediction" without the leave-one-out. C.1's rows are its v3 answer's predictions, so the note now says there is no leave-one-out prediction from a v4 answer.

## What this establishes, and what it does not

**It establishes** that C.1 (v3)'s registered predictions for tph-1 and tax-4 fail, at all three restarts that passed v3's wild type. The rows are read once, under rules fixed before the run, on a wild type shown to reproduce exactly.

- In C.1's model, tph-1's encoding moves roaming the wrong way.
- tax-4's encoding makes the worm leave the lawn, which matches what Logbook 136's probe found for restart 6.

It also publishes the fidelity table. On benchmark v4, no Phase 10 controller passes the wild type and every encodable mutant, and none has a passing mutant prediction.

**It does not establish**:

- **No ranking:** which controller is closer to the worm. The contrast is description, under four named conditions.
- **Nothing beyond the model:** that C.1's mechanism class cannot predict these mutants. The predictions come from three answers of one fitted model, under its registered encodings.
- **No held-out reading for C.1:** its held-out dates were not read.
- **No explanation of C.2's pass:** C.2's wild-type pass may not reflect a reachable fit. Its recovery control failed, and its mutant passes are in-sample.

**What it conditions.** Phase 10's synthesis receives the table as published. The phase's headline is fidelity, and on it the record is:

- a benchmark that real worms pass and every negative fails;
- a connectome controller that matches the wild type and two of its three mutants only by fitting to them;
- a mechanistic baseline whose two predictions both fail.

## Registered consequences

- **C.1 (v3)'s predicted tph-1:** *met*. Read: `fail`, all three restarts 0 of 5.
- **C.1 (v3)'s predicted tax-4:** *met*. Read: `fail`, all three restarts 0 of 5.
- **The reproduction check:** *met*. 0 mismatches, and the repeat was identical.
- **C.4, the fidelity table:** *met*. Published as [fidelity_table.md](supporting/137-fidelity-table/fidelity_table.md) and [fidelity_table.json](supporting/137-fidelity-table/fidelity_table.json).
- **C.1's held-out reading:** *unmet-with-reason*. It was not read, by the maintainer's decision of 2026-11-27, since the benchmark's controls allow a held-out read only inside a registered fit.
- **C.3's row:** *unreachable-with-reason*. C.3 is not run in Phase 10, by the maintainer's decision of 2026-11-25.
- **C.2 against C.1:** *met*, as description, with no verdict.
- **The roadmap's exit criterion "C.1–C.4 read and the fidelity table published":** *deferred-with-destination* (the Phase 10 synthesis). C.1, C.2 and C.4 are read. The synthesis assigns the criterion its status with C.3's.

## Artefacts

In [supporting/137-fidelity-table/](supporting/137-fidelity-table/):

- [launch.md](supporting/137-fidelity-table/launch.md): Part A, the rules, seeds, guards, bound records and disclosures.
- [grade.json](supporting/137-fidelity-table/grade.json): the reproduction check, every population's wild type on v3 and v4, every mutant's differences and readings, the predictions and the 10 more populations.
- [fidelity_table.json](supporting/137-fidelity-table/fidelity_table.json) and [fidelity_table.md](supporting/137-fidelity-table/fidelity_table.md): the table, with every source's SHA-256.
- [probes/reproduction_preflight.py.txt](supporting/137-fidelity-table/probes/reproduction_preflight.py.txt) and [its output](supporting/137-fidelity-table/probes/reproduction_preflight.txt): the wild-type-only check before Part A.
- `scripts/analysis/fidelity_table.py` (`grade`, `table`, `smoke`), with its tests in `test_fidelity_table.py`.
