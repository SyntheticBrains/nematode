## Why

Phase 10's exit criteria need C.1–C.4 read and the fidelity table published. C.1 and C.2 have both been read, and C.3 is not run in Phase 10. The table that sets them beside the negatives and the real-worm controls does not exist yet.

Almost every cell is already in a committed record:

- **C.2's registered answer:** its wild type, mutants, held-out reading and pass rates ([Logbook 133](../../../docs/experiments/logbooks/133-c2-fit.md)).
- **C.1 (v3)'s answer on v4:** the wild type fails in 0 of 5 ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)).
- **B v4's controls:** eight negatives, each failing in 5 of 5, and the real-worm controls, all passing ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)).
- **C.1's v4 refit:** stopped at its pilot ([Logbook 136](../../../docs/experiments/logbooks/136-c1-v4-fit.md)).

One reading is missing. C.1 (v3) was fitted to the wild type alone ([Logbook 121](../../../docs/experiments/logbooks/121-c1-v3-fit.md)), so its mutant rows are predictions, and they have never been graded. Under the registered encoding table (`supporting/113-c1-fit/encodings.md`), they are its only predictions in C.4.

## What Changes

- **`scripts/analysis/fidelity_table.py`,** with three stages:
  - **`grade`:** the one new registered reading.
    - C.1 (v3)'s restarts 0, 4 and 6, the three that pass v3's wild type, run on their own registered populations (525,000,000 + 200·g + i, g < 5) on the arm they were fitted on (`mlp`).
    - tph-1 and tax-4 run on the small lawn on the same animals.
    - The wild type must first reproduce its committed v3 distances and passes exactly. Otherwise nothing is read.
    - Restart 6, the chosen answer, also runs 10 more populations (555,000,000), as description.
  - **`table`:** assembles every row from the committed records and the grade record.
  - **`smoke`:** runs on discarded seeds.
- **The rows:**
  - C.1 (v3), whose mutants are predicted and pdfr-1 not encodable;
  - C.2, whose mutants are fitted;
  - C.3, not run;
  - the eight negatives;
  - the real-worm controls and W's minimum;
  - C.1's v4 pilot answer, as description.
- **The columns:** the wild type on v4 by family, D and D_fit, the reversal share, the fitted families, the recovery control, each mutant's kind and verdict, and the held-out reading.
- **The rules:**
  - C.1's mutant verdict follows the encoding table: a restart passes a mutant in at least 3 of 5 populations, and then the `unidentified` rule runs across restarts 0, 4 and 6.
  - C.1's held-out cell is "not read". The benchmark-controls exception covers only a registered fit's `read` stage.
  - C.2 against C.1 is description, with its conditions stated.
- **Launch record 137, Part A,** committed before `grade` runs.
- **Logbook 137,** with the close-out.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `controller-fit`: C.4's fidelity table. It adds the reading of C.1 (v3)'s predicted mutants under the encoding table's rule, and the assembly of the table from committed records.

## Impact

- **Code:** `scripts/analysis/fidelity_table.py` (new) and its tests. No package code changes.
- **Records:**
  - `docs/experiments/logbooks/supporting/137-fidelity-table/`: the launch record, the grade record, the table as JSON and Markdown;
  - Logbook 137 and its index row;
  - the roadmap (the C row, the exit criterion, a D37 correction);
  - the tracker (C.4);
  - AGENTS.md and the CHANGELOG.
- **Cost:** about 25 C.1 evaluations with the mutants, a few minutes on 16 workers.
