## 1. The script

- [x] 1.1 `scripts/analysis/fidelity_table.py`:
  - **`grade`:** C.1 (v3)'s restarts 0, 4 and 6 on their registered populations with tph-1 and tax-4, the reproduction check, each restart's verdict, the `unidentified` rule, and restart 6 on 10 more populations;
  - **`table`:** the rows from the bound records and the grade record, as JSON and Markdown;
  - **`smoke`:** on discarded seeds.
- [x] 1.2 Tests, each naming its spec scenario: the reproduction check, the simulation path (stubbed), the rule, the rows and every copied cell, the held-out cell, the git guards and the seeds.
- [x] 1.3 The smoke (99,870,000, discarded): it ran end to end in 53 s.
- [x] 1.4 The reproduction preflight: the wild type alone at the registered populations, no mutant, 0 mismatches.

## 2. Registration

- [x] 2.1 Launch record 137, Part A: the reading, its rules, the reproduction check, the table, the verdict map, the seeds, the guards, the bound records and the disclosures.
- [x] 2.2 An independent review of the change, the script and Part A, then commit. *(The review found one blocking ambiguity in the restart rule, a failed reproduction that would still have simulated the mutants, and gaps in the tests: 21 of 40 planted bugs survived. All are fixed. The wild type now runs alone first, and the planted bugs re-run against the revised tests are caught, except one that is equivalent on the committed record.)*

## 3. The reading

- [x] 3.1 `grade`: `grade.json`. *(2026-11-27, 4.2 min: reproduced, 0 mismatches, the repeat identical. tph-1 `fail` and tax-4 `fail`, every restart 0 of 5.)*
- [x] 3.2 `table`: `fidelity_table.json` and `fidelity_table.md`.

## 4. Close-out

- [x] 4.1 Logbook 137 and its index row. Update:
  - the roadmap: the C row, the exit criterion, and the D37 note of 2026-11-27, whose "carries no prediction" for C.1's tax-4 row is corrected, since C.1's rows are its v3 answer's predictions;
  - the tracker (C.4);
  - AGENTS.md;
  - the CHANGELOG.
- [ ] 4.2 The full suite, then `pre-commit run --all-files` after `git add -A`, then `openspec validate --strict`. Archive the change and open the PR.
