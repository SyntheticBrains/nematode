# Tasks: The 8a synthesis

Phase 8 task **S8a**. The decisions taken before this change are in `design.md`.

## Logbook

- [ ] 1. **Logbook 076, the 8a synthesis:**

  - the status walkthrough (Decision A), every 8a criterion and open thread with one status and a
    destination;
  - block V in one sentence (Decision B);
  - what 8a established, what it did not, and each negative's cause;
  - the A.5 package inventory;
  - the reproducibility statement (Decision E);
  - the D20 gate and D21 (Decision C).

  Every figure is cited to the committed logbook, CSV or JSON it comes from; nothing is computed
  afresh.

## Records

- [ ] 2. **The roadmap:**
  - the Phase 8 row reads **8a complete / 8b pending**;
  - the 8a synthesis exit criterion is ticked;
  - the SHOULD line and the overshoot row carry their statuses (B.2 carried, with its reason);
  - **D21** is added to the decisions table, with dated notes at C.1 and C.1e;
  - the Phase 8 minimum-viable success level is marked met.
- [ ] 3. **The tracker:**
  - 8a status COMPLETE, and 8b ready to start (gate GO);
  - S8a ticked;
  - A.3, A.5, B.2a, B.2b, M.5, M.6 and M.7 each given a status and destination;
  - the open threads listed with their destinations.
- [ ] 4. **The index row, and `README.md`'s status line.**

## Close-out

- [ ] 5. **Verify:**
  - every figure in Logbook 076 traces to a committed file;
  - every 8a criterion carries exactly one status in the logbook, the roadmap and the tracker, and
    the three agree;
  - every new link resolves;
  - `openspec validate --strict`;
  - `git add -A`, then pre-commit, judged by its exit code.
- [ ] 6. **Archive and open the PR.**
