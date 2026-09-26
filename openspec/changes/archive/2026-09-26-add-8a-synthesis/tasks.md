# Tasks: The 8a synthesis

Phase 8 task **S8a**. The decisions taken before this change are in `design.md`.

## Logbook

- [x] 1. **done: Logbook 076.** Original scope: **Logbook 076, the 8a synthesis:**

  - the status walkthrough (Decision A), every 8a criterion and open thread with one status and a
    destination;
  - block V in one sentence, each clause naming its draw, with the thermal cell's untested rungs
    stated (Decision B);
  - what 8a established, what it did not, and each negative's cause;
  - the A.5 package inventory;
  - the limitations (Decision E2) and the reproducibility statement (Decision E);
  - the phase-wide optional items M.1–M.4 named as assigned at S8b;
  - the D20 gate and D21 (Decision C).

  Every figure is cited to the committed logbook, CSV or JSON it comes from; nothing is computed
  afresh.

## Records

- [x] 2. **done: the Phase 8 row, the Phase 8 shipment table's 8a rows, the 8a exit criterion, the SHOULD line, the overshoot row, D21 with notes at C.1, C.4 and B.2, and the minimum-viable success level.** Original scope: **The roadmap:**
  - the Phase 8 row reads **8a complete / 8b pending**;
  - the 8a synthesis exit criterion is ticked;
  - the SHOULD line and the overshoot row carry their statuses (B.2 carried, with its reason);
  - **D21** is added to the decisions table, covering every 8b wiring contrast (C.1e, C.4, and B.2's
    carried arms), with dated notes at C.1, C.4 and B.2;
  - the Phase 8 minimum-viable success level is marked met.
- [x] 3. **done: 8a COMPLETE and 8b ready to start; S8a ticked; A.3, A.5, B.2a, B.2b, M.5, M.6 and M.7 each given a status note; 8a's OpenSpec changes listed.** Original scope: **The tracker:**
  - 8a status COMPLETE, and 8b ready to start (gate GO);
  - S8a ticked;
  - A.3, A.5, B.2a, B.2b, M.5, M.6 and M.7 each given a status and destination;
  - the open threads listed with their destinations.
- [x] 4. **done.** Original scope: **The index row, and `README.md`'s status line.**

## Close-out

- [x] 5. **done: every figure in 076 cited to 070–075; the logbook, roadmap and tracker agree (8a complete, gate GO, D21); links resolve; hooks pass; validated strict.** Original scope: **Verify:**
  - every figure in Logbook 076 traces to a committed file;
  - every 8a criterion carries exactly one status in the logbook, the roadmap and the tracker, and
    the three agree;
  - every new link resolves;
  - `openspec validate --strict`;
  - `git add -A`, then pre-commit, judged by its exit code.
- [x] 6. **done: archived; the PR is opened from this branch.** Original scope: **Archive and open the PR.**
