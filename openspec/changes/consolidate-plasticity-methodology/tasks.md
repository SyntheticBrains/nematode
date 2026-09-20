# Tasks: Consolidate the Phase 7 methodology rules

Phase 8 task **A.4**. The plan is in this change's `proposal.md` and `design.md`; the 44-row mapping
is `design.md` § The mapping. Runs before any other 8a registration
(`openspec/changes/phase8-tracking/tasks.md` § Block A).

- [x] 1. **Author the change** — proposal, design with the 44-row mapping table, and the two delta
  specs: `## REMOVED Requirements` × 37 against `plasticity-evaluation` (each with a `**Reason**`
  and a `**Migration**`), and `## ADDED Requirements` × 11 against
  `architecture-comparison-protocol` (verbatim text). `openspec validate --strict` passes.

- [ ] 2. **Fold the eighteen into the phase protocol** — `docs/research/phase-protocol.md`. Clauses
  under principles 1, 4, 6, 7, 10, 11 and 12 per the mapping; principle 10's scope extended to
  matching the statistic and the metric to the outcome's shape. No new principle, **no
  renumbering** (the roadmap, the literature-watch brief and the Phase 8 tracker cite principles by
  number). Each clause carries its logbook citation the way the existing principles do.

- [ ] 3. **Widen the receiving spec's Purpose** — `architecture-comparison-protocol`'s Purpose
  statement currently names only cross-architecture ranking; it gains a clause saying it also holds
  the comparison methodology that applies whatever is being compared.

- [ ] 4. **Audit the analysis harnesses for orphaned clauses** — the ~20 modules under
  `scripts/analysis/` and their tests under
  `packages/quantum-nematode/tests/quantumnematode_tests/analysis/` are the implementation of these
  rules, and their docstrings restate the reasoning by paraphrase. For each removed requirement with
  a live harness, check whether a specific clause survives only in a test assertion. Where the clause
  is general, promote it to the protocol; where it is specific to that analysis, leave it and add a
  comment recording the test as its authority. Named candidates: the feature-ablation minimum stated
  as a fraction of the established effect, and `survives_without_it` meaning a failure to detect.

- [ ] 5. **Correct the counts** — `docs/roadmap.md` A.4 deliverable (§ Required deliverables 3) and
  technical-debt item 14 describe A.4 in terms of a count this change changes; restate as what
  happened. `openspec/changes/phase8-tracking/tasks.md` A.4 likewise. **Leave untouched**: Logbook
  069's "42 requirements" and the roadmap's inheritance-list note, which are historical and already
  carry their own dated explanation.

- [ ] 6. **Tick A.4 in the tracker** — `openspec/changes/phase8-tracking/tasks.md`, with the
  outcome recorded the way Phase 7's tracker records a closed task: what the consolidation did, in
  one sentence, with the resulting requirement counts.

- [ ] 7. **Verify** —
  `openspec validate consolidate-plasticity-methodology --strict`;
  after archiving, `plasticity-evaluation` has **7** requirements and
  `architecture-comparison-protocol` has **22**, confirmed by
  `grep -c '^### Requirement:'` on each;
  no count claimed anywhere in `docs/` or `openspec/` contradicts those;
  `pre-commit run --all-files` clean;
  the phase protocol still reads in one sitting (past roughly 250 lines the fold became relocation
  and should be tightened);
  three logbook provenance lines (060, 067, 068) resolve against the mapping table.

- [ ] 8. **Archive** — `openspec archive consolidate-plasticity-methodology`, which applies both
  deltas to the live specs. Re-run the counts afterwards.

## Not in this change

- **The six capability requirements are not touched.** Their code (`scripts/run_plasticity_test.py`,
  `quantumnematode/plasticity/`) has been dormant since April 2026 with no live config, and whether
  the forgetting-protocol harness should itself be retired is a separate decision about a separate
  capability. Folding it into a methodology task would bury it. Recorded as a follow-up for the
  Phase 8 housekeeping window.
