# Tasks: Consolidate the Phase 7 methodology rules

Phase 8 task **A.4**. The plan is in this change's `proposal.md` and `design.md`; the 44-row mapping
is `design.md` § The mapping. Runs before any other 8a registration
(`openspec/changes/phase8-tracking/tasks.md` § Block A).

**Ordering note**: `openspec archive` updates the main specs, so anything that edits a live spec by
hand — the receiving spec's Purpose statement — happens **after** the archive, not before, and the
verification splits either side of it.

- [x] 1. **Author the change** — proposal, design with the 44-row mapping table, and the two delta
  specs: `## REMOVED Requirements` × 37 against `plasticity-evaluation` (each with a `**Reason**`
  and a `**Migration**`), and `## ADDED Requirements` × 11 against
  `architecture-comparison-protocol` (verbatim text, no preamble — provenance belongs in the
  receiving spec's Purpose and in `design.md`, not in a delta whose position after archiving is not
  predictable). `openspec validate --strict` passes.

- [x] 2. **Fold the eighteen into the phase protocol** — **done 2026-09-20**: eight clause blocks
  under principles 1, 4, 6, 7, 10, 11 and 12, the protocol at **222** lines against the 250 budget,
  thirteen principles and their numbering unchanged. Original scope: — `docs/research/phase-protocol.md`. Clauses
  under principles 1, 4, 6, 7, 10, 11 and 12 per the mapping; principle 10's scope extended to
  matching the statistic and the metric to the outcome's shape. No new principle, **no
  renumbering** (the roadmap, the literature-watch brief and the Phase 8 tracker cite principles by
  number). Each clause carries its logbook citation the way the existing principles do.

- [x] 3. **Audit the analysis harnesses for orphaned clauses** — **done 2026-09-20, nothing orphaned.**
  Across **27** harness/test pairs, every normative clause in a test's module docstring also appears
  in the harness it tests; the only phrase the sweep flagged, "the instruments are read-only",
  describes the test approach rather than a methodology rule. Both named candidates are stated in the
  harness source, not only in a test: `l4_feature_ablations.py` carries `MIN_CARRY = 0.123` with its
  two-thirds rationale in the constant's own comment and in the module docstring, and defines
  `survives_without_it` as "no significant interaction -- a FAILURE TO DETECT, its size and interval
  carried". No clause needed promoting and no test needed marking as an authority. *(Original scope:
  **25** modules under
  `scripts/analysis/` matching the `l4_*`, `l1b_*` and `wiring_*` families, with their tests under
  `packages/quantum-nematode/tests/quantumnematode_tests/analysis/`; the sweep covered 27 pairs.)*

- [ ] 4. **Correct the counts** — `docs/roadmap.md` A.4 deliverable (§ Required deliverables 3) and
  technical-debt item 14 describe A.4 in terms of a count this change changes; restate as what
  happened. `openspec/changes/phase8-tracking/tasks.md` A.4 likewise. **Leave untouched**: Logbook
  069's "42 requirements" and the roadmap's inheritance-list note, which are historical and already
  carry their own dated explanation.

- [ ] 5. **Tick A.4 in the tracker** — `openspec/changes/phase8-tracking/tasks.md`, with the
  outcome recorded the way Phase 7's tracker records a closed task: what the consolidation did, in
  one sentence, with the resulting requirement counts.

- [ ] 6. **Verify, before archiving** —
  `openspec validate consolidate-plasticity-methodology --strict`;
  every relative link in the change resolves (the delta specs sit two directories deeper than
  `proposal.md`, so their paths to `docs/` need five `../` segments, not three);
  `pre-commit run --all-files` clean;
  the phase protocol still reads in one sitting (past roughly 250 lines the fold became relocation
  and should be tightened);
  three logbook provenance lines (060, 067, 068) resolve against the mapping table.

- [ ] 7. **Archive** — `openspec archive consolidate-plasticity-methodology`, which applies both
  deltas to the live specs.

- [ ] 8. **After archiving** — widen `architecture-comparison-protocol`'s Purpose statement, which
  currently names only cross-architecture ranking, to say it also holds the comparison methodology
  that applies whatever is being compared, naming the Phase 7 consolidation as its provenance. Then
  confirm the final counts: `plasticity-evaluation` at **7** requirements and
  `architecture-comparison-protocol` at **22**, by `grep -c '^### Requirement:'` on each, with no
  count claimed anywhere in `docs/` or `openspec/` contradicting them.

## Not in this change

- **The six capability requirements are not touched.** Their code (`scripts/run_plasticity_test.py`,
  `quantumnematode/plasticity/`) has been dormant since April 2026 with no live config, and whether
  the forgetting-protocol harness should itself be retired is a separate decision about a separate
  capability. Folding it into a methodology task would bury it.
- **`openspec/specs/l4-plasticity-panel/`** is the other Phase 7-era capability whose scope ended
  with the phase; whether it retires alongside the forgetting harness is the same kind of question.

Both are recorded as follow-ups for the Phase 8 housekeeping window.
