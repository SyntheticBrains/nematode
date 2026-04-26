## Why

Phase 4 (multi-agent complexity) is complete. Phase 5 (Evolution & Adaptation) begins now and will span multiple AI sessions over months across many milestone PRs. Without an explicit cross-session tracking artefact, future sessions will have to re-derive the milestone plan, re-read recent logbooks, and re-discover decisions already made.

Phase 4 used the per-milestone OpenSpec change directory and Logbook 011 to coordinate work, but lacked a single living checklist tying milestones together. This change introduces that scaffold for Phase 5 so any AI session can resume Phase 5 work by reading two files: this change's `tasks.md` and the roadmap Phase 5 block.

This change adds a new `phase5-tracking` spec capability whose requirements commit the project to (a) maintaining a single living checklist for all of Phase 5, (b) keeping the roadmap status block in sync with that checklist, and (c) recording Phase 5 design decisions in this change directory rather than re-deriving them per session. There is no runtime / source-code impact — the commitment is documentation discipline enforced by future PR review.

## What Changes

### 1. Phase 5 Milestone Tracking Change

Create `openspec/changes/2026-04-26-phase5-tracking/` with proposal/tasks/design. The `tasks.md` is a living checklist of every Phase 5 milestone (M0–M7) at sub-task granularity. Each subsequent Phase 5 milestone PR (M0, M1, M2, …) updates this checklist as part of its diff.

This change is intentionally not archived after merge — it stays open until the entire Phase 5 is complete (M7 logbook published). At that point it gets archived alongside the M7 evaluation change.

### 2. Roadmap Status Block Update

Edit `docs/roadmap.md` Phase 5 section:

- Update Phase 5 row in Timeline Overview table from `🔲 PLANNED` to `🟡 IN PROGRESS`
- Add a **Phase 5 Milestone Tracker** sub-section directly under the existing Phase 5 deliverables list, with one bullet per milestone (M-1 through M7) and current status

### 3. Three Recorded Phase 5 Decisions

The plan that generated these milestones made three decisions that should be visible in `proposal.md` and `design.md` so they aren't re-litigated:

- **Pilot-first**: M2 hyperparameter + M3 Lamarckian as lightweight pilots (20-30 gens, small population). Subsequent scientific milestones (M4 Baldwin, M5 co-evolution, M6 transgenerational) are gated on pilot evidence.
- **No QVarCircuit backwards compatibility**: existing `scripts/run_evolution.py` is hardcoded to `QVarCircuitBrain` — the M0 framework rebuild does not preserve byte-equivalent behaviour. Old script moves to `scripts/legacy/` unmaintained. Quantum brain support deferred to Phase 6.
- **LSTMPPO + klinotaxis as first-class brain**: M2/M3 use cheap MLPPPO for fast iteration, but the headline scientific milestones (M4/M5/M6) all target LSTMPPO+klinotaxis because Phase 4 produced 20+ validated configs with this combination, recurrent state allows transmission of meaningful priors (relevant for Lamarckian/Baldwin/transgenerational), and bilateral head-sweep sensing is the most biologically realistic configuration we have.

## Capabilities

**Added**: `phase5-tracking` (new) — three requirements covering the living milestone checklist, the roadmap Phase 5 status block, and the Phase 5 decision record. This capability lives until M7 archives alongside it.

## Impact

**Docs:**

- `openspec/changes/2026-04-26-phase5-tracking/proposal.md` — this file
- `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — living Phase 5 milestone checklist
- `openspec/changes/2026-04-26-phase5-tracking/design.md` — tracking strategy notes
- `openspec/changes/2026-04-26-phase5-tracking/specs/phase5-tracking/spec.md` — new capability
- `docs/roadmap.md` — Phase 5 status → IN PROGRESS, milestone tracker sub-section

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase5-tracking` capability is documentation-only and has no consumers in code.
