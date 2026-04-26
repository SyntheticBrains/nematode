## Overview

Phase 5 (Evolution & Adaptation) spans 8 milestones (M0–M7) plus this scaffold (M-1) over many months and AI sessions. This design documents the cross-session tracking strategy so the choice of artefacts is explicit and can be revisited if it proves insufficient.

## Goals / Non-Goals

**Goals:**

- A future AI session can answer "what's the next Phase 5 milestone?" by reading two files (this `tasks.md` and the roadmap Phase 5 block)
- Each Phase 5 milestone PR has a single canonical place to mark progress
- Three Phase 5 design decisions (pilot-first, no QVarCircuit backwards compat, LSTMPPO+klinotaxis first-class) are recorded once, not re-derived per session
- The tracking artefact decays gracefully: when M7 is published, this change archives alongside it

**Non-Goals:**

- Real-time progress dashboards (the roadmap status table is enough)
- Automated milestone status (humans/agents update the checklist manually as part of milestone PRs)
- Replacing per-milestone OpenSpec changes — those still happen, this scaffold is *additional* coordination
- GitHub Issues / Project boards — optional mirror, decided at PR time per milestone

## Design Decisions

### Decision 1: One change directory for all of Phase 5, not one per milestone

Each Phase 5 milestone (M0–M7) gets its own OpenSpec change directory (e.g. `2026-04-28-add-evolution-framework`). This M-1 change is *additional* — it's the meta-tracking layer above the per-milestone changes.

**Why:** Per-milestone changes are scoped to one PR with concrete code/spec deltas; they archive on merge. The Phase 5 tracker needs to outlive any single milestone — it spans the entire phase. Putting it in its own change directory means it lives until M7 archives it, while individual milestones come and go.

**Alternative considered:** Embed milestone tracking in `docs/roadmap.md` only. Rejected because the roadmap is the public-facing strategy doc; sub-task granularity belongs in a working artefact. The roadmap gets a one-line status per milestone (the "Milestone Tracker" sub-section); the OpenSpec `tasks.md` carries the full sub-task checklist.

### Decision 2: tasks.md uses sub-task granularity matching Phase 4

Phase 4's `2026-04-11-add-phase4-evaluation/tasks.md` used numbered sub-tasks (1.1, 1.2, … 7.x) with `[x]` checkboxes. M-1 follows the same convention — each Phase 5 milestone gets a numbered section with its sub-tasks listed. This means subsequent milestone PRs that update the checklist look stylistically identical to Phase 4 evaluation work.

**Why:** Familiarity reduces cognitive load. Future AI sessions that have seen Phase 4 patterns will read the Phase 5 tracker the same way.

### Decision 3: Three recorded design decisions surface in proposal.md, not buried in this file

The three decisions (pilot-first, no QVarCircuit backwards compat, LSTMPPO+klinotaxis first-class) are stated in `proposal.md` so they show up in `openspec change show 2026-04-26-phase5-tracking`. The rationale lives here in `design.md` for anyone who wants the reasoning.

**Why:** A future session reading the proposal first should see *what was decided* without having to open the design doc. The design doc explains *why those decisions* but isn't required reading.

## Tracking Strategy

Three artefacts answer "where are we in Phase 5?":

1. **`openspec/changes/2026-04-26-phase5-tracking/tasks.md`** — sub-task checklist updated by every milestone PR
2. **`openspec/changes/<milestone>/`** — per-milestone proposal/tasks/design/specs, archived on milestone merge
3. **`docs/roadmap.md` Phase 5 section** — one-line milestone status, updated as part of every milestone PR

A future AI session orients by:

- Reading the roadmap Phase 5 block first (one-line current status per milestone)
- Reading this `tasks.md` for sub-task granularity
- Reading active `openspec/changes/<milestone>/` if a specific milestone is in flight
- Reading the latest published `artifacts/logbooks/0XX/` if a milestone has completed evaluation

## Maintenance

- Every Phase 5 milestone PR updates `tasks.md` (mark sub-tasks complete) and `docs/roadmap.md` Phase 5 milestone tracker (one-line status update)
- This change does not archive until M7 completes
- If Phase 5 deviates substantially from this plan (e.g. a milestone is dropped or reordered), update `tasks.md` to reflect reality — the checklist is descriptive, not aspirational

## Risks

1. **The checklist drifts from reality if PRs forget to update it.** Mitigation: include "update phase5-tracking tasks.md" in the per-milestone PR template / checklist.
2. **The three recorded decisions become stale if circumstances change.** Mitigation: any decision reversal must amend `proposal.md` in a follow-up commit to this same change directory, leaving git history as the audit trail.
3. **Sub-task granularity becomes too fine and burdensome.** Mitigation: the M-1 initial `tasks.md` keeps each milestone to ~5–10 sub-tasks; expand only if necessary.
