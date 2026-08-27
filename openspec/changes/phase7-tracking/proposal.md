## Why

Phase 6a closed with a Gate 3 GO ([Logbook 037](../../../docs/experiments/logbooks/037-phase6a-synthesis.md)), satisfying the hard phase boundary (Phase 7 never gates on Phase 6b, which has its own tracker). The Phase 7 plan was then reviewed, adversarially critiqued, and amended before start — roadmap v4.2 (PR #300) carries the ratified design decisions **D1–D13**, the pre-registered **2×2 central hypothesis** (rule × wiring: does the connectome's wiring become load-bearing under a biologically-plausible three-factor rule, when Logbook 034 showed it is inert under PPO?), and the rescoped cross-species deliverable (the Cook 2025 *P. pacificus* data is head-only, chemical-synapse-only).

Per the house workflow (a tracking change per phase: `phase5-tracking`, `phase6-tracking`, `phase6b-tracking`), Phase 7 needs its living tracker before implementation begins, so that the committed work has an honest home as `[ ]` not-started tasks, the D1–D13 decisions have a binding amendment path, and the Phase 6 execution-protocol learnings are pinned where milestone changes must inherit them rather than relearn them.

## What Changes

### 1. Phase 7 Tracking Change

Create `openspec/changes/phase7-tracking/` with proposal/design/tasks/spec. The `tasks.md` is the living checklist for Phase 7's three shipments — **7a-i** (pre-L4 platform freeze + trace substrate + minimal three-factor rule + the D10 2×2 panel), **7a-ii** (receptor-grounded neuromodulator stack + panel re-run), and **7b** (cross-species comparative learning at head-circuit scope, dauer-first) — plus the SHOULD items (6a preprint, imitation-warm-start arm, learnable-gap-junction ablation, weight-transplant transfer, species-appropriate third behaviour, co-primary biological validation) and the synthesis. Every Phase 7 milestone PR updates this checklist as part of its diff.

This is a **tracking scaffold**, not an implementation change. Load-bearing implementation choices already decided are recorded in `docs/roadmap.md` § Phase 7 § Pre-registered design decisions (D1–D13, authoritative); choices deliberately left open (rule-seam refactor shape, exact panel budget, transplant spec details) are recorded in `design.md` here as open questions, resolved inside per-milestone changes.

### 2. Scope explicitly NOT in Phase 7

- **Phase 6b (T8 NEAT)** — tracked in `phase6b-tracking`, decoupled from Phase 7's early window per D13 (opportunistic, pending a dated GPU/cloud decision). Phase 7 never gates on it.
- **Co-evolution** — deferred with no scheduled destination (unchanged from Phase 6; roadmap § Research Questions RQ4).
- **Neuropeptide signalling layer as a rule substrate** — recorded in the roadmap as the largest deferred fidelity gap; a candidate Phase-7+ layer, not a Phase 7 deliverable.
- **Full `∂C/∂t = D∇²C` dynamic-diffusion PDE** and ***C. briggsae*** — unchanged deferrals.

### 3. Roadmap

`docs/roadmap.md` § Phase 7 gains a one-line pointer to this tracker's `tasks.md` (the same "how to orient" pattern Phases 5/6 use). No plan content changes — the plan was finalised in PR #300.

## Capabilities

**Added**: `phase7-tracking` (new) — requirements covering the living Phase 7 checklist, the binding status of the roadmap's D1–D13 decisions, the pre-panel substrate freeze (D5/D7 + validation gate + re-baseline), the inherited execution-protocol standards, the shipment-completion semantics (7a GO ≠ Phase 7 COMPLETE), and the scope exclusions. This capability lives until the Phase 7 synthesis archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase7-tracking/proposal.md` — this file
- `openspec/changes/phase7-tracking/design.md` — tracker-level decisions + open questions
- `openspec/changes/phase7-tracking/tasks.md` — living Phase 7 checklist (shipments 7a-i / 7a-ii / 7b + synthesis)
- `openspec/changes/phase7-tracking/specs/phase7-tracking/spec.md` — new capability
- `docs/roadmap.md` — one-line tracker pointer in § Phase 7

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase7-tracking` capability is documentation-only and has no consumers in code.
