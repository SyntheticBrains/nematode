## Why

Phase 5 closed 2026-05-23 with one headline-positive result (M3 Lamarckian inheritance) and three substrate-grounded STOP diagnoses (M4 Baldwin, M5 co-evolution, M6.x transgenerational memory). The roadmap was revamped to v4 with Phase 6 (Connectome Substrate & Architecture Comparison) reframed as a four-layer build (L0 connectome → L1 architecture-plugin → L2 PPO weight search → L3 NEAT topology search). Phase 6 will span ~6-10 months across multiple AI sessions and multiple milestone-level OpenSpec changes (the first being `add-connectome-substrate` for L0). Without a single living checklist, future sessions will re-derive the layer plan, re-read the roadmap, and re-discover decisions already made.

Phase 5 used a `phase5-tracking` change for exactly this purpose; Phase 6 follows the same discipline. Any AI session can resume Phase 6 work by reading two files: this change's `tasks.md` and the roadmap Phase 6 block.

This change adds a new `phase6-tracking` spec capability whose requirements commit the project to (a) maintaining a single living checklist for all of Phase 6, (b) keeping the roadmap status block in sync with that checklist, and (c) making each mid-phase gate's go/no-go decision visible from this tracker. There is no runtime / source-code impact — the commitment is documentation discipline enforced by future PR review.

## What Changes

### 1. Phase 6 Milestone Tracking Change

Create `openspec/changes/phase6-tracking/` with proposal/design/tasks/spec. The `tasks.md` is a living checklist of every Phase 6 deliverable at sub-task granularity, organised by the roadmap's L0/L1/L2/L3 layers plus continuous-physics + Rung 2 gradients + corrected ASH/ADL nociception + real-worm validation + three mid-phase gates + Phase 6 synthesis logbook. Each subsequent Phase 6 milestone PR updates this checklist as part of its diff.

This change is **intentionally not archived after merge** — it stays open until the entire Phase 6 is complete (synthesis logbook published). At that point it gets archived alongside the synthesis change.

### 2. Roadmap Status Block Update

Edit `docs/roadmap.md` Phase 6 section:

- Update Phase 6 row in Timeline Overview table from `🔲 PLANNED` to `🟡 IN PROGRESS`
- Add a **Phase 6 Milestone Tracker** sub-section to the Phase 6 block, listing layers L0-L4 (with L4 marked deferred to Phase 7) and their status, with links to this change's `tasks.md` (sub-task detail) and `proposal.md` (design-decision record). Placement and style mirror the existing Phase 5 Milestone Tracker block.

### 3. Five Recorded Phase 6 Decisions

The plan that generated these layers made five decisions that should be visible in `proposal.md` and `design.md` so they aren't re-litigated mid-phase:

- **Tranching policy**: Phase 6 is decomposed into tranches that map to the roadmap's L0/L1/L2/L3 layers. Tranche 1 is L0-only (connectome data import + validation, no plugin work, no training); Tranche 2 is L1 + first PPO-on-connectome attempt (Gate 1 trigger); subsequent tranches follow. This split is deliberate so each layer's Risk-mitigation pivot has its own evidence and its own decision artefact.
- **Connectome data source**: Cook 2019 hermaphrodite via OpenWorm `cect` (ConnectomeToolbox) is the L0 primary. NeuroML/c302 is deferred to a future *export* path for OpenWorm/Sibernetic interop (Future Directions). Rationale is documented; do not re-litigate inside individual L0/L1/L2 changes.
- **Mid-phase gate discipline**: each of the three roadmap gates (L0 month-2, L1 month-4-5, L2 month-7-8) produces a written go/no-go decision *inside the relevant OpenSpec change*, not as a silent continuation. This tracking change indexes where each gate decision lives.
- **Architecture-family scope is fixed**: Phase 6's MUST architecture-family set is the roadmap-defined eight families (connectome-constrained, MLP-PPO, LSTM/GRU-PPO, spiking, reservoir, quantum, hybrid, NEAT-evolved). Transformer is the documented MAY. Do not expand this set inside individual milestone changes; expansion must amend *this* tracking change.
- **Behavioural scope is fixed**: Phase 6 commits to three behaviours (klinotaxis, thermotaxis, predator evasion). Aerotaxis/pheromones/multi-agent are *deferred*. Same anti-scope-creep discipline: don't add behaviours inside milestone changes.

## Capabilities

**Added**: `phase6-tracking` (new) — requirements covering the living Phase 6 layer/milestone checklist, the roadmap Phase 6 status block, and gate-decision visibility. This capability lives until the Phase 6 synthesis logbook archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase6-tracking/proposal.md` — this file
- `openspec/changes/phase6-tracking/design.md` — tranching, data-source, gate-discipline, and scope-fix rationale
- `openspec/changes/phase6-tracking/tasks.md` — living Phase 6 layer / sub-task checklist
- `openspec/changes/phase6-tracking/specs/phase6-tracking/spec.md` — new capability
- `docs/roadmap.md` — Phase 6 status → IN PROGRESS, Phase 6 Milestone Tracker sub-section

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase6-tracking` capability is documentation-only and has no consumers in code.
