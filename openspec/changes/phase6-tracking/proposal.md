## Why

Phase 5 closed 2026-05-23 with one headline-positive result (M3 Lamarckian inheritance) and three substrate-grounded STOP diagnoses (M4 Baldwin, M5 co-evolution, M6.x transgenerational memory). The roadmap was revamped to v4 with Phase 6 (Connectome Substrate & Architecture Comparison) reframed as a four-layer build (L0 connectome → L1 architecture-plugin → L2 PPO weight search → L3 NEAT topology search). Phase 6 will span ~6-10 months across multiple AI sessions and multiple milestone-level OpenSpec changes (the first being `add-connectome-substrate` for L0). Without a single living checklist, future sessions will re-derive the layer plan, re-read the roadmap, and re-discover decisions already made.

Phase 5 used a `phase5-tracking` change for exactly this purpose; Phase 6 follows the same discipline. Any AI session can resume Phase 6 work by reading two files: this change's `tasks.md` and the roadmap Phase 6 block.

This change adds a new `phase6-tracking` spec capability whose requirements commit the project to (a) maintaining a single living checklist for all of Phase 6, (b) keeping the roadmap status block in sync with that checklist, and (c) making each mid-phase gate's go/no-go decision visible from this tracker. There is no runtime / source-code impact — the commitment is documentation discipline enforced by future PR review.

## What Changes

### 1. Phase 6 Tracking Change

Create `openspec/changes/phase6-tracking/` with proposal/design/tasks/spec. The `tasks.md` is a living checklist of every Phase 6 deliverable at sub-task granularity, organised by the eight tranches in [design.md § Decision 1](design.md) (L0 ingest → L1 plugin refactor → corrected ASH/ADL nociception → L2 first pass on grid → env upgrades → L2 re-run on upgraded substrate + real-worm validation → L3 NEAT → Phase 6 synthesis logbook), with the three mid-phase gates mapped to tranche boundaries (Gate 1 closes T2, Gate 2 closes T5, Gate 3 closes T6). Each subsequent Phase 6 milestone PR updates this checklist as part of its diff.

This change is **intentionally not archived after merge** — it stays open until the entire Phase 6 is complete (synthesis logbook published). At that point it gets archived alongside the synthesis change.

### 2. Roadmap Edits (Four)

Edit `docs/roadmap.md`:

- Update Phase 6 row in the Timeline Overview table from `🔲 PLANNED` to `🟡 IN PROGRESS`.
- Add a **Phase 6 Tranche Tracker** sub-section to the Phase 6 block, listing the eight tranches with status (T1 in progress as P6-0 scaffold ships; subsequent tranches not started; L4 marked deferred to Phase 7) and links to this change's `tasks.md` (sub-task detail) and `design.md` (design-decision record). Placement and style mirror the existing Phase 5 Milestone Tracker block.
- Annotate the existing Phase 6 architecture-families table (around lines 597-611) to mark the four MUST families per [design.md § Decision 4](design.md) and demote the four others to SHOULD/MAY with a brief rationale citation pointing at the design.
- Add a brief note to the Phase 6 layered-platform table (around lines 575-585) that env upgrades (Rung 2 + continuous-2D + continuous-action heads) sit in their own tranche (Tranche 5) between the L2 first pass (Tranche 4) and the L2 re-run (Tranche 6).

### 3. Seven Recorded Phase 6 Decisions

The plan that generated these tranches made seven decisions that should be visible in `proposal.md` and `design.md` so they aren't re-litigated mid-phase:

- **Decision 1 — Tranching policy**: Phase 6 has eight tranches with deliberate ordering (full table in `design.md`). Load-bearing ordering choices: corrected ASH/ADL (T3) before L2 first pass (T4); L2 first pass (T4) before env upgrades (T5); env upgrades (T5) between two L2 passes so the env-upgrade delta is itself a finding; continuous-action heads sit in T5 not T4; real-worm validation lives in T6 after defensible behavioural numbers exist; L3 NEAT (T7) runs against the upgraded substrate, not the grid.
- **Decision 2 — Connectome data source**: Cook 2019 hermaphrodite via OpenWorm `cect` (ConnectomeToolbox) is the L0 primary. NeuroML/c302 is deferred to a future *export* path for OpenWorm/Sibernetic interop.
- **Decision 3 — L1 plugin parity is real refactor work**: the existing `setup_brain_model()` dispatcher and `Brain` Protocol do not meet the "≤ 1 week to add a new architecture" parity test. L1 = dispatcher → registry refactor + topology/rule factoring. The `Brain` Protocol surface does NOT change.
- **Decision 4 — Architecture-family scope tightened to four MUSTs**: connectome-constrained, MLP-PPO, LSTM/GRU-PPO, NEAT-evolved. Quantum and spiking demote to SHOULD; reservoir, hybrid, transformer are MAY. Cuts the L2 sweep from 96 runs to 48. Promotion of a SHOULD/MAY to MUST requires amending this change.
- **Decision 5 — Behavioural scope is fixed at three**: klinotaxis, thermotaxis, predator evasion. Aerotaxis/pheromones/multi-agent are deferred.
- **Decision 6 — Mid-phase gate discipline**: each of the three roadmap gates produces a written go/no-go decision *inside the relevant OpenSpec change*, not as a silent continuation. This tracking change indexes where each gate decision lives.
- **Decision 7 — L2 connectome-substrate semantics**: strict-mask (Cook 2019 adjacency as hard connectivity constraint; PPO tunes only along existing edges) is the headline; soft-prior (synaptic counts as weight initialisation; PPO free to grow new connections) ships as the documented ablation in the same sweep.

## Capabilities

**Added**: `phase6-tracking` (new) — five requirements covering the living Phase 6 tranche checklist, the roadmap Phase 6 status block, gate-decision visibility, fixed architecture-family scope, and the deliberate tranche sequence. This capability lives until the Phase 6 synthesis logbook archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase6-tracking/proposal.md` — this file
- `openspec/changes/phase6-tracking/design.md` — seven decisions (tranching, data source, L1-refactor scope, tightened architecture-family scope, behavioural scope, gate discipline, L2 substrate semantics) + "what this change does not decide" boundary
- `openspec/changes/phase6-tracking/tasks.md` — living Phase 6 tranche / sub-task checklist (T1–T8)
- `openspec/changes/phase6-tracking/specs/phase6-tracking/spec.md` — new capability with five requirements
- `docs/roadmap.md` — Phase 6 Timeline Overview status → IN PROGRESS; Phase 6 Tranche Tracker sub-section; architecture-families table MUST/SHOULD/MAY annotations; layered-platform table env-upgrade tranche note

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase6-tracking` capability is documentation-only and has no consumers in code.
