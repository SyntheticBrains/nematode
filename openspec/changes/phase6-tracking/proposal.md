## Why

Phase 5 closed 2026-05-23 with one headline-positive result (M3 Lamarckian inheritance) and three substrate-grounded STOP diagnoses (M4 Baldwin, M5 co-evolution, M6.x transgenerational memory). The roadmap was revamped to v4 with Phase 6 (Connectome Substrate & Architecture Comparison) reframed as a four-layer build (L0 connectome → L1 architecture-plugin → L2 PPO weight search → L3 NEAT topology search). Phase 6 will span ~6-10 months across multiple AI sessions and multiple milestone-level OpenSpec changes (the first being `add-connectome-substrate` for L0). Without a single living checklist, future sessions will re-derive the layer plan, re-read the roadmap, and re-discover decisions already made.

Phase 5 used a `phase5-tracking` change for exactly this purpose; Phase 6 follows the same discipline. Any AI session can resume Phase 6 work by reading two files: this change's `tasks.md` and the roadmap Phase 6 block.

This change adds a new `phase6-tracking` spec capability whose requirements commit the project to (a) maintaining a single living checklist for all of Phase 6, (b) keeping the roadmap status block in sync with that checklist, and (c) making each mid-phase gate's go/no-go decision visible from this tracker. There is no runtime / source-code impact — the commitment is documentation discipline enforced by future PR review.

## What Changes

### 1. Phase 6 Tracking Change

Create `openspec/changes/phase6-tracking/` with proposal/design/tasks/spec. The `tasks.md` is a living checklist of every Phase 6 deliverable at sub-task granularity, organised by the nine tranches in [design.md § Decision 1](design.md) (L0 ingest → L1 plugin refactor → corrected ASH/ADL nociception → L2 first pass on grid → platform refactor (continuous-2D + continuous-action heads + parity) → env fidelity (Rung 2 + log-concentration adaptation) → L2 re-run on upgraded substrate + real-worm validation → L3 NEAT → Phase 6 synthesis logbook), with the three mid-phase gates mapped to tranche boundaries (Gate 1 closes T2, Gate 2 closes T5, Gate 3 closes T7). Each subsequent Phase 6 milestone PR updates this checklist as part of its diff.

This change is **intentionally not archived after merge** — it stays open until the entire Phase 6 is complete (synthesis logbook published). At that point it gets archived alongside the synthesis change.

### 2. Roadmap Edits (Four)

Edit `docs/roadmap.md`:

- Update Phase 6 row in the Timeline Overview table from `🔲 PLANNED` to `🟡 IN PROGRESS`.
- Add a **Phase 6 Tranche Tracker** sub-section to the Phase 6 block, listing the nine tranches with status (P6-0 in progress as scaffold ships; subsequent tranches not started; L4 marked deferred to Phase 7) and links to this change's `tasks.md` (sub-task detail) and `design.md` (design-decision record). Placement and style mirror the existing Phase 5 Milestone Tracker block.
- Annotate the existing Phase 6 architecture-families table (around lines 597-611) to mark the four MUST families per [design.md § Decision 4](design.md) and demote the four others to SHOULD/MAY with a brief rationale citation pointing at the design.
- Add a brief note to the Phase 6 layered-platform table (around lines 575-585) that env upgrades (continuous-2D + continuous-action heads + Rung 2 chemical gradients) split across two tranches (T5 platform refactor + T6 env fidelity) between the L2 first pass (T4) and the L2 re-run (T7).

### 3. Seven Recorded Phase 6 Decisions

The plan that generated these tranches made seven decisions that should be visible in `proposal.md` and `design.md` so they aren't re-litigated mid-phase. Decisions 1, 6, and 7 were tightened on a self-critique iteration (T5 split out from T6, quantitative gate criteria added, connection-type taxonomy spelled out):

- **Decision 1 — Tranching policy**: Phase 6 has **nine** tranches with deliberate ordering (full table in `design.md`). Load-bearing ordering choices: corrected ASH/ADL (T3) before L2 first pass (T4); L2 first pass (T4) before any env upgrade (T5 + T6); platform refactor (T5) split from env fidelity (T6) so Gate 2 closes against a single verifiable platform outcome; env upgrades (T5 + T6) between two L2 passes so the env-upgrade delta (T4→T7) is itself a finding; continuous-action heads sit in T5; real-worm validation lives in T7 after defensible behavioural numbers exist; L3 NEAT (T8) runs against the upgraded substrate, not the grid.
- **Decision 2 — Connectome data source**: Cook 2019 hermaphrodite via OpenWorm `cect` (ConnectomeToolbox) is the L0 primary. NeuroML/c302 is deferred to a future *export* path for OpenWorm/Sibernetic interop.
- **Decision 3 — L1 plugin parity is real refactor work**: the existing `setup_brain_model()` dispatcher (459 LOC, 19 elif branches) and `Brain` Protocol do not meet the "≤ 5 working days to add a new architecture" parity test. L1 = dispatcher → registry refactor + topology/rule factoring + 19-architecture migration with a Phase-5-M1-style byte-equivalence regression bar. The `Brain` Protocol surface does NOT change.
- **Decision 4 — Architecture-family scope tightened to four MUSTs**: connectome-constrained, MLP-PPO, LSTM/GRU-PPO, NEAT-evolved. Quantum and spiking demote to SHOULD; reservoir, hybrid, transformer are MAY. Cuts the per-L2-pass MUST sweep from 96 runs to 48 (× 2 passes = 96 total across T4 + T7). Promotion of a SHOULD/MAY to MUST requires amending this change.
- **Decision 5 — Behavioural scope is fixed at three**: klinotaxis, thermotaxis, predator evasion. Aerotaxis/pheromones/multi-agent are deferred.
- **Decision 6 — Mid-phase gate discipline with quantitative pre-registered criteria**: each of the three gates has explicit numerical pass criteria (G1.a-d; G2.a-d; G3.a-d) in the spirit of Phase 5's M2/M3/M4 quantitative gates. Each gate decision is recorded in the triggering tranche's published *logbook* (not in `tasks.md`, which becomes hard to amend post-archive).
- **Decision 7 — L2 connectome-substrate semantics with explicit connection-type taxonomy**: strict-mask applies to chemical synapses (directed, weighted) — PPO tunes weights only along existing chemical edges. Gap junctions are fixed-weight electrical couplings (not learnable — biologically faithful). Extra-synaptic/peptidergic signalling is out of Phase 6 scope (reserved for Phase 7 L4 plasticity). Soft-prior on chemical synapses ships as a documented ablation in the same sweep.

## Capabilities

**Added**: `phase6-tracking` (new) — five requirements covering the living Phase 6 tranche checklist, the roadmap Phase 6 status block, gate-decision visibility, fixed architecture-family scope, and the deliberate tranche sequence. This capability lives until the Phase 6 synthesis logbook archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase6-tracking/proposal.md` — this file
- `openspec/changes/phase6-tracking/design.md` — seven decisions (tranching, data source, L1-refactor scope, tightened architecture-family scope, behavioural scope, gate discipline, L2 substrate semantics) + "what this change does not decide" boundary
- `openspec/changes/phase6-tracking/tasks.md` — living Phase 6 tranche / sub-task checklist (T1–T9)
- `openspec/changes/phase6-tracking/specs/phase6-tracking/spec.md` — new capability with five requirements
- `docs/roadmap.md` — Phase 6 Timeline Overview status → IN PROGRESS; Phase 6 Tranche Tracker sub-section; architecture-families table MUST/SHOULD/MAY annotations; layered-platform table env-upgrade tranche note

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase6-tracking` capability is documentation-only and has no consumers in code.
