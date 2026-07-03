## Why

Phase 6 was split into two shipments at the 2026-07 endgame re-scope (the pre-registered Gate-3 Phase 6a/6b sub-phase split, invoked **by success rather than by overrun**). Phase 6a (T1–T7 + connectome-structure controls + real-worm validation) is a complete, self-contained, citable result — the platform plus the six-family architecture ranking ([Logbook 029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md)). Phase 6b is the deferred completion: the L3 NEAT topology search (T8) and its synthesis addendum (T9b).

The `phase6-tracking` spec's *"Phase 6a / 6b sub-phase split triggered at Gate 3"* scenario pre-commits the rule: because Phase 6b contains more than one tranche (T8 NEAT + T9b synthesis), it **SHALL inherit a fresh `phase6b-tracking` change**. This change is that tracker. It exists now — rather than being authored at 6b start — so the committed 6b work has an honest home as `[ ]` not-started tasks, letting `phase6-tracking` archive cleanly at 6a synthesis instead of holding open on ticked-but-undone boxes.

This is a **tracking scaffold**, not a NEAT implementation change. The load-bearing 6b design decisions (environment vectorisation, TensorNEAT integration specifics, population/generation budget) are recorded here as **open questions**, resolved inside the per-milestone 6b OpenSpec change when 6b actually starts.

## What Changes

### 1. Phase 6b Tracking Change

Create `openspec/changes/phase6b-tracking/` with proposal/design/tasks/spec. The `tasks.md` is the living checklist for Phase 6b — the T8 NEAT topology-search sub-tasks (moved out of `phase6-tracking`, where they were blocking archival) plus the T9b synthesis addendum. Each Phase 6b milestone PR updates this checklist as part of its diff, the same discipline `phase5-tracking` and `phase6-tracking` used.

Phase 6b is **gated on three preconditions**: Gate 3 GO (Phase 6a close) + GPU availability + an environment-vectorisation decision (the ~500× TensorNEAT speedup assumes a vmappable env; `Continuous2DEnvironment` is not one yet, so env throughput — not the GPU — is the binding constraint).

### 2. Scope moved in from `phase6-tracking`

- **T8.1–T8.7 (NEAT topology search)** — relocated here as genuine `[ ]` not-started tasks. `phase6-tracking`'s T8 section becomes a pointer stub with no checkboxes.
- **T9b (synthesis addendum + Phase 6 COMPLETE marker)** — relocated here; `phase6-tracking` retains only T9a (the Phase 6a synthesis).

### 3. Scope explicitly NOT in Phase 6b

- **Co-evolution (ex-T8.4, matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP)** is **deferred with no scheduled destination** — it is *not* a Phase 6b deliverable. It is recorded as a ticked scoping decision in `phase6-tracking` (repo house style for dropped scope) and in the roadmap (§ Research Questions RQ4). The lag-matrix cross-pairing instrument (Phase 5 logbook 017) is retained for whenever a future phase commits to it.
- **The connectome-structure controls** (degree-preserving rewired-null + learnable-gap-junction) are **Phase 6a**, not 6b — they are PPO weight-search, not topology search, so they live under `phase6-tracking` T7.controls.\* and reuse the existing T7 pipeline (no NEAT / GPU / env-vectorisation). Phase 6b's "is the connectome a local optimum?" analysis cross-references the 6a rewired-null result.

### 4. Roadmap

`docs/roadmap.md` already documents the split (§ Phase 6a/6b split, tranche tracker, exit criteria). This change points the T8/6b OpenSpec-change references at `phase6b-tracking`.

## Capabilities

**Added**: `phase6b-tracking` (new) — requirements covering the living Phase 6b checklist, the roadmap Phase 6b/L3 status sync, the NEAT preconditions (Gate 3 GO + GPU + env-vectorisation decision), and the explicit exclusion of co-evolution from Phase 6b. This capability lives until the Phase 6b synthesis addendum archives alongside it.

**Modified**: none. (`phase6-tracking`'s proposal/design/tasks are amended for the split under that change, per its own spec's Gate-3-split scenario.)

## Impact

**Docs:**

- `openspec/changes/phase6b-tracking/proposal.md` — this file
- `openspec/changes/phase6b-tracking/design.md` — 6b-specific decisions + open questions (env-vectorisation, GPU tier, co-evolution exclusion)
- `openspec/changes/phase6b-tracking/tasks.md` — living Phase 6b checklist (T8 + T9b)
- `openspec/changes/phase6b-tracking/specs/phase6b-tracking/spec.md` — new capability
- `docs/roadmap.md` — T8/6b OpenSpec-change references point at `phase6b-tracking`

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase6b-tracking` capability is documentation-only and has no consumers in code.
