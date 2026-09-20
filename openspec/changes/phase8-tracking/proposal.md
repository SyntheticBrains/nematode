## Why

Phase 7 closed 2026-09-19 as **SPLIT** ([Logbook 069](../../../docs/experiments/logbooks/069-phase7-synthesis.md)): no biologically plausible rule that writes the connectome learns the substrate to any benefit, and what shipped instead is a replicated wiring advantage on learning speed under PPO (carrying the standing condition that rewiring varies with initialisation), a rate-robust null on the wiring as fixed features, and a readout-width positive that holds at one learning rate. The Phase 8 plan was then reviewed, corrected at review (D15's premise, PR #395), and ratified as roadmap **v4.3** — § Phase 8, *ground, then embody*, with design decisions **D15–D20**, two shipments, and an explicit scope decision against whole-organism fidelity.

Per the house workflow (a tracking change per phase: `phase5-tracking`, `phase6-tracking`, `phase6b-tracking`, `phase7-tracking`), Phase 8 needs its living tracker before the first registration, so the committed work has an honest home as `[ ]` not-started tasks, D15–D20 have a binding amendment path, and the execution-protocol standards Phases 6 and 7 paid for are pinned where every rung change must inherit them.

## What Changes

### 1. Phase 8 Tracking Change

Create `openspec/changes/phase8-tracking/` with proposal/design/tasks/spec. The `tasks.md` is the living checklist for the two shipments — **8a** (block A: the init-vs-rewiring control, the operating-point robustness surface, frozen-operator predictors, methodology consolidation, the publication decision; B.1 measured synaptic weights; B.2 the dynamics rung; the 8a synthesis) and **8b** (C.0 body prerequisites and D19; C.1 the anatomical motor-to-muscle readout into a kinematic body; C.2 the rod-chain body; C.3 body-level validation; C.4 the ranking through the body; C.5 rendering; B.3 + D.1 internal state and patchy lawns; the phase synthesis) — plus the MAY items. Every Phase 8 milestone PR updates this checklist as part of its diff.

This is a **tracking scaffold**, not an implementation change. Load-bearing choices already decided are in `docs/roadmap.md` § Phase 8 § Pre-registered design decisions (D15–D20, authoritative); choices deliberately left open (the dense-draw-then-mask code path, the A.2 sweep budget, the B.1 unit-scale mapping and licence, the B.2 integrator, the C.0 step–time constant, C.1's resistive-force formulation and segment count, C.2's cost-budget number, D19's body-level generator) are recorded in `design.md` as open questions, resolved inside per-milestone changes.

### 2. Scope explicitly NOT in Phase 8

- **Cross-species work** — 7b's *P. pacificus* comparison, the dauer pathfinder, weight transplant. Deferred to the phase after 8; C.1 (a readout that does not carry the learning) is its precondition.
- **Multi-agent, pheromones, red-queen and ecological co-evolution** — Phase 4/5 verdicts stand.
- **Evolution in every form** — 6b NEAT stays deferred unscheduled in `phase6b-tracking` (a body makes the env-vectorisation decision harder, not easier); Lamarckian and transgenerational work stays closed.
- **The male–hermaphrodite wiring contrast** (data confirmed on disk), **structural plasticity across development**, the **neuropeptide layer as a rule substrate**, **spiking-STDP and neuromorphic deployment**, **3D environments, the Sibernetic body and ion-channel neurons**, and the **uniform substrate-writing rule programme** (closed with a diagnosed cause, [Logbook 063](../../../docs/experiments/logbooks/063-l4-eprop.md)).

### 3. Roadmap, Logbook 069 and the literature-watch context

`docs/roadmap.md` § Phase 8 gains the tracker pointer (the "how to orient" pattern Phases 5–7 use). Riding along, because they were agreed together and are small: (a) the **Lee 2026 preprint** (bioRxiv 2026.09.06.749731, found by the re-aimed literature watch on the day it was seeded) is recorded where it bears — the C.1/C.2 novelty row is **narrowed** (the "body + learning + wiring-control cell is empty for any organism" claim is withdrawn), a **risk is registered against B.1** (atlas grounding produced no functional sensory-to-command step in that model), and its gap-junction-shuffle result is cited as **convergent** with B.2's focus — plus an entry under § Worm body and whole-organism models; (b) **Logbook 069** § What Phase 8 opens on, item 1, gains a **dated correction footnote** for the degree-preservation claim corrected at PR #395's review, the original sentence left in place; (c) the roadmap's Future Directions note that said the logbook was left as written is updated to match; (d) `openspec/config.yaml`'s focus line moves from Phase 7 to Phase 8.

## Capabilities

**Added**: `phase8-tracking` (new) — requirements covering the living Phase 8 checklist, the binding status of D15–D20, a positive control before any connectome arm on a new component, operating-point discipline (D16), the new-reference-frame rule for body-substrate results, the two-shipment completion semantics (D20), and the scope exclusions. This capability lives until the Phase 8 synthesis archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase8-tracking/proposal.md` — this file
- `openspec/changes/phase8-tracking/design.md` — tracker-level decisions + open questions
- `openspec/changes/phase8-tracking/tasks.md` — living Phase 8 checklist (8a / 8b + syntheses)
- `openspec/changes/phase8-tracking/specs/phase8-tracking/spec.md` — new capability
- `docs/roadmap.md` — tracker pointer in § Phase 8; the Lee 2026 edits (novelty map, B.1 risk, B.2 convergence, § Worm body entry); the Future Directions correction note updated
- `docs/experiments/logbooks/069-phase7-synthesis.md` — one dated correction footnote under § What Phase 8 opens on, item 1
- `openspec/config.yaml` — focus line

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase8-tracking` capability is documentation-only and has no consumers in code.
