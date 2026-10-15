## Why

Phase 8 closed 2026-10-14 as **COMPLETE** with the shipment decision **GO to a plan review** ([Logbook 094](../../../docs/experiments/logbooks/094-phase8-synthesis.md)): its first half took block V apart — on hard350 the wiring's learning-speed lead over a degree-preserving null came from the null's rewired gap junctions and its rewired sensory–motor boundary — and what survived its second half's embodiment is **gap-junction placement**, +0.096 `auc_success` on the point worm ([Logbook 082](../../../docs/experiments/logbooks/082-plastic-gaps.md)) and +0.126 through the kinematic body ([Logbook 092](../../../docs/experiments/logbooks/092-thermal-body.md)). The Phase 9 plan was then reviewed on 2026-10-14 and put through a three-lens adversarial critique on 2026-10-15 — a hostile referee, an execution-and-protocol lens and a research-strategy lens, each given the draft rather than its conclusions — which together ratified decisions **D23–D32** as roadmap **v4.5** § Phase 9, *Screen, then Mechanism — the Gap Junctions, and the Ranking Under a Trigger*. The critique's load-bearing finding is that the surviving effect has an untested model-level explanation — in the settling loop a gap junction is additive drive scaled by raw synapse counts, with the junctions touching ALA, CAN and PVD carrying 38% of all effective coupling and a gap scale that has never been swept under PPO — so the phase opens with an operating-point screen (G.0) and nothing in the mechanism block registers until it has read.

Per the house workflow (a tracking change per phase: `phase5-tracking`, `phase6-tracking`, `phase6b-tracking`, `phase7-tracking`, `phase8-tracking`), Phase 9 needs its living tracker before the first registration, so the committed work has an honest home as `[ ]` not-started tasks, D23–D32 have a binding amendment path, the seed bands are reserved per registration before any pilot touches them, and the execution-protocol standards Phases 6, 7 and 8 paid for — the thirteen principles of the [phase protocol](../../../docs/research/phase-protocol.md) and the five additions 8b made to it — are pinned where every rung change must inherit them.

## What Changes

### 1. Phase 9 Tracking Change

Create `openspec/changes/phase9-tracking/` with proposal/design/tasks/spec. The `tasks.md` is the living checklist for the two shipments — **9a** (*Screen*: block H housekeeping; block M, the instrument at long horizons — M.1 late collapse, M.2 wild type against wild type, M.3 variance components, M.4 the secondary body statistic; G.0 the gap model's operating-point screen; the mechanism block G.1–G.3b gated on G.0; R.1 the per-family positive controls through the body; the 9a synthesis) and **9b** (*Mechanism's consequences and the ranking*: R.2 the six-family ranking through the body under D27's trigger; G.5–G.8; the K block; the P block; E.1 and T.1; the phase synthesis) — plus the MAY items. Every Phase 9 milestone PR updates this checklist as part of its diff.

This is a **tracking scaffold**, not an implementation change. Load-bearing choices already decided are in `docs/roadmap.md` § Phase 9 § Pre-registered design decisions (D23–D32, authoritative); choices deliberately left open (the gap-scale knob's form and the unit-weight arm's normalisation, the totals-held construction, the gap nulls' boundary sets on the point worm and the body, the scatter rule's degree stratum width, the collapse definition's parameters, M.2's transplant rule and dataset 7's licence, R.1's per-family entropy rule, the set-point task's cultivation phase, E.1's drag coefficients, T.1's measurement protocol) are recorded in `design.md` as open questions, resolved inside per-milestone changes. The seed-band table — pilots in the 9xxx band, panels in the 2xxx band and above, one reserved row per registration — lives in `design.md`.

### 2. Scope explicitly NOT in Phase 9

Copied from the roadmap's *Not in Phase 9* paragraph; each item is kept in § Future Directions with its reason:

- **Cross-species work** — the *P. pacificus* data have no gap junctions and no loader, so they cannot carry the one wiring effect that holds; the dauer, whose gap junctions are unreleased (D28).
- **NEAT and evolution on the wiring beyond G.3's two-sided reading** — a placement search is 1,280 training runs at a fitness noise half the effect; MAY behind G.3b and T.1.
- **A further chemical-wiring rung** — four readings inert or unresolved under fair nulls, no named confound removed.
- **The dynamical substrate as built** — [Logbook 080](../../../docs/experiments/logbooks/080-across-step-state.md) failed its control by 9–19× the margin; the ohmic-coupling form returns only as G.0's `artefact` remedy.
- **Multi-agent, pheromones, red queen, 3D, Sibernetic, neuropeptides, spiking and neuromorphic work; the uniform substrate-writing rule programme; whole-organism fidelity.**

### 3. Roadmap, README, config, Logbook 082 and the literature watch

These ride along in the same PR because they were agreed together at the review and are small against the Phase 9 section itself:

- `docs/roadmap.md` — the **Phase 9 section** (goal, scope decision, what the review established, the shipment tracker, D23–D32, the MUST/SHOULD/MAY deliverables, validation targets, exit criteria, risk table, novelty map, go/no-go), with the tracker pointer in its *How to orient* paragraph; the version and date header and the table of contents; the **North Star restatement** (evolution kept, its condition named — a wiring result that makes the search space affordable — served at zero cost by G.3's two-sided reading); the **timeline row** for Phase 9 and the status paragraph; the **Future Directions absorption note** mapping Logbook 094's nine open items onto G.0, G.1, G.2, G.3/G.3b, G.8, M.1, K, E.1, R.1/R.2, M.2, M.3, P, G.0(g), D28 and D30; the **dated record corrections** — Hong et al. 2019 (eLife 8:e47155) for the *P. pacificus* chemotaxis homologs where § Phase 7 D3 cited "Han et al."; **RIM–AIB** for the learning-tuned electrical synapse where § Phase 8 recorded RMG/AIB; **ElegansBot's code licence is MIT** (CC BY is the eLife paper's); Lee 2026 at **v2** — each original sentence left as written; the **new related-work paragraph** (Churchland & Garcia-Ojalvo 2025; "The digital sphinx" 2026; An 2026; the `arterialist/active-inference` repository; version notes for the 2026 preprints); **Technical Debt § Active for Phase 9** (items 16–22: the gap-coupling scale knob and G.0 wirings; per-update PPO statistics and checkpoints; the gap-null family as rewiring modes; body action-space heads for the remaining families; the male sheets and the Witvliet chimera; the repaired behavioural references; issue #206 linked to T.1); **Phase 9 success levels** (minimum viable, target, stretch); the **fidelity-ladder row** (level 5++, no new fidelity rung by design); the **architecture-comparison protocol note on frame stability** (through a body the sweep re-runs once per stable reference frame, under D27's four conditions — "wait for fidelity" is not a trigger); and the closing paragraph.
- `README.md` — the status sentence gains Phase 9 (roadmap v4.5, *screen, then mechanism*, planned 2026-10-15).
- `openspec/config.yaml` — the focus line moves from Phase 8 to Phase 9.
- `docs/experiments/logbooks/082-plastic-gaps.md` — one dated correction note (RIM–AIB, Choi et al. 2020), the original sentence left in place.
- `docs/research/literature-watch/context.md` and `seeds.toml` — re-aimed at Phase 9's questions (protocol principle 13; task H.1).

## Capabilities

**Added**: `phase9-tracking` (new) — requirements covering the living Phase 9 checklist; the binding status of D23–D32; a positive control before any connectome arm on a new component (the body heads for the new families, any corrected gap model, the pirouette head, the set-point task); operating-point discipline, now including that no gap contrast runs before G.0 has read and that every block-G result cites its scale and depth; the screen gating the mechanism block; the new-reference-frame rule for body-substrate results; the ranking running only under its trigger (D27); collapse as a registered statistic (D25); every null stating what it does not preserve, with its matched-statistics table and shortcut audit (D24); artefact retention registered at phase start; the two-shipment completion semantics; and the scope exclusions. This capability lives until the Phase 9 synthesis archives alongside it.

**Modified**: none.

## Impact

**Docs:**

- `openspec/changes/phase9-tracking/proposal.md` — this file
- `openspec/changes/phase9-tracking/design.md` — tracker-level decisions, the execution-protocol standards (B.1–B.21), the seed-band table, open questions, risks
- `openspec/changes/phase9-tracking/tasks.md` — living Phase 9 checklist (9a / 9b + syntheses + MAY)
- `openspec/changes/phase9-tracking/specs/phase9-tracking/spec.md` — new capability
- `docs/roadmap.md`, `README.md`, `openspec/config.yaml`, `docs/experiments/logbooks/082-plastic-gaps.md`, `docs/research/literature-watch/context.md`, `docs/research/literature-watch/seeds.toml` — as listed under § What Changes 3

**Code:** None.

**Configs:** None.

## Breaking Changes

None.

## Backward Compatibility

No runtime behaviour affected. The new `phase9-tracking` capability is documentation-only and has no consumers in code.
