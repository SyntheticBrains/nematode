## Overview

Phase 6 (Connectome Substrate & Architecture Comparison) is a four-layer build (L0 connectome substrate → L1 architecture-plugin → L2 PPO weight search → L3 NEAT topology search) plus continuous-physics + Rung 2 chemical gradients + corrected ASH/ADL nociception + ≥1 real-worm validation, with three mid-phase decision gates. The roadmap (`docs/roadmap.md` § Phase 6, approximately lines 569-710) is the canonical strategic document; this tracking change is its sub-task working artefact. Phase 6 spans ~6-10 months and many AI sessions. This design records the cross-session decisions whose re-litigation would consume disproportionate session budget, so each subsequent milestone change can pick up the framing without re-deriving it.

## Goals / Non-Goals

**Goals:**

- A future AI session can answer "what's the next Phase 6 layer/milestone?" by reading two files (this `tasks.md` and the roadmap Phase 6 block).
- Each Phase 6 milestone PR has a single canonical place to mark progress (this `tasks.md`).
- Five Phase 6 design decisions (tranching, connectome data source, gate discipline, fixed architecture-family scope, fixed behavioural scope) are recorded once, not re-derived per session.
- The three mid-phase gate decisions are indexed from this change so a reader can see, at a glance, where each go/no-go landed in writing.
- The tracking artefact decays gracefully: when the Phase 6 synthesis logbook publishes, this change archives alongside it.

**Non-Goals:**

- Real-time progress dashboards (the roadmap status table + this tracker's status headers are enough).
- Automated milestone status (humans/agents update the checklist manually as part of milestone PRs).
- Replacing per-layer / per-milestone OpenSpec changes — those still happen; this scaffold is *additional* coordination.
- Inventing the Phase 6 strategy — the strategy lives in the roadmap. This change records the *tracking discipline* around it.
- GitHub Issues / Project boards — optional mirror, decided at PR time per milestone.

## Design Decisions

### Decision 1: Tranching policy — Phase 6 splits along the L0/L1/L2/L3 layer boundaries

Phase 6 is decomposed into tranches that map directly to the roadmap's four-layer stack:

- **Tranche 1 — L0 only.** Connectome data import + validation + cross-checks + forward-pass smoke. No plugin work; no training; no behaviour evaluation. First OpenSpec change: `add-connectome-substrate`.
- **Tranche 2 — L1 + first L2 attempt.** Architecture-plugin interface refactor; plugin-parity test; the first PPO-on-connectome training run that exercises L0+L1+L2 end-to-end. This tranche triggers **Gate 1** (month ~2 — L0 working in anger).
- **Tranche 3 — L2 across the MUST architecture-family set.** Full weight-search sweep on all eight MUST families × three behaviours. Triggers **Gate 2** (month ~4-5 — L1 plugin parity proven) and feeds **Gate 3** (month ~7-8 — L2 results in hand).
- **Tranche 4 — L3 NEAT topology search + supporting infrastructure.** TensorNEAT integration; matched-vs-asymmetric capacity head-to-head; the unconstrained-vs-connectome ranking.
- **Tranche 5 — Cross-cutting substrate work.** Continuous-2D physics; Rung 2 chemical gradients with log-concentration adaptation kinetics; corrected ASH/ADL nociception; ≥1 real-worm validation. Some of this may interleave with Tranches 2-3 if a target behaviour needs it earlier (e.g. predator evasion needs the corrected nociception); the tracker accommodates re-sequencing without re-design.
- **Tranche 6 — Phase 6 synthesis logbook.** Closes Phase 6; archives this change.

**Why this split.** Each tranche has a separable failure mode and a separable pivot path. The L0 Risk-mitigation row ("if c302 takes > 2 months, drop to hand-curated subset") needs Tranche 1's evidence in isolation — bundling L0 with L1 buries the substrate-import diagnosis under a plugin-design diagnosis. Same logic at every layer: Gate 2 (L1 parity) is decidable only if L1 ships before the full L2 sweep starts.

**Alternative considered.** A single mega-change `add-phase6` covering all of L0+L1+L2+L3. Rejected because it would (a) be impossible to review and (b) defeat the per-gate decision discipline that Decision 3 below relies on.

### Decision 2: Connectome data source — Cook 2019 hermaphrodite via OpenWorm `cect` is L0 primary; c302/NeuroML is deferred to an export path

The L0 connectome import has two plausible primary sources:

- **Option A (chosen): OpenWorm `cect` / ConnectomeToolbox.** Python-native; ships Cook et al. 2019 hermaphrodite + Witvliet 2021 + multiple other datasets behind one API; widely cited; format is already-parsed adjacency/connectivity matrices, not raw NeuroML XML.
- **Option B (rejected as primary): NeuroML 2 / c302.** The OpenWorm c302 pipeline lives upstream of `cect` and is the canonical format for Sibernetic body-physics interop. Useful as an *export* target if/when Phase 6 needs to hand connectomes to Sibernetic for behavioural-fidelity validation — but using it as the primary *import* path means Phase 6 absorbs the NeuroML parsing + c302 metadata + element-tree schema complexity before it can validate a single neuron count.

**Why this matters now and not inside `add-connectome-substrate`.** The Risk-mitigation row "L0 c302 import takes > 2 months → drop to hand-curated subset" assumes c302 is the import path. With Option A as primary, the failure mode tightens to "Cook 2019 via `cect` doesn't expose the metadata we need" — a much narrower diagnosis, and one where the hand-curated-subset pivot becomes a deliberate substrate-engineering decision rather than a c302-format escape valve. Recording the choice here means `add-connectome-substrate` can frame its scope around Option A from the start; it does not have to argue for it.

**Export path remains in scope.** If a later Phase 6 sub-task needs to hand a topology to OpenWorm Sibernetic for body-physics validation, NeuroML/c302 returns as the export format. That's a different concern from import-time data quality and lives in the Future Directions arc.

### Decision 3: Mid-phase gate discipline — every gate produces a written go/no-go inside the relevant OpenSpec change

The roadmap defines three mid-phase decision gates (see `docs/roadmap.md` § Phase 6 § Mid-phase decision gates):

- **Gate 1 (month ~2)**: L0 import working — connectome substrate loaded, validated, basic-MLP-PPO trainable on it.
- **Gate 2 (month ~4-5)**: L1 plugin parity — adding a new architecture demonstrably ≤ 1 week of work.
- **Gate 3 (month ~7-8)**: L2 results across architectures — weight-search results across the MUST architecture-family set and all three behaviours in hand.

Each gate is the decision boundary between a tranche and its successor. Each must produce a **written** go/no-go decision inside the relevant OpenSpec change's `tasks.md` or a dedicated decision section in its logbook — not a silent continuation, not "the next milestone just started so I guess Gate 1 passed". The `tasks.md` here indexes those decisions by linking to them once they land.

**Why.** Phase 5's M4/M5/M6.x STOP verdicts were valuable specifically because the diagnosis was written down at the gate point — substrate constraint, architecture asymmetry, wrong abstraction. The same discipline applied at Phase 6's mid-phase gates protects against the failure mode where a layer "kind of works" and the project slides into the next layer without the underlying gate being clearly passed or pivoted.

**Pivot path is part of the gate.** Each gate has a documented pivot (the Risk-mitigation rows). A gate that triggers its pivot is also a written decision — it produces an amended scope and a new tranche definition. "Gate FAILED → pivot to hand-curated subset" is a successful gate outcome by this design; only an undocumented slide past the gate is a failure of discipline.

### Decision 4: Architecture-family scope is fixed at the roadmap-defined eight MUST + one MAY

Phase 6's L1 plugin parity test is meaningful only if the set of architecture families it accommodates is fixed. The roadmap pins this set at:

- **MUST (eight families):** connectome-constrained, MLP-PPO, LSTM/GRU-PPO, spiking (PPO-trained), reservoir, quantum (Phase 2 representatives), hybrid quantum-classical, NEAT-evolved.
- **MAY (one family):** transformer / attention-based.

**Why this matters as a tracker-level decision.** It's tempting to add a ninth architecture family mid-Phase-6 because some new variant looks interesting — Phase 0-3 added 19 architectures total under exactly this pressure. Phase 6's value proposition is the *comparison*, which requires that the set of compared rows is stable across the sweep. Expansion mid-phase invalidates already-completed L2/L3 results for the rows that ran first.

**Mechanism.** A new family proposed mid-Phase-6 must amend *this* tracking change (a follow-up commit to `proposal.md` + `tasks.md`), not be added inside an individual milestone change. The amend forces the project to look at the cross-family budget impact (each new family is +N seeds × three behaviours × two layers of search) before the family is added.

### Decision 5: Behavioural scope is fixed at three (klinotaxis, thermotaxis, predator evasion)

The same anti-scope-creep logic as Decision 4, applied on the behaviour axis. Phase 6 commits to three behaviours; aerotaxis / pheromones / multi-agent dynamics are explicitly *deferred*.

**Why this matters as a tracker-level decision.** Each behaviour added is +1 axis on the L2 sweep, +1 axis on the L3 sweep, and +1 set of training configs and analysis scripts per architecture family. A fourth behaviour added mid-phase pushes Phase 6 past 10 months without strengthening the headline platform claim.

**Mechanism.** Same as Decision 4 — a new behaviour must amend this tracking change. The amend forces the cross-tranche budget impact to be looked at before the behaviour is added to any milestone change.

## Tracking Strategy

Three artefacts answer "where are we in Phase 6?":

1. **`openspec/changes/phase6-tracking/tasks.md`** — sub-task checklist updated by every Phase 6 milestone PR.
2. **`openspec/changes/<phase-6-milestone>/`** — per-milestone proposal/tasks/design/specs, archived on milestone merge.
3. **`docs/roadmap.md` Phase 6 section** — layer-level status, updated as part of every Phase 6 milestone PR.

A future AI session orients by:

- Reading the roadmap Phase 6 block first (per-layer current status).
- Reading this `tasks.md` for sub-task granularity.
- Reading active `openspec/changes/<milestone>/` if a specific milestone is in flight.
- Reading the latest published `docs/experiments/logbooks/0XX/` if a milestone has completed evaluation.

## Maintenance

- Every Phase 6 milestone PR updates `tasks.md` (mark sub-tasks complete) and `docs/roadmap.md` Phase 6 Milestone Tracker (one-line status update).
- This change does not archive until the Phase 6 synthesis logbook ships.
- If Phase 6 deviates substantially from this plan (e.g. a tranche is dropped, gate criteria are softened, an architecture family is added or removed), update `tasks.md` + `proposal.md` + this `design.md` to reflect reality — the checklist is descriptive, not aspirational. Git history is the audit trail of the change.

## Risks

1. **The checklist drifts from reality if PRs forget to update it.** Mitigation: each per-milestone OpenSpec change's `tasks.md` includes "update phase6-tracking tasks.md" as an explicit sub-task; spec scenario 1 below makes this a requirement.
2. **A mid-phase gate slides by without a written decision.** Mitigation: spec requirement 3 below makes the written gate decision a hard requirement; this `tasks.md` has explicit Gate 1 / Gate 2 / Gate 3 checkboxes that cannot be ticked without a link to the decision artefact.
3. **The fixed architecture-family and behavioural scopes (Decisions 4 + 5) get expanded informally inside individual milestone changes, bypassing the amend mechanism.** Mitigation: reviewers of per-milestone PRs check the proposed architecture/behaviour set against this `proposal.md`; any addition must amend `proposal.md` first.
4. **Tranche 5 (cross-cutting substrate work) gets dropped if Tranches 2-3 consume more time than budgeted.** Mitigation: Rung 2 gradients + corrected ASH/ADL nociception + ≥1 real-worm validation are roadmap exit criteria (MUST), not nice-to-haves; the tracker's Tranche 5 sub-tasks remain visible and unticked, and Gate 3 is the natural point to re-budget if Tranche 5 hasn't started.
5. **Phase 6 overshoots 10 months and the Phase 6a/6b sub-phase split (roadmap Risk-mitigation row) is needed.** Mitigation: that split is a documented pivot, not a tracker failure. If triggered, this change amends to reflect 6a's exit criteria and 6b's deferred scope; archive happens on 6a's synthesis logbook publication and 6b inherits a fresh tracking change if scope warrants.
