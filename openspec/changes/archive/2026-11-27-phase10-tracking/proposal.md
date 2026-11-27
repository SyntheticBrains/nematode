## Why

Phase 9 closed 2026-10-22 as **COMPLETE** on its STOP branch
([Logbook 103](../../../docs/experiments/logbooks/103-phase9-synthesis.md)): the connectome's
learning-speed advantage over its nulls was an artefact of the gap model's degree normaliser. The
maintainer chose the next direction on 2026-10-22, **fidelity on the plate**:

- standard on-lawn assays graded against real-worm data;
- controllers fitted to match the animal;
- the architecture ranking halted.

On 2026-10-23 the maintainer took the plan review's decisions:

- the project's question re-framed (D33);
- a 1-s point worm (D35);
- a wiring test that rewires the neuromodulator layer as well (D40);
- the body opening Phase 11 (D41).

Per the house workflow (a tracking change per phase: `phase5-tracking` to `phase9-tracking`), Phase
10 needs its living tracker before the first registration.

## What Changes

### 1. The Phase 10 tracking change

`openspec/changes/phase10-tracking/` holds a proposal, a design, tasks and a spec. `tasks.md` is the
living checklist, block by block:

- **H**, housekeeping;
- **E + T**, the 1-s worm, primitives, assays and the batched evaluator;
- **B**, the benchmark;
- **N**, the neuromodulator layer from data;
- **F.0**, the fitting methods' positive control;
- **C**, controllers and the fidelity table;
- **W**, the wiring question on fidelity;
- the synthesis.

`design.md` makes D33–D41 binding, carries the inherited execution standards with Phase 10's
additions, and holds the seed-band table. This is a tracking scaffold, not an implementation change.

### 2. The roadmap, AGENTS.md, README, config and literature watch

- `docs/roadmap.md` is now **v5.0**:
  - the vision and the Executive Summary's question are re-framed, with the earlier wording kept as
    the record;
  - the phase table gains Phase 10;
  - the **Phase 10 section**: goal, why, what the review established, the shipment tracker, D33–D41,
    the deliverables, validation targets, exit criteria, risks, novelty map and go/no-go;
  - Phase 10's success levels.
- `AGENTS.md`, `README.md` and `openspec/config.yaml` carry the re-framed question, and the README's
  wiring result and status are restated after Phase 9.
- `docs/research/literature-watch/context.md` and `seeds.toml` are re-aimed at Phase 10's questions
  (protocol principle 13; the Phase 9 close deferred this to the plan review).

### 3. Not in Phase 10

- the architecture ranking on learning speed (halted 2026-10-22);
- new thermal, predator, multi-agent or oxygen work in the required blocks. Thermotaxis is the second
  tier, and the optional MAY items (thermotaxis on the new primitives; bordering against an O₂ field
  with npr-1) remain allowed;
- the body and posture (Phase 11);
- reward-only panels beyond C.3;
- npr-1 and cat-2 as **required** mutant tests: O₂ sensing and dopamine are not modelled in the required
  blocks. npr-1 may still enter as the optional O₂-field bordering comparison.

## Capabilities

**Added**: `phase10-tracking`. Its requirements cover:

- the living Phase 10 checklist;
- the binding status of D33–D41;
- a benchmark that is void unless its controls behave;
- a fitting method that clears parameter recovery before it fits real data;
- the wiring reading at equal budget, on graded distance, with every outcome named.

**Modified**: none.
