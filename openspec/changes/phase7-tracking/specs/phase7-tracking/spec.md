## ADDED Requirements

### Requirement: Phase 7 Living Shipment Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase7-tracking/tasks.md` covering Phase 7's three shipments (7a-i platform freeze + minimal rule + 2×2 panel; 7a-ii receptor-grounded neuromodulator stack; 7b cross-species comparative learning) plus the synthesis, at sub-task granularity. Every Phase 7 milestone PR SHALL update this file as part of its diff. This change SHALL remain unarchived until the Phase 7 synthesis publishes, at which point it archives alongside that synthesis change.

#### Scenario: Future session orients to Phase 7

- **GIVEN** a fresh AI session resumes Phase 7 work
- **WHEN** the agent reads `openspec/changes/phase7-tracking/tasks.md` and `docs/roadmap.md` § Phase 7
- **THEN** the agent SHALL be able to identify the current shipment, sub-task, and applicable design decisions (D1–D13) without re-deriving the plan or the adversarial-review amendments

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 7 milestone PR (e.g. `add-l4-trace-substrate`, a panel change, or the synthesis change) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase7-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant shipment status header

### Requirement: Design Decisions D1–D13 Are Binding

Phase 7 milestone changes SHALL conform to the ratified design decisions D1–D13 recorded in `docs/roadmap.md` § Phase 7 § Pre-registered design decisions. Amending a D-decision SHALL require a dated note in the `phase7-tracking` change (with a matching roadmap edit) before the affected milestone change merges.

#### Scenario: Milestone change contradicts a D-decision

- **GIVEN** a Phase 7 milestone change proposes work that contradicts a D-decision (e.g. a spiking-STDP MUST arm against D1, an L4 package inside `quantumnematode/plasticity/` against D8, or a quantitative pass/fail comparison to a PPO-trained arm against D2/D10)
- **WHEN** the change is reviewed
- **THEN** the change SHALL be blocked until either it conforms, or a dated amendment to the relevant D-decision is recorded in `phase7-tracking` and `docs/roadmap.md`

### Requirement: Substrate Freeze Before the L4 Panel

The L4 2×2 panel SHALL NOT run until the pre-panel platform tranche is complete and the substrate is declared frozen: D5 dead-key removal (behaviour unchanged), D7 state-dependent action `std`, the D7 validation gate (Logbook 036 thermotaxis assay re-run with the klinokinesis signature present), the post-D7 n=8 re-baseline of the reference arms, and load-time config validation extended to the new plasticity keys with the freeze declared (tasks P.1–P.5). Any comparison spanning a subsequent substrate change SHALL be reported as qualitative.

#### Scenario: Panel proposed before the D7 validation gate

- **GIVEN** a milestone change proposes launching the 2×2 panel
- **WHEN** the change is reviewed and the D7 validation gate has not recorded a present klinokinesis signature (or the post-D7 re-baseline is absent)
- **THEN** the panel SHALL be blocked until the pre-panel tranche tasks (P.1–P.5) are complete and the freeze is declared

### Requirement: Execution-Protocol Standards for Panels and Sweeps

Every Phase 7 panel or sweep SHALL conform to the inherited execution-protocol standards (design.md § Decision B): n ≥ 8 paired-seed with BH-FDR within-pass and converged-fraction reported; a uniform budget set by the slowest converger with a convergence audit (level-agnostic metric) before any ranking is read; a metric audit before every panel on a new behaviour regime; byte-identical-when-off for every new mechanism; load-time config validation covering new keys; and no wiring claim from a single seed or single fit (ensemble-invariance, claim-discipline bar (a)).

#### Scenario: Ranking proposed from an unaudited or under-powered run

- **GIVEN** a milestone change reports a ranking, a 2×2 verdict, or a cross-species comparison
- **WHEN** the results rest on unaudited convergence, a threshold-calibrated detector on a new learning regime, n < 8, or a single-seed wiring claim
- **THEN** the result SHALL be treated as preliminary and SHALL NOT close a shipment task or exit criterion until the protocol standards are met

### Requirement: Shipment Completion Semantics

A 7a GO decision SHALL record the phase state as "7a complete / 7b pending" and SHALL NOT mark Phase 7 complete. Phase 7 SHALL be marked ✅ COMPLETE only when the 7b comparative cross-connectome sweep has shipped and the synthesis publishes; "well underway" SHALL never satisfy completion. Splits SHALL be invoked on pre-registered criteria (including by success), never on month counts.

#### Scenario: 7a closes while 7b is pending

- **GIVEN** shipments 7a-i and 7a-ii are complete with D2-bar results in hand
- **WHEN** the 7a shipment decision is recorded
- **THEN** the roadmap Phase 7 status SHALL read "7a complete / 7b pending" (mirroring the 6a/6b pattern)
- **AND** Phase 7 SHALL NOT be marked COMPLETE until the 7b sweep and the synthesis land

### Requirement: Scope Exclusions

Phase 7 SHALL NOT include: the Phase 6b NEAT topology search (tracked in `phase6b-tracking`, decoupled per D13); the co-evolution test (deferred with no scheduled destination); the neuropeptide signalling layer as a plasticity-rule substrate; the full dynamic-diffusion PDE; or *C. briggsae* transfer.

#### Scenario: Excluded scope proposed as Phase 7 work

- **GIVEN** a Phase 7 milestone change proposes any excluded item above as Phase 7 scope
- **WHEN** the change is reviewed
- **THEN** the addition SHALL be blocked as out of Phase 7 scope; committing to it requires a separate re-scoping decision documented in the owning tracker (`phase6b-tracking` for NEAT) or a dated D-decision amendment with its budget impact
