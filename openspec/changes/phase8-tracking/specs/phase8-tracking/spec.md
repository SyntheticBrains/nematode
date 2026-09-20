## ADDED Requirements

### Requirement: Phase 8 Living Shipment Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase8-tracking/tasks.md` covering Phase 8's two shipments (8a: block A, B.1, B.2 and the 8a synthesis; 8b: C.0–C.5, B.3 + D.1 and the phase synthesis) plus the MAY items, at sub-task granularity. Every Phase 8 milestone PR SHALL update this file as part of its diff. This change SHALL remain unarchived until the Phase 8 synthesis publishes, at which point it archives alongside that synthesis change.

#### Scenario: Future session orients to Phase 8

- **GIVEN** a fresh AI session resumes Phase 8 work
- **WHEN** the agent reads `openspec/changes/phase8-tracking/tasks.md` and `docs/roadmap.md` § Phase 8
- **THEN** the agent SHALL be able to identify the current shipment, sub-task, and applicable design decisions (D15–D20) without re-deriving the plan or the review corrections

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 8 milestone PR (e.g. the A.1 control change, a rung change, or the synthesis change) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase8-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant shipment status header

### Requirement: Design Decisions D15–D20 Are Binding

Phase 8 milestone changes SHALL conform to the ratified design decisions D15–D20 recorded in `docs/roadmap.md` § Phase 8 § Pre-registered design decisions. Amending a D-decision SHALL require a dated note in the `phase8-tracking` change (with a matching roadmap edit) before the affected milestone change merges.

#### Scenario: Milestone change contradicts a D-decision

- **GIVEN** a Phase 8 milestone change proposes work that contradicts a D-decision (e.g. a single unstated definition of shared initialisation against D15, a per-arm muscle gain against D18, the connectome generating the undulation at C.1 against D19, or 8b work before the 8a synthesis against D20)
- **WHEN** the change is reviewed
- **THEN** the change SHALL be blocked until either it conforms, or a dated amendment to the relevant D-decision is recorded in `phase8-tracking` and `docs/roadmap.md`

### Requirement: Positive Control Before Any Connectome Arm on a New Component

Each new substrate component — the dynamical substrate (B.2), the anatomical muscle readout and kinematic body (C.1), the rod-chain body (C.2), and any body-level rhythm generator (D19) — SHALL clear an MLP-PPO positive control on the target cell before any connectome arm runs on it. A rung whose positive control fails SHALL close with the diagnosis as its deliverable and SHALL NOT report a wiring result.

#### Scenario: Connectome arm proposed on an uncontrolled component

- **GIVEN** a milestone change proposes a wiring contrast on a component with no recorded MLP-PPO positive control on that cell
- **WHEN** the change is reviewed
- **THEN** the contrast SHALL be blocked until the positive control has run and its outcome is recorded in the tracker

### Requirement: Operating-Point Discipline

No Phase 8 wiring contrast SHALL run at a pin that has not been swept on the learner it uses (D16). The A.2 sensitivity surface SHALL be recorded before B.1 registers, and every later rung SHALL cite the swept point it runs at. A setting inherited from a previous rung SHALL be treated as a hypothesis at the new one and re-checked when the substrate or the readout changes.

#### Scenario: Contrast registered at an inherited pin

- **GIVEN** a milestone change registers a wiring contrast at a value of `plasticity_rate`, readout width, `forward_pass_depth`, `initial_log_std` or `trace_decay` that A.2 did not sweep for that learner and substrate
- **WHEN** the change is reviewed
- **THEN** the registration SHALL be blocked until the pin is swept or the A.2 surface is extended to cover it

### Requirement: Body-Substrate Results Are a New Reference Frame

Results measured through the C.1 or C.2 body SHALL be reported against floors and baselines re-established on that substrate, and SHALL NOT be reported as controlled deltas against Logbook 029, block V, or any pre-body result. Every C.0 platform change (step–time calibration, signed speed, the proprioceptive channel) SHALL land, validate and freeze before C.1 registers.

#### Scenario: Body result compared to a pre-body number

- **GIVEN** a milestone change reports a body-substrate result as a percentage change against a pre-body baseline
- **WHEN** the change is reviewed
- **THEN** the comparison SHALL be restated as qualitative, and the rung's own baseline campaign SHALL be cited as the quantitative frame

### Requirement: Shipment Completion Semantics

An 8a GO decision SHALL record the phase state as "8a complete / 8b pending" and SHALL NOT mark Phase 8 complete. 8b work SHALL NOT start until the 8a synthesis has assigned every 8a criterion one of the five statuses (met, unmet-with-reason, deferred-with-destination, superseded-by-result, unreachable-with-reason). Phase 8 SHALL be marked ✅ COMPLETE only when the 8b synthesis publishes with every criterion assigned a status; "well underway" SHALL never satisfy completion. Splits and stops SHALL be invoked on pre-registered criteria, never on month counts.

#### Scenario: 8a closes while 8b is pending

- **GIVEN** A.1, A.2, A.4 and B.1 are complete with statuses assigned and the 8a synthesis writes GO
- **WHEN** the 8a shipment decision is recorded
- **THEN** the roadmap Phase 8 status SHALL read "8a complete / 8b pending"
- **AND** Phase 8 SHALL NOT be marked COMPLETE until the 8b synthesis lands

#### Scenario: A rung is deferred out of the phase

- **GIVEN** a rung the completion predicate names becomes unrunnable on evidence (e.g. C.2 exceeds its registered cost budget), and a dated decision defers it
- **WHEN** that decision is recorded
- **THEN** the rung SHALL carry the status *deferred-with-destination* with the destination named, in the tracker and the roadmap in the same PR, and SHALL NOT be dropped silently

### Requirement: Scope Exclusions

Phase 8 SHALL NOT include: cross-species work (the *P. pacificus* comparison, the dauer pathfinder, weight transplant); the male–hermaphrodite wiring contrast; multi-agent, pheromone, red-queen or ecological co-evolution work; evolution in any form (6b NEAT stays in `phase6b-tracking`; Lamarckian and transgenerational work stays closed); structural plasticity across development; the neuropeptide layer as a rule substrate; spiking-STDP or neuromorphic deployment; 3D environments, the Sibernetic body or ion-channel neurons; or the uniform substrate-writing rule programme.

#### Scenario: Excluded scope proposed as Phase 8 work

- **GIVEN** a Phase 8 milestone change proposes any excluded item above as Phase 8 scope
- **WHEN** the change is reviewed
- **THEN** the addition SHALL be blocked as out of Phase 8 scope; committing to it requires a dated D-decision amendment with its budget impact, recorded in `phase8-tracking` and `docs/roadmap.md`
