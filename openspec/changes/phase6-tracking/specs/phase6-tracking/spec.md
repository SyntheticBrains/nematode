## ADDED Requirements

### Requirement: Phase 6 Living Milestone Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase6-tracking/tasks.md` covering every Phase 6 layer (L0–L4), cross-cutting deliverable (continuous-2D physics, Rung 2 chemical gradients, corrected ASH/ADL nociception, real-worm validation), mid-phase gate, and the Phase 6 synthesis logbook, at sub-task granularity. Every Phase 6 milestone PR SHALL update this file as part of its diff.

#### Scenario: Future session orients to Phase 6

- **GIVEN** a fresh AI session resumes Phase 6 work
- **WHEN** the agent reads `openspec/changes/phase6-tracking/tasks.md` and the `docs/roadmap.md` Phase 6 block
- **THEN** the agent SHALL be able to identify the current in-progress layer / tranche and the next sub-task to start without further codebase exploration

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 6 milestone PR (e.g. `add-connectome-substrate`, an L2 architecture-family sweep, an L3 NEAT run, a Tranche 5 substrate-work change) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase6-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant layer / tranche status header

#### Scenario: Checklist outlives individual milestones

- **GIVEN** a Phase 6 milestone OpenSpec change (e.g. `add-connectome-substrate`) is archived
- **WHEN** archival completes
- **THEN** the `phase6-tracking` change SHALL remain unarchived and continue to receive updates from subsequent milestone PRs
- **AND** archival of `phase6-tracking` itself SHALL only occur alongside the Phase 6 synthesis logbook change

#### Scenario: Scope expansion requires amending this change

- **GIVEN** a Phase 6 milestone change proposes adding a ninth architecture family or a fourth behaviour beyond the fixed scopes in [proposal.md § Decisions 4 + 5](proposal.md)
- **WHEN** the milestone change is reviewed
- **THEN** the addition SHALL be deferred until `openspec/changes/phase6-tracking/proposal.md` is amended in a separate commit to widen the fixed scope
- **AND** the amend SHALL document the cross-tranche budget impact of the addition

### Requirement: Roadmap Phase 6 Status Block

The `docs/roadmap.md` Phase 6 section SHALL include a Phase 6 Milestone Tracker sub-section listing each layer (L0–L4) and cross-cutting deliverable with current status, updated as part of every Phase 6 milestone PR.

#### Scenario: Roadmap reflects current layer progress

- **GIVEN** a Phase 6 layer (e.g. L0 connectome substrate) has just been completed
- **WHEN** the milestone PR is merged
- **THEN** the roadmap Phase 6 Milestone Tracker SHALL show that layer's status as `✅ complete` with a one-line summary
- **AND** the next layer in dependency order SHALL show status `🟡 in progress` if work has begun, otherwise `🔲 not started`

#### Scenario: Roadmap timeline table reflects phase status

- **GIVEN** Phase 6 has at least one layer in progress (P6-0 onward)
- **WHEN** a reader views the Timeline Overview table at the top of `docs/roadmap.md`
- **THEN** the Phase 6 row SHALL show status `🟡 IN PROGRESS` (not `🔲 PLANNED`)
- **AND** when the Phase 6 synthesis logbook publishes, the row SHALL change to `✅ COMPLETE`

### Requirement: Mid-Phase Gate Decision Visibility

Each of the three roadmap-defined mid-phase decision gates (Gate 1: L0 working at month ~2; Gate 2: L1 plugin parity at month ~4-5; Gate 3: L2 results across MUST architectures at month ~7-8) SHALL produce a written go/no-go decision inside the OpenSpec change that triggers the gate, and `openspec/changes/phase6-tracking/tasks.md` SHALL link to each decision once it lands.

#### Scenario: Gate decision is recorded in writing

- **GIVEN** a Phase 6 mid-phase gate (Gate 1, Gate 2, or Gate 3) is reached
- **WHEN** the triggering layer (L0 / L1 / L2) closes
- **THEN** the OpenSpec change for that layer SHALL include a written go/no-go decision in its `tasks.md` or its published logbook
- **AND** the decision SHALL state explicitly GO / PIVOT (with the chosen pivot path) / STOP (with the diagnosis)
- **AND** `openspec/changes/phase6-tracking/tasks.md` SHALL be updated to link to the decision

#### Scenario: Pivot triggered at a mid-phase gate

- **GIVEN** a mid-phase gate triggers its documented Risk-mitigation pivot (e.g. Gate 1 triggers the hand-curated-subset pivot for L0; Gate 2 triggers the L1 refactor pivot; Gate 3 triggers the Phase 6a / Phase 6b sub-phase split)
- **WHEN** the pivot is executed
- **THEN** the pivot SHALL be documented as a successful gate outcome (not a failure of discipline)
- **AND** the OpenSpec change for the triggering layer SHALL document the pivot's amended scope
- **AND** `openspec/changes/phase6-tracking/tasks.md` and `proposal.md` SHALL be amended to reflect the new tranche definitions

#### Scenario: A gate is reached without a written decision

- **GIVEN** the next layer's OpenSpec change opens before the prior layer's gate decision is written
- **WHEN** a reviewer observes the missing gate decision
- **THEN** the new layer's PR SHALL be blocked until the prior gate's decision is recorded in the prior layer's OpenSpec change and linked from `openspec/changes/phase6-tracking/tasks.md`
