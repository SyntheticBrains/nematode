## ADDED Requirements

### Requirement: Phase 5 Living Milestone Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/2026-04-26-phase5-tracking/tasks.md` covering every Phase 5 milestone (M-1 through M7) at sub-task granularity, updated by every Phase 5 milestone PR as part of its diff.

#### Scenario: Future session orients to Phase 5

- **GIVEN** a fresh AI session resumes Phase 5 work
- **WHEN** the agent reads `openspec/changes/2026-04-26-phase5-tracking/tasks.md` and the `docs/roadmap.md` Phase 5 block
- **THEN** the agent SHALL be able to identify the current in-progress milestone and the next milestone to start without further codebase exploration

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 5 milestone PR (e.g. M0, M1, …, M7) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/2026-04-26-phase5-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the milestone status header

#### Scenario: Checklist outlives individual milestones

- **GIVEN** a Phase 5 milestone OpenSpec change (e.g. `2026-04-28-add-evolution-framework`) is archived
- **WHEN** archival completes
- **THEN** the `2026-04-26-phase5-tracking` change SHALL remain unarchived and continue to receive updates from subsequent milestone PRs
- **AND** archival of `2026-04-26-phase5-tracking` itself SHALL only occur alongside the M7 synthesis evaluation change

### Requirement: Roadmap Phase 5 Status Block

The `docs/roadmap.md` Phase 5 section SHALL include a Phase 5 Milestone Tracker sub-section listing each milestone with current status, updated as part of every Phase 5 milestone PR.

#### Scenario: Roadmap reflects current milestone progress

- **GIVEN** a Phase 5 milestone has just been completed
- **WHEN** the milestone PR is merged
- **THEN** the roadmap Phase 5 Milestone Tracker SHALL show that milestone's status as `complete` with a one-line summary
- **AND** the next milestone in sequence SHALL show status `in progress` if work has begun, otherwise `not started`

#### Scenario: Roadmap timeline table reflects phase status

- **GIVEN** Phase 5 has at least one milestone in progress (M-1 onward)
- **WHEN** a reader views the Timeline Overview table at the top of `docs/roadmap.md`
- **THEN** the Phase 5 row SHALL show status `🟡 IN PROGRESS` (not `🔲 PLANNED`)
- **AND** when M7 completes the row SHALL change to `✅ COMPLETE`

### Requirement: Recorded Phase 5 Decisions

The `proposal.md` of `2026-04-26-phase5-tracking` SHALL record the three Phase 5 design decisions (pilot-first, no QVarCircuit backwards compatibility, LSTMPPO+klinotaxis as first-class brain) so that subsequent sessions do not re-litigate them.

#### Scenario: Decision is recorded once

- **GIVEN** a Phase 5 design decision has been made (e.g. "no QVarCircuit backwards compatibility")
- **WHEN** the decision is finalized
- **THEN** it SHALL appear in `openspec/changes/2026-04-26-phase5-tracking/proposal.md`
- **AND** the rationale SHALL appear in `openspec/changes/2026-04-26-phase5-tracking/design.md`

#### Scenario: Decision reversal preserves audit trail

- **GIVEN** a recorded Phase 5 decision is reversed during the phase
- **WHEN** the reversal is committed
- **THEN** `proposal.md` and `design.md` SHALL be amended in a follow-up commit to the same change directory
- **AND** git history SHALL serve as the audit trail of the change
