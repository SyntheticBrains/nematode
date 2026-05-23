## ADDED Requirements

### Requirement: Phase 6 Living Tranche Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase6-tracking/tasks.md` covering every Phase 6 tranche (T1–T8) and every mid-phase gate, at sub-task granularity. Every Phase 6 milestone PR SHALL update this file as part of its diff.

#### Scenario: Future session orients to Phase 6

- **GIVEN** a fresh AI session resumes Phase 6 work
- **WHEN** the agent reads `openspec/changes/phase6-tracking/tasks.md` and the `docs/roadmap.md` Phase 6 block
- **THEN** the agent SHALL be able to identify the current in-progress tranche and the next sub-task to start without further codebase exploration

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 6 milestone PR (e.g. `add-connectome-substrate`, an L1 plugin refactor, a T4 L2 first-pass run, a T5 env-upgrade change, a T6 L2 final-pass run, a T7 NEAT run) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase6-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant tranche status header

#### Scenario: Checklist outlives individual milestones

- **GIVEN** a Phase 6 milestone OpenSpec change (e.g. `add-connectome-substrate`) is archived
- **WHEN** archival completes
- **THEN** the `phase6-tracking` change SHALL remain unarchived and continue to receive updates from subsequent milestone PRs
- **AND** archival of `phase6-tracking` itself SHALL only occur alongside the Phase 6 synthesis logbook change (Tranche 8)

### Requirement: Roadmap Phase 6 Status Block

The `docs/roadmap.md` Phase 6 section SHALL include a Phase 6 Tranche Tracker sub-section listing the eight tranches (T1–T8) plus L4 (marked deferred to Phase 7) with current status, updated as part of every Phase 6 milestone PR. The Phase 6 architecture-families table SHALL annotate each family as MUST / SHOULD / MAY per the scope in this change's `design.md` Decision 4.

#### Scenario: Roadmap reflects current tranche progress

- **GIVEN** a Phase 6 tranche (e.g. T1 L0 connectome substrate) has just been completed
- **WHEN** the milestone PR is merged
- **THEN** the roadmap Phase 6 Tranche Tracker SHALL show that tranche's status as `✅ complete` with a one-line summary
- **AND** the next tranche in dependency order SHALL show status `🟡 in progress` if work has begun, otherwise `🔲 not started`

#### Scenario: Roadmap timeline table reflects phase status

- **GIVEN** Phase 6 has at least one tranche in progress (P6-0 onward)
- **WHEN** a reader views the Timeline Overview table at the top of `docs/roadmap.md`
- **THEN** the Phase 6 row SHALL show status `🟡 IN PROGRESS` (not `🔲 PLANNED`)
- **AND** when the Phase 6 synthesis logbook publishes, the row SHALL change to `✅ COMPLETE`

### Requirement: Mid-Phase Gate Decision Visibility

Each of the three roadmap-defined mid-phase decision gates (Gate 1 at the close of Tranche 2; Gate 2 at the close of Tranche 5; Gate 3 at the close of Tranche 6) SHALL produce a written go/no-go decision inside the OpenSpec change that triggers the gate, and `openspec/changes/phase6-tracking/tasks.md` SHALL link to each decision once it lands.

#### Scenario: Gate decision is recorded in writing

- **GIVEN** a Phase 6 mid-phase gate (Gate 1, Gate 2, or Gate 3) is reached
- **WHEN** the triggering tranche (T2 / T5 / T6) closes
- **THEN** the OpenSpec change for that tranche SHALL include a written go/no-go decision in its `tasks.md` or its published logbook
- **AND** the decision SHALL state explicitly GO / PIVOT (with the chosen pivot path) / STOP (with the diagnosis)
- **AND** `openspec/changes/phase6-tracking/tasks.md` SHALL be updated to link to the decision

#### Scenario: Pivot triggered at a mid-phase gate

- **GIVEN** a mid-phase gate triggers its documented Risk-mitigation pivot (e.g. Gate 1 triggers the hand-curated-subset pivot for L0; Gate 2 triggers the L1 refactor pivot; Gate 3 triggers the Phase 6a / Phase 6b sub-phase split)
- **WHEN** the pivot is executed
- **THEN** the pivot SHALL be documented as a successful gate outcome (not a failure of discipline)
- **AND** the OpenSpec change for the triggering tranche SHALL document the pivot's amended scope
- **AND** `openspec/changes/phase6-tracking/tasks.md` and `proposal.md` SHALL be amended to reflect the new tranche definitions

#### Scenario: A gate is reached without a written decision

- **GIVEN** the next tranche's OpenSpec change opens before the prior tranche's gate decision is written
- **WHEN** a reviewer observes the missing gate decision
- **THEN** the new tranche's PR SHALL be blocked until the prior gate's decision is recorded in the prior tranche's OpenSpec change and linked from `openspec/changes/phase6-tracking/tasks.md`

### Requirement: Architecture-Family Scope is Fixed at the Four MUSTs

Phase 6's architecture-family MUST set is fixed at the four families documented in this change's `design.md` Decision 4: connectome-constrained, MLP-PPO, LSTM/GRU-PPO, and NEAT-evolved. SHOULD families (quantum, spiking) and MAY families (reservoir, hybrid, transformer) MAY be evaluated opportunistically in Tranche 6 if substrate-change × architecture-family interaction looks scientifically worthwhile, but their results SHALL NOT gate any Phase 6 exit criterion.

#### Scenario: Adding a new architecture family mid-Phase-6

- **GIVEN** a Phase 6 milestone change proposes adding a tenth architecture family beyond the nine in `design.md` Decision 4
- **WHEN** the milestone change is reviewed
- **THEN** the addition SHALL be deferred until `openspec/changes/phase6-tracking/proposal.md` + `design.md` are amended in a separate commit to widen the scope
- **AND** the amend SHALL document the cross-tranche budget impact of the addition (each new MUST family is +12 L2 cells at minimum, before NEAT topology search)

#### Scenario: Promoting a SHOULD or MAY family to MUST mid-Phase-6

- **GIVEN** a Phase 6 milestone change proposes promoting a SHOULD or MAY family (e.g. quantum, spiking, reservoir, hybrid, transformer) to MUST so that family's results gate a Phase 6 exit criterion
- **WHEN** the promoting tranche merges
- **THEN** the promotion SHALL require an amendment to `openspec/changes/phase6-tracking/design.md` Decision 4 before the promoting tranche merges
- **AND** the amend SHALL document why the cross-tranche budget impact of promotion is justified

#### Scenario: Behavioural scope expansion

- **GIVEN** a Phase 6 milestone change proposes adding a fourth behaviour beyond the three fixed in `design.md` Decision 5 (klinotaxis, thermotaxis, predator evasion)
- **WHEN** the milestone change is reviewed
- **THEN** the addition SHALL follow the same amendment-blocking mechanism as architecture-family scope expansion above

### Requirement: Tranche Sequence is Load-Bearing

The eight Phase 6 tranches SHALL execute in the order documented in this change's `design.md` Decision 1. Re-ordering — most notably collapsing Tranche 4 (L2 first pass on grid) and Tranche 6 (L2 re-run on upgraded substrate) into a single L2 sweep — SHALL require an amendment to `proposal.md` + `design.md` + `tasks.md` with explicit rationale, because the T4-then-T5-then-T6 ordering produces the env-upgrade delta finding as a Phase 6 result in its own right.

#### Scenario: Proposal to skip Tranche 3 (corrected ASH/ADL nociception)

- **GIVEN** a Phase 6 milestone change proposes to skip Tranche 3 and run the L2 first pass against the existing (biologically wrong) nociception model
- **WHEN** the milestone change is reviewed
- **THEN** the skip SHALL be blocked because every predator-evasion L2 cell in Tranche 4 would need to be rerun once Tranche 3 lands later
- **AND** the only acceptable alternative is to drop predator-evasion from Tranche 4 entirely and run it for the first time in Tranche 6, which itself requires amending `tasks.md` Tranche 4 scope

#### Scenario: Proposal to collapse Tranche 4 (L2 first pass) and Tranche 6 (L2 re-run) into one L2 sweep on the upgraded substrate

- **GIVEN** a Phase 6 milestone change proposes to skip Tranche 4 and run L2 only once on the upgraded substrate, citing schedule pressure
- **WHEN** the milestone change is reviewed
- **THEN** the collapse SHALL require amending `design.md` Decision 1 with explicit rationale acknowledging that the env-upgrade delta finding (the load-bearing reason for the T4-vs-T6 split) is being given up
- **AND** Phase 6 exit criteria SHALL be reviewed to confirm the collapsed L2 sweep still satisfies the seven roadmap MUSTs

#### Scenario: Re-ordering env upgrades (T5) to follow L2 re-run (T6)

- **GIVEN** a Phase 6 milestone change proposes to run L2 (T6) before env upgrades (T5)
- **WHEN** the milestone change is reviewed
- **THEN** the re-ordering SHALL be blocked because Gate 2's plugin-parity verification depends on T5's env-upgrades work exercising the L1 plugin interface in practice, and because T6's L2 re-run is defined as the L2 sweep on the upgraded substrate (no upgrade, no T6 as defined)
