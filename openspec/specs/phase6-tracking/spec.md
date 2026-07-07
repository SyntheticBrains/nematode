# phase6-tracking Specification

## Purpose

The living tranche checklist + decision-gate record for Phase 6 (connectome substrate + architecture
comparison), delivered in two sub-phases: **6a** (T1–T7 + connectome-structure controls + real-worm
validation) closed at Gate 3 GO ([Logbook 037](../../../docs/experiments/logbooks/037-phase6a-synthesis.md)),
and **6b** (T8 NEAT topology search + the T9b synthesis addendum) tracked separately under
`phase6b-tracking`. Records the Phase-6a/6b sub-phase split triggered at Gate 3.

## Requirements

> **Frozen — historical record.** Phase 6a closed at Gate 3 GO ([Logbook 037](../../../docs/experiments/logbooks/037-phase6a-synthesis.md)) and the `phase6-tracking` change is **archived (read-only)**. The requirements below describe the tracking contract as it governed Phase 6a and are retained as a record of what was done — they no longer impose live obligations. The **live checklist contract for the remaining Phase 6 work (6b — T8 NEAT + the T9b synthesis) is `phase6b-tracking`**; direct all forward "maintain / update the checklist" obligations there.

### Requirement: Phase 6a Living Tranche Checklist (now frozen)

Through Phase 6a the repository maintained a single living checklist at `openspec/changes/phase6-tracking/tasks.md` covering the Phase-6a tranches (T1–T7) + the mid-phase gates at sub-task granularity, updated by every Phase-6a milestone PR. That change is now **archived** on the Phase-6a synthesis; the checklist is frozen. Ongoing tracking continues under `phase6b-tracking`.

#### Scenario: Future session orients to Phase 6

- **GIVEN** a fresh AI session resumes Phase 6 work
- **WHEN** the agent reads `openspec/changes/phase6-tracking/tasks.md` and the `docs/roadmap.md` Phase 6 block
- **THEN** the agent SHALL be able to identify the current in-progress tranche and the next sub-task to start without further codebase exploration

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 6 milestone PR (e.g. `add-connectome-substrate`, an L1 plugin refactor, a T4 L2 first-pass run, a T5 platform-refactor change, a T6 Rung 2 gradients change, a T7 L2 final-pass run, a T8 NEAT run) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase6-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant tranche status header

#### Scenario: Checklist outlives individual milestones

- **GIVEN** a Phase 6 milestone OpenSpec change (e.g. `add-connectome-substrate`) is archived
- **WHEN** archival completes
- **THEN** the `phase6-tracking` change SHALL remain unarchived and continue to receive updates from subsequent milestone PRs
- **AND** archival of `phase6-tracking` itself SHALL only occur alongside the Phase 6 synthesis logbook change (Tranche 9)

#### Scenario: Phase 6a / 6b sub-phase split triggered at Gate 3

- **GIVEN** Gate 3 PIVOT-scope is triggered (partial MUST-cell coverage or 10-month overshoot per `design.md` § Decision 6 § Gate 3)
- **WHEN** the Phase 6a / Phase 6b split is documented in the T7 logbook
- **THEN** this change's `proposal.md` + `design.md` + `tasks.md` SHALL be amended to mark Phase 6a as the scope that ships first (T1–T7) and Phase 6b as the deferred follow-on (T8 NEAT + T9 synthesis)
- **AND** archival of `phase6-tracking` SHALL occur on Phase 6a's synthesis logbook publication (an interim synthesis logbook scoped to T1–T7)
- **AND** Phase 6b's tracker arrangement SHALL follow this concrete rule: if Phase 6b contains more than one tranche (e.g. T8 NEAT + T9 synthesis = two tranches) it SHALL inherit a fresh `phase6b-tracking` change; if Phase 6b is a single-tranche continuation (e.g. T9 synthesis only, with T8 already shipped under Phase 6a) it SHALL be appended as a new `phase6b-tracking` change with a single-tranche scope OR scoped inside the synthesis change directly — the choice MAY be left to the Phase 6b OpenSpec change's author

### Requirement: Roadmap Phase 6 Status Block

The `docs/roadmap.md` Phase 6 section SHALL include a Phase 6 Tranche Tracker sub-section listing the nine tranches (T1–T9) plus L4 (marked deferred to Phase 7) with current status, updated as part of every Phase 6 milestone PR. The Phase 6 architecture-families table SHALL annotate each family as MUST / SHOULD / MAY per the scope in this change's `design.md` Decision 4.

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

Each of the three roadmap-defined mid-phase decision gates (Gate 1 at the close of Tranche 2; Gate 2 at the close of Tranche 5; Gate 3 at the close of Tranche 7) SHALL produce a written go/no-go decision in the published logbook of the OpenSpec change that triggers the gate, evaluated against the quantitative pass criteria pre-registered in this change's `design.md` § Decision 6. `openspec/changes/phase6-tracking/tasks.md` SHALL link to each decision once it lands.

#### Scenario: Gate decision is recorded in writing, in the logbook

- **GIVEN** a Phase 6 mid-phase gate (Gate 1, Gate 2, or Gate 3) is reached
- **WHEN** the triggering tranche (T2 / T5 / T7) closes
- **THEN** the OpenSpec change for that tranche SHALL include a written go/no-go decision in its **published logbook** (not in `tasks.md`, which becomes hard to amend post-archive)
- **AND** the decision SHALL evaluate each of the gate's pre-registered numerical pass criteria from `design.md` § Decision 6 (G1.a-d / G2.a-d / G3.a-d) explicitly
- **AND** the decision SHALL state GO / PIVOT (with the chosen pivot path) / STOP (with the diagnosis)
- **AND** `openspec/changes/phase6-tracking/tasks.md` SHALL be updated to link to the logbook decision

#### Scenario: Gate decision lacks quantitative criterion evaluation

- **GIVEN** a tranche-close logbook claims a gate as GO but does not address every pre-registered numerical criterion from `design.md` § Decision 6
- **WHEN** a reviewer observes the missing criterion
- **THEN** the gate SHALL be treated as not-yet-decided and the next tranche's PR SHALL be blocked until the logbook is amended to evaluate each criterion explicitly

#### Scenario: Amendment mechanism — definition of "the gate fires" and recording rules

This scenario anchors the normative content of `design.md` § Decision 6 § Amendment mechanism; design.md is rationale-only and references back here for enforceable rules.

- **GIVEN** any Phase 6 mid-phase gate (Gate 1, Gate 2, or Gate 3)
- **WHEN** the project needs to determine whether a Decision 6 criterion amendment is pre-fire (allowed) or post-fire (prohibited)
- **THEN** "the gate fires" SHALL be defined as the moment the triggering tranche's logbook PR is opened with the gate-decision section drafted
- **AND** an amendment commit landing before that PR opens (including a separate prior PR landing the previous day) SHALL be treated as pre-fire
- **AND** an amendment commit landing in the same PR as the gate-decision draft, OR in any PR opened after that PR opens, SHALL be treated as post-fire
- **AND** a Decision 6 amendment SHALL land as a separate commit (and SHALL NOT be bundled with the gate-decision-shipping PR diff), so amendments are visible independently of the verdicts they affect
- **AND** when a gate decision evaluates against a pre-fire amended criterion, the triggering tranche's logbook SHALL record both the original criterion and the amended one

#### Scenario: Gate criterion recalibrated before the gate fires (allowed)

- **GIVEN** an in-flight tranche surfaces evidence that one of Decision 6's pre-registered criteria was empirically miscalibrated (e.g. T1 reveals the G1.c frozen-random-control baseline is itself unstable, or T2 reveals the G2.b "≤ 6 files" floor is too tight against the chosen registry pattern)
- **WHEN** the criterion is amended *before* the gate fires (definition: "fires" = the moment the triggering tranche's logbook PR is opened with the gate-decision section drafted; see `design.md` § Decision 6 § Amendment mechanism)
- **THEN** the amendment SHALL land as a **separate prior PR** (not bundled with the gate-decision-shipping PR) that commits to `design.md` § Decision 6 the criterion being recalibrated, the in-flight evidence motivating the change, and the alternative criterion that replaces it
- **AND** the triggering tranche's logbook gate-decision SHALL record both the original criterion and the amended one when evaluating

#### Scenario: Gate criterion changed after the gate has fired (prohibited — goalpost-moving)

- **GIVEN** a tranche has produced data evaluated against a Decision 6 criterion, and the criterion is failing
- **WHEN** a PR proposes lowering or relaxing that criterion to make the gate pass
- **THEN** the PR SHALL be blocked; the gate verdict stays as it stood under the original criterion
- **AND** the only acceptable amendment after a gate has fired is one that *raises* the bar (e.g. tightening tolerances on the basis of better-than-expected data) — this rule prevents goalpost-moving while permitting evidence-driven calibration before the gate fires

#### Scenario: Pivot triggered at a mid-phase gate

- **GIVEN** a mid-phase gate triggers its documented Risk-mitigation pivot (e.g. Gate 1 triggers the hand-curated-subset pivot for L0; Gate 2 triggers the L1 refactor pivot; Gate 3 triggers the Phase 6a / Phase 6b sub-phase split)
- **WHEN** the pivot is executed
- **THEN** the pivot SHALL be documented as a successful gate outcome (not a failure of discipline)
- **AND** the OpenSpec change for the triggering tranche SHALL document the pivot's amended scope in its logbook
- **AND** `openspec/changes/phase6-tracking/tasks.md` and `proposal.md` SHALL be amended to reflect the new tranche definitions

#### Scenario: A gate is reached without a written decision

- **GIVEN** the next tranche's OpenSpec change opens before the prior gate's logbook decision is written
- **WHEN** a reviewer observes the missing gate decision
- **THEN** the new tranche's PR SHALL be blocked until the prior gate's decision is recorded in the prior tranche's published logbook and linked from `openspec/changes/phase6-tracking/tasks.md`

### Requirement: Architecture-Family Scope is Fixed at the Four MUSTs

Phase 6's architecture-family MUST set is fixed at the four families documented in this change's `design.md` Decision 4: connectome-constrained, MLP-PPO, LSTM/GRU-PPO, and NEAT-evolved. SHOULD families (quantum, spiking) and MAY families (reservoir, hybrid, transformer) MAY be evaluated opportunistically in Tranche 7 if substrate-change × architecture-family interaction looks scientifically worthwhile, but their results SHALL NOT gate any Phase 6 exit criterion.

#### Scenario: Adding a new architecture family mid-Phase-6

- **GIVEN** a Phase 6 milestone change proposes adding a tenth architecture family beyond the nine in `design.md` Decision 4
- **WHEN** the milestone change is reviewed
- **THEN** the addition SHALL be deferred until `openspec/changes/phase6-tracking/proposal.md` + `design.md` are amended in a separate commit to widen the scope
- **AND** the amend SHALL document the cross-tranche budget impact of the addition (each new MUST family is +3 behaviours × 4 seeds × 2 L2 passes = +24 cells at minimum, before NEAT topology search)

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

The nine Phase 6 tranches SHALL execute in the order documented in this change's `design.md` Decision 1. Re-ordering — most notably collapsing Tranche 4 (L2 first pass on grid) and Tranche 7 (L2 re-run on upgraded substrate) into a single L2 sweep, or re-merging Tranche 5 (platform refactor) and Tranche 6 (env fidelity) back into a single env-upgrade tranche — SHALL require an amendment to `proposal.md` + `design.md` + `tasks.md` with explicit rationale, because the T4-then-T5-then-T6-then-T7 ordering produces the env-upgrade delta finding as a Phase 6 result in its own right AND because the T5/T6 split places Gate 2 against a single verifiable platform outcome rather than a bundled env-upgrade tranche.

#### Scenario: Proposal to skip Tranche 3 (corrected ASH/ADL nociception)

- **GIVEN** a Phase 6 milestone change proposes to skip Tranche 3 and run the L2 first pass against the existing (biologically wrong) nociception model
- **WHEN** the milestone change is reviewed
- **THEN** the skip SHALL be blocked because every predator-evasion L2 cell in Tranche 4 would need to be rerun once Tranche 3 lands later
- **AND** the only acceptable alternative is to drop predator-evasion from Tranche 4 entirely and run it for the first time in Tranche 7, which itself requires amending `tasks.md` Tranche 4 scope

#### Scenario: Proposal to collapse Tranche 4 (L2 first pass) and Tranche 7 (L2 re-run) into one L2 sweep on the upgraded substrate

- **GIVEN** a Phase 6 milestone change proposes to skip Tranche 4 and run L2 only once on the upgraded substrate, citing schedule pressure
- **WHEN** the milestone change is reviewed
- **THEN** the collapse SHALL require amending `design.md` Decision 1 with explicit rationale acknowledging that the env-upgrade delta finding (the load-bearing reason for the T4-vs-T7 split) is being given up
- **AND** Phase 6 exit criteria SHALL be reviewed to confirm the collapsed L2 sweep still satisfies the seven roadmap MUSTs

#### Scenario: Proposal to re-merge Tranche 5 (platform refactor) and Tranche 6 (env fidelity)

- **GIVEN** a Phase 6 milestone change proposes to re-merge T5 and T6 back into a single env-upgrade tranche, citing engineering coupling between continuous-action heads and Rung 2 gradient consumers
- **WHEN** the milestone change is reviewed
- **THEN** the re-merge SHALL require amending `design.md` Decision 1 with explicit rationale acknowledging that Gate 2 will now close against a bundled tranche outcome (parity + chemosensory-adaptation iteration) rather than a clean platform-refactor outcome
- **AND** the rationale SHALL address why the Phase 5 M4/M4.5/M4.6 iteration precedent (which motivated the split) does not apply to the bundled scope

#### Scenario: Re-ordering env upgrades (T5 + T6) to follow L2 re-run (T7)

- **GIVEN** a Phase 6 milestone change proposes to run L2 (T7) before env upgrades (T5 + T6)
- **WHEN** the milestone change is reviewed
- **THEN** the re-ordering SHALL be blocked because Gate 2's plugin-parity verification depends on T5's continuous-action work exercising the L1 plugin interface in practice, and because T7's L2 re-run is defined as the L2 sweep on the upgraded substrate (no upgrade, no T7 as defined)
