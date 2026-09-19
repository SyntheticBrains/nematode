# phase6b-tracking Specification

## Purpose

The living checklist for **Phase 6b** — the deferred completion of Phase 6 after the 2026-07 Gate-3
sub-phase split: Tranche 8 (L3 NEAT topology search on the upgraded continuous substrate) plus Tranche
9b (the synthesis addendum that would have flipped Phase 6 to COMPLETE). Phase 6a's tracker,
`phase6-tracking`, owned T1–T7 and archived at the Phase 6a synthesis.

**Phase 6b never started, and was closed off on 2026-09-19 at the Phase 7 close.** Every item is
recorded **deferred-with-destination, unscheduled**, in the five-status vocabulary
[Logbook 069](../../../docs/experiments/logbooks/069-phase7-synthesis.md) introduced. The binding
precondition was never decided — whether to port `Continuous2DEnvironment` to a vmappable form, since
TensorNEAT's speedup assumes one — and neither Phase 7's results nor Phase 8's opening scope depends on
it. **Phase 6's L3 exit criterion therefore stays unmet**, and Phase 6 remains *6a COMPLETE / 6b
pending* rather than COMPLETE, which is the honest state.

## Requirements

> **Frozen — historical record.** `phase6b-tracking` is **archived (read-only)** and its scope was
> never exercised. The requirements below describe the checklist contract as it was written and impose
> no live obligations; none may be satisfied by editing the frozen `tasks.md`. If the NEAT
> topology-search question is taken up, a **fresh change is authored** rather than this one revived.

### Requirement: Phase 6b Living Tranche Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase6b-tracking/tasks.md` covering Phase 6b (Tranche 8 L3 NEAT topology search + Tranche 9b synthesis addendum) at sub-task granularity. Every Phase 6b milestone PR SHALL update this file as part of its diff. This change SHALL remain unarchived until the Phase 6b synthesis addendum publishes, at which point it archives alongside that synthesis change.

#### Scenario: Future session orients to Phase 6b

- **GIVEN** a fresh AI session resumes Phase 6b work
- **WHEN** the agent reads `openspec/changes/phase6b-tracking/tasks.md` and the `docs/roadmap.md` Phase 6 block
- **THEN** the agent SHALL be able to identify the current Phase 6b sub-task and its preconditions without re-deriving the 6a/6b split or the NEAT plan

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 6b milestone PR (e.g. `add-l3-neat-topology-search`, or the 6b synthesis change) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase6b-tracking/tasks.md` marking completed sub-tasks as `[x]` and updating the relevant tranche status header

### Requirement: NEAT Execution Preconditions

Phase 6b Tranche 8 (NEAT topology search) execution SHALL NOT begin until three preconditions all hold: (1) Gate 3 GO (Phase 6a closed); (2) GPU availability; (3) a documented environment-vectorisation decision. The environment-vectorisation decision is load-bearing because the TensorNEAT speedup assumes a vmappable environment and `Continuous2DEnvironment` is not one — environment throughput, not the GPU, is the binding constraint.

#### Scenario: NEAT campaign proposed before the env-vectorisation decision

- **GIVEN** a Phase 6b milestone change proposes launching the TensorNEAT campaign
- **WHEN** the change is reviewed and no environment-vectorisation decision has been documented (vmappable-env-port vs reduced-population/scoped-search)
- **THEN** the campaign SHALL be blocked until the decision and its resulting population / generation / behaviour / seed budget are recorded (task T8.0)

#### Scenario: Phase 6a not yet closed

- **GIVEN** Gate 3 has not recorded a GO for Phase 6a
- **WHEN** a Phase 6b milestone change proposes NEAT execution
- **THEN** the change SHALL be blocked until the Phase 6a Gate 3 decision is recorded in the Phase 6a synthesis logbook

### Requirement: Co-evolution is Excluded from Phase 6b

Phase 6b SHALL NOT include the co-evolution test (matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP; the Phase 5 M5 architecture-asymmetry follow-up, ex-T8.4). It is deferred with no scheduled destination and is recorded as a resolved scoping decision in `phase6-tracking` and `docs/roadmap.md` § Research Questions RQ4.

#### Scenario: Co-evolution proposed as Phase 6b scope

- **GIVEN** a Phase 6b milestone change proposes adding the co-evolution / architecture-asymmetry test to Tranche 8
- **WHEN** the change is reviewed
- **THEN** the addition SHALL be blocked as out of Phase 6b scope; committing to co-evolution requires a separate future-phase decision that re-scopes it, documenting the compute budget and why the already-corroborated M5 diagnosis (Resendez Prado 2026) now warrants an in-house test

### Requirement: Roadmap Phase 6b / L3 Status Sync

The `docs/roadmap.md` Phase 6 Tranche Tracker SHALL reflect Phase 6b (T8 / T9b) status, updated as part of every Phase 6b milestone PR. When the Phase 6b synthesis addendum publishes, the roadmap Phase 6 status SHALL change to `✅ COMPLETE` and the L3 exit criterion SHALL be recorded as satisfied.

#### Scenario: Phase 6 marked complete at 6b close

- **GIVEN** Tranche 8 (NEAT) and Tranche 9b (synthesis addendum) are both complete
- **WHEN** the 6b synthesis PR merges
- **THEN** the roadmap Phase 6 Timeline Overview row SHALL change to `✅ COMPLETE`
- **AND** the L3 NEAT topology-search exit criterion SHALL be recorded with its verdict, cross-referencing the Phase 6a `T7.controls.rewired_null` degree-preserving control
