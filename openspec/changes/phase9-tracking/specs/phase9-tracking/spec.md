## ADDED Requirements

### Requirement: Phase 9 Living Shipment Checklist

The repository SHALL maintain a single living checklist file at `openspec/changes/phase9-tracking/tasks.md` covering Phase 9's two shipments (9a: block H, block M, G.0, G.1–G.3b, R.1 and the 9a synthesis; 9b: R.2, G.5–G.8, the K block, the P block, E.1, T.1 and the phase synthesis) plus the MAY items, at sub-task granularity, with the seed band reserved for every registration recorded in the change's `design.md`. Every Phase 9 milestone PR SHALL update this file as part of its diff. This change SHALL remain unarchived until the Phase 9 synthesis publishes, at which point it archives alongside that synthesis change.

#### Scenario: Future session orients to Phase 9

- **GIVEN** a fresh AI session resumes Phase 9 work
- **WHEN** the agent reads `openspec/changes/phase9-tracking/tasks.md` and `docs/roadmap.md` § Phase 9
- **THEN** the agent SHALL be able to identify the current shipment, sub-task, G.0's verdict if read, the applicable design decisions (D23–D32) and the seed band of the next registration without re-deriving the plan or the review's corrections

#### Scenario: Milestone PR updates the checklist

- **GIVEN** a Phase 9 milestone PR (e.g. the G.0 screen change, a null-family change, or a synthesis change) is being prepared
- **WHEN** the PR is opened
- **THEN** the PR diff SHALL include updates to `openspec/changes/phase9-tracking/tasks.md` marking completed sub-tasks as `[x]` with their status and updating the relevant shipment status header

### Requirement: Design Decisions D23–D32 Are Binding

Phase 9 milestone changes SHALL conform to the ratified design decisions D23–D32 recorded in `docs/roadmap.md` § Phase 9 § Pre-registered design decisions. Amending a D-decision SHALL require a dated note in the `phase9-tracking` change (with a matching roadmap edit) before the affected milestone change merges.

#### Scenario: Milestone change contradicts a D-decision

- **GIVEN** a Phase 9 milestone change proposes work that contradicts a D-decision (e.g. a point-worm arm on Cook 2019 against D23, the degree-preserving null carried as a standing beside-arm against D24, a 6,000-episode body panel other than M.1(c) before M.1 has read against D25, R.2 at reduced n against D27, or a placed arm without its two-arm control against D32)
- **WHEN** the change is reviewed
- **THEN** the change SHALL be blocked until either it conforms, or a dated amendment to the relevant D-decision is recorded in `phase9-tracking` and `docs/roadmap.md`

### Requirement: Positive Control Before Any Connectome Arm on a New Component

Each new component — the body action-space heads for LSTM/GRU, CfC, Transformer and GA (R.1), any corrected gap model that lands after a G.0 `artefact` verdict, the Bernoulli pirouette head (the K 2×2) and the set-point task with its cultivation phase (P) — SHALL clear a positive control on the target cell before any connectome arm runs on it: an MLP-PPO control for a substrate, task or head change, and each family's own control for its body head. A component whose positive control fails SHALL close with the diagnosis as its deliverable and SHALL NOT report a wiring result; a family whose control fails SHALL be ranked last with its reason and SHALL NOT be dropped. A positive control that fails its fixed rule SHALL be retried only as a new registration, with the failed verdict kept, the change fixed before running, new seeds and the same rule.

#### Scenario: Connectome arm proposed on an uncontrolled component

- **GIVEN** a milestone change proposes a wiring contrast on a corrected gap model, a pirouette head or the set-point task with no recorded positive control on that cell
- **WHEN** the change is reviewed
- **THEN** the contrast SHALL be blocked until the positive control has run and its outcome is recorded in the tracker

#### Scenario: A family fails its body control

- **GIVEN** R.1's control for a family reads below its floor after its entropy calibration
- **WHEN** R.2's registration lists the families
- **THEN** that family SHALL appear ranked last with its reason, and the ranking SHALL ship with the rest

### Requirement: Operating-Point Discipline

No Phase 9 gap-junction contrast SHALL run before G.0 has read, and every block-G result SHALL cite the gap scale and settling depth it runs at, on the learner it runs, in the same sentence as its claim (D16 carried: a setting inherited from a previous rung is a hypothesis at the new one). The phase SHALL run on Emmons 2024 everywhere (D23), and the gap-placement effect SHALL be re-anchored on that substrate in its own pilot before G.1 registers, never inside G.1's panel; G.1 SHALL carry Logbook 082's 0.0577 minimum only if the anchor's effect sits in [0.058, 0.13] and both levels read `readable`, and SHALL otherwise re-fix the minimum from the anchor. Every gate a registration reads SHALL be evaluated at the registered point before launch with `gate_preflight.py`, on committed runs where they exist and otherwise on a pilot on disjoint seeds, and a gate carried from another cell SHALL be re-checked on the new one.

#### Scenario: Gap contrast registered before the screen

- **GIVEN** a milestone change registers a contrast between the wild type and any gap null
- **WHEN** the change is reviewed and G.0 has not recorded a verdict
- **THEN** the registration SHALL be blocked until G.0 has read

#### Scenario: Block-G claim without its operating point

- **GIVEN** a block-G logbook states a lead over a gap null without citing the gap scale and depth it ran at
- **WHEN** the logbook is reviewed
- **THEN** the claim SHALL be restated with its scale and depth, and with G.0's verdict, before it is cited anywhere

#### Scenario: G.1 registers without the Emmons anchor

- **GIVEN** G.1's change registers its 2×2 citing Logbook 082's minimum
- **WHEN** the change is reviewed and G.0(a)'s anchor pilot has not read on Emmons 2024
- **THEN** the registration SHALL be blocked until the anchor has read and the minimum is carried or re-fixed by its rule

### Requirement: The Screen Gates the Mechanism Block

G.1, G.2, G.3 and G.3b SHALL register only after G.0 has read `survives`, or on the corrected gap model after G.0 has read `artefact` and the named substrate change has landed with its own MLP positive control and Logbook 092's body contrast has been re-run on it. An `unresolved` verdict SHALL license neither; how the block then proceeds SHALL be a dated decision recorded in this tracker and the roadmap. G.2's reading SHALL set G.3's aim: if the boundary carries the effect, the class map SHALL be drawn over the boundary pairs rather than the interior. G.3b SHALL run only if G.3 names a class, and SHALL otherwise close unreachable-with-reason with the effect recorded as distributed.

#### Scenario: Mechanism-block change registered after an `artefact` verdict

- **GIVEN** G.0 has read `artefact` and a milestone change proposes G.1 on the uncorrected gap model
- **WHEN** the change is reviewed
- **THEN** the registration SHALL be blocked until the named substrate change has landed with its positive control and Logbook 092's contrast has been re-run on the corrected model, or G.1 is recorded deferred-with-destination

#### Scenario: G.3b proposed when G.3 named no class

- **GIVEN** G.3's readout names no class, or two
- **WHEN** G.3b's status is recorded
- **THEN** G.3b SHALL be closed unreachable-with-reason with the branch (distributed or non-additive) named, and the landscape search SHALL stay MAY

### Requirement: Body-Substrate Results Are a New Reference Frame

Results measured through the kinematic body SHALL be reported against floors and baselines re-established on that substrate, and SHALL NOT be reported as controlled deltas against Logbook 029, block V or any point-worm result. R.2's claim SHALL be labelled "through the level-5++ kinematic body (imposed rhythm, synthetic lateral sampling), a new reference frame" and SHALL state that the families differ in interface as well as architecture.

#### Scenario: Body result compared to a point-worm number

- **GIVEN** a milestone change reports a body-substrate result as a percentage change against a point-worm baseline
- **WHEN** the change is reviewed
- **THEN** the comparison SHALL be restated as qualitative, and the rung's own baseline campaign SHALL be cited as the quantitative frame

### Requirement: The Ranking Runs Under Its Trigger

R.2 SHALL run once, last in the phase, only if all four of D27's conditions hold: (1) M.1 is closed — a collapse rule and the secondary statistic registered, or the three-round stop rule has fired; (2) a dated body freeze is signed with no C.x change, no body-side klinokinesis rung or shorter step, and no E.1 change pending; (3) every family has passed R.1 or is ranked last with reason; (4) a primary is registered beyond the tiers with a minimum — whether the connectome's standing relative to the MLP and to its gap-only null through the body differs from the point worm's. If any condition cannot be met this phase, R.2 SHALL be recorded deferred-with-destination with the unmet condition named, and SHALL NOT run at reduced n. R.1's per-family positive controls SHALL run in 9a regardless of the trigger.

#### Scenario: R.2 proposed with a condition unmet

- **GIVEN** R.2's registration is proposed while M.1 is open, the body freeze is unsigned, a family is uncontrolled, or no primary beyond the tiers is registered
- **WHEN** the change is reviewed
- **THEN** the registration SHALL be blocked, and if the condition cannot be met this phase R.2 SHALL be recorded deferred-with-destination with that condition named in the tracker and the roadmap in the same PR

#### Scenario: Reduced-n ranking proposed to fit the window

- **GIVEN** a milestone change proposes R.2 at fewer than 32 seeds per family, or with a family dropped, to fit the phase's window
- **WHEN** the change is reviewed
- **THEN** the change SHALL be blocked: the ranking runs at its registered size with every family present or ranked last with reason, or is deferred

### Requirement: Collapse Is a Registered Statistic

Every Phase 9 panel over 3,000 episodes SHALL report collapse incidence under D25's pre-registered per-seed definition (a trailing-200-episode peak; onset when the 100-episode mean falls below half of it and stays there for at least 300 episodes; censored at the horizon) with Wilson intervals, time-to-collapse censored at the horizon, and `auc_success` to 3,000 and to the horizon separately, with IQM and performance profiles beside the mean. No 6,000-episode body panel SHALL register before M.1 reads out, except M.1(c), M.1's own confirmatory panel, which is how M.1 reads out; every later 6,000-episode body panel stays gated. A collapse remedy SHALL be chosen by rescue of the known collapsers across both wirings, never tuned per arm; SHALL hold the MLP positive control, read on fresh seeds in M.1's confirmatory panel against a stated equivalence margin on `auc_success` to 3,000 and on collapse incidence; SHALL be applied identically to every arm, byte-identical-when-off and frozen for the phase; and at most three remedy rounds SHALL run. A collapsed seed SHALL stay in the primary reading. Best-so-far and best-block statistics SHALL NOT be comparison metrics, and M.4's secondary statistic SHALL be reported beside the primary and SHALL NOT re-read a verdict.

#### Scenario: Long body panel registered before M.1 reads

- **GIVEN** a milestone change proposes a 6,000-episode body panel other than M.1(c)
- **WHEN** the change is reviewed and M.1 has neither closed nor fired its stop rule
- **THEN** the registration SHALL be blocked until M.1 reads out, or the panel is re-registered at 3,000 episodes with the collapse statistic

#### Scenario: Remedy read on the pilot seeds

- **GIVEN** a remedy's effect on `auc_success` is reported from the 16 M.9 seeds it was chosen on
- **WHEN** the reading is reviewed
- **THEN** it SHALL be restated as a rescue reading, and the effect SHALL be read only from M.1's confirmatory panel on fresh seeds

### Requirement: Every Null States What It Does Not Preserve

A Phase 9 panel SHALL carry only the nulls its question needs — gap questions the gap-only family, chemical questions the chemical-only and boundary-preserving nulls — and the degree-preserving null SHALL NOT be carried as a standing beside-arm (D24). Every null SHALL state what it does not preserve, SHALL publish a matched-statistics table in the model's effective units (the effective maximum and 99th percentile, effective per-neuron totals for ALA, CAN, PVD, AVA and AVB, sensor–motor gap pairs, and the count–degree correlation), and SHALL carry the shortcut audit (`sensory_motor_hops.py`, gap edges included) in its launch record. The number of distinct nulls and their distance from the wild type SHALL be computed before a constrained swap registers.

#### Scenario: Gap null registered without its table

- **GIVEN** a milestone change registers a gap null (totals-held, sizes-permuted, boundary-held, interior-held, a class scatter, or contact-constrained) whose launch record lacks the matched-statistics table or the shortcut audit
- **WHEN** the change is reviewed
- **THEN** the registration SHALL be blocked until both are in the launch record and the null's non-preserved properties are stated

### Requirement: Artefact Retention Registered at Phase Start

Before the first Phase 9 campaign runs, task H.2 SHALL re-register the retention rule in A.0's wording — the parsed per-seed CSVs, analysis JSON and launch records committed under `docs/experiments/logbooks/supporting/`; raw logs, step-level exports and trained weights archived outside the repository; a campaign directory deleted only after its CSV is committed; a field whose source is gone named as uncompared — and 8b's campaign directories SHALL be archived under it. Every campaign SHALL cite the rule, and a phase synthesis SHALL state plainly what is re-derivable from git alone and what is not.

#### Scenario: Campaign runs with no retention rule recorded

- **GIVEN** a Phase 9 milestone change proposes launching a campaign
- **WHEN** the change is reviewed and task H.2 has not recorded the retention rule
- **THEN** the campaign SHALL be blocked until the rule is recorded

### Requirement: Shipment Completion Semantics

A 9a GO decision SHALL record the phase state as "9a complete / 9b pending" and SHALL NOT mark Phase 9 complete. The 9a synthesis SHALL read G.0's verdict as the gate for 9b's gap work and SHALL write D27's body-freeze decision as a go/no-go. 9b work SHALL NOT start until the 9a synthesis has assigned every 9a criterion one of the five statuses (met, unmet-with-reason, deferred-with-destination, superseded-by-result, unreachable-with-reason). Phase 9 SHALL be marked ✅ COMPLETE only when the 9b synthesis publishes with every criterion assigned a status and the publication decision (D30) recorded; "well underway" SHALL never satisfy completion. Splits and stops SHALL be invoked on pre-registered criteria, never on month counts.

#### Scenario: 9a closes while 9b is pending

- **GIVEN** H, M.1–M.4, G.0, G.1–G.3b and R.1 are complete with statuses assigned and the 9a synthesis writes GO
- **WHEN** the 9a shipment decision is recorded
- **THEN** the roadmap Phase 9 status SHALL read "9a complete / 9b pending"
- **AND** Phase 9 SHALL NOT be marked COMPLETE until the 9b synthesis lands

#### Scenario: A rung is deferred out of the phase

- **GIVEN** a rung the completion predicate names becomes unrunnable on evidence (e.g. R.2's body freeze cannot be signed, or G.8's gate never opens), and a dated decision defers it
- **WHEN** that decision is recorded
- **THEN** the rung SHALL carry the status *deferred-with-destination* (or *unreachable-with-reason* where nothing in the phase could have made it attemptable) with the destination or reason named, in the tracker and the roadmap in the same PR, and SHALL NOT be dropped silently

### Requirement: Scope Exclusions

Phase 9 SHALL NOT include: cross-species work (the *P. pacificus* comparison, the dauer pathfinder, weight transplant); NEAT or evolution on the wiring beyond G.3's two-sided reading (the placement search stays MAY behind G.3b and T.1); a further chemical-wiring rung; the dynamical substrate as built (the ohmic-coupling form returns only as G.0's `artefact` remedy); multi-agent, pheromone, red-queen, 3D, Sibernetic, neuropeptide, spiking or neuromorphic work; the uniform substrate-writing rule programme; or whole-organism fidelity.

#### Scenario: Excluded scope proposed as Phase 9 work

- **GIVEN** a Phase 9 milestone change proposes any excluded item above as Phase 9 scope
- **WHEN** the change is reviewed
- **THEN** the addition SHALL be blocked as out of Phase 9 scope; committing to it requires a dated D-decision amendment with its budget impact, recorded in `phase9-tracking` and `docs/roadmap.md`
