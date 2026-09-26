## Context

Phase 7's synthesis (`openspec/changes/archive/2026-09-19-add-phase7-synthesis/`, Logbook 069) is the
precedent. It introduced the five-status vocabulary and the standing-condition requirement, and both
apply here.

Decisions already taken:

- synthesize now, and carry B.2 to 8b;
- A.5 is a separate step after this synthesis;
- 8b's primary null is the chemical-only null.

## Goals / Non-Goals

**Goals:**

- a status for every 8a criterion;
- block V's full conditions in one sentence;
- the D20 gate;
- the 8b null decision;
- a codified sizing rule.

**Non-Goals:**

- any new run;
- the publication decision (A.5);
- designing 8b's rungs beyond what the gate and D21 record.

## Decisions

### Decision A: The status walkthrough

Every 8a criterion gets exactly one of the five statuses: *met*, *unmet-with-reason*,
*deferred-with-destination*, *superseded-by-result* or *unreachable-with-reason*.

**MUST:**

| criterion | status | basis |
|---|---|---|
| A.0 retention rule | met | registered before the phase's first campaign |
| A.1 init-vs-rewiring control | met | Logbook 070 |
| A.2 operating-point surface | met, with one named gap (readout width under PPO saturates) | Logbook 071 |
| A.4 methodology consolidation | met | `consolidate-plasticity-methodology` |
| B.1 measured weights (a, b, c) | met, without a positive | Logbooks 072–073 |
| A.6 null-strength control, and its split | met | Logbooks 074–075 |
| S8a | met | this logbook |

**SHOULD:**

| criterion | status | destination and reason |
|---|---|---|
| A.3 structural predictors | deferred-with-destination | the A.5 decision, which says whether the package needs the registered hop-predictor test; otherwise the 8b window |
| A.5 publication decision | deferred-with-destination | its own step, directly after this synthesis |
| B.2a / B.2b dynamics rung | deferred-with-destination | 8b. The reason is not the overshoot clause, since 8a took about a week against a 9–12 week budget. A.6 and Logbook 075 made gap junctions central, so B.2's design should absorb that (B.2b's plastic gap junctions, B.2a's stiffness warning); B.2c and C.2's second half need 8b's body anyway. |
| B.2c bout durations | deferred-with-destination | 8b, after C.0b, as the tracker already records |

**MAY:**

| criterion | status | destination |
|---|---|---|
| M.5 across-seed variance components | deferred-with-destination | the phase synthesis (S8b) |
| M.6 depth follow-ups | deferred-with-destination | the phase synthesis (S8b); the crossing was deprioritised, recorded rather than dropped |
| M.7 chemotaxis file | deferred-with-destination | its own small change, any time |

**Open threads each get a status and a destination:**

| thread | status | destination |
|---|---|---|
| B.1c's reading-learner interaction, unresolved | deferred-with-destination | A.5, only if the package claims it |
| gap-junction placement against strength | deferred-with-destination | A.5; B.2b's plastic gap junctions in 8b are the natural next test |
| the ~128-seed learning-speed panel against the chemical-only null | deferred-with-destination | A.5, only if the package needs the learning-speed form against that null |
| readout width under PPO | deferred-with-destination | carried as A.2's named gap into any 8b PPO arm |

### Decision B: Block V in one sentence

Every condition travels together:

> On hard350 under PPO, at a settling depth of 4 or more (reversed at 2, abolished at 3), with the
> pooled readout and the edge-order draw, the wild-type connectome learns faster than its
> degree-preserving null; that survives a shared initialisation (A.1) and a measured-weight prior
> (B.1c); about half of its `auc_success` lead came from the null's rewired gap junctions, and against
> a null with the wild type's gap junctions it is +0.025 `auc_success` and +235 episodes to competence
> (Logbook 075); and its magnitude is not stable across seed sets (A.1).

Every figure in the sentence is cited to the logbook it comes from.

### Decision C: The D20 gate is GO, and D21 is registered

**The gate.** D20's GO condition is met: 8b will embody a substrate with

- a known initialisation story (A.1),
- a known operating point (A.2),
- a measured-weight verdict (B.1),
- and a null-strength readout (A.6 and its split).

**D21, the 8b primary null.** Every wiring contrast from C.1e on reads against the chemical-only
null (`rewired_chemical_only`), with the current null reported beside it.

- *Why:* the chemical-only null differs from the wild type only in chemical placement. The current
  null also moves gap-junction strength, which Logbook 075 showed carries about half of block V's lead.
- *Comparability:* it is Wormlight's primary null, so the two projects' wiring tests compare.
- *What it costs:* the body is a new reference frame anyway, so no committed result is re-read.

D21 goes into the roadmap's decisions table.

### Decision D: The sizing rule

B.1c, A.6 and the split each sized a panel from a proxy, reported the achieved sensitivity beside
it, and did not re-read a verdict against the achieved figure. That is now one requirement.

### Decision E: The reproducibility statement

- **Committed:** every figure in 070–075 comes from committed per-seed CSVs and analysis JSON.
- **Archived:** the campaign directories, exports and experiment records are archived off-repo
  under A.0.
- **What the committed data supports:** the identity checks and the re-scoring checks in 074–075
  were run against files still on disk. They are recorded in the committed JSON, so a reader can see
  that they passed, but cannot repeat them without the archived raw logs.

## Risks / Trade-offs

- **The synthesis could hide open threads by carrying them.** Each carried item names its
  destination in the tracker and the roadmap as well as in the logbook. Several are routed to A.5,
  which is the next step, not an open-ended future.
- **D21 changes 8b before it starts.** That is the right time: a null chosen before 8b's first
  registration cannot have been chosen for its result.
