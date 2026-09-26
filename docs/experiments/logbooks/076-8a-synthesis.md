# 076: Shipment 8a — Block V Holds With Its Conditions Named, and Half of It Was the Null's Gap Junctions (Phase 8 8a Synthesis)

**Status**: completed. **Every 8a criterion is assigned a status, and the D20 gate is GO.** 8a set
out to put three controls under block V, Phase 7's strongest result: an initialisation control, an
operating-point surface and a measured-weight rung. A fourth control arrived from outside. Where 8a
leaves block V:

- **It survives what was registered.**
  - No dissolution under a shared initialisation (A.1).
  - Measured weights under PPO leave it unmoved (B.1c).
- **It is narrower than the claim first made.**
  - It is **depth-critical**: present at settling depths 4 and 6, abolished at 3, reversed at 2
    (A.2).
  - **About half of its `auc_success` lead came from the null's rewired gap junctions** (A.6, then
    the split).
  - Against a null that keeps the wild type's gap junctions, it is **+0.025 `auc_success` and +235
    episodes to competence** on hard350.
- **On the thermal cell** only the initialisation control has been run.

8b starts on a substrate whose initialisation story, operating point, measured-weight verdict and
null strength are all known. It starts with a new primary null, D21.

**Date**: 2026-09-20 to 2026-09-26.

**OpenSpec change**: `add-8a-synthesis`, which extends `architecture-comparison-protocol` with the
rule that a panel sized from a proxy reports its achieved sensitivity and never re-reads a verdict
against it.

**Pre-registration**: none, and none is needed. This synthesis reads committed records and runs
nothing. Every figure below is cited to the logbook, CSV or JSON it comes from.

## Objective

Phase 7 closed ([Logbook 069](069-phase7-synthesis.md)) with block V as its strongest citable result:
under PPO the wild-type connectome reaches competence faster than its degree-preserving rewired null.
It was shipped with a standing condition, because rewiring and initialisation varied together.

Phase 8's shipment 8a was scoped to close that exposure before any body work began:

- **A.1**, an initialisation control;
- **A.2**, an operating-point surface;
- **B.1**, a measured-weight rung.

D20 makes this synthesis the gate: 8b does not start until every 8a criterion has a status.

## The status walkthrough

Statuses use the five words Logbook 069 introduced.

### Required (MUST)

| criterion | status | record |
|---|---|---|
| A.0 artefact-retention rule | **met** | registered before the phase's first campaign |
| A.1 init-vs-rewiring control | **met** | [070](070-init-sharing-control.md): 768 runs, 32 paired seeds; no dissolution detected on any arm; survival established on five of eight readings |
| A.2 operating-point surface | **met**, with one named gap | [071](071-operating-point-surface.md): 1,600 runs. The gap is readout width under PPO, which saturates at 96.5% and 97.7% |
| A.4 methodology consolidation | **met** | `consolidate-plasticity-methodology` |
| B.1 measured weights (a, b, c) | **met, without a positive** | [072](072-measured-prior-pilot.md) and [073](073-measured-prior-contrast.md): 448 and 960 runs |
| A.6 null-strength control, and its split | **met** | [074](074-null-strength-control.md) and [075](075-gap-only-split.md): 480 runs, then 160 new runs on A.6's seeds |
| S8a | **met** | this logbook |

### Recommended (SHOULD)

| criterion | status | destination and reason |
|---|---|---|
| A.3 frozen-operator structural predictors | **deferred-with-destination** | the A.5 decision, which says whether the package needs A.2's hop distance registered as a predictor; otherwise the 8b window |
| A.5 publication decision | **deferred-with-destination** | its own step, directly after this synthesis |
| B.2a / B.2b dynamics rung | **deferred-with-destination** | 8b. See below. |
| B.2c bout durations | **deferred-with-destination** | 8b, after C.0b, as the tracker already recorded |

**Why B.2 moves to 8b.** It is **not** the overshoot clause. 8a took about a week against a budget of
9–12 active weeks. The reason is that A.6 and its split made gap junctions central to the wiring
effect, and B.2 is the rung where gap junctions stop being a fixed matrix:

- B.2a turns them into ohmic coupling, and the stiffness warning from the Wormlight review applies
  there.
- B.2b makes them plastic under PPO, and is the natural next test of gap placement against gap
  strength.

That design should be written with 075 in hand, and B.2c and C.2's second half need 8b's body anyway.

### Optional (MAY)

| criterion | status | destination |
|---|---|---|
| M.5 across-seed variance components | **deferred-with-destination** | the Phase 8 synthesis (S8b) |
| M.6 the depth finding's follow-ups | **deferred-with-destination** | S8b; the depth crossing was deprioritised once 071's hop probe explained depth, recorded rather than dropped |
| M.7 the chemotaxis reference file | **deferred-with-destination** | its own small change, at any time |

M.1 (placed plasticity), M.2 (wild type against wild type), M.3 (gait) and M.4 (reproducibility
artefacts) are phase-wide optional items rather than 8a work. **They get their statuses at S8b.**

### Open threads

| thread | status | destination |
|---|---|---|
| B.1c's reading-learner interaction (+0.082, spanning its 0.140 minimum) | **deferred-with-destination** | A.5, only if the package claims it |
| gap-junction placement against strength | **deferred-with-destination** | A.5; B.2b in 8b is the natural test |
| the learning-speed lead against the chemical-only null (+94 episodes, spanning zero; resolving it would take about 128 seeds) | **deferred-with-destination** | A.5, only if the package needs that form |
| readout width under PPO | **deferred-with-destination** | carried as A.2's named gap into every 8b PPO arm |

## Block V, with every condition in one sentence

> On hard350 under PPO, with the pooled readout, at settling depths 4 and 6 (abolished at 3,
> reversed at 2, nothing above 6 tested; [071](071-operating-point-surface.md)), the wild-type
> connectome learns faster than its degree-preserving null. Under the edge-order draw, no dissolution
> was detected under a shared initialisation, with survival established on five of eight readings
> ([070](070-init-sharing-control.md)). Under the per-neuron fan-in draw, a measured-weight prior
> leaves it unmoved ([073](073-measured-prior-contrast.md)). Under the edge-order draw, about half of
> its `auc_success` lead came from the null's rewired gap junctions, and against a null with the wild
> type's gap junctions it is +0.025 `auc_success` [+0.012, +0.038] and +235 episodes to competence
> [+62, +409] ([074](074-null-strength-control.md), [075](075-gap-only-split.md)). Its magnitude is
> not stable across seed sets ([070](070-init-sharing-control.md)).

**The thermal cell.** Block V was established on two cells. On thermal only A.1 has been run: the
direction replicated, at 134.6 episodes against a committed 382.3 (35%). The operating point, the
measured weights and the null-strength question are **all untested there**. On hard350 A.1 read
580.4 against a committed 553.1 (105%).

## What 8a established

- **The wiring advantage survives a shared initialisation** on the readings that resolved: five of
  eight, and no dissolution anywhere (A.1). The published critique that motivated the control is
  answered on its own terms.
- **It exists at a point, not across a region** (A.2).
  - Under PPO it replicates at the centre, +536 episodes, 92% of A.1's figure. It holds at depth 6
    and is gone below 4.
  - A graph measurement accounts for the depth dependence. The wild type has no motor neuron one hop
    from a food sensor; a rewiring manufactures about nine. A.6's hop probe then showed this
    mechanism is **chemical**: the chemical-only null still manufactures 8.0.
  - Under the reading learner the null is ahead at the same point, and only readout width moves it,
    by +0.268 against Logbook 066's +0.2818.
- **Measured synaptic weights do not change what the wiring is worth under PPO** (B.1).
  - The Creamer–Leifer–Pillow fitted weights cover 1,049 of Cook 2019's 3,709 chemical edges, and
    none onto a body motor neuron.
  - They leave the pathway learnable at every scale tried.
  - Against random weights, measured × wiring is +0.013 [−0.001, +0.027], inside the minimum both
    ways.
  - The wild type's lead is present at every prior.
- **The degree-preserving null carried an unnamed difference**, found by the Wormlight review and
  measured here (A.6, the split).
  - Gap-junction counts are coupling weights, and they travel with the edges.
  - Holding the null's gap junctions at the wild type's moves the wiring gap toward the null. On the
    exact pairing the move is −0.024 under PPO (about 87% of A.6's combined move) and −0.131 under
    the reading learner (all of it).
  - That became a protocol requirement: a null states every structural property it does not
    preserve.

## What 8a did not establish

- **That the wild type's learning-speed advantage survives every null.** Against A.6's chemical-only
  null, which also keeps the autapses and draws a different chemical sample, it is +94 episodes with
  the interval spanning zero.
- **That the measured weights make the wiring legible, or hide it.** Neither learner reached either
  verdict, and the reading learner's reading is unresolved.
- **Which property of the gap junctions matters**, placement or strength.
- **Anything about thermal** beyond A.1.

## Each negative, with its cause

- **The measured prior does not move block V under PPO.** PPO rewrites the weights it starts from,
  and the prior barely moved the pilot's plateau (74–77% at every level, Logbook 072). The fitted
  table also reaches the head only: none of its edges lands on the motor layer that A.2's depth
  finding locates the effect's mechanism in.
- **A.6's combined control fell short of its minimum** (`below_minimum`). Its move was real
  (q = 0.019) but sized at about half of block V's effect, and the minimum was 2/3 of the whole
  committed effect (0.041 against a move of 0.028). The split then measured the move against itself
  and found the gap junctions carry it.
- **The reading learner's measured-weight contrast is unresolved.** Its per-seed spread (sd ≈ 0.4
  `auc_success`) left the panel able to detect only about 0.7 of the reference effect, at 48 seeds.

## What 8a changed in the protocol

Five requirements were added to `architecture-comparison-protocol` across 8a, each from a defect or
near-miss it met:

1. **A swept level is shown to reach the learner it is set on** (A.2).
2. **A pin is chosen on the learner's own gate, never on the contrast it will carry** (B.1b).
3. **A measured-weight positive is read against its placement-shuffled control** (B.1c).
4. **A null states every structural property it does not preserve** (A.6).
5. **A panel sized from a proxy reports its achieved sensitivity and never re-reads a verdict against
   it** (this change).

The fifth codifies a practice B.1c, A.6 and the split already followed. Each registered a sensitivity
from a proxy, reported the achieved figure, and let the verdict stand: 0.50–0.72 against about 0.75
for B.1c, and 0.84 and 1.11 against 0.97 and 1.03 for the split.

## Limitations

- **One cell for most rungs.** A.2, B.1c, A.6 and the split ran on hard350 alone.
- **One substrate**, Cook 2019. The Emmons 2024 release is considered for 8b, at C.0.
- **Different random-weight draws across rungs.** A.1, A.2, A.6 and the split used edge order;
  B.1c's PPO arms used the per-neuron fan-in draw.
- **Gap-junction placement and strength were never separated.**
- **The effect's size is unstable across seed sets** (A.1).
- **Under PPO, a pooled readout throughout**, with width unresolved there.

## What A.5 inherits: a package inventory, not a decision

*(**A.5 decided 2026-09-26: later.** The package waits for a combined 8a + 8b paper. It ships on its own if C.1's MLP positive control fails or C.1e has not read out by 2026-12-31, and the decision reopens at once if the literature watch finds the null-strength point or block V's claim published with comparable controls. Thermal coverage of A.6 and A.3's registered hop predictor run early in 8b. See the tracker's A.5.)*

| claim | its conditions | committed data | what an open thread would add |
|---|---|---|---|
| Block V: the wild-type wiring learns faster, with its conditions | the one-sentence form above | 070, 071, 073, 074, 075 per-seed CSVs and JSON | thermal coverage for A.2 and A.6; the ~128-seed learning-speed panel against the chemical-only null |
| The operating-point finding and its chemical mechanism | hard350 | 071; 074's hop probe | A.3 registering hop distance as a predictor |
| The null-strength finding: a null's gap junctions carry about half of the lead | combined, then paired; placement and strength joint | 074, 075, the identity checks | the placement/strength split (B.2b) |
| Measured weights leave the wiring effect unmoved under PPO | head coverage only; fan-in draw; a preprint source | 072, 073 | the reading learner resolved |
| The rule-programme negative with a diagnosed cause (Phase 7) | as in 069 | 069 and earlier | — |

## Reproducibility

- **Committed:** every figure in 070–075 is re-derivable from the per-seed CSVs and analysis JSON in
  each logbook's supporting directory.
- **Archived:** the campaign directories, exports and experiment records are archived off-repo under
  A.0.
- **Checks recorded but not repeatable from git alone.** The identity checks and re-scoring checks in
  074 and 075 ran against files still on disk. The committed JSON records that they passed, but a
  reader cannot repeat them without the archived raw logs, and this logbook says so rather than
  implying otherwise.

## The D20 gate: **GO**

D20's GO condition is met. The substrate 8b will embody has:

- **a known initialisation story** (A.1);
- **a known operating point** (A.2);
- **a measured-weight verdict** (B.1);
- **a null-strength readout** (A.6 and its split).

**D21, registered here.** Every 8b wiring contrast reads against the **chemical-only null**
(`rewired_chemical_only`), with the current null reported beside it. That covers C.1e, C.4 and B.2's
carried arms.

- **Why:** the chemical-only null differs from the wild type only in chemical placement, and it is
  Wormlight's primary null.
- **Cost:** the body is a new reference frame, so no committed result is re-read.
- **Timing:** chosen before 8b's first registration, so it cannot have been chosen for a result.

**8b starts with:**

- **C.0**, the body prerequisites, together with the decision on an Emmons 2024 substrate;
- **D19's generator**, from the documented rhythm sources;
- the null that D21 fixes.

## Artefacts

No new data. The records this synthesis rests on:

- [070](070-init-sharing-control.md) and
  [supporting/070-init-sharing-control/](supporting/070-init-sharing-control/)
- [071](071-operating-point-surface.md) and
  [supporting/071-operating-point-surface/](supporting/071-operating-point-surface/)
- [072](072-measured-prior-pilot.md) and
  [supporting/072-measured-prior-pilot/](supporting/072-measured-prior-pilot/)
- [073](073-measured-prior-contrast.md) and
  [supporting/073-measured-prior-contrast/](supporting/073-measured-prior-contrast/)
- [074](074-null-strength-control.md) and
  [supporting/074-null-strength-control/](supporting/074-null-strength-control/)
- [075](075-gap-only-split.md) and [supporting/075-gap-only-split/](supporting/075-gap-only-split/)
