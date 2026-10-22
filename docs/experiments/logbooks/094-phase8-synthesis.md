# 094: Phase 8 — The Connectome's Measurable Wiring Advantage Is Gap-Junction Placement, and It Survives the Body (Phase 8 Synthesis)

*(**Restated 2026-10-22**, [Logbook 102](102-decomposition-restated.md): the headline is withdrawn. The gap-junction placement advantage was an artefact of the gap model's degree normaliser (Logbooks 095–101). What stands of block V is the chemical wiring at the sensory–motor boundary on hard350, with no interior-wiring advantage on either cell and none through the body on the corrected model.)*

**Status**: completed. **Every 8b criterion is assigned a status, and Phase 8 is COMPLETE.** The
shipment decision is **GO to a plan review** for the phase after 8.

Phase 8 set out to ground the connectome brain's wiring advantage over a degree-preserving null
(block V, Phase 7's strongest result) and then embody it. Where it leaves that advantage:

- **On hard350, nothing of it is left in the interior chemical wiring.**
  - About half came from the null's rewired gap junctions (Logbook 075).
  - The rest came from how the null rewires the edges out of the sensors and into the motor neurons
    (Logbook 079).
  - Against a null that keeps both, no interior advantage is detectable, within ±0.014
    `auc_success`.
- **On the thermal cell, about 84% came from the null's gap junctions** (Logbook 078).
- **Gap-junction placement is the effect that holds.**
  - Against a null whose gap junctions alone are rewired, the wild type leads by **+0.096** on the
    point worm (Logbook 082).
  - That lead survives letting both wirings tune every gap strength (`placement`).
  - Through the kinematic body it is **+0.126** (`move_wt`, Logbook 092). That is mostly faster
    learning and a higher peak, together with a stronger weathervane toward the thermal setpoint
    (Logbook 093).
- **Grounding the chemical wiring moved no registered reading.**
  - Measured weights left the wiring effect unmoved (8a, Logbook 073).
  - Real synapse signs, per neuron and per connection, gave the chemical wiring no lead (Logbook
    091).
  - The serotonin and PDF knockouts read, on on-lawn speed, no effect at the minimum (*tph-1*) and
    unresolved (*pdfr-1*), though both lowered intake (Logbook 090).
- **The body works.** Both MLP-PPO and the connectome learn to forage through a 12-segment kinematic
  body driven by the anatomical motor-to-muscle map. Its crawl passes the sourced body checks.
  Learning produces a weathervane in every arm and every sense, and never klinokinesis.

**Date**: 2026-10-03 to 2026-10-14 (8b); Phase 8 from 2026-09-20.

**OpenSpec change**: `add-phase8-synthesis`. It adds two requirements to
`architecture-comparison-protocol`: a panel sized from a small pilot sizes from an upper bound on the
pilot's spread, and a positive control that fails its fixed rule is retried only as a new
registration.

**Pre-registration**: none, and none is needed. This synthesis reads committed records and runs
nothing. Every figure below is cited to the logbook it comes from.

## Objective

D20 makes this synthesis Phase 8's close: Phase 8 is COMPLETE only when every criterion carries one
of the five statuses Logbook 069 introduced. The 8a synthesis
([Logbook 076](076-8a-synthesis.md)) did that for 8a and opened 8b with the GO gate and D21.

**8b was re-planned on 2026-10-03** (roadmap v4.4):

1. **The carried controls first:** thermal coverage of the null-strength point (A.6t), the hop
   predictor with a boundary-preserving null (A.3), and the dynamics rung (B.2).
2. **Then the body:** C.0 to C.3, with C.2 gated on a cost pilot and C.5's rendering optional.
3. **Then internal state:** B.3 with D.1.
4. **Then this synthesis.**

Two items were added along the way: M.10 (per-connection signs) and M.9 (the thermal body cell).

## The status walkthrough

### Housekeeping and the carried controls

| criterion | status | record |
|---|---|---|
| H.1 literature watch repaired and re-aimed | **met** | PR #417; the first digest since September, issue #418 |
| H.2 per-connection sign table vendored | **met** | `add-per-connection-signs`: it reproduces Wormlight's table on 3,686 of 3,709 edges |
| H.3 wall-proximity exclusion in the chemotaxis validation | **met** | `add-wall-proximity-exclusion`; [035](035-realworm-chemotaxis-validation.md)'s arms re-read and unmoved |
| A.6t thermal null strength | **met** | [077](077-thermal-null-strength.md): `unreadable`, saturated (a registration error, stated); [078](078-thermal-split.md): `gap_junctions`, −0.0865 [−0.097, −0.075], lead `lead_below_minimum` |
| A.3 hop predictor with the boundary-preserving null | **met** | [079](079-boundary-null.md): `boundary`, `no_lead`; predictor `unresolved` |
| B.2a across-step dynamics | **unmet-with-reason** | [080](080-across-step-state.md): `inferior` on both cells (−0.139 and −0.216 `auc_success`) |
| B.2b plastic gap junctions on the dynamical substrate | **unreachable-with-reason** | B.2a's positive control failed; M.8 answered the question on the settling substrate |
| B.2c bout durations | **deferred-with-destination** | the next phase. Through the body it is unreachable, since a reversal lasts one step by construction; on the point worm it remains possible |

### The body

| criterion | status | record |
|---|---|---|
| C.0a step calibration | **met** | [081](081-body-prerequisites.md): a full-speed step is 5.0 worm-seconds |
| C.0b signed speed | **met** | [081](081-body-prerequisites.md): MLP 97.1%, connectome 80.2%, against 0% floors |
| C.0c proprioceptive channel | **met**, via C.1c | D22 moved it into C.1c, which landed the generator's curvature-and-phase interface |
| C.0d the rhythm source | **met** | D19, decided 2026-10-07 |
| C.1a–c the motor-to-muscle map and the body | **met** | `add-neuromuscular-body` |
| C.1d the MLP positive control | **met** | [083](083-body-control.md): passes through the fallback, 92.5% at 500 steps |
| C.1e the wiring contrast through the body | **met**, with an unresolved answer | [084](084-body-wiring-pilot.md), [085](085-body-wiring.md): +0.016 [−0.012, +0.044], `unresolved`. The contrast ran and read out, which is the criterion; the answer was not positive |
| C.2a–c the rod-chain body | **deferred-with-destination** | the next phase. Closed without its cost pilot: the recorded arithmetic puts a rod-chain run at 34–1,450 CPU-hours against about one hour on the kinematic body, and C.1e gave it no question it is needed to answer now |
| C.3 body validation | **met** | [086](086-body-validation.md): body checks pass, speed partial, no omega postures; thermotaxis half [093](093-thermal-body-validation.md) |
| C.4 the ranking through the body | **deferred-with-destination** | the next phase. Each family needs its own positive control through the body, which costs weeks of machine time while the central connectome contrast was unresolved (085) |
| C.5 rendering | **met** | `add-body-rendering`, then the realistic replay (`add-realistic-replay`) |

### Internal state and lawns

| criterion | status | record |
|---|---|---|
| D.1 patchy lawns | **met** | [087](087-patchy-lawns.md): `dwells`, +0.603 over the floor |
| D.1b patch-leaving out of roaming | **unmet-with-reason** | [088](088-patch-leaving.md): `unreadable`; 9 of 16 learners never roam on a lawn |
| B.3 stage 0, the connectome on lawns | **met** | [089](089-connectome-lawns.md): `dwells`, +0.514 |
| B.3 serotonin and PDF | **met** | [090](090-modulators.md): acute *tph-1* no effect at the minimum; acute *pdfr-1* unresolved |

### Optional (MAY)

| criterion | status | destination and reason |
|---|---|---|
| M.1 placed plasticity | **deferred-with-destination** | the plan review. It needed per-connection signs, which now exist (M.10) |
| M.2 wild type against wild type | **deferred-with-destination** | the plan review. A cheap control that calibrates every null; never run |
| M.3 gait transition | **deferred-with-destination** | with C.2, the body it needs |
| M.4 reproducibility artefacts | **met** | this logbook's reproducibility statement, under A.0's rule |
| M.5 across-seed variance components | **deferred-with-destination** | the plan review. The variance-components design was never run |
| M.6 the depth follow-ups | crossing **superseded-by-result**; thermal depths 2–3 **deferred-with-destination** | [079](079-boundary-null.md) put the hard350 lead in the sensory–motor boundary, the mechanism the depth surface tracked. The thermal check goes to the plan review |
| M.7 the chemotaxis reference set | **met** | `fix-chemotaxis-references` |
| M.8 plastic gap strengths | **met** | [082](082-plastic-gaps.md): `placement` |
| M.9 the thermal body cell | **met** | [092](092-thermal-body.md): `move_wt`, +0.126 [+0.067, +0.178] |
| M.10 per-connection signs | **met** | [091](091-per-connection-signs.md): `no_move` on all three readings |

### Carried from 8a

| criterion | status | destination |
|---|---|---|
| A.5 the publication decision | **deferred-with-destination** | the next phase's plan review, with the inventory below. The literature-watch trigger stays armed |
| S8b | **met** | this logbook |

**Open threads from Logbook 076:**

| thread | status | destination |
|---|---|---|
| gap-junction placement against strength | **met** | [082](082-plastic-gaps.md): placement, not per-junction strength |
| B.1c's reading-learner interaction | **deferred-with-destination** | A.5, only if the package claims it |
| the learning-speed lead against the chemical-only null on hard350 | **superseded-by-result** | [079](079-boundary-null.md): that lead is the boundary's |
| readout width under PPO | **deferred-with-destination** | the plan review; not reached in 8b |

## Block V, with every condition in one sentence

> On hard350 under PPO with the pooled readout, at settling depths 4 and 6, the wild-type connectome
> learns faster than its degree-preserving null. About half of that lead came from the null's
> rewired gap junctions ([075](075-gap-only-split.md)), and the rest from the null's rewired
> sensory–motor boundary: against a null that holds both, the wild type has no detectable lead
> (−0.0032 [−0.010, +0.003], [079](079-boundary-null.md)). On the thermal cell at target 35, about
> 84% came from the gap junctions ([078](078-thermal-split.md)), and against a gap-held null a lead
> below the minimum remains (+0.0164 [+0.006, +0.027]).

## What Phase 8 established

- **Block V is decomposed** (075, 078, 079). Its advantage over the degree-preserving null is, on
  hard350, the null's rewired gap junctions plus its rewired sensory–motor boundary. On thermal it is
  mostly the gap junctions. **No interior chemical-wiring advantage is detectable** on either cell
  once those are held.
- **Gap-junction placement is a robust wiring effect**:
  - +0.096 `auc_success` on the point worm on thermal at target 35 (082);
  - it is placement rather than per-junction strength, since a null free to tune every gap still
    trails by +0.086 (082);
  - it survives embodiment, +0.126 through the kinematic body at target 20 (092);
  - through the body it is mostly faster learning (+0.115 over the first 3,000 episodes) and a higher
    peak (+7.6 points), with a stronger weathervane toward the setpoint (+0.0052, 093);
  - the two wirings crawl alike, so it is not the gait.
- **Grounding the chemical layer changes nothing measurable.**
  - Measured weights do not move the wiring effect under PPO (073).
  - Atlas or per-connection signs give the chemical wiring no lead on thermal; every reading falls
    inside ±0.058 once each sign level is calibrated to the same operating point (091).
  - An uncalibrated sign effect (+0.10 to +0.14) was an operating-point artefact (091).
- **The connectome can be embodied.** The anatomical motor-to-muscle map drives a 12-segment
  kinematic body:
  - MLP-PPO learns hard350 through it, at 92.5% (083);
  - the connectome learns it on every seed (085);
  - the crawl passes frequency, wavelength, eigenworm and bout checks, and speed is partial (086).
- **Learning produces steering, never run modulation.** The weathervane is learned in every arm, for
  food (086) and temperature (093), and survives without the synthetic head-sweep (086).
  Klinokinesis is weak or absent everywhere: the point worm (036), food through the body (1.03–1.24
  against the literature's 1.5–3.0, 086) and temperature through the body (ABSENT, 093).
- **Internal state gives dwelling on lawns.** A learner reading its satiety dwells where its
  untrained policy never does: the MLP +0.603 (087) and the connectome +0.514 (089). The connectome
  uses serotonin and PDF (removing either costs food), but not in the mutants' way (090).

## What Phase 8 did not establish

- **Anything about the connectome's interior wiring.** Against the boundary-preserving null it has
  no detectable lead on hard350. No boundary-preserving *gap* null was run.
- **What in gap placement matters**: which neurons are coupled, or each neuron's total gap strength.
  A degree-preserving swap changes both together (082).
- **A chemical-wiring lead through the body.** On hard350 it is at most about 0.044 (085). On the
  thermal body cell it is +0.086 (p = 0.14, 092), which was not registered.
- **The size of the body's gap advantage on the point worm's scale.** The comparison crosses cells
  (targets 20 and 35), and the point worm saturates at target 20 (092).
- **Anything for a learner that reads the wiring without writing it.** That learner failed its
  floor through the body (084).
- **A dynamical substrate that learns as well as the settling one** (080).
- **Patch-leaving's coupling to roaming** (088), and the PDF knockout's effect (090).

## Each negative, with its cause

- **A.6t's first panel was unreadable (077)** because its saturation gate was imported from hard350
  without checking the thermal point; both learners plateaued at 94–96%. That was a registration
  error, stated, and it produced the gate preflight (protocol, 2026-10-05).
- **B.2a is inferior (080).** The leaky substrate passed the sensory signal about 400 times more
  weakly than settling, and learned only after a recalibration. Even then it learned slower, and
  the less it carried across steps the better it did. B.2b had no substrate left to run on.
- **C.1e is unresolved (085).** On hard350 the point worm's lead over the chemical-only null was
  already small (+0.022), and the body's spread was large (sd 0.175). Resolving it would take about
  140 seeds.
- **D.1b is unreadable (088).** Most learners leave lawns without ever roaming, so a coupling to
  roaming has nothing to read.
- **B.3's *pdfr-1* reading is unresolved (090)** at its sensitivity. Its *tph-1* reading is a
  no-effect: the network uses serotonin, but not to set speed the mutant's way.
- **M.10 finds no chemical lead (091).** On thermal, the cell's large effect is gap placement, and
  the chemical wiring's lead is about zero under every sign model. Grounded signs also saturate the
  settled network, so each level needed calibration, and a fifth of grounded runs never learn.
- **M.9's first positive control failed (092).** At 3,000 episodes, two seeds started learning late
  and were still climbing. It passed when retried at 6,000 under a new registration.
- **C.2 and C.4 were deferred on cost** (2026-10-10).
  - **C.2:** a rod-chain run costs 34–1,450 CPU-hours against about one hour on the kinematic body,
    and once C.1e read out the rod chain had no question it was needed to answer.
  - **C.4:** each family would need its own positive control through the body, weeks of machine
    time, while its central contrast was unresolved.

## What 8b changed in the protocol

1. **Every gate a registration reads is evaluated at the registered point before launch**, with
   `gate_preflight.py` (phase protocol, 2026-10-05, after 077).

2. **A control repeated on another cell is sized and judged on that cell's own effect**
   (`add-thermal-null-strength`).

3. **A difficulty pin for a follow-up is chosen by the gates alone, under a rule fixed before the
   pilot that decides it** (`add-thermal-split`).

4. **A panel sized from a small pilot sizes from an upper bound on the pilot's spread** (this
   change). Four of the five readings listed below, each sized from a pilot in 8b, achieved a larger
   spread than the pilot's:

   | reading | achieved spread over the pilot's |
   |---|---|
   | C.1e | 1.09 (085) |
   | M.10's lead | 1.5 (091) |
   | M.10's grounding reading | 1.4 (091) |
   | M.10's receptor reading | 0.95 (091) |
   | M.9 | **3.6** (092) |

   M.9's point estimate gave 9 seeds; the panel's achieved spread would have needed about 111.

5. **A positive control that fails its fixed rule is retried only as a new registration** (this
   change). M.9's retry is the precedent: the failed verdict kept, the change fixed before running,
   new seeds, and the same rule.

## Limitations

- **Two cells and one learner for the wiring readings.** Every registered wiring contrast in 8b ran
  under PPO, on hard350 or thermal.
- **Two substrates.** The point-worm controls and M.9 ran on Cook 2019. C.0 to C.3's body work and
  B.3 ran on Emmons 2024, which differs from Cook in four gap-junction pairs.
- **The body is kinematic**, not muscle-driven. Its rhythm comes from a body-level generator (D19),
  and its left–right thermal and chemical signal is partly synthetic.
- **Large per-seed spreads through the body**, from late PPO collapses among other causes (092).
- **Sign-only behavioural references.** Thermal and chemical bias curves are graded on direction,
  not magnitude.

## What A.5 inherits: a package inventory, not a decision

| claim | its conditions | committed data | what an open thread would add |
|---|---|---|---|
| Block V decomposed: no interior chemical advantage once gap junctions and the boundary are held | hard350 and thermal, PPO, settling depth 4 | 074, 075, 077–079 | a boundary-preserving gap null |
| Gap-junction placement is a wiring effect, and not per-junction strength | thermal, point worm, target 35 | 078, 082 | a null moving placement with per-neuron totals held |
| It survives the kinematic body | thermal, target 20, 6,000 episodes, Cook 2019 | 092, 093 | the collapse rate on more seeds; a gap boundary null through the body |
| Grounding the chemical layer moves nothing | measured weights (8a); calibrated signs | 073, 091 | — |
| Learning produces steering, never klinokinesis | every substrate and sense tested | 036, 086, 093 | a thermal derivative control |
| Phase 7's rule-programme negative | as in 069 | 069 | — |

## Reproducibility

- **Committed:** every figure in Logbooks 077–093 is re-derivable from the per-seed CSVs, analysis
  JSON and launch records in each logbook's supporting directory.

- **Archived:** the campaign directories, session exports and trained weights are archived off-repo
  under A.0's rule. The trained weights behind 086, 092 and 093's evaluations are among them.

- **Not repeatable from git alone:**

  - re-scoring a panel from raw logs, and any evaluation on trained weights (the kinematics in 086,
    092 and 093);
  - the identity checks run against files on disk.

  The committed JSON records their results.

## The shipment decision: **GO to a plan review**

The roadmap names no phase after 8 and gives it no gate. Phase 7's close is the precedent: this
synthesis writes the inventory, and the next phase is designed in its own plan review, as Phase 8 was
(roadmap v4.3). Two other branches were open, and neither was taken:

- **Scoping the next phase here** would mix design into a reading of the record.
- **Leaving the decision open** would leave the destinations above with no one to read them.

### What the phase after 8 opens on

Ordered by dependency:

1. **What in gap placement matters.**
   - A null that moves gap placement with each neuron's total gap strength held (082's open
     question).
   - A boundary-preserving gap null: 247 of Cook 2019's 1,093 gap pairs are interior under the
     body's boundary, which is enough to rewire. This needs either a lower-spread substrate or the
     collapses understood first. At M.9's spread, 32 seeds resolve only 0.107.
2. **Late PPO collapse at long horizons.** Every arm collapsed sometimes after episode 4,000 (092).
   This is a methods question, and every 6,000-episode panel inherits it.
3. **Why learning never yields klinokinesis.** It is consistent across substrates and senses. The
   candidates are the reward, the action space and the step length.
4. **The body's carried items:**
   - C.2's rod-chain body behind its cost pilot, with M.3's gait transition;
   - C.4's ranking through the body;
   - B.2c on the point worm.
5. **Cheap calibration controls:** M.2 (wild type against wild type) and M.5 (variance components).
   Both bound how much of any null contrast is null variance.
6. **M.1 placed plasticity** on the klinotaxis circuit, now that per-connection signs exist.
7. **M.6's thermal depth check**, and readout width under PPO.
8. **Cross-species transfer** (7b), held behind the body.
9. **A.5**, the publication decision, with the inventory above.

## Artefacts

No new data. The records this synthesis rests on are Logbooks
[077](077-thermal-null-strength.md) to [093](093-thermal-body-validation.md), each with its supporting
directory, and the 8a synthesis [076](076-8a-synthesis.md).
