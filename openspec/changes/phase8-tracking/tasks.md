# Tasks: Phase 8 (Ground, then Embody — Measured Substrate & Body) Shipment Tracker

This is the living checklist for Phase 8. The plan is authoritative in `docs/roadmap.md`
§ Phase 8 (v4.3, PR #395): the scope decision against whole-organism fidelity, the rungs, design
decisions D15–D20, exit criteria, risk table and novelty map. Phase 8 ships in two cuts —
**8a** (block A + B.1 + B.2 + synthesis) and **8b** (block C + D + B.3 + synthesis) — split by
success (D20). Every Phase 8 milestone PR updates this checklist as part of its diff.

**Status legend**: `[ ]` not started, `[x]` closed — done, or (for unexercised SHOULD/MAY scope)
dropped/deferred with a dated note and one of the five statuses, so `openspec archive` is never
blocked by an honestly-unexercised item.

**Preconditions (both hold):**

1. ✅ **Phase 7 closed** — SPLIT, every criterion assigned a status
   ([Logbook 069](../../../docs/experiments/logbooks/069-phase7-synthesis.md), 2026-09-19).
2. ✅ **Plan finalised** — roadmap v4.3 with D15–D20 ratified, D15's premise corrected at review
   (PR #395, merged 2026-09-20).

> **Not in Phase 8.** Cross-species (7b: pacificus, dauer, transplant — the phase after 8, behind
> C.1). Male–hermaphrodite contrast (data on disk; Future Directions). Multi-agent, pheromones,
> red queen, ecological co-evolution. Evolution in every form (6b NEAT in `phase6b-tracking`,
> deferred unscheduled). Structural plasticity across development. Neuropeptide layer as a rule
> substrate. Spiking-STDP, neuromorphic. 3D, Sibernetic, ion-channel neurons. The uniform
> substrate-writing rule programme (closed, Logbook 063).

<!-- -->

> **Execution-protocol standards (Decision B in design.md) apply to every rung, panel and sweep
> below** — paired seeds with BH-FDR within-pass; uniform budget with a convergence audit;
> metric audit on a new regime; byte-identical-when-off; load-time validation; **a positive
> control per new component before any connectome arm**; **sweep before pin** (D16); re-baseline
> at a moved operating point in its own campaign; a registered minimum effect in both
> directions; **body-substrate results are a new reference frame**; artefact retention
> registered at phase start.

<!-- -->

> **Execution**: paired-seed arms launch through `scripts/run_campaign.py`; each run is
> byte-for-byte the single-run entry point, so only wall-clock moves. The machine is 18 cores
> with 16-way parallelism as the measured ceiling ([Logbook 039](../../../docs/experiments/logbooks/039-runtime-acceleration-audit.md));
> the GPU was rejected on measurement and is not assumed anywhere below.

<!-- -->

> **Coarse-grained by design.** These sub-tasks are the load-bearing shape; per-milestone OpenSpec
> changes elaborate them (first: A.4's consolidation change, then the A.1 control change).

## Shipment 8a — Ground

**OpenSpec changes**: placeholders; created per milestone
**Status**: ⬜ not started
**Roadmap layer**: substrate (init control, operating point, measured weights, dynamics)
**Approx effort**: ≈ 9–12 active weeks (A ≈ 3–4, B.1 ≈ 3–4, B.2 ≈ 3–4)
**Roadmap reference**: `docs/roadmap.md` § Phase 8 § Required deliverables (8a), D15/D16/D17

### Block A — close Phase 7's exposed result

**Execution order**: A.0 → **A.4** → A.1 → A.2, with A.3 and A.5 opportunistic. The list below
keeps the roadmap's numbering; A.4 runs first because it lands before the first 8a registration.

- [x] **A.0 Artefact-retention rule** — **registered 2026-09-20**, before the phase's first
  campaign, per design.md B.13. Logbook 069 records what the absence of this rule cost: the
  step-level exports for every campaign before the readout-width era are **gone**, and during L.1b a
  metric could not be compared against L.0's run because that run's export had been deleted, which
  was recorded as uncompared rather than counted as matching. The rule, cited by every campaign
  below:
  - **Committed to git**, under `docs/experiments/logbooks/supporting/<logbook>/`: the **parsed
    per-seed CSV** carrying every field the analysis reads, the analysis JSON the logbook's figures
    are derived from, and the launch record. A headline figure that cannot be re-derived from these
    alone does not ship.
  - **Archived off-repo**: raw campaign logs and step-level exports. They are the maintainer's
    private archive, not a public artefact, and form no part of the reproducibility surface —
    stated plainly rather than implied, as Logbook 069 states it.
  - **Deleted deliberately, never incidentally**: a campaign directory is removed only after its
    per-seed CSV is committed. Where a field cannot be compared because its source is gone, it is
    named as **uncompared** rather than counted as matching — the parsed-field identity rule's own
    wording.
  - **Registered before the campaign**, not after: a campaign that starts without its retention
    line recorded is blocked, per the `phase8-tracking` requirement.
- [x] **A.1 The init-vs-rewiring control** — **done 2026-09-21** ([Logbook 070](../../../docs/experiments/logbooks/070-init-sharing-control.md), 768/768 runs, 32 paired seeds): **no dissolution detected on any arm**, survival **established** on five of eight readings, three **unresolved at the panel's sensitivity and not evidence of survival**. The **pairing half** of block V's standing condition is **partially** discharged; the across-seed half is registered as a follow-up. *(Two corrections are recorded in the logbook rather than quietly applied: the `dense_mask` arms shared a generator with PPO's minibatch sampler and were **re-run** — the other sixteen arms were shown unaffected and their reuse licensed by a field-by-field identity check — and the first write-up's tally was an **over-count by one**, four survivals rather than five. The re-run independently moves the true count to five, so the figure cited elsewhere is right but was not right when written.)* Second finding, ranked above the first in the record: block V's **magnitude is not stable across seed sets** — thermal replicated in direction at **35%** of its committed size against hard350's 105% — so with power registered against V.4's magnitude the thermal cell was underpowered against its own observed effect even at 32 seeds. Original scope: (D15; own change): both definitions of shared
  initialisation — (i) dense-draw-then-mask, (ii) per-neuron fan-in sharing — as arms,
  byte-identical-when-off; n ≥ 16 paired seeds on the thermal and hard food-only block-V cells,
  `rewire_seed` fixed per pair, through the committed block-V harnesses. Registered outcome: the
  +35.4/+23.5/+55.3/+40.1 learning-speed effects survive (wiring result) or dissolve
  (initialisation result); a shrunken effect is reported as shrunken, against a registered
  minimum in both directions. The learning-speed result is restated with its status wherever
  it is cited.
- [x] **A.2 The calibration-and-robustness surface** — **done 2026-09-23** ([Logbook 071](../../../docs/experiments/logbooks/071-operating-point-surface.md), `add-operating-point-surface`, 1,600 runs). **The committed point sits on an edge, not a plateau.** Under PPO the wiring advantage replicates at the centre (+536 episodes, 92% of A.1's) and is **depth-critical** — present at depths 4 and 6, abolished at 3, reversed at 2 with every arm learning — and robust to initial action noise. Under the reading learner the rewired null is ahead at the same point, reproducing 064/066, and only readout width moves it, by +0.268 against 066's +0.2818. A hop measurement accounts for depth: the wild type has no motor neuron one hop from a food sensor, a rewiring manufactures about nine. The A.1 re-read at depth 6 (32 seeds) is consistent with A.1. **One named gap**: readout width under PPO saturates on hard350 and is unresolved there (see B.1c). Follow-ups recorded at A.3 and M.6. *(**Correction 2026-09-23** to this item's wording below: the three pins are "shared by both learners" only across the learners of `ConnectomePPOBrain`. `readout_width` and `forward_pass_depth` are not declared on `MLPPPOBrainConfig`, and a YAML setting them under `mlpppo` is dropped with a warning rather than an error, so no later rung may read this surface onto an MLP arm.)* Original scope: (D16 as amended 2026-09-20; own change).
  **Two halves, because a pin swept on one learner is not swept for another.**
  *(a) The reading learner* — `readout_only`, the `PlasticTensors` literal in
  `brain/arch/_plasticity_config.py` that freezes the chemical matrix and leaves the readout
  learning — across all five pins: `plasticity_rate`, readout width, `forward_pass_depth`,
  `initial_log_std`, `trace_decay`.
  *(b) PPO* — across the **three shared** pins only (readout width, `forward_pass_depth`,
  `initial_log_std`; the other two are the plasticity rule's), at a **reduced grid**, because
  PPO runs are the expensive ones and because the phase's most exposed claim is a PPO result
  whose depth and log-std [Logbook 069](../../../docs/experiments/logbooks/069-phase7-synthesis.md)
  lists as never varied.
  Both halves one-factor-at-a-time around the current point first, a full crossing only for pins
  that move the wiring effect's sign; reported as a sensitivity surface, not a new pin. Fixes the
  point every later rung cites, per learner. If (b) moves the sign, **A.1 is re-read at the new
  point** — a registered outcome, not a surprise. Churchland et al. 2026 (arXiv:2609.07355) is
  the external reason to report it this way.
- [ ] **A.3 Frozen-operator structural predictors** (SHOULD; probe or small change): routing
  confinement and mode-driver metrics (Therianos 2026, arXiv:2606.17745) on Cook 2019
  synapse-count weights against every rewired null block V generated, registered as predictors
  of time-to-competence. No training.
  **A.2 handed this a concrete candidate** *(added 2026-09-23)*. Logbook 071's hop probe measured
  sensory-to-motor distance over the graph the simulation propagates through and found the wild type
  has **no** motor neuron one hop from a food sensor while a degree-preserving rewiring manufactures
  about nine, so at a settling budget of 2 the wild type reaches 26 of 39 motor neurons and the null
  reaches essentially all 39. That tracks the depth surface exactly. **It was computed after the
  fact and registers nothing**, which is what A.3 is for: a statistic, a direction and a minimum
  named before the correlation, per the structural-predictor requirement. Logbook 069 recorded that
  no graph property measured there predicted learning time; this one predicts the operating point at
  which the effect exists, which is a different and testable claim.
- [x] **A.4 Methodology consolidation** (`consolidate-plasticity-methodology`, 2026-09-20): the
  **44** `plasticity-evaluation` requirements redistributed four ways — **7 stay** (the six the
  spec's own Purpose statement describes, plus the delayed-reward control, which describes live
  code in `quantumnematode/plasticity/positive_control.py` and had been misfiled as methodology);
  **11 move** to `architecture-comparison-protocol`, each binding live Phase 8 work and following
  the home Phase 7 already used for its wiring rules; **18 fold** into the
  [phase protocol](../../../docs/research/phase-protocol.md) as clauses under existing principles,
  no new principle and no renumbering, the document at 222 lines against a 250 budget; **8 retire**
  with a reason and a migration each, being specific to the rule programme that closed with a
  diagnosed cause. The reading surface is now **thirteen principles plus the eleven requirements
  that stayed enforceable** in `architecture-comparison-protocol`, rather than thirty-eight rules in
  a capability whose Purpose never described them.
  The harness audit found nothing orphaned across 27 harness/test pairs. *(The 44 was counted
  2026-09-20; Logbook 069 and the roadmap's inheritance list say 42, which was true at the close —
  archiving the Phase 7 synthesis change merged its own two requirements into this spec.)*
- [ ] **A.5 The publication decision** (SHOULD; taken after A.1 reads out): the package is block V
  with its initialisation control, the rule-programme negative with a diagnosed cause, and the
  operating-point finding with its surface. Not a gate for anything after it.

### B.1 — measured synaptic signs and strengths (D17)

- [x] **B.1a Data sub-deliverable** — **done 2026-09-23** (`add-measured-weight-prior`): the fitted table vendored with provenance and its licence notice, a SHA-checked loader with a coverage report, and `weight_prior` (`random` | `measured` | `measured_signs` | `measured_shuffled`) plus the `measured_weight_scale` pin, byte-identical when off, with the rewired null receiving each neuron's wild-type values. B.1b is now configs only. *(**Corrected 2026-09-23** at B.1a's licence check: the Randi 2023 atlas is **cited, not vendored** — its OSF deposit states no licence and its only licensed copy is a GPL-3.0 file, in an Apache-2.0 repository. The Creamer–Leifer–Pillow table is vendored under MIT. It covers **125 neurons** (the fitted model holds 154; 156 is the paper's recording figure), and on Cook 2019 it reaches 1,049 of 3,709 chemical edges — 77.0% of the 1,363 coverable at head scope, none onto the body motor neurons.)* Original scope: (own change): the Creamer–Leifer–Pillow fitted weights (bioRxiv
  2024.09.22.614271, preprint — never stands alone) and the Randi 2023 atlas as raw source,
  vendored under `data/connectome/` with `PROVENANCE.md` and a **licence check before
  vendoring**; a `weight_prior` key (`random` | `measured` | `measured_shuffled`),
  byte-identical-when-off; head-scope coverage (156 neurons) stated, with the command-to-motor
  and motor layers left on the random draw.
- [x] **B.1b Pilot-scale arms** — **met 2026-09-24** ([Logbook 072](../../../docs/experiments/logbooks/072-measured-prior-pilot.md), `add-measured-prior-pilot`, 448 runs). **The pathway is learnable at every scale tried, and both learners take multiplier 1.0.** On hard350, the sign-only prior and the measured prior at 0.25, 0.5, 1, 2 and 4 times the random draw's magnitude each beat their own frozen floor on both wirings, 8 of 8 seeds, under PPO (fan-in draw, seeds 113–120) and under the reading learner (A.2's centre, seeds 121–128); none saturates. The Lee branch did not fire, and the registered rule, which never reads the wiring gap, selected the default. No level's gap crossed zero against the random level's side, so B.1c inherits no condition from the pilot. Under the reading learner small multipliers cost learning without preventing it, and the null's lead there closes at 2 and 4, recorded as descriptive only. Original scope: sign-only vs sign-plus-magnitude; the LDS-to-rate-model unit
  scale swept (a pin); disjoint pilot seeds.
- [ ] **B.1c The 2×3** (own change): wiring {wild type, rewired null} × prior {random, measured,
  measured-shuffled}, on `readout_only` at A.2's swept point and again under PPO **under D15's
  shared-init protocol**; head scope and full scope reported separately. Payoff either way. **Risk
  registered** (roadmap § Required deliverables 4, from Lee 2026): atlas grounding produced no
  functional sensory-to-command step in an atlas-fitted c302; if measured weights leave the
  klinotaxis pathway unlearnable here, the rung closes *unmet-with-reason* with the pathway named.
  **Readout width is unresolved under PPO, and this arm cites A.2 for it** *(added 2026-09-23)*.
  A.2's PPO half ran the `per_neuron` level, but both arms saturated — 96.5% and 97.7% plateau
  full-clear against the instrument's 90% bar — so the wiring gap there reads +1.25 episodes with a
  tight interval, which is two arms tied at the ceiling rather than a swept reading. The
  operating-point requirement blocks a PPO contrast registered against a sweep that does not cover
  its pins, so **B.1's PPO arm either runs at the pooled width and says so in the same sentence as
  its claim, or buys the width reading on a cell that does not saturate first.** The reading half
  has no such problem: it resolves width cleanly and finds it the one pin that moves that learner.
  **What B.1b hands this arm** *(added 2026-09-24, [Logbook 072](../../../docs/experiments/logbooks/072-measured-prior-pilot.md))*:
  - `measured_weight_scale` **1.0** on both learners, chosen on each learner's own gate;
  - the PPO arm runs under `per_neuron_fanin`, the pairing B.1b defined so every neuron keeps its
    wild-type multiset on the null. That draw is not the one A.2 swept, so the arm's depth and
    initial-noise settings come from a surface measured under `edge_order`, and it says so in the
    same sentence as its PPO claim;
  - no condition from the pilot's sign-movement rule. The reading learner's gap closing at
    multipliers 2 and 4 is offered to this change's registration as a candidate sensitivity arm,
    not carried as a condition.

### B.2 — the dynamics rung (SHOULD)

- [ ] **B.2a Across-step state** (own change): per-neuron leaky-integrator state with intrinsic
  time constants (a global τ swept first); gap junctions as ohmic coupling inside that dynamics;
  byte-identical-when-off. **Positive controls, in this order**: first MLP-PPO on the target cell
  (the spec's blocking control for any new component), then — once that passes — PPO on the
  dynamical connectome against PPO on the settling connectome, which must learn the cell at least
  as well before any wiring contrast runs. The second is itself a connectome arm, so it comes
  after the first rather than beside it.
- [ ] **B.2b Plastic gap junctions under PPO** (D4's surviving destination): the electrical
  synapses learnable on the block-V cells, against the null, with the *Nat. Commun.* 2020
  olfactory-learning precedent as the biological motivation. External convergence to cite, not
  lean on: Lee 2026's gap-junction-only shuffle collapses chemotaxis where the chemical-only
  shuffle barely moves it, in a different model with different controls.
- [ ] **B.2c Validation target — deferred to 8b by construction**: forward/reverse bout-duration
  statistics per Morrison & Young 2025, registered as a behavioural sign/shape-level claim.
  Bout durations need reversal, which is **C.0b in 8b**, and D20 forbids 8b work before the 8a
  synthesis — so this task cannot complete inside 8a and is listed here only because it belongs
  to B.2. It is scheduled immediately after C.0b and carries B.2's status until then. *(The
  alternative, pulling signed speed forward into 8a, was rejected: it would drag C.0's substrate
  freeze into 8a with it.)*

### 8a synthesis

- [ ] **S8a** The 8a synthesis logbook: every 8a criterion assigned one of the five statuses; the
  D20 gate written as a go/no-go decision; the roadmap Phase 8 row set to "8a complete / 8b
  pending" on GO; `phase8-tracking` status headers updated.

## Shipment 8b — Embody

**OpenSpec changes**: placeholders; created per milestone
**Status**: ⬜ not started — **does not start until S8a has assigned every 8a criterion a status (D20)**
**Roadmap layer**: body (C), environment (D), internal state (B.3)
**Approx effort**: ≈ 10–15 active weeks (C ≈ 8–12, D + B.3 ≈ 2–3)
**Roadmap reference**: `docs/roadmap.md` § Phase 8 § Required deliverables (8b), D18/D19/D20
**Dependencies**: S8a GO; B.2 for C.2's second half

### C.0 — body prerequisites (MUST; each lands, validates and freezes before C.1 registers)

- [ ] **C.0a Step–time calibration**: one recorded constant relating an environment step to worm
  seconds, from the validated 0.2 mm/s crawl, the ~1.6 s undulation period and the arena scale;
  cited by every kinematic target and cost estimate below. (The current cap is one body length
  per step, ≥ 5 s of worm time at full speed.)
- [ ] **C.0b Signed speed**: reversal as a first-class continuous action (speed is clamped to
  `[0, max_step_mm]` today), byte-identical-when-off, so VA/DA and VB/DB mean different things
  and escape can be reversal-plus-turn.
- [ ] **C.0c Proprioceptive channel**: posture or stretch fed back as sensory input, with the
  target neurons stated with a biological argument (the predator-projection precedent).
- [ ] **C.0d D19 decided and recorded** — the body-level proprioceptive wave generator's form and
  parameters, calibrated once on the MLP positive control (design.md open question).

### C.1 — the anatomical motor-to-muscle readout into a kinematic body (MUST)

- [ ] **C.1a Loader keeps the muscle cells**: the Cook 2019 body-wall muscle columns (`dBWML*`,
  `vm*`) that `connectome/loader.py` drops become a first-class motor-neuron-to-muscle tensor
  with smoke tests and a `PROVENANCE.md` note.
- [ ] **C.1b Muscle readout**: the NMJ matrix pooled into four quadrants × N segments, learnable
  *gains* only (D18: a gain vector identical in size across arms, calibrated once on the MLP
  control and frozen, sensitivity-checked).
- [ ] **C.1c Kinematic body**: muscle drive → segmental curvature; displacement per step from the
  change of posture between steps by resistive-force theory (no ODE), with D19's body-level
  generator supplying the wave. Renderer hook for C.5.
- [ ] **C.1d Positive control**: MLP-PPO forages through C.1 on the target cell. **Blocks every
  connectome arm below it.**
- [ ] **C.1e The wiring contrast through the body**: wild type vs rewired null under PPO and under
  `readout_only`, with **floors and baselines re-established on this substrate** — a new
  reference frame, never a delta against 029 or block V.

### C.2 — the rod-chain body (SHOULD)

- [ ] **C.2a Cost budget registered** before the rung: "a 16-seed panel in roughly a day at 16
  workers" pinned to a wall-clock number from a pilot configured the way the campaign will be.
- [ ] **C.2b The chain**: an ElegansBot-class 2D rod chain (Chung, Chang & Kim, eLife 2024; 8–12
  rods, anisotropic drag, torsional-spring muscles) in `Continuous2DEnvironment`, driven by
  C.1's muscle drive; MLP positive control; if the budget fails after the reduced chain, coarser
  integrator and Numba/JAX have been tried, **stop at C.1** and record *deferred-with-destination*
  (behind D6).
- [ ] **C.2c Second half** (MAY; needs B.2 and a shorter step): remove the body-level generator
  and ask whether the connectome's motor circuit plus proprioception produces the wave, against
  the rewired null.

### C.3 — body-level validation (SHOULD)

- [ ] **C.3** Eigenworm posture spectrum (Stephens et al. 2008), undulation frequency and
  amplitude, omega-turn geometry, and the Logbook 035/036 klinokinesis and weathervane curves
  re-derived from *emergent* kinematics; swimming vs crawling gait if C.2 ships (MAY).

### C.4 — the architecture ranking through the body (SHOULD)

- [ ] **C.4** The six MUST families of Logbook 029 re-run on the frozen body substrate with new
  baselines, per the architecture-comparison protocol's re-run rule. First SHOULD to drop if 8b
  overloads.

### C.5 — rendering (SHOULD)

- [ ] **C.5** `pixel_continuous` draws the segmented body from curvature or rod state (head,
  tail, reversals and omega turns visible) with an optional posture overlay; headless unchanged.

### B.3 + D.1 — internal state, the modulator field, patchy lawns (SHOULD, coupled)

- [ ] **B.3** Internal-state sensory module and the modulator concentration field (Phase 7's
  B.2/B.3, deferred): satiety already crosses the brain boundary; serotonin/PDF gating of roaming
  vs dwelling as the first behavioural consequence.
- [ ] **D.1** Patchy bacterial lawns: geometry with edges, per-patch depletion (the
  `source_depletion_enabled` mechanism, config-gated), food quality; the roaming/dwelling
  readout B.3 gates; validation against Flavell-lab roaming/dwelling fractions. The 2D agar plate
  is kept (D.2); the three behaviours stay the comparison set (D.3).

### MAY (not gates)

- [ ] **M.1 Placed plasticity** on the klinotaxis circuit, on the B.1 substrate, against a
  degree-stratified random subset of the same size, with the three confounds specified.

- [ ] **M.2 Wild-type-vs-wild-type control**: Cook 2019 against Witvliet dataset 8 (adult,
  nerve-ring scope, already vendored).

- [ ] **M.3 Swimming/crawling gait transition** as a body validation target (with C.2).

- [ ] **M.6 The depth finding's two registered follow-ups** *(added 2026-09-23 after A.2)*.
  A.2's PPO surface found the wiring effect **depth-critical**: replicated at depths 4 and 6,
  abolished at 3, reversed at 2 with the rewired null reaching 62% full-clear against the wild
  type's 10%. Every arm cleared its own frozen floor, so no level is a broken-arm artefact. Two
  things were registered before that readout and are owed:

  - **Thermal confirmation at depths 2 and 3** — branch 2 of
    [071's launch record](../../../docs/experiments/logbooks/supporting/071-operating-point-surface/launch.md):
    thermal runs at levels where hard350 shows the sign move. A.1 measured thermal's detectable
    effect at 1.6-3.4x its observed one even at 32 seeds, so this confirms a direction rather than
    resolving a magnitude, and it should be sized for that.
  - **The full crossing for `forward_pass_depth`** — D16 gives a crossing to any pin that moves the
    sign, and 071's launch record makes the crossing a second registration drawing its sensitivity
    from the one-factor pass as prior committed data.
    **The crossing's priority fell once the mechanism was measured.** 071's hop probe explains the
    depth surface on its own — the wild type has no motor neuron one hop from a food sensor and a
    rewiring manufactures about nine — so a depth-by-pin crossing is now less likely to be where the
    answer is than A.3's registered predictor test. Recorded rather than dropped, because it was
    registered before the readout and dropping it afterwards is the move the protocol forbids.

- [ ] **M.5 The across-seed half of block V's standing condition** *(added 2026-09-21 after A.1)*.
  A.1 discharged the **within-seed** half — the two wirings putting the same drawn values on
  different edges. The other half is untouched: with `rewire_seed` unset, a null's graph **and** its
  weights both derive from the run seed, so the null arm carries graph-variance the wild-type arm
  does not, and no panel has separated them.
  **The obvious design is wrong.** Pinning `rewire_seed` to a single constant holds the graph fixed
  but makes the result about **one** rewiring, which reintroduces the shared-nulls caveat
  [V.4](../../../docs/experiments/logbooks/065-wiring-fresh-rewiring.md) closed by moving to fresh
  rewirings. The right shape is a **variance-components design** — several pinned graphs, several
  weight seeds within each — which separates graph variance from weight variance instead of trading
  one confound for the other.
  **A.1 made this more interesting, not less.** Its second finding is that block V's magnitude moves
  substantially across seed sets (thermal at 35% of its committed size against hard350's 105%), and
  one candidate explanation is that some rewirings are simply easier to beat than others. That is
  exactly the quantity this design would measure.
  MAY, not SHOULD: A.1 already answered the half that addresses the published critique, so this is
  about understanding the effect's stability rather than defending it, and Phase 8's committed scope
  is the substrate and body ladder.

- [ ] **M.4 Reproducibility artefacts** current to the Phase 8 platform state, under A.0's rule.

### Phase 8 synthesis

- [ ] **S8b** The Phase 8 synthesis logbook: every criterion assigned one of the five statuses;
  the roadmap Phase 8 row set to ✅ COMPLETE only then; the literature watch re-aimed (protocol
  principle 13); this change archived alongside the synthesis change.
