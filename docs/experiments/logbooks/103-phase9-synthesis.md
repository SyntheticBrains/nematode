# 103: Phase 9 — The Connectome's Wiring Advantage Was the Gap Model's Normaliser (Phase 9 Synthesis)

**Status**: completed. **Every Phase 9 criterion is assigned a status, and Phase 9 is COMPLETE.** The
shipment decision is **GO to a plan review** for the next phase, whose direction the maintainer chose on
2026-10-22: standard lawn assays graded against real-worm data, with the architecture ranking halted.

Phase 9 set out to screen the gap model's operating point before building a mechanism on the one wiring
effect that survived Phase 8: gap-junction placement. The screen found the effect was the model's.

> No gap model tested shows the wild type's gap-placement lead at the minimum. The lead the project
> carried from Logbook 082 to 092 came from the model's degree normaliser, which let a few high-count
> pairs couple far above 1. Fixing the normaliser alone removes it, as do unit weights and
> log-compressed counts, on the point worm and through the body. Under ohmic coupling at its
> calibrated, weak conductance, a wild-type lead below the minimum (+0.021 [+0.007, +0.035]) is
> unresolved.

That is the phase's registered STOP condition, met as registered
([Logbooks 099](099-logcount-contrast.md) and [100](100-body-logcount.md)), opened as a modelling
question ([101](101-gap-model-question.md)) and restated across the record
([102](102-decomposition-restated.md)).

**Date**: 2026-10-22.

**OpenSpec change**: none; the tracker `phase9-tracking` is archived with this close.

## Objective

Phase 9 was planned as *screen, then mechanism* (roadmap v4.5):

- **block M**, the instrument at long horizons;
- **block G**, the mechanism of gap-junction placement, gated on G.0's screen of the gap model's
  operating point;
- **block R**, the architecture ranking through the body, under a trigger;
- smaller blocks K (klinokinesis), P (placed plasticity), E.1 (the rod-chain body) and T.1 (a JIT
  spike).

G.0 read `artefact` on 2026-10-17, and the phase followed the branch its plan named for that case.

## The status walkthrough

The vocabulary is the protocol's: met, unmet-with-reason, deferred-with-destination,
superseded-by-result, unreachable-with-reason. "The next phase" means the direction decided on
2026-10-22. Its stages, from the direction review, are:

- **stage 0:** the lawn-assay benchmark;
- **stage 1:** locomotor primitives and slow state;
- **stage 2:** controllers fitted to match the animal;
- **stage 3:** the wiring question on fidelity;
- **stage 4:** the body and posture.

### H — housekeeping

| criterion | status | record |
|---|---|---|
| H.1 the literature watch re-aimed and seeded | **met** | the brief and seeds re-aimed with the tracker (PR #454); digests W42 (#448) and W43 (#462) ran against them and were reflected |
| H.2 the retention rule; 8b's campaigns archived | **met** | A.0's wording, cited by every launch record since |
| H.3 D23 and D24 recorded | **met** | every Phase 9 launch record runs on Emmons 2024 and carries a D24 table |
| H.4 the Cook 2019 male sheets for G.6 | **superseded-by-result** | they served G.6, which replicates a gap effect there is no longer one of |
| H.5 the record corrections as dated notes | **met** | landed with the tracker's PR, 2026-10-15 |
| H.6 the klinokinesis reference repaired (K.1) | **deferred-with-destination** | the next phase's stage 0, where the pirouette-rate curve against dC/dt is a graded statistic |

### M — the instrument

| criterion | status | record |
|---|---|---|
| M.1 late collapse | **met** | [096](096-collapse-diagnosis.md): no registered cause indicated; [098](098-collapse-remedies.md): no remedy beat a placebo re-roll, and the stop rule fired. The collapse statistic ships (D25), and collapse follows the training stream, not the seed |
| M.1(c) the confirmatory panel | **superseded-by-result** | there was no passing remedy to confirm |
| M.2 wild type against wild type | **deferred-with-destination** | the next phase's stage 3, where wiring contrasts return under a fidelity metric |
| M.3 variance components | **deferred-with-destination** | the next phase's stage 3, for the same reason |
| M.4 the secondary body statistic | **superseded-by-result** | [100](100-body-logcount.md) fixed the body verdict on episodes 1–3,000, with the full horizon and collapse beside it, under a protocol requirement that answers what M.4 was for |

### G — the gap junctions

| criterion | status | record |
|---|---|---|
| G.0 the gap model's operating point | **met** | [095](095-gap-operating-point.md): `artefact`; +0.104 under counts, +0.003 under unit weights |
| G.0-A the corrected model | **met** | [097](097-gap-substrate.md): log counts chosen by learnability; [099](099-logcount-contrast.md) and [100](100-body-logcount.md): `no_wiring_effect` on the point worm and through the body; [102](102-decomposition-restated.md): the decomposition restated |
| G.0-B the modelling question (added 2026-10-20) | **met** | [101](101-gap-model-question.md): strength normalisation `no_wiring_effect`; ohmic `unresolved` below the minimum; the STOP statement stands |
| G.1, G.2, G.3, G.3b the mechanism block | **superseded-by-result** | there is no gap-placement effect left to explain |
| D30 the publication decision | **deferred-with-destination** | the first result of the next phase. Decided by the maintainer, 2026-10-22: the STOP result is the methods motivation of a first fidelity paper, not a stand-alone preprint. The literature-watch trigger stays armed |

### R — the ranking

| criterion | status | record |
|---|---|---|
| R.1 per-family positive controls through the body | **unmet-with-reason** | halted by the maintainer's direction decision of 2026-10-22: the ranking answers a question the STOP has emptied of its main motivation |
| R.2 the ranking through the body | **deferred-with-destination** | the trigger's condition (2), a signed body freeze (D27), is unmet. Condition (1), M.1 closed, is met; (3) and (4) were never evaluated, not shown unmet. Destination: the next phase's stage 2, where controller families are compared on the fidelity benchmark rather than ranked on learning speed through the body |
| D27 the body freeze | **unmet-with-reason** | the go/no-go decision is **NO-GO**: the freeze is not signed, because no ranking through the body is scheduled to need it |

### The syntheses

| criterion | status | record |
|---|---|---|
| S9a the 9a synthesis | **met** | this logbook |
| S9b the Phase 9 synthesis | **met** | this logbook. The maintainer chose on 2026-10-22 that one synthesis closes both shipments, because the direction decision empties 9b |
| the watch re-aimed at the close | **deferred-with-destination** | the plan review, once the next phase's open questions are written; the brief should name them, and they do not exist yet |

### SHOULD

| criterion | status | record |
|---|---|---|
| G.5 the boundary-held gap null through the body | **superseded-by-result** | no gap effect through the body ([100](100-body-logcount.md)) |
| G.3b on hard350 | **superseded-by-result** | G.3b has no class to test |
| G.6 the male's gap layer | **superseded-by-result** | there is no gap effect to replicate |
| G.7 the reading learner on G.3b's nulls | **superseded-by-result** | G.3b did not run |
| G.8 a contact-constrained gap null | **superseded-by-result** | it was gated on G.0 reading `survives` |
| K the bearing × primitive 2×2 | **deferred-with-destination** | the next phase's stage 1, which adds explicit locomotor primitives, with reward-trained PPO kept as stage 2's comparator |
| P(a)–(c) a placed, state-gated set-point memory | **deferred-with-destination** | the next phase's stage 2, as a candidate mechanism for salt-set-point chemotaxis (Kunitomo's assay is a plate assay). W43's AFD co-transmission paper means a placed rule must state its transmitter form |
| E.1 the rod-chain cost pilot | **deferred-with-destination** | the next phase's stage 4, where omega and delta postures need a body that can make them |
| T.1 the whole-step JIT spike | **deferred-with-destination** | the plan review, if fitting controllers to behavioural statistics needs the speed |

### MAY

| criterion | status | record |
|---|---|---|
| the full G.1/G.2 replication on hard350 | **superseded-by-result** | no effect to replicate |
| a co-location-stratified gap null | **superseded-by-result** | no effect to stratify |
| the gap-placement landscape search | **superseded-by-result** | it was gated on G.3b naming a class |
| the klinokinesis ladder's remaining rungs | **deferred-with-destination** | the next phase's stage 1 |
| readout width under PPO on a non-saturating cell | **unmet-with-reason** | a PPO wiring-contrast question, out of scope after the direction decision |
| B.2c's forward/reverse bout durations on the point worm | **deferred-with-destination** | the next phase's stage 0, where run and reversal bouts are graded statistics |
| the ohmic-coupling settling substrate | **met** | built and calibrated in G.0-A ([097](097-gap-substrate.md)); its contrast read in G.0-B ([101](101-gap-model-question.md)) |

## What Phase 9 established

1. **The gap-placement advantage was an artefact of the gap model's degree normaliser.**

   | gap model, thermal t35, Emmons 2024 | wild type − gap-only null (80% CI) |
   |---|---|
   | count weights, degree normaliser (the original) | +0.104 [+0.087, +0.123] |
   | unit weights | +0.003 [−0.012, +0.018] |
   | log-compressed counts | +0.003 [−0.009, +0.016] |
   | strength normaliser | −0.009 [−0.026, +0.009] |
   | ohmic ×0.03 | +0.021 [+0.007, +0.035], unresolved and below the minimum |
   | log counts, through the body (t20, 128 seeds, episodes 1–3,000) | −0.022 [−0.045, +0.005] |

   The strength normaliser keeps the counts and replaces only the degree-based denominator, so the
   artefact sits in that denominator.

2. **What stands of block V** (Logbook 102):

   - on hard350, the chemical wiring at the sensory–motor boundary, with no interior-wiring advantage
     there;
   - on thermal, a residual below the minimum, with the chemical wiring and the autapses not
     separated;
   - through the body, nothing on the corrected model.

3. **Late PPO collapse on the body cell** strikes about one run in eight by 6,000 episodes. Which run
   collapses follows the training stream, not the seed. No registered cause precedes it, and no
   standard remedy beats a placebo re-roll (096, 098). The collapse statistic, and the windowed
   verdict that reads before the hazard, now serve every long panel.

4. **Strength-normalised gap junctions** are a brain option, with every coupling at most 1, a spectral
   radius of 1 and an exact identity under unit weights. They are the standard symmetric
   normalisation for a weighted graph. The project's default remains the original, for byte identity.

## What Phase 9 did not establish

- **That gap placement is irrelevant in any model.** The corrections read here keep the coupling
  additive, apart from ohmic at a weak conductance. Stronger ohmic scales failed the learnability probe
  and could not be read under this learner.
- **The boundary lead under a corrected gap model.** 079 read it only on the original model, with the
  gap layer identical in both arms.
- **075's gap component on hard350.** It was never re-read, so it is uncited rather than refuted.
- **Through the body, the substrate's part.** The body re-run moved from Cook to Emmons; the count
  model's body lead on Emmons was not measured.

## The re-read: question or instrument

- **About the question:** 095, 099, 100, 101 and 102. Each reads what the wiring does, under a stated
  gap model.
- **About the instrument:** 096, 097 and 098. Collapse, the choice of corrected model by learnability
  with its wiring gap never read, and the remedies.
- **Neither moves the other.** The collapse window (100) was fixed before the body ran. G.0-A's selection
  pilot never read the wiring gap, and G.0-B read ohmic only on fresh seeds.

## What Phase 9 changed in the protocol

1. **A wiring effect traced to a model artefact is re-read on the corrected model, under its own
   registration, before mechanism work builds on it** (`architecture-comparison-protocol`, from
   `add-logcount-gap-contrast`).
2. **A panel longer than the collapse hazard may fix its verdict on an earlier window.** The window is
   fixed in advance, the gates read the same window, and the rest is reported beside
   (`add-body-logcount-contrast`).
3. **Collapse incidence is reported beside every long learning curve, under one definition**
   (`add-collapse-instrument`, D25 as amended).
4. **A remedy is read against a placebo re-roll of the same seeds**, never against the runs it was
   chosen to fix (`add-collapse-remedies`).
5. **Operating characteristics are simulated with the registered instrument**, not a normal
   approximation. The shared classifier's `unresolved` band takes about 15% of the probability at a
   true null, whatever the seed count. A family of two readings needing two nulls is therefore
   "stands" only about 59% of the time (101).
6. **A harness reads a run only if its own log records the config its name claims** (from PR #457's
   review), applied in every Phase 9 harness since.

## Limitations

- **One learner.** Every wiring reading in Phase 9 ran under PPO.
- **One cell for the corrections.** The four gap models were read on thermal t35, and log counts also
  through the body. hard350 was not re-read on a corrected model.
- **Additive coupling, or weak ohmic.** No gap model with strong diffusive coupling could be read, since
  stronger ohmic scales fail to learn under this learner.
- **Sign-only behavioural references** for thermal and chemical bias curves, as in Phase 8.

## Reproducibility

- **Committed:** every reading in Logbooks 095–102 is re-derivable from the per-seed CSVs, analysis
  JSON, probes, preflights, D24 tables and launch records in each logbook's supporting directory.
  Every launch record names the commit its harness ran at.
- **Archived off-repo** under A.0's rule: the campaign directories, session exports and trained
  weights.
- **Not repeatable from git alone:** re-scoring any panel from its raw logs.

## The shipment decision: GO to a plan review

Phase 9 is COMPLETE. Its deliverable, under the STOP branch, is the statement above, with the controls
that make it citable.

The maintainer decided the next phase's direction on 2026-10-22, from a direction review (gitignored):
**fidelity on the plate**.

- **The benchmark:** a graded benchmark of standard on-lawn assays, built from named public datasets:
  - roaming and dwelling (Scheer & Bargmann 2023's released HMM);
  - local-search decay (López-Cruz 2019);
  - lawn leaving (Scheer; Haley 2025);
  - the chemotaxis navigation components (Iino & Yoshida 2009);
  - posture later (the Open Worm Movement Database).
- **Order:** the point worm first, with explicit locomotor primitives and the serotonin/PDF slow state;
  the body and posture later.
- **Controllers:** fitted to match the animal, with reward-only PPO kept as a comparator.
- **The connectome question,** re-asked on behavioural fidelity and mutant phenocopy rather than on
  learning speed.
- **Out of scope:** the architecture ranking is halted. Thermotaxis is deferred to a second tier, not
  dropped. Predator and other non-standard scenarios leave the scope.

The plan review writes that phase, re-aims the literature watch at its questions, and decides whether
the project's research question in AGENTS.md is re-framed.

## Artefacts

- Logbooks [095](095-gap-operating-point.md) to [102](102-decomposition-restated.md), and their
  supporting directories.
- The archived tracker: `openspec/changes/archive/2026-10-22-phase9-tracking/`.
