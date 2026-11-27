# 138: Phase 10 — The Benchmark Is Valid and a Worm Can Be Fitted to Pass It, but No Controller Predicts a Mutant, and What the Wiring Contributes Is Unread (Phase 10 Synthesis)

**Status**: `completed`. **Every Phase 10 criterion is assigned a status, and Phase 10 is COMPLETE.** It closes between its minimum-viable and target success levels. The shipment decision is **GO to the Phase 11 plan review**.

**The publication decision is taken.** Phase 10's publication is the `wormbench` release: the grader on PyPI and the real tracks on Zenodo under a DOI. Whether to write a benchmark paper goes to the plan review, with a cross-lab check of the bands as its first step.

> Phase 10 asked how faithfully a simulated worm reproduces standard *C. elegans* behaviour, graded against real worms, and what the connectome contributes to that fidelity.
>
> - **The first half has an instrument and a partial answer.** The benchmark is valid: real worms pass it and every negative fails. A connectome controller can be fitted to pass its wild type.
> - **No controller predicts a mutant.** The one controller with predicted mutant rows, C.1 (v3), gets both wrong. C.2's registered fit fails tph-1, though tph-1 was in its objective.
> - **The second half is unread on behaviour.** Phase 10 closes without W. The one reading in which the specific wiring beats its nulls is neural, not behavioural.

**Date**: 2026-11-27.

**OpenSpec change**: none. The tracker `phase10-tracking` is archived with this close.

## Objective

Phase 10 was planned as *fidelity on the plate* (roadmap v5.0, 2026-10-23, D33–D41):

- **H:** the housekeeping;
- **E + T:** the 1-s worm, its locomotor primitives and assays, and the batched evaluator;
- **B:** the benchmark, valid only if its positive controls pass and every negative fails;
- **N:** the neuromodulator layer from data;
- **F.0:** the fitting method's parameter-recovery control;
- **C:** the controllers and the fidelity table:
  - C.1, a mechanistic baseline;
  - C.2, the connectome;
  - C.3, the connectome trained by reward;
  - C.4, the table;
- **W:** the wiring question on fidelity;
- **the synthesis.**

The maintainer decided on 2026-11-23 that Phase 10 closes without W. C.2's registered fit then ran once, capped ([Logbook 133](133-c2-fit.md)). C.1's v4 refit closed at its pilot ([Logbook 136](136-c1-v4-fit.md)), and C.4 was read ([Logbook 137](137-fidelity-table.md)).

## The status walkthrough

The vocabulary is the protocol's: met, unmet-with-reason, deferred-with-destination, superseded-by-result, unreachable-with-reason. "Phase 11" is the next phase, which D41 opens on the body and posture. Its plan review writes it.

### H — housekeeping

| criterion | status | record |
|---|---|---|
| H.1 the literature watch re-aimed | **met** | the brief and seeds re-aimed in PR #470; digest #498, the first run against them, ran 2026-10-26 |
| H.2 the research question re-framed (D33) | **met** | AGENTS.md, the README and the roadmap, PR #470 |
| H.3 the scenario scope recorded | **met** | `configs/README.md` § Scope in Phase 10 |
| H.4 the seed-band table | **met** | the tracker's design.md. W's band moved to 7201–7456 after Logbook 104's calibration overlapped it |
| H complete before the first registration | **met** | H closed on 2026-10-24, before B.3, the first registration to cite a band. It ran after E + T and B.1, neither of which drew a registered band |

### E + T — the worm, its assays and the batched evaluator

| criterion | status | record |
|---|---|---|
| E.1 one step-duration value (D35) | **met** | change `add-step-duration` |
| E.2 locomotor primitives (D36) | **met** | change `add-locomotor-primitives`. On-lawn reversal kinematics from data ([108](108-food-dependent-kinematics.md)), and reversals that follow the run speed as an option ([132](132-speed-following-reversals.md)) |
| E.3 assay protocols | **met** | change `add-assay-protocols` |
| T.1 the batched evaluator, its parity test and cost pilot | **met** | [104](104-batched-evaluator.md): parity in three layers, 48–98 times faster per core than the per-step runner |

### B — the benchmark

| criterion | status | record |
|---|---|---|
| B.1 the reference re-extraction | **met** | [105](105-reference-extraction.md); the primitives checked on food, [108](108-food-dependent-kinematics.md) |
| B.2 the readouts | **met** | [107](107-benchmark-readouts.md); the reversal detector scoped to off-food tracks |
| B.3 the bands and the grading rule | **met** | [109](109-benchmark-grading.md); the rule's own check passed on a retry registered before it ran |
| B.4 the controls: B valid | **met** | [110](110-benchmark-controls.md): 14 of 15 held-out units pass against an allowance of 2, and six negatives fail |
| B.5 track import | **met** | [114](114-track-import.md): CSV or WCON, with no simulator code in the grading |
| B v2 | **met** | [115](115-benchmark-v2.md), VALID: 19 statistics, a measurement model, and kinematics and chemotaxis families |
| B v3 | **met** | [120](120-benchmark-v3.md), VALID: 21 statistics, adding local search's log ratio, the off-food onset rate and the flip fraction |
| B v4 | **met** | [124](124-benchmark-v4.md), VALID: 24 statistics in six families, with on-lawn activity |

### N — the neuromodulator layer

| criterion | status | record |
|---|---|---|
| N.1 the licences | **met** | [106](106-neuromodulator-layer.md): Bentley's S1 Dataset and Ripoll-Sánchez's figshare items are CC BY 4.0 |
| N.2 the layer | **met** | [106](106-neuromodulator-layer.md): 55 pools, 121 couples, 16 fixed monoamine signs and 105 fitted strengths |
| N.3 its null family | **met** | receptor-placement nulls, audited over seeds 7501–7564 |

### F.0 — the fitting methods' positive control

| criterion | status | record |
|---|---|---|
| F.0a, NPE's recovery on C.1's model | **met** | read: it failed as registered, its informative gate sitting below the simulator's own noise ([111](111-npe-recovery.md)) |
| NPE as C.1's fitting method (D38) | **superseded-by-result** | after F.0a, D38 was amended on 2026-10-25 so that C.1 is fitted by sep-CMA |
| F.0b-i, sep-CMA on C.1's model | **met** | [112](112-cma-recovery.md): gain 0.991 on v1. On v2, [116](116-cma-recovery-v2.md) failed (0.690) and [117](117-cma-recovery-v2b.md) passed (0.873). On v3, [121](121-c1-v3-fit.md)'s reduced recovery gave 0.985 |
| F.0b-ii, sep-CMA on C.2's encoder | **met** | read: its gate failed, gain 0.635 against 0.75 ([133](133-c2-fit.md)). F.0 does not clear for C.2 |

### C — the controllers and the fidelity table

| criterion | status | record |
|---|---|---|
| C.1 the mechanistic baseline | **met** | <ul><li>v1: passed, [113](113-c1-fit.md)</li><li>v2: C.1b failed, reversal-dominated, [119](119-c1b-fit.md)</li><li>v3: passed 3 of 5, [121](121-c1-v3-fit.md)</li><li>v4: passes 0 of 5, [124](124-benchmark-v4.md)</li><li>the v4 refit: closed at its pilot's no-go, [136](136-c1-v4-fit.md)</li></ul> |
| C.2 the connectome controller | **met** | read: its verdict fails ([133](133-c2-fit.md)). The wild type passes 5 of 5 and tph-1 0 of 5, while pdfr-1 and tax-4 pass 5 of 5. Read against the failed F.0b-ii |
| C.3 the reward-trained comparator | **unreachable-with-reason** | Its entry was tied to a C.2 pass (the maintainer's decision of 2026-11-23). C.2's fit, the phase's last C.2 run, failed (decision of 2026-11-25). Its question, the C.2 architecture trained by reward, goes to the Phase 11 plan review |
| C.4 the fidelity table | **met** | [137](137-fidelity-table.md). C.1 (v3)'s two predicted mutants fail. No controller passes the wild type with every encodable mutant |
| the exit criterion "C.1–C.4 read and the table published" | **unmet-with-reason** | C.1, C.2 and C.4 are read and the table is published. C.3 was not run, for the reason above |

### W — the wiring question on fidelity

| criterion | status | record |
|---|---|---|
| W.1 the nulls | **deferred-with-destination** | a later phase, scheduled at the Phase 11 plan review. It can run only once a fitting procedure passes its own recovery control. The nulls from Phase 9 and N.3 are built and stay |
| W.2 the registration | **deferred-with-destination** | The same destination. By the maintainer's decision of 2026-11-27, no registration is written now. Since F.0b-ii failed, W's arms do not inherit C.2's procedure (D38's note of 2026-11-24), so a registration would have no tested procedure to run. Its notes and sizing stay: T.1's inputs and C.2's measured 36.75 h a fit ([133](133-c2-fit.md)) |
| W.3 the panel | **deferred-with-destination** | the same destination |
| the exit criterion "W read on graded distance" | **deferred-with-destination** | Phase 10 closes without W, by the maintainer's decision of 2026-11-23 |

### The synthesis

| criterion | status | record |
|---|---|---|
| S10 every criterion assigned a status | **met** | this logbook |
| D30 the publication decision | **met** | Taken on 2026-11-27. Phase 10's publication is the `wormbench` release, citable alone: the grader on PyPI and the real tracks on Zenodo under a DOI. The tracks were deposited on 2026-11-27, [doi:10.5281/zenodo.24585652](https://doi.org/10.5281/zenodo.24585652), and the PyPI release follows this close. Whether to write a benchmark paper goes to the Phase 11 plan review, with a cross-lab check of the bands on Haley 2025's real tracks as its first step |
| the watch re-aimed at the close (principle 13) | **deferred-with-destination** | the Phase 11 plan review, as at Phase 9's close. The brief should name the next phase's open questions, which that review writes |

### SHOULD

| criterion | status | record |
|---|---|---|
| Haley 2025's patch acceptance against encounter number | **deferred-with-destination** | the Phase 11 plan review, as the benchmark's first extension, a new family. Haley's tracks are not vendored. The plan review recorded them as CC BY, and the deposit's own licence is checked before they are vendored. The same tracks serve the cross-lab check above |
| WCON export | **deferred-with-destination** | the `wormbench` release that follows this close: the writer in `scripts/analysis/track_import_check.py` moves into the package, with a command and a test |
| the novelty spot-checks owed at B's registration | **met** | [109](109-benchmark-grading.md): NeuroSparkSNT, PGOB and Morrison & Young 2025 do not occupy the claim. The search of 2026-10-30 covered 2023 to October 2026 ([its note](../../research/novelty-and-external-models-2026-10-30.md)) |

### MAY

| criterion | status | record |
|---|---|---|
| thermotaxis on the new primitives | **unmet-with-reason** | Not taken. The loader refuses the 1-s step and the primitives away from lawns. `data/thermotaxis/` holds bias curves only, so the benchmark's standard (bands from cohorts, held-out controls) needs a track dataset first. D41 keeps thermotaxis the second tier |
| bordering against an O₂ field, with npr-1 | **unmet-with-reason** | Not taken. It needs an oxygen field on the 1-s worm, npr-1 bordering data (none vendored) and an npr-1 encoding for each controller |
| a Creamer-dynamics controller | **deferred-with-destination** | the Phase 11 plan review, as a candidate controller whose internals are fitted to neural data rather than behaviour. The weights are vendored (`data/connectome/creamer_lds_2026_model_weights.csv`), but its readout still needs a fit |

## What Phase 10 established

1. **A valid benchmark of standard on-lawn behaviour for simulated worms.**

   - **What v4 grades:** 24 statistics at z\* 3.0781 in six equal families: Scheer's small lawn and uniform well, López-Cruz's local search, Chen's chemotaxis, kinematics and on-lawn activity.

   - **Its controls:**

     - N2's 26 unread dates pass on the activity family, v4's registered real-worm control. PD1074's 24 held-out dates pass too, as description, since v4's planning probe had read them ([124](124-benchmark-v4.md)).
     - All eight negatives fail the wild type in 5 of 5 populations, at a median D of 4.30 to 7.72 ([137](137-fidelity-table.md)'s table).

   - **What it grades:** outside tracks, as CSV or WCON ([114](114-track-import.md)). It ships as `wormbench`, which needs only numpy and scipy.

   - **How it got there:** it was revised three times. Each time a fit had passed through a gap, which the revision closed:

     - v2 added kinematics and chemotaxis, which C.1's v1 pass did not survive ([115](115-benchmark-v2.md));
     - v3 graded reorientation from positions, after C.1b's fits reversed 52–56% of the time off food ([119](119-c1b-fit.md), [120](120-benchmark-v3.md));
     - v4 graded on-lawn activity ([124](124-benchmark-v4.md)).

     Each version was found VALID under its own controls, and each earlier pass stands as read at its version.

2. **A connectome controller can be fitted to pass the wild type.** C.2's registered answer passes v4's wild type in 5 of 5 registered populations and 15 of 15 in all ([133](133-c2-fit.md)). Three conditions attach:

   - its recovery control failed (gain 0.635), so the fit's reach is unvalidated;
   - its procedure was tuned through six pilots on discarded seeds;
   - its mutants were fitted, so they are in-sample.

   C.1 (v3), fitted to v3's wild type alone, passes v3 in 3 of 5 and fails v4 in all 5, on activity ([121](121-c1-v3-fit.md), [124](124-benchmark-v4.md)).

3. **No controller predicts a mutant.**

   - **C.1 (v3)'s two predictions both fail** at every restart that passed v3's wild type ([137](137-fidelity-table.md)):
     - tph-1 goes the wrong way. Removing the intake signal lowers roaming by 0.12 to 0.20, where real tph-1 worms roam more.
     - tax-4 leaves the lawn 5.7 to 15.8 more times an hour, against a real +0.31.
   - **C.2's registered fit fails tph-1,** though tph-1 was in its objective ([133](133-c2-fit.md)). That does not show C.2's model cannot represent it: the sixth pilot's answer, under the same rules, passes tph-1 in 15 of 15.
   - **Overall:** no Phase 10 controller passes the wild type with every encodable mutant.

4. **In C.2, wild-type behaviour does not identify the knockout responses** (from probes and description, not a registered reading).

   - **The principle** is [Logbook 128](128-c2-identifiability.md)'s, and D37's amendment of 2026-11-10 rests on it.
   - **What is new** is the ensemble below, tabulated at this close.
   - **Its consequence:** fitted on the wild type alone, C.2's tph-1 is inert. Fitting the mutant found the real response in three of five runs, at a D_fit within about 0.1 of the wild-type-only answer's. Because every start began from the same point, the 24 starts are not independent tests.

5. **What the wiring contributes to behavioural fidelity is unread.** W did not run.

   - **The one reading where the specific wiring beats its nulls is neural.** C.2's encoder at the box's centre, unfitted, predicts Creamer et al.'s measured responses to single-neuron stimulation at r 0.065, and beats all 19 degree-preserving rewirings ([134](134-fit-first-neural-readings.md)).
   - **A readout ceiling** of about 0.42 bounds that reading ([135](135-neural-readout-ceiling.md)).
   - **Behaviour fitting's part is unsettled.** At Logbook 131's answer it does not detectably add ([135](135-neural-readout-ceiling.md)). At C.2's registered answer it raised r by 0.023, its interval clearing 0 by 0.001 ([133](133-c2-fit.md)), on one answer, as description.

6. **The platform for fidelity work:**

   - the 1-s point worm;
   - locomotor primitives with kinematics from data;
   - assay protocols;
   - the batched evaluator, 48–98 times faster per core, with parity to the per-step runner;
   - a neuromodulator layer placed from data, with its null family;
   - the compact encoder that makes the connectome brain a function of named parameters ([122](122-c2-encoder.md), [130](130-c2-encoder-procedure.md)).

## Two readings added at this close (description)

Neither is a registered reading, and no status above depends on either.

### The identifiability ensemble

`supporting/138-phase10-synthesis/probes/identifiability.py.txt` reads C.2's committed pilot, recovery and fit records, and simulates nothing.

- **The ensemble:** 24 starts reached stage 2a, which fits the wild type on D_fit's four families, with no mutant in the objective.

- **tph-1 at the end of 2a:** its roaming difference sits between −0.017 and +0.006 in every start. Over all 432 2a readouts it is −0.051 to +0.046. The real band centre is +0.27.

- **What the starts share:**

  - Every start begins at the graded box's centre (launch record 133).
  - Stage 1 fits 77 parameters and holds the other 60 there, among them the neuromodulator layer's strengths.
  - So every start enters 2a with the same serotonin couples: ser-5 at 0.055 at stage 1's end in every run (`133-c2-fit/probes/serotonin_history_complete.json`).
  - 2a frees them and moves them widely. At 2a's end ser-5 sits between 0.0039 and 2.31 across the five kept starts ([133](133-c2-fit.md)'s table). tph-1 stays inert in every one, the sixth pilot's included at ser-5 2.31.
  - The starts share an origin, not an end state, so they are not independent tests. What persists is that the wild-type objective never makes the response real.

- **The lineage is the stronger evidence:**

  | Run (kept start) | 2a's end: D_fit, tph-1 Δ | 2b's end: D_fit, tph-1 Δ | The final stage's end: D_fit, tph-1 Δ |
  |---|---|---|---|
  | fourth pilot | 1.84, −0.013 | 2.15, +0.281 | 1.95, +0.371 |
  | fifth pilot | 1.91, +0.001 | 1.96, +0.232 | 1.89, +0.275 |
  | sixth pilot | 1.68, −0.000 | 1.80, +0.278 | 1.77, +0.269 |
  | F.0b-ii's recovery | 1.95, +0.006 | 2.00, +0.001 | 1.98, −0.003 |
  | the registered fit | 2.17, −0.007 | 2.55, +0.001 | 1.67, −0.002 |

  - **The fifth pilot:** its 2a answer and its final answer sit at nearly the same distance from the wild type's bands, D_fit 1.91 and 1.89. One has no tph-1 response, and the other has the real one. Its final answer failed its go rule's wild type on the on-lawn onset rate ([133](133-c2-fit.md)), so neither answer is shown to pass the wild type.
  - **The other runs:** three of the five runs that reached 2b found the response there, and two never did. [Logbook 133](133-c2-fit.md) shows that ser-5's strength does not order which.

- **tax-4:** its roaming difference at 2a's end varies in sign even among the starts nearest the wild type. Below D_fit 2 it is +0.69 once (the fourth pilot, D_fit 1.84) and −0.03 to −0.04 three times. The largest negative value, −0.278, is at D_fit 4.78. The real band centre is −0.12.

### The likelihood check

`supporting/138-phase10-synthesis/probes/likelihood_check.py.txt` was the one-hour check decided on 2026-11-23, before any architecture work ranked by likelihood. Does a real-worm likelihood rank C.1's four committed answers as the benchmark does?

- **How it works:** no behaviour is simulated. Real Scheer tracks are replayed along their paths, and each answer scores every second by the forward algorithm over its two hidden states. A second contributes through two channels:

  - the event channel: whether a reversal or omega started;
  - the speed channel: the 1-s displacement.

- **The rule:** Spearman ρ ≥ 0.8 tracks the benchmark; ρ ≤ 0.4 drops the route. It was fixed before the first run, on held-out cohorts. The band-cohort re-run below came after that run's result had been seen.

- **The result:** ρ is 1.0 on 975 small-lawn and 44 uniform-well animals. With C.1 (v1) placed last by rule, ρ orders three answers by their D.

  | Answer | Log-likelihood per scored second | The three Scheer families' mean D |
  |---|---|---|
  | C.1 (v3) | −14.63 | 1.19 |
  | C.1's v4 warm probe | −14.70 | 1.37 |
  | C.1b | −22.84 | 1.54 |
  | C.1 (v1) | −194.93 | fails every population, so its D is undefined and it is ranked last by rule |

- **Three conditions attach:**

  - **It is not a held-out test.** The animals are band cohorts, the ones the v2–v4 bands were built from, and every C.1 answer was fitted against those bands. It shows the likelihood agreeing with the benchmark on the data both were tuned to.

  - **The speed channel dominates:** about −15.5 per second at C.1 (v3), against −0.33 for the events. The probe does not say why:

    - the model's speed spread may be too narrow for real worms' per-second variation;
    - or the probe's own stated approximations may inflate it: the spread is redrawn every second rather than every 10 s, under a fixed tracker-noise σ.

    Either way, a likelihood fit on this emission would chase speed, so the emission needs checking before any likelihood route.

  - **On the event channel alone**, the v4 warm probe ranks first (−0.328 against −0.334), as v4's D_fit does.

- **A breach, disclosed:**

  - The check's first run, on 2026-11-24, read PD1074's held-out cohorts, which the benchmark's controls forbid (`benchmark-controls`, "The new statistics are validated on unread data").
  - It was description only, and nothing from it is committed or used here.
  - The record above is the re-run on band cohorts, with its labels corrected when it was committed.

## What Phase 10 did not establish

- **Which controller is closer to the worm.** C.2 against C.1 is description under four conditions ([137](137-fidelity-table.md)):
  - C.2's control failed;
  - the bodies differ;
  - C.2's mutants are fitted and C.1's predicted;
  - the populations are unpaired.
- **What the connectome contributes to behavioural fidelity.** W did not run, and no null was fitted at equal budget.
- **That a fitted controller holds the worm's mechanism.** C.2's mutant passes are in-sample.
- **That C.1's mechanism class cannot predict the mutants.** The predictions come from three answers of one fitted model, under its registered encodings.
- **That the lawn bands hold on another lab's rig.** The small-lawn, uniform-well and mutant families come from Scheer & Bargmann's recordings alone, and the cross-lab check was not run.
- **How reward training fares on fidelity.** C.3 was not run.

## The re-read: question or instrument

- **About the instrument:**
  - the benchmark and its controls ([105](105-reference-extraction.md)–[110](110-benchmark-controls.md), [114](114-track-import.md), [115](115-benchmark-v2.md), [120](120-benchmark-v3.md), [124](124-benchmark-v4.md));
  - the evaluator and its speed-up ([104](104-batched-evaluator.md), [118](118-v2-speedup.md));
  - the recovery controls ([111](111-npe-recovery.md), [112](112-cma-recovery.md), [116](116-cma-recovery-v2.md), [117](117-cma-recovery-v2b.md), and F.0b-ii in [133](133-c2-fit.md));
  - the encoder and C.2's procedure ([122](122-c2-encoder.md), [123](123-c2-speed-noise.md), [126](126-c2-run-mode.md)–[132](132-speed-following-reversals.md)).
- **About the question:**
  - the fits and the table ([113](113-c1-fit.md), [119](119-c1b-fit.md), [121](121-c1-v3-fit.md), [125](125-c1-v4-diagnosis.md), [133](133-c2-fit.md), [136](136-c1-v4-fit.md), [137](137-fidelity-table.md));
  - the neural readings ([134](134-fit-first-neural-readings.md), [135](135-neural-readout-ceiling.md)).
- **The instrument moved question readings, by design and in the open.**
  - Each benchmark revision followed a fit that passed through a gap, and conditioned that fit's pass. C.1's v1 pass fails v2, and its v3 pass fails v4.
  - The earlier verdicts stand as read at their versions, with dated notes at every site.
  - Every reading in the fidelity table is on v4, fixed before C.4 was registered.

## Success level

- **Minimum viable: met.** The benchmark is valid, with the 1-s worm, its primitives and assays, and track import.
- **Target: not met.** C.1, C.2 and C.4 are read and the table is published, but C.3 was not run and W was not read.
- **Stretch: not met,** because W did not run. Its controller clause, a controller that passes every first-tier assay and reproduces the pdfr-1 and tax-4 dissociations, is met only in-sample: C.2, fitted to both mutants, passes the wild type and both. No controller reproduces them by prediction.

Against the novelty map:

- **The benchmark is delivered.** The 2026-10-30 search found the claim unoccupied for the worm.
- **The connectome controller graded against equal-budget nulls is not.** No null was fitted.
- **The pdfr-1 and tax-4 dissociation** is reproduced only in-sample, by C.2 fitted to both mutants.

## What Phase 10 changed in the protocol

1. **Eleven corollaries dated 2026-10-25** ([phase-protocol.md](../../research/phase-protocol.md)) come from building a benchmark from data and controlling the fitting methods:
   - **For a positive control:**
     - a positive control's bar is met by its oracle before launch;
     - an instrument is shown to fail what it must fail, part by part;
     - where validity needs several controls to pass, the joint rule is registered;
     - the preflight covers every gate, including one added in review, with a bar between simulations set against the simulator's own noise.
   - **For held-out data and seeds:**
     - held-out data are read once, under a guard in the code;
     - every seed drawn enters the seed table.
   - **For pilots and minimums:**
     - pilots run long enough to reach the machine's sustained rate;
     - where the claim is fidelity to data, the minimum is anchored to the data;
     - the uncertainty is matched to the question and checked before grading.
   - **For retries and records:**
     - a failed registered gate is retried only under an amendment committed first;
     - every number in a record is read from a committed record or script.
2. **The controllers' rules, in the `controller-fit` and `benchmark-controls` specs:**
   - every fit's stages refuse unless their launch record's parts are committed with their bound records unchanged;
   - fits record every statistic's value and z, not only failures;
   - a held-out read happens only inside a registered fit;
   - a mutant's prediction follows the encoding table's `unidentified` rule.

**Three lessons are proposed for the protocol at the plan review.** Each has a committed record, and none is written into the protocol here:

- **A fit to the wild type does not identify its perturbation responses.** Fit the perturbations or predict them, say which, and never read a fitted one as a prediction ([128](128-c2-identifiability.md); D37's amendment; the ensemble above).
- **A benchmark is revised against the fits that pass through it.** Register each revision before the next fit. Read every earlier pass as conditioned, not overturned. Freeze the version a registered table reads.
- **A smoke or probe that exercises a held-out reading's code passes band-cohort stand-ins, never the defaults.** Two breaches show why: the likelihood check's first run above, and C.1's refit smoke ([136](136-c1-v4-fit.md)'s launch record, Part A).

## Limitations

- **The point worm only.** There is no body and no posture, by D41.
- **One rig for the lawn families.** Scheer & Bargmann's small lawn, uniform well and mutants come from one lab. López-Cruz's search and Chen's chemotaxis come from two others.
- **Three mutants.** tph-1 is graded on Scheer only, pdfr-1 is not encodable for C.1, and npr-1 and cat-2 are second tier (D37).
- **Every answer reverses less than real worms.** Median reversal shares are 3.4% (C.1 (v3)) and 5.5% (C.2), against Roberts' 17.3%. The share is reported, not graded ([137](137-fidelity-table.md)).
- **C.2's procedure is unvalidated.** It was tuned over six pilots, and its recovery control failed.

## Reproducibility

- **Committed:** every reading in Logbooks 104–137 is re-derivable from the launch records, records, probes and scripts in each logbook's supporting directory. The fidelity table's JSON records every source's SHA-256 ([137](137-fidelity-table.md)).
- **This close's two probes** re-run from the repository root in seconds, from committed records and the vendored band-cohort tracks.
- **Archived off-repo:** C.2's fit and pilot run directories and logs, by the retention rule.
- **Not repeatable from git alone:** C.2's guarded stages. Since the grader moved into `wormbench`, C.2's manifest no longer matches, so those stages refuse and its records are final ([133](133-c2-fit.md)).

## Corrections made in the open

- **The likelihood check's labels.** Its band-cohort record named its animals `held_out_animals` and its question "held-out real-worm likelihood".
  - The committed copy calls them band cohorts and states that they are in-sample to C.1's fits. Its rule's text is unchanged, with a note that the rule was written for held-out cohorts.
  - Its figures are identical. Only its run time differs, and C.1 (v1)'s infinite D is written as null.
- **An independent review before commit** checked every figure against its record and found statements the records do not support. Each is fixed here:
  - the shared origin of the 24 starts, first stated as a shared serotonin state at 2a's end, which 2a does not keep;
  - behaviour fitting's neural effect at C.2's registered answer, first stated as no detectable gain;
  - PD1074's held-out dates, first listed among v4's controls rather than as description;
  - C.1b's off-food reversal share, first given as 42–56% after Logbook 120's wording (42% is the small lawn; Logbook 120 and AGENTS.md now carry the correction);
  - C.2's tph-1, first stated for the model rather than its registered fit;
  - the likelihood rule's timing, the cost of fitting the mutants, tax-4's sign, and the speed channel's cause.
- **The identifiability count.** The direction review of 2026-11-23 and its handover (both gitignored) counted fewer starts. The handover's 23 is the committed probe's 24 without the registered fit, which ran after it.

## The shipment decision: GO to the Phase 11 plan review

Phase 10 is COMPLETE. Its deliverable is the benchmark, with the controls that make it citable, and the fidelity table that grades the phase's controllers on it.

**D41 opens Phase 11 on the body and posture.** The plan review writes that phase, and weighs, from this close:

1. **The paper question,** with the cross-lab check of the bands on Haley 2025's real tracks first. If another lab's real worms fail Scheer's bands, the bands are rig-specific. Haley's patch-acceptance family is the benchmark's candidate extension.
2. **What a controller needs to predict a mutant,** since wild-type fitting does not identify the responses. The candidates:
   - neural constraints, where the one positive wiring reading sits ([134](134-fit-first-neural-readings.md), [135](135-neural-readout-ceiling.md));
   - a controller fitted to neural data, Creamer's dynamics;
   - mechanism ablations on C.1-class models, to say which statistic needs which mechanism.
3. **W,** once a fitting procedure passes its own recovery control.
4. **C.3's question,** the C.2 architecture trained by reward.
5. **The speed emission,** before any likelihood route.
6. **The literature watch,** re-aimed at the review's open questions.

The real tracks were deposited on Zenodo on 2026-11-27, [doi:10.5281/zenodo.24585652](https://doi.org/10.5281/zenodo.24585652). The `wormbench` release on PyPI, with WCON export, follows this close.

## Artefacts

- Logbooks [104](104-batched-evaluator.md) to [137](137-fidelity-table.md), and their supporting directories.
- [supporting/138-phase10-synthesis/probes/](supporting/138-phase10-synthesis/probes/):
  - `identifiability.py.txt` and `identifiability.json`: C.2's ensemble and lineage, read from committed records;
  - `likelihood_check.py.txt` and `likelihood_check.json`: the likelihood check on band cohorts.
- The archived tracker: `openspec/changes/archive/2026-11-27-phase10-tracking/`.
