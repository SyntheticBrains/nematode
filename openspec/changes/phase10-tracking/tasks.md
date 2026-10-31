# Tasks: Phase 10 (Fidelity on the Plate — Standard Lawn Assays, Graded Against the Worm) Shipment Tracker

This is the living checklist for Phase 10. The plan is authoritative in `docs/roadmap.md` § Phase 10
(v5.0, 2026-10-23), from the plan review of 2026-10-23 and its adversarial critique: the goal, what the
review established, the deliverables, design decisions D33–D41, exit criteria, risk table, novelty map
and go/no-go. Every Phase 10 milestone PR updates this checklist as part of its diff.

**Status legend**: `[ ]` not started; `[x]` closed (done, or given a status at the synthesis).

**Execution order** (binding where the roadmap says so):

01. H.
02. E + T.
03. B, with N in parallel.
04. B valid.
05. F.0.
06. C.1.
07. C.2.
08. C.3.
09. C.4.
10. W.
11. The synthesis.

Binding: nothing is fitted before B reads valid; nothing is fitted on real data before F.0 clears;
nothing in W registers before C.2 has read. *(Amended 2026-10-25, the maintainer's decision: F.0
clears per model. C.1 may be fitted once F.0b-i passes, and C.2 waits for F.0b-ii.)*

## H — housekeeping (MUST, first)

- [x] **H.1 The literature watch re-aimed** at Phase 10's questions (`context.md`, `seeds.toml`; lands with
  this tracker's PR). The item closes when the first digest has run against the new brief.

  *(2026-10-24: the brief and seeds were re-aimed in PR #470. The last digest, on 2026-10-19, ran
  before it, so this waits for the scheduled run of Monday 2026-10-26. Two topics were added to the
  brief with H.3 and H.4: locomotor kinematics on food against off food, raised by B.1's kinematics
  check, and differences between the N2 and PD1074 wild-type strains, since tph-1's batch-matched
  control is N2.)*

  *(Closed 2026-10-28: digest #498, the first run against the new brief, ran 2026-10-26. It raised Yad &
  Nabila 2026's reconstruction-sensitivity result, acted on in W.1 and W.2 below. B v2 adds one topic:
  public chemotaxis track datasets.)*

- [x] **H.2 The research question re-framed** (D33) in AGENTS.md, the README and the roadmap (lands with
  this tracker's PR; closes on merge). *(Closed 2026-10-23: PR #470 merged.)*

- [x] **H.3 Scenario scope recorded**: thermal, predator, multi-agent and oxygen configs frozen for the
  required blocks, kept and tested; thermotaxis named as the second tier; the MAY items below stay
  allowed. It closes when the docs say so in the scenario index.

  *(Closed 2026-10-24: `configs/README.md` § Scope in Phase 10. Where Phase 10's configs live, what is
  frozen, the second tier and the MAY items. The two non-spatial memory families, which D41 does not
  name, are recorded as kept and tested, not extended.)*

- [x] **H.4 The seed-band table** (design.md), cited by every registration.

  *(Closed 2026-10-24. The table has been in design.md since PR #470. One amendment, the maintainer's
  decision: W's band moves from 6001–6256 to 7201–7456. Logbook 104's calibration ran simulated
  animals on 6001–7132, in twelve blocks of 32, which are now recorded as used. Seeds that belong to a
  record, such as a split's or a bootstrap's, are stated not to be bands.)*

  *H ran after E + T and B.1, not first.* No registration was harmed: E + T's parity and pilot used their own band, E's smokes used the smoke band, its unit tests drew environment seeds for placement checks only, and B.1 drew no simulated seed (its split seed
  belongs to its record). B.3 is the first registration to cite a band, and H closes before it.

## E + T — the worm, its assays and the batched evaluator (MUST)

- [x] **E.1 One step-duration value** (D35): every step-coupled constant derived from it. These include:

  - modulator decay, body step, the classifier's window and step;
  - the STAM length, the adaptive-sensor α, `membrane_tau_steps`;
  - satiety decay, intake and regrowth;
  - `max_turn_rad`, `max_steps`, and PPO γ.

  The 5-s default stays byte-identical, and a test asserts the derivation. A 1-s point worm configured.

  *(Closed 2026-10-23, change `add-step-duration`, archived as `2026-10-23-add-step-duration`.)*

  - A top-level `step_seconds` is added. Configs stay written for their 5-s reference step, and the
    loader converts every step-coupled constant. Unset, nothing is derived.
  - The positive control passes: distance, heading, satiety, a grazed cell, the STAM weights and the
    adaptive baseline match at 5 s and 1 s. An unconverted 1-s worm covers five times the ground.
  - Generated configs `*_lawns_modulated_s1.yml` and `*_lawns_internal_s1.yml`.
  - The classifier reads 1-s captures at 5 s, and the bias-curve harness refuses them.

  **Dated amendments (2026-10-23):**

  - "Body step" moves to Phase 11 (D41). The kinematic body is refused at a non-default step: its
    reversal and refractory counts have no rule.
  - The training budget is not converted: rollout buffer, BPTT chunk, epochs, learning rate,
    entropy. C.3's registration sets it.
  - `membrane_tau_steps` is converted only under leaky dynamics.

  **Smoke note for T.1**, three untrained episodes per arm (base seed 5991):

  - A worm-hour is 3,600 steps at 1 s against 720 at 5 s.
  - The untrained 1-s worms starved later in worm time than the 5-s ones, and ate more. A policy
    taking five times the decisions per worm-second, with the same per-decision noise, diffuses
    less. This is learning-relevant, not a conversion fault.

- [x] **E.2 Locomotor primitives** (D36): run, reversal bout, omega/delta reorientation and pause, with
  kinematics fixed from data (Broekmans 2016; Gray 2005) and timing chosen by the controller. The
  lawn-edge question is decided in its spec.

  *(Closed 2026-10-24, change `add-locomotor-primitives`, archived as
  `2026-10-24-add-locomotor-primitives`.)*

  - `continuous.locomotion: primitives` and `action_space: primitives`, for the connectome and
    MLP-PPO brains.
  - Five numbers per second drive them: run speed, run curvature, and start hazards for reversal,
    omega and delta.
  - Kinematics are in `data/locomotion/primitives.json`.
  - The **lawn-edge question** is answered with a food-contact sense, the lawn density under the worm.
    A `locomotor_state` sense is added beside it.
  - Two non-learning controllers: `ConstantHazards`, B.4's Poisson negative, and `PirouetteHazards`.
  - **Positive control.** Executed bouts match their sources, read from trajectories. The pirouette
    rule reaches a lawn 6 mm away more often than constant hazards at its mean rate, and the inverted
    rule less often.
  - Configs: `*_lawns_modulated_s1_primitives.yml` and `*_lawns_internal_s1_primitives.yml`.
  - D36 amended (dated): a pause is a run at zero speed; delta turns are deep dorsal coils; the
    sources are off food.
  - **Note for C.3:** the untrained policies' sampled hazards average about 0.1/s per kind, so untrained
    worms spend most of their time in bouts.

- [x] **E.3 Assay protocols**: Scheer's small lawn (40 min); food removal onto a bare plate; an off-food
  start; multi-hour episodes with satiety re-calibrated.

  *(Closed 2026-10-24, change `add-assay-protocols`, archived as `2026-10-24-add-assay-protocols`.)*

  - `environment.assay` is a list of phases. Each sets its food (lawns or a bare plate) and the worm's
    placement (keep, the lawn's centre, bare agar, or the arena centre as a transfer). Satiety and the
    controller's state carry across phases.
  - New options: a circular arena, a centred lawn, and `starvation_ends_episode`.
  - Eight generated configs on the 1-s primitives worm: Scheer, López-Cruz, Calhoun and Flavell, each
    for the MLP and the connectome.
  - Captures record each step's phase, and the file records the protocol's timing.

  **Dated corrections (2026-10-24):**

  - Scheer's worms start on bare agar and find the lawn; the "off-food start" is Scheer's.
  - López-Cruz 2019 reports no exponential fit (see B.2).
  - Hunger runs on hours, the food memory on minutes: satiety is a 12-h reserve that never ends an
    assay, and the memory is the controller's.
  - The assay lawn halves under a still worm in about 2 h, against D.1's 3 min. This is the
    maintainer's choice, the first parameter to revisit if dwelling or leaving statistics fail every
    controller.

- [x] **T.1 The batched evaluator**, about 8–12 engineering days: many animals per call, no per-step
  Python, carrying E.2's primitives and E.3's assays. A parity test against the per-step runner, and
  a cost pilot (seeds 5001–5032) that sizes W.

  *(Closed 2026-10-24, change `add-batched-evaluator`, archived as `2026-10-24-add-batched-evaluator`;
  [Logbook 104](../../../docs/experiments/logbooks/104-batched-evaluator.md).)*

  - `quantumnematode/evaluation/`: `BatchedEvaluator` and four controllers, with per-animal streams.

  - **Parity** in three layers:

    - state within 1e-5 over 50 steps;
    - bouts exact over 300 steps, positions within 1e-3 over 150;
    - full-assay statistics with no detected bias, Bonferroni-corrected over the twenty checks after
      the per-check 95% rule false-alarmed on CI (6 of 12 null bands failed it, none fail the
      corrected rule).

    Two deliberate faults were caught.

  - **Cost pilot**, per core at 256 animals, on seeds 5001–5032:

    - MLP: 2.6–3.8 µs per animal-step (77–98×);
    - connectome: 6.1–7.3 µs (48–52×);
    - hazard rules: 1.6–2.8 µs.

    One 200-animal evaluation over all four assays, timed at N = 200, takes 13 s on the MLP and 27 s
    on the connectome.

  - **W's sizing inputs:** panel cost = (1 + null arms × draws) × population × generations × 27 s. At
    20 × 200, 5 arms and 8–16 draws, that is 1,210–2,390 core-hours. Default threads speed the
    connectome by about 15% on up to 18 threads, so the fits parallelise by process.

## B — the benchmark (MUST; the gate for everything after)

- [x] **B.1 Reference re-extraction**: Scheer & Bargmann 2023 from the Zenodo mirror (8310289), with
  positions and cohort IDs, through the restricted unpickler. Calhoun 2014 and Roberts 2016 tracks;
  López-Cruz 2019 targets. Each with its licence, unit convention and provenance.

  *(Added 2026-10-24 by E.2.)* Check the primitives' off-food kinematics (`data/locomotion/primitives.json`)
  against Scheer's on-food movement flags (reversal, pause, omega, with the lawn-boundary distance), on
  cohorts held out of the benchmark. A disagreement beyond the sources' spread is a dated amendment
  before any fit.

  *(Closed 2026-10-24, change `add-reference-extraction`, archived as
  `2026-10-24-add-reference-extraction`;
  [Logbook 105](../../../docs/experiments/logbooks/105-reference-extraction.md).)*

  - `data/benchmark/`, read by `quantumnematode.benchmark.references`:

    - Scheer: 2,847 animals in eight arms at 1 s. N2 was added as tph-1's batch-matched control, since
      tph-1 was recorded only beside N2;
    - Calhoun (pixels: no scale is published) and Roberts search tracks;
    - López-Cruz's reorientation bins. pdfr-1 is present; tax-4 and tph-1 are not.

  - The store reproduces D.1's windows: decoded states agree on 99.96% of windows.

  - **The split** (`scheer2023_split.json`): recording dates, a third held out, from counts alone. The
    OD 2 wild type has 24 of 65 dates held out, tph-1 2 of 5; OD 1 has one date and none. Held out of
    the bands, not unseen by the classifier, which D.1 calibrated across every date.

  - **Kinematics check, a disagreement:** on-lawn reversal bouts last 2.05 s [2.01, 2.08], below the
    published off-food span 2.10–3.71 s; off the lawn, 3.30 s, inside it. `primitives.json` is
    unchanged.

  - **Owed before B.3 registers:** the maintainer's choice of the D36 amendment's form, either
    food-dependent kinematics keyed on food contact, or the off-food values kept with the on-food gap
    stated.

    *(Closed 2026-10-24: the maintainer chose food-dependent kinematics, as change
    `amend-food-dependent-kinematics`, archived as `2026-10-24-amend-food-dependent-kinematics`;
    [Logbook 108](../../../docs/experiments/logbooks/108-food-dependent-kinematics.md).)* A reversal
    that starts on a lawn draws 2.047 ± 0.899 s at 0.069 mm/s, measured on the held-out cohorts; one off
    every lawn keeps the published values. `continuous.reversal_kinematics: off_food` reproduces every
    earlier record. The pilot (seeds 9001–9064) is description.

- [x] **B.2 Readouts**: a trajectory-based reorientation detector, validated against Calhoun's and
  Roberts's omega flags; the classifier run on tracks downsampled to 5 s; the local-search curve; the
  repaired pirouette quintile ratio (the old K.1).

  *(Added 2026-10-24 by E.3.)* López-Cruz 2019 has no αe^(−γt)+β fit. On its deposited CC BY data
  (Mendeley 10.17632/prdtw5jgcd.1), the wild type gives γ = 0.115 ± 0.011/min only from the 4–6 min
  bin: 0.089 from 2 min and 0.137 from 6 min. The curve rises before it decays. B registers:

  - the readout: reorientations that begin with a reversal, per 2-min bin;
  - the fit's start bin;
  - the time zero, after checking the recording's 4-min offset from removal against the bins;
  - γ, from the deposited data.

  *(Closed 2026-10-24, change `add-benchmark-readouts`, archived as
  `2026-10-24-add-benchmark-readouts`;
  [Logbook 107](../../../docs/experiments/logbooks/107-benchmark-readouts.md).)*

  - **The reversal-onset detector** reads 1-s positions alone and was tuned on Roberts.
    - The registered preflight stopped the first run before any held-out data were read (in-sample
      0.62/0.68 against 0.70).
    - Retried at bars set from the instrument (0.55/0.60), registered before running, it passes
      Roberts held-out (0.66/0.72), simulated primitives (0.87/0.92), Calhoun (0.53, ρ 0.67) and the
      dead worms.
    - It fails Scheer on food (0.43/0.45), so the maintainer **scoped it to off-food readouts**. B.3
      grades on-lawn behaviour through the classifier and lawn statistics, never on-lawn reversal
      counts.
  - **López-Cruz:** the bins are recording time (removal + 4 min). The fit runs from the wild type's
    peak bin over 8–44 min after removal: γ = 0.114 [0.101, 0.127] per min
    (`data/benchmark/lopezcruz2019_reference.json`).
  - **The quintile ratio's reference**, digitised from Pierce-Shimomura 1999's Figure 8, is 2.26–2.58.
    Lee 2026's 9.5 is cited beside it, unverified.
  - **The classifier** reads 1-s captures through its 5-s path unchanged.

- [x] **B.3 Bands and the grading rule** (D34): from batch-matched cohorts with the project's
  classifier; mutant effects relative to the controller's own wild type; composite rules that depend
  on closed-loop contingencies.

  *(Closed 2026-10-24, change `add-benchmark-grading`, archived as
  `2026-10-24-add-benchmark-grading`;
  [Logbook 109](../../../docs/experiments/logbooks/109-benchmark-grading.md).)*

  - **The rule.** 11 wild-type statistics:

    - Scheer's small lawn: roaming, both bout means, exits per on-lawn hour, pre-exit enrichment,
      time on the lawn;
    - Scheer's uniform well (new protocol `uniform_lawn_scheer`): roaming and both bout means;
    - López-Cruz: γ and early/late.

    Mutant Δs against their own wild type (tph-1/N2, pdfr-1 and tax-4/PD1074). A statistic passes
    at |z| ≤ 2.84 (Bonferroni over 11). D and D_fit are weighted RMS z's
    (`quantumnematode.benchmark.grading`, `data/benchmark/bands.json`).

  - **The classifier**, refitted on band dates, passes its gate at κ = 0.625.

  - **The rule's own check failed its first attempt** (0.80/0.86 against 0.90): the bands were too
    narrow. The maintainer chose a registered retry (centre uncertainty, a calibration factor
    κ = 1.19–1.30, a t prediction interval), committed before it ran. It passes at 0.925 and 0.962.

  - **Preflight** (seeds 9101–9164): ConstantHazards and PirouetteHazards fail every family (D 6.6
    and 7.2). Four single statistics pass for the Poisson worm, so the family rule and the distance
    carry the discrimination.

  - **Only two mutant directions are resolved:** tph-1 roams more, and tax-4 roams less. Leaving and
    search differences are graded by magnitude alone.

  - **Exits read 0.71 per on-lawn hour** under the geometric window rule (D.1's 0.21 came from the
    authors' out-of-lawn runs).

  *(Amended 2026-10-24, change `harden-benchmark-grading`.)* Before B.4:

  - mutant directions by a paired t-test across dates;
  - bands at any size, one generator per band, graded per family (scales moved −2.9% to +4.0%);
  - γ undefined without a decay;
  - pdfr-1's local search reported, not graded (pdfr-1's bar 2.24);
  - a registered decay-presence test for local search: a flat population passed the family 12–13%
    of the time; every band batch is present.

  Logbook 109's addendum has every changed value.

- [x] **B.4 The controls**:

  - positive: cohort-held-out real tracks pass, and held-out real mutant cohorts pass the mutant
    grading;
  - matched negatives (the open-loop Scheer HMM; Poisson-timed primitives) fail;
  - trivial negatives (a random walk, a correlated random walk, the untrained policy) fail.

  **The benchmark is void unless all of these behave.**

  *(Added 2026-10-24 by B.3's hardening, change `harden-benchmark-grading`.)* B.4's registration
  must fix:

  - **A rule for multiplicity across its controls.** About six positive controls, each graded at a
    95% family-wise rate, would void the benchmark by chance about a quarter of the time
    (1 − 0.95⁶). For example, Bonferroni across the controls, or a registered count of allowed
    failures.
  - **The thin controls, stated:**
    - the uniform well has one held-out date (31 animals);
    - tph-1 has two held-out paired dates;
    - local search has ten held-out batches, graded one by one (the rule's check passed 0.962 per
      batch);
    - pdfr-1's local search has no held-out cohort and is reported only.
  - **Grading at the controls' own sizes,** now built (`benchmark_grade.py`, bands on demand).
  - **Each statistic's discrimination, as description.** Four single statistics passed for the
    preflight's Poisson worm, so the family rule and D carry the discrimination. Local search must
    also be present: early/late above 1 at its own lower bound.

  *(Closed 2026-10-25, change `add-benchmark-controls`, archived as `2026-10-25-add-benchmark-controls`;
  [Logbook 110](../../../docs/experiments/logbooks/110-benchmark-controls.md).)* **The benchmark is
  VALID.**

  - **Positive controls:** 14 of 15 held-out units pass, against an allowance of 2. The wild type
    passes on the small lawn (573 animals, largest |z| 1.29), on the uniform well (31) and in all 10
    López-Cruz batches. tph-1 fails on leaving: held-out Δ exits +4.0/h, against its three band
    dates' +0.07.
  - **Negatives:** all six fail the wild type, with D 4.49–8.02 (open-loop HMM, Poisson, random and
    correlated walks, untrained MLP and connectome). Staying on the small lawn separates every one.
  - **W's minimum: 1.42 on D** (95th percentile of real half-splits; median 0.89).
  - **Carried forward:**
    - C's mutant table: tph-1's leaving reference is unreliable;
    - local search alone is weak: the untrained connectome passes that family, and D_fit leaves it
      out;
    - C.2's weights: `time_on_lawn` and exits carry the discrimination.
  - **Seeds used:**
    - 5101–5300 (animals) and 5301–5316 (untrained networks);
    - pilot 9201–9280.

- [x] **B.5 Track import** (WCON or CSV), independent of the simulator, positioned against Tierpsy and
  OWMD feature sets. NeuroSparkSNT and a wider novelty search checked at registration.

  *(2026-10-24: the novelty check is done, in B.3's registration ([Logbook 109](../../../docs/experiments/logbooks/109-benchmark-grading.md)):
  NeuroSparkSNT, PGOB and Morrison & Young 2025 do not occupy the claim. Track import remains.)*

  *(Closed 2026-10-26: change `add-track-import`;
  [Logbook 114](../../../docs/experiments/logbooks/114-track-import.md).)*

  - **What it reads:** `quantumnematode.benchmark.track_import` reads CSV (with a sidecar) and
    WCON 1.0's core, resamples to 1 s, and marks lawn membership by a column or Scheer's boundary
    rule. `benchmark_grade.py tracks` grades the result.
  - **The round trip is exact in both formats:**
    - 1,013 small-lawn and 69 uniform-well band-cohort Scheer animals, at tallies and statistics;
    - 78 Roberts calibration-cohort worms, at the detector.
  - **Independence:** grading calls no simulator code. *(2026-10-26, change
    `refactor-benchmark-light-imports`:)* importing the grader also loads none of the simulator and
    no PyTorch, so the benchmark works without the `torch` extra.
  - **Positioning** against Tierpsy and OWMD is in `data/benchmark/IMPORT.md`. **The B block is
    complete.**

- [x] **B v2 The benchmark revised once** (change `revise-benchmark-v2`, with
  `benchmark-v2-empirical-noise`), after C.1's v1 pass relied on ungraded routes.

  *(Closed 2026-10-28: [Logbook 115](../../../docs/experiments/logbooks/115-benchmark-v2.md).)*
  **B v2 is VALID**, and `benchmark_grade.py` defaults to it.

  - **Added:**
    - **a measurement model** for simulated tracks. On Scheer's rig it resamples real paused-worm
      noise and draws losses as runs; on Chen's it adds jitter and cuts fragments. Local search
      stays exact;
    - **a kinematics family** (window speeds and roaming-bout survival);
    - **a chemotaxis family** from Chen 2025, with presence required;
    - **19 statistics,** z\* 3.0078.
  - **Every registered check holds:** the measurement checks, reachability on the four speeds,
    fresh N2 (0 of 4), the chemotaxis leave-one-out (0 of 6), B.4's negatives, the blind worm (fails
    5 of 5) and C.1's v1 answer (fails 5 of 5).
  - **Four registered stops before Part B** changed the denoiser test, the noise model, the
    reversal floor and the reachability worm, each in a launch amendment.
  - **Declared non-goals:**
    - not graded: the on-lawn reversal rate (reported; no open-loop point worm reaches it), the
      omega rate, and local search's rise;
    - not modelled: the tracker's noise while the worm moves, and López-Cruz's tracker.
  - **Next:** the F.0b-i re-run on v2, then C.1b on v2, each its own change.

## N — the neuromodulator layer from data (MUST)

- [x] **N.1 Licences confirmed**: Bentley 2016 (CC BY, S1 Dataset) and Ripoll-Sánchez 2023 (unstated on
  the toolbox and repository pages). Vendored only if confirmed; otherwise what stands in is stated.

  *(Closed 2026-10-24, change `add-neuromodulator-layer`;
  [Logbook 106](../../../docs/experiments/logbooks/106-neuromodulator-layer.md).)* Bentley's S1 Dataset
  and Ripoll-Sánchez's figshare items are CC BY 4.0. The 2024 corrections (dmsr-5, npr-34) and the
  per-neuron expression table are MIT, on GitHub only, and are not used. They are not needed: every
  long-range network is exactly its producer set times its receptor set.

- [x] **N.2 The layer built**: receptor expression and producers per neuron, replacing the hand-placed
  four-pathway circuit behind a byte-identical default, with the diffusion-range model chosen in its
  spec.

  *(Closed 2026-10-24.)* `modulator_layer: data`, under the long-range model (one pool per ligand):

  - 55 pools and 121 couples, the union of both sources under WormBase names, every cell tagged;
  - 16 fixed monoamine signs and 105 fitted strengths;
  - gene knockouts, a potency sensitivity and one-source sensitivities, and batched support;
  - the restricted layer reproduces the hand-placed field, and τ = 540 s holds for nine minutes.

  **The two sources each support half of the hand-placed circuit:** PDF-1's producers from Bentley,
  PDFR-1's sites (and NSM) from Ripoll-Sánchez. MOD-1 on ASI and RIF is in neither.

- [x] **N.3 Its null family**: receptor-placement nulls, with a D24-style matched-statistics table.

  *(Closed 2026-10-24.)* Receptor cells move by class to same-size classes on the same member positions,
  keeping every gene's cell and class counts. Audit over 7501–7564:

  - **Reach is saturated:** sensory-produced couples reach all 39 motor neurons in the data and in every
    draw, so the null cannot change sensor-to-motor reach.
  - **Ventral-cord classes of unique size pin their receptors:** npr-34 59%, seb-3 and gnrr-1 53%.

  W reads both beside its result.

## F.0 — the fitting methods' positive control (MUST)

- [ ] **F.0** parameter recovery at known parameters, by sep-CMA on C.1's model and C.2's encoder
  (seeds 5401–5464). *(Amended 2026-10-25: C.1's half was first NPE's; D38 as amended.)* It reports recovery and spread, and the spread sizes W.

  *(2026-10-25, F.0a: change `add-npe-recovery`, archived as `2026-10-25-add-npe-recovery`;
  [Logbook 111](../../../docs/experiments/logbooks/111-npe-recovery.md).)* **F.0a fails as
  registered.**

  - C.1's 20-parameter mechanistic baseline (`BatchedMechanistic`) and NPE (`sbi`) are built.
  - Both flows are calibrated (coverage 0.89) but weakly informative: posterior-predictive D 5.3
    against the bar of 1.42.
  - The diagnosis, read after the gates: two simulations of the same parameters differ by a median
    D of 4.08, so the gate was unreachable. Local-search and bout statistics flip onto the z cap for
    unrealistic prior worms.
  - **The maintainer's decisions:**
    - F.0a-ii: truncated sequential NPE, truths from the region C.1 is fitted in, and a gate set
      against each truth's measured floor, preflighted;
    - F.0b's statistic-space gate is also floor-relative and preflighted.
  - *(2026-10-25, superseded the same day.)* **F.0a-ii was withdrawn before its spec review**, and
    **C.1 is fitted by sep-CMA** (the maintainer's decision; D38 as amended). Planning reads, committed
    as `scripts/analysis/npe_region_check.py` and its `region_check.json` (Logbook 111, addendum),
    showed the planned gate out of reach for the method rather than the oracle: near the wild type a
    nearest-neighbour posterior gains far less than the bar, and a first truncated round keeps most of
    the prior box.
    **F.0's remaining control is sep-CMA's, on C.1's model and C.2's encoder**, with floor-relative,
    preflighted statistic-space gates.
  - *(2026-10-26, F.0b-i: change `add-cma-recovery`;
    [Logbook 112](../../../docs/experiments/logbooks/112-cma-recovery.md).)* **F.0b-i passes, and F.0
    clears for C.1.**
    - Four near-wild-type truths were each fitted twice by sep-CMA from the box centre. The median
      gain of each truth's best restart is 0.991, against 0.75, and every truth is at least 0.81.
    - The oracle scored 0.996 and the unfitted start 0.032. The pilot plateaued at 140, which set the
      cap at 210, and all eight fits plateaued before it, in 10.9 h.
    - Only weathervane steering and the search time are recovered as parameters.
    - **For W:** two restarts on the same data differ by a median D of 1.12 (0.48–1.92), close to
      W's minimum of 1.42, so W's arms need several fits each.
    - **For C.1:** keep at least two restarts. The registered plateau rule carries over.
    - **Still open:** F.0b-ii, C.2's encoder and its recovery.

  *(2026-10-29: change `rerun-cma-recovery-v2`; [Logbook 116](../../../docs/experiments/logbooks/116-cma-recovery-v2.md).)*
  **F.0b-i on v2 fails its gate**, with a median gain of 0.690 against 0.75. sep-CMA recovers the two
  truths with stable statistics (0.914, 1.007) but not the two noisy ones far from the wild type
  (0.466, 0.056).

  - **The cause is the fit's objective.** The capped distance is flat beyond the cap, so the first
    pilot stopped at 0.511. Uncapped (amendment 1), an undefined statistic's flat 10 undercut far
    defined ones, so those fits escaped into undefined families.
  - **Next:** a new recovery change. It ranks on a slowly growing uncapped z (for example
    10·asinh(|z|/10)), makes an undefined statistic cost more than any defined one, is checked on
    the two failed truths with discarded seeds, and re-runs. C.1b waits for it.
  - **The 1.76× speed-up of a v2 simulation** (identical output) carries to every later fit.

  *(2026-10-31: change `rerun-cma-recovery-v2b`; [Logbook 117](../../../docs/experiments/logbooks/117-cma-recovery-v2b.md).)*
  **F.0b-i on v2 passes on its second attempt**, with a median gain of 0.873 against 0.75, and F.0
  clears for C.1 on v2.

  - **The fix:** fits rank on 1,000 × (statistics undefined in one record) + the weighted RMS of
    10·asinh(z/10), and every gain stays on the capped D. Attempt 1's truths and observations are
    reused, the latter pinned by SHA-256.
  - **The probe** (discarded seeds) fixed the escape (6985 0.056 → 0.981), but 1164 reached 0.787,
    under its 0.80 bar. The maintainer accepted it and raised the restarts to 4 per truth.
  - **The fits:** best restarts 0.773, 0.968, 0.778 and 1.040. Selection by final objective picked
    the highest-gain restart for every truth. With 2 restarts alone the median would have been 0.822.
    9 of 16 restarts reach 0.75. Parameters are not recovered (median errors 0.13–0.23).
  - **For C.1b**, listed under C.1b below.

  *(Added 2026-10-28, B v2.)* **F.0b-i is re-run in full on v2**, in its own change. F.0b-i was read
  on v1. Its recovery has to hold under the benchmark C.1b and C.2 are fitted against: 19 statistics,
  five families and the measurement model. C.1b follows it.

  - **C.1's note:** its real fit must simulate at the observation's size, or add between-date
    noise matched to the bands.
  - **Seeds:**
    - animals for simulation k at `1_000_000 + 200·k + i` (pilot `2_000_000`, floor `3_000_000`);
    - prior draws 5401, flows 5402–5403, prior-predictive draws 5404.

## C — controllers and the fidelity table (MUST)

- [x] **C.1 The mechanistic baseline**: a stochastic run/reversal switch, a roaming/dwelling state with
  intake and satiety dependence, pirouettes and weathervane, ≤ 30 parameters. Fitted by NPE, with
  priors set without search data; posterior-predictive checks.

  *(Amended 2026-10-25, D38 as amended.)* Fitted by sep-CMA on the benchmark's distance, like C.2,
  with its search box and x₀ set without search data. Its predictive checks re-simulate the
  restarts' fits.

  *(Closed 2026-10-27: change `add-c1-fit`;
  [Logbook 113](../../../docs/experiments/logbooks/113-c1-fit.md).)* **C.1 passes the wild type** at
  the registered threshold.

  - The chosen restart (3) passes 3 of its 5 registered populations, at D 0.95–1.26 from the bands.
  - Its weak statistic is the small lawn's roaming-bout length (z 3.55 and 4.68 where it fails). The
    uniform well's runs short, so one roaming state is stretched between the two lawns.
  - Restarts 1 and 5 pass 5 of 5; two restarts stopped early on a false plateau.
  - Weathervane steering is strong and toward food (κ_w −47.3 °/s), and the search time is 8.0 min.
  - **The observation-size note is resolved:** the bands' n = 200 scales already carry the
    between-date spread.

  *(Added 2026-10-28, B v2.)* **C.1's v1 answer fails B v2** on all five populations (D 6.1–6.2),
  failing the small lawn, the uniform well, kinematics and chemotaxis. Its v1 pass relied on
  routes that v2 grades, so C.1 is refitted on v2 as C.1b, after the F.0b-i re-run on v2.

  *(Added 2026-10-31, F.0b-i on v2; [Logbook 117](../../../docs/experiments/logbooks/117-cma-recovery-v2b.md).)*
  **C.1b may now run**, as its own change. Its plan carries five points from the recovery:

  - **the objective:** fit with the ranking score (definedness first, then 10·asinh(z/10)), and
    grade on the capped D;
  - **the plateau guard:** two of 16 recovery restarts stopped at generations 47 and 52 while their
    window still held medians with an undefined statistic, since a rise in S reads as a plateau. The
    rule must not stop a restart until its window has none;
  - **presence:** the ranking score leaves it out, and the wild-type pass needs it, so restarts are
    chosen on the registered grade with presence;
  - **reachability first:** a short pilot fit on the real bands, on discarded seeds, before
    registration. Both of 1164's probe misses were on the kinematics speed medians;
  - **the restarts:** keep 8. Per restart, 9 of 16 recovered, and 6985's first two reached at best 0.360.

- [ ] **C.2 The connectome controller**: on the primitives, with the data-derived neuromodulator layer.
  Its slow variable is stated and its feasibility for a 9-minute decay checked. It is fitted by
  sep-CMA on the benchmark's distance at ≥ 200 animals per evaluation, with the budget pre-registered
  and a plateau criterion.

  *(Added 2026-10-27, C.1.)* **The plateau rule needs a guard.** Two of C.1's restarts stopped at
  generations 40 and 43, still at D 5–6. C.2's registration adds a minimum generation or a σ
  condition, so a stuck restart is not counted as converged.

  *(Added 2026-10-24 by B.3's hardening.)* `time_on_lawn`'s band scale is 0.006 at n = 200, so a
  controller that leaves the lawn a little reaches the z cap of 10 there, and an unweighted D_fit is
  dominated by it. C.2's registration states its weights and cap with that in view.

- [ ] **C.3 The comparator**: the C.2 architecture trained by reward on the same primitives and lawns,
  16 seeds, γ adjusted to the 1-s step.

- [ ] **C.4 The fidelity table**: C.1–C.3 and the negatives on held-out assays and the mutant set
  (tph-1 on Scheer only, pdfr-1, tax-4), under a per-controller encoding table registered before any
  fit, and with C.2 against C.1 as a secondary contrast.

  *(Added 2026-10-24 by B.3's hardening.)* Only tph-1's (more) and tax-4's (less) roaming directions
  are resolved by the references; every other mutant difference is graded by magnitude. The
  dissociation claim is narrowed accordingly in the roadmap.

  *(Added 2026-10-27, change `add-c1-fit`.)* **The encoding table is registered**:
  `docs/experiments/logbooks/supporting/113-c1-fit/encodings.md`, committed before any fit.

  - **C.1's rows** are structural knockouts: tph-1 removes the intake dependence; pdfr-1 is not
    encodable; tax-4 removes the chemosensory gains.
  - **C.2's and C.3's rows** are gene and neuron knockouts.
  - **tph-1's leaving** is left out and reported beside.
  - **The `unidentified` rule** applies across the restarts that pass the wild type. For C.1 these
    are restarts 1, 3 and 5.

## W — the wiring question on fidelity (MUST, after C.2 reads)

- [ ] **W.1** The nulls: degree-preserving; gap-only on the corrected gap model; boundary-held;
  class-preserving left–right-symmetric; and N.3's receptor-placement nulls. Each has a fresh rewiring
  draw per fit, with parameters counted per arm.

  *(Added 2026-10-28, B v2, from digest #498: Yad & Nabila 2026.)* **Two robustness checks.**

  - **The reconstruction:** the wild type is also fitted on Witvliet 2021's adult dataset 8 beside
    Cook 2019, since connectome conclusions can depend on the reconstruction.
  - **The weights:** the fitted C.2 is re-graded under 1% weight noise.

- [ ] **W.2** The registration: the minimum is the held-out distance between two real wild-type cohorts,
  both directions. Draws per arm are sized from F.0 and the cost pilot. Every outcome is named,
  including "the nulls pass too" and "a null beats the wild type".

  *(Added 2026-10-27, C.1.)* Restart-to-restart spread is a median D of 1.09 on real data (six
  converged restarts) and 1.12 on F.0b-i's synthetic truths, against W's minimum of 1.42. A single fit
  per arm would read fitting noise of about the minimum, so W's arms need several fits each, sized
  from C.2's own spread.

  *(Added 2026-10-28, B v2.)* **W's minimum is recorded on both family sets.**

  - **The fitted families** (small lawn and kinematics): 2.97, between halves of N2's unread dates.
  - **The held-out families** (local search and chemotaxis): 2.02.

  W's registration reads its arms on the same family set as the minimum it cites, and carries the
  reconstruction and weight-noise checks above.

  *(Added 2026-10-31; [Logbook 118](../../../docs/experiments/logbooks/118-v2-speedup.md).)*
  **The compute plan.** T.1 sized W at 1,210–2,390 core-hours at v1's 27 s per connectome
  evaluation. On this machine 16 workers deliver about 6 cores' worth of work, so that is about
  200–400 h, too long to run as sized. The levers, each keeping every output identical unless
  stated:

  - **done:** the High Power energy mode (1.35×) and the v2 speed-up of Logbook 118 (1.54× on one
    core, 1.35× on 16 workers);
  - **before C.2's cost pilot:** a whole generation (16 candidates × 200 animals) in one evaluator
    call, with a GPU check at that size, since the earlier MPS check found its gain only at
    4,000 or more animals per call;
  - **in W's registration:** a registered sequential stopping rule for the null draws, and W's
    cost re-sized from v2 timings rather than v1's;
  - **dropped:** surrogate-assisted CMA. The objective is noisy, and it would need its own
    recovery control.

- [ ] **W.3** The panel, read on graded distance whatever any gate says.

## The synthesis (MUST)

- [ ] **S10** Every criterion assigned one of the five statuses. The publication decision taken (D30's
  destination: the first fidelity result).

## SHOULD

- [ ] Haley 2025's patch acceptance against encounter number.
- [ ] WCON export.

## MAY

- [ ] Thermotaxis on the new primitives (the second tier).
- [ ] Bordering against an O₂ field, with npr-1 (an optional second-tier comparison; npr-1 is excluded
  only as a required mutant test).
- [ ] A Creamer-dynamics controller graded on the benchmark.
