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

- [ ] **H.1 The literature watch re-aimed** at Phase 10's questions (`context.md`, `seeds.toml`; lands with
  this tracker's PR). The item closes when the first digest has run against the new brief.

  *(2026-10-24: the brief and seeds were re-aimed in PR #470. The last digest, on 2026-10-19, ran
  before it, so this waits for the scheduled run of Monday 2026-10-26. Two topics were added to the
  brief with H.3 and H.4: locomotor kinematics on food against off food, raised by B.1's kinematics
  check, and differences between the N2 and PD1074 wild-type strains, since tph-1's batch-matched
  control is N2.)*

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

- [ ] **B.5 Track import** (WCON or CSV), independent of the simulator, positioned against Tierpsy and
  OWMD feature sets. NeuroSparkSNT and a wider novelty search checked at registration.

  *(2026-10-24: the novelty check is done, in B.3's registration ([Logbook 109](../../../docs/experiments/logbooks/109-benchmark-grading.md)):
  NeuroSparkSNT, PGOB and Morrison & Young 2025 do not occupy the claim. Track import remains.)*

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
  - **C.1's note:** its real fit must simulate at the observation's size, or add between-date
    noise matched to the bands.
  - **Seeds:**
    - animals for simulation k at `1_000_000 + 200·k + i` (pilot `2_000_000`, floor `3_000_000`);
    - prior draws 5401, flows 5402–5403, prior-predictive draws 5404.

## C — controllers and the fidelity table (MUST)

- [ ] **C.1 The mechanistic baseline**: a stochastic run/reversal switch, a roaming/dwelling state with
  intake and satiety dependence, pirouettes and weathervane, ≤ 30 parameters. Fitted by NPE, with
  priors set without search data; posterior-predictive checks.

  *(Amended 2026-10-25, D38 as amended.)* Fitted by sep-CMA on the benchmark's distance, like C.2,
  with its search box and x₀ set without search data. Its predictive checks re-simulate the
  restarts' fits.

- [ ] **C.2 The connectome controller**: on the primitives, with the data-derived neuromodulator layer.
  Its slow variable is stated and its feasibility for a 9-minute decay checked. It is fitted by
  sep-CMA on the benchmark's distance at ≥ 200 animals per evaluation, with the budget pre-registered
  and a plateau criterion.

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

## W — the wiring question on fidelity (MUST, after C.2 reads)

- [ ] **W.1** The nulls: degree-preserving; gap-only on the corrected gap model; boundary-held;
  class-preserving left–right-symmetric; and N.3's receptor-placement nulls. Each has a fresh rewiring
  draw per fit, with parameters counted per arm.
- [ ] **W.2** The registration: the minimum is the held-out distance between two real wild-type cohorts,
  both directions. Draws per arm are sized from F.0 and the cost pilot. Every outcome is named,
  including "the nulls pass too" and "a null beats the wild type".
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
