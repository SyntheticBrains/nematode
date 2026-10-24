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
nothing in W registers before C.2 has read.

## H — housekeeping (MUST, first)

- [ ] **H.1 The literature watch re-aimed** at Phase 10's questions (`context.md`, `seeds.toml`; lands with
  this tracker's PR). The item closes when the first digest has run against the new brief.
- [x] **H.2 The research question re-framed** (D33) in AGENTS.md, the README and the roadmap (lands with
  this tracker's PR; closes on merge). *(Closed 2026-10-23: PR #470 merged.)*
- [ ] **H.3 Scenario scope recorded**: thermal, predator, multi-agent and oxygen configs frozen for the
  required blocks, kept and tested; thermotaxis named as the second tier; the MAY items below stay
  allowed. It closes when the docs say so in the scenario index.
- [ ] **H.4 The seed-band table** (design.md), cited by every registration.

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

- [ ] **B.2 Readouts**: a trajectory-based reorientation detector, validated against Calhoun's and
  Roberts's omega flags; the classifier run on tracks downsampled to 5 s; the local-search curve; the
  repaired pirouette quintile ratio (the old K.1).

  *(Added 2026-10-24 by E.3.)* López-Cruz 2019 has no αe^(−γt)+β fit. On its deposited CC BY data
  (Mendeley 10.17632/prdtw5jgcd.1), the wild type gives γ = 0.115 ± 0.011/min only from the 4–6 min
  bin: 0.089 from 2 min and 0.137 from 6 min. The curve rises before it decays. B registers:

  - the readout: reorientations that begin with a reversal, per 2-min bin;
  - the fit's start bin;
  - the time zero, after checking the recording's 4-min offset from removal against the bins;
  - γ, from the deposited data.

- [ ] **B.3 Bands and the grading rule** (D34): from batch-matched cohorts with the project's
  classifier; mutant effects relative to the controller's own wild type; composite rules that depend
  on closed-loop contingencies.

- [ ] **B.4 The controls**:

  - positive: cohort-held-out real tracks pass, and held-out real mutant cohorts pass the mutant
    grading;
  - matched negatives (the open-loop Scheer HMM; Poisson-timed primitives) fail;
  - trivial negatives (a random walk, a correlated random walk, the untrained policy) fail.

  **The benchmark is void unless all of these behave.**

- [ ] **B.5 Track import** (WCON or CSV), independent of the simulator, positioned against Tierpsy and
  OWMD feature sets. NeuroSparkSNT and a wider novelty search checked at registration.

## N — the neuromodulator layer from data (MUST)

- [ ] **N.1 Licences confirmed**: Bentley 2016 (CC BY, S1 Dataset) and Ripoll-Sánchez 2023 (unstated on
  the toolbox and repository pages). Vendored only if confirmed; otherwise what stands in is stated.
- [ ] **N.2 The layer built**: receptor expression and producers per neuron, replacing the hand-placed
  four-pathway circuit behind a byte-identical default, with the diffusion-range model chosen in its
  spec.
- [ ] **N.3 Its null family**: receptor-placement nulls, with a D24-style matched-statistics table.

## F.0 — the fitting methods' positive control (MUST)

- [ ] **F.0** parameter recovery at known parameters, for NPE on C.1's model and sep-CMA on C.2's
  encoder (seeds 5401–5464). It reports recovery and spread, and the spread sizes W.

## C — controllers and the fidelity table (MUST)

- [ ] **C.1 The mechanistic baseline**: a stochastic run/reversal switch, a roaming/dwelling state with
  intake and satiety dependence, pirouettes and weathervane, ≤ 30 parameters. Fitted by NPE, with
  priors set without search data; posterior-predictive checks.
- [ ] **C.2 The connectome controller**: on the primitives, with the data-derived neuromodulator layer.
  Its slow variable is stated and its feasibility for a 9-minute decay checked. It is fitted by
  sep-CMA on the benchmark's distance at ≥ 200 animals per evaluation, with the budget pre-registered
  and a plateau criterion.
- [ ] **C.3 The comparator**: the C.2 architecture trained by reward on the same primitives and lawns,
  16 seeds, γ adjusted to the 1-s step.
- [ ] **C.4 The fidelity table**: C.1–C.3 and the negatives on held-out assays and the mutant set
  (tph-1 on Scheer only, pdfr-1, tax-4), under a per-controller encoding table registered before any
  fit, and with C.2 against C.1 as a secondary contrast.

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
