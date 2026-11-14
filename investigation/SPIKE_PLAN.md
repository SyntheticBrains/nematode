# Spike plan: reproduce one connectome-constrained fit, then bridge it to the closed loop

*Status, added 2026-11-14 at merge: superseded. The plan was never registered. It ran only as
fact-finding sessions 0 and 0b, and the spike stopped there (A30; SPIKE_REPORT.md). The answer
vector named below as `best` is in fact `z`; `best` holds the per-generation scores.*

Written 2026-11-14 and revised the same day after the review of Part A (REVIEW_PART_A.md § 3) and
the maintainer's seven decisions (DECISIONS.md A15 to A21). **Proposed, not run.** Part B runs it
after approval. Once approved, this file is the spike's registration: its bars and rules are not
changed after any scored run, and anything else is reported beside them as description. Numbers
that this plan says a session computes by a fixed rule (the split's checksum, the ceiling, the
reference score S_C) enter by committed amendment before any bar that uses them is read; an amendment records
numbers and never changes a rule.

## 1. What the spike tests, and what it cannot

Two questions, in order:

1. **Can this project reproduce a published connectome-constrained fit to worm perturbation data,
   by gradient descent, and score it on held-out perturbations and held-out neurons against a
   degree-preserving wiring null?** This measures the method's cost and reliability here, on real
   data, before anything is planned around it.
2. **What does that fitted model do, unchanged, in the closed loop on the project's benchmark?**

**What the bridge cannot do.** The fitted model is linear, covers only the head, was fitted to
immobilised animals, and carries no slow state of the kind roaming and dwelling need. It will very
likely fail the benchmark whatever fit-first's merit, so **the bridge cannot support or refute
fit-first's closed-loop prospects.** Its value is in where the interface breaks, and in whether the
wiring shows at all. A pass would license only the narrow statement in § 6.

**What the spike is mainly for.** Part B's estimates must rest on measurement. Beside its bars the
spike keeps an effort ledger and takes three cheap readings (§ 7), including the first held-out
neural test of the project's current model.

## 2. The fit to reproduce

**Creamer, Leifer and Pillow, bioRxiv 2024.09.22.614271 (v4, 2026).** It is the only published fit of
a connectome-constrained model to worm perturbation data (LANDSCAPE.md § 1.1). Its code is MIT
(`Nondairy-Creamer/Creamer_LDS_2026`). The project vendors only its off-diagonal weights
(`data/connectome/creamer_lds_2026_model_weights.csv`: 2,011 entries over 125 neurons, with no AWC,
AIB or AIA). The full model, with its diagonal dynamics, input filters, noise and offsets, is in the
repository's released pickles. The review reports, from fetch summaries, that the repository's
`data/` folder also holds his measured STAMs (`measured_stams.pkl`, with train and test arrays for 30
held-out recordings), `measured_corr.pkl`, `example_recording.pkl` and `initial_conditions.pkl`, and
that his scoring script is `quick_start_examples/predict_stams.py`. These are inside his MIT
repository, not on OSF. `submission_params/exp_test.yml` is probably a 10-recording test
configuration rather than the paper's split, so the paper's 30 test animals are probably not listed
anywhere. Session 0 confirms each of these.

**The published result:**

- A latent linear dynamical system at 2 Hz over 154 head neurons. Each neuron's next activity is a
  weighted sum of all neurons' current activity plus a 45-tap filtered history of its own optogenetic
  input. Weights are free only on the connectome mask (chemical and gap junctions summed, from White
  1986 and Witvliet 2020): 9.4% of pairs, which is about 2,200 weights by my arithmetic. Gaussian
  process and observation noise; an offset.
- Fitted by expectation maximisation with a Kalman smoother on 80 animals; scored on 30 held-out
  animals.
- Held-out stimulus-triggered response magnitudes (STAMs): model against test data r = 0.14, and
  the 80 training animals' STAMs against the 30 test animals' r = 0.17, so a **relative correlation
  of 0.82**. v3 of the preprint reported 0.92.
- A relabelled connectome (the same graph with the neuron labels permuted, "topologically
  identical") "performed poorly". A fully connected model "performed no better" on perturbation
  responses and slightly better on correlations.
- Masked neurons in held-out animals reconstructed at mean r about 0.25 (0.30 in v3).
- About 10 hours on 10 CPUs.

**What changes in the reproduction:** the fitting method only. The same model and objective are
fitted by gradient descent on the exact marginal likelihood, computed by a Kalman filter written in
PyTorch. The data split, the scoring code and the hyperparameters are Creamer's wherever session 0
recovers them, so that a difference can be attributed to the fitter.

## 3. Data and the repository's artefacts

- **Primary data: DANDI 001075** (Randi et al. 2023 atlas, version 0.240930.1859, **CC-BY-4.0**,
  checked twice). 110 segmentation NWB files of 1.9 to 42.1 MB (median 15.9; about 2 GB in all) and
  113 imaging files of 4.4 to 60.5 GB. Three sessions have no segmentation file.
- **Creamer's repository** (MIT), pinned to one commit recorded on session 0: his data-preparation,
  model and scoring code; his released models; his measured STAMs and held-out list if present.
- **The pickles.** Each is first listed with `pickletools` without loading it. It is
  then loaded in an empty directory with `python -I` through an unpickler with an allow-list: NumPy's
  array reconstruction, Python's built-in containers, and classes from his repository's modules
  whose source has been read. Any other global stops the spike and is reported to the maintainer.
  The parameters are exported to an `.npz` in `investigation/spike/` with a SHA-256 and his MIT
  notice. Nothing is added to `data/` (decision A19).
- **No unlicensed data.** Creamer's and Randi's OSF deposits carry no licence and are not used
  (decision A20).
- **If DANDI cannot supply the traces** (gate G0 in § 5), the target becomes Mi et al. 2022 on
  Yemini 2021 (DANDI 000541, CC-BY-4.0; code TuragaLab/wormvae), **only under its own committed
  amendment** with its own oracle, its own degree-preserving null and a bar set from the spread over
  held-out neurons rather than seed standard errors. That target has no optogenetic perturbation,
  so it would answer the brief's "held-out neurons" branch only.
- **Preprocessing:** Creamer's data-preparation logic, read from his pinned code and re-implemented
  on the NWB input. What his code drops at the start of each recording, and how it normalises and
  encodes stimulation, is taken from the code, not from the paper's text.
- **The split:** Creamer's 30 held-out recordings, if session 0 finds a list of them. If not, session
  1 tries for at most two hours to infer them from which neuron pairs his test array covers and its
  values, and accepts an inferred set only if the spike's STAMs on it reproduce his test array to
  10⁻⁶. Failing that, 80 and 30 are drawn with `default_rng(9721)`: the **unrecovered branch**, which
  § 5 treats separately and which is probably the main path. Nothing is chosen on the 30 test
  animals, and pipeline debugging uses training animals only.
- **Nothing large is committed.** Downloads go to a gitignored folder. Committed results are the
  fitted weights (about 50 KB each), score JSONs, the read log, the effort ledger and the report,
  with CC-BY attribution to Randi et al. and MIT attribution to Creamer et al.

## 4. Method

**Code** lives in `investigation/spike/` on this branch. It imports the package read-only (the
neuron tables and classes, the connectome loaders, the batched evaluator and its controllers'
transforms, `scripts/analysis/benchmark_grade.py`'s `simulate_v2` and `grade_populations_v4`, and
`scripts/analysis/benchmark_v4_controls.py`'s distance rule) and changes no
package or script file. NWB reading uses `pynwb` and `dandi` through `uv run --with`, so no project
dependency is added.

**Framework: PyTorch on CPU, float64.** It is already a dependency, and the model is small. JAX would
add a dependency for no gain at this size; Apple's GPU path offers no float64 in PyTorch, and
jax-metal is abandoned (LANDSCAPE.md § 5).

**The likelihood.** Each animal has a fixed set of recorded neurons. Where no frames are missing
within a recording, the filter's covariance reaches a steady state, computed by the doubling
algorithm for the Riccati equation with a convergence test (relative change below 10⁻¹²; iterations
reported). The steady-state gain then filters the whole recording. Any animal with a missing frame
inside a recording uses the exact time-varying filter, with gradient checkpointing. Session 1
measures how often that happens; if more than 5% of frames are missing in more than 10% of animals,
the compute table is re-costed and the ledger records it.

**Hyperparameters, the same for every arm** (decision A22): Creamer's own ridge
penalty and pruning step, read from his code on session 0. If his released models used different
settings per model, each arm takes its counterpart's (the dense arm the fully connected model's, the
relabelled arm the shuffled model's), and session 0 records whether his pruning changes the dense
model's density. The only choice the spike makes is the optimiser's learning rate, chosen from two
values on the synthetic data set from `default_rng(9701)`, which is neutral to every arm. Adam, then
L-BFGS. **Convergence:** a fit stops when the training negative log-likelihood changes by less than
10⁻⁷ relative over 100 steps, or at 5,000 steps. A fit at the cap is extended once to 10,000 steps.
If it is still unconverged, every comparison that uses it is unreadable, and the report says so. The
dense model is also fitted at two other ridge values. **The dense fit that counts is the one at
Creamer's ridge**; the other two are description only.

**The arms** (each fitted identically on the same 80 animals):

- **Wild type:** Creamer's mask from his four NemaNode files.
- **Relabelled:** the paper's control, the same graph with labels permuted (`default_rng(9710)`).
- **Degree-preserving nulls, 8 draws** (`default_rng(9711)` to `default_rng(9718)`). The model gives
  every entry of its mask its own free directed weight, whatever the synapse type, so the null
  rewires the mask itself.
  - Entries whose reverse is also in the mask (mostly gap pairs) are rewired among themselves by
    undirected double-edge swaps. The other entries are rewired by directed double-edge swaps.
  - Any swap that would create an entry already present, a self-connection, or a new reciprocal
    pair is rejected. Each draw attempts 10 swaps per mask entry.
  - This preserves exactly the number of free weights, every neuron's in- and out-degree, and the
    number of reciprocal pairs, and it always terminates. Each draw asserts all three.
  - The diagonal is kept. Each draw reports the fraction of entries moved, and how many gap and
    chemical edges of the source files its entries still coincide with.
  - The swap code is spike-local and is tested on small graphs. `rewire_degree_preserving` is not
    used, because it rewires typed edges rather than the model's untyped mask.
- **Dense:** every off-diagonal weight free.
- **Coupling-free:** off-diagonal weights fixed at zero, with the same diagonal, filters and noise.
  Its in-silico STAMs off the diagonal are exactly zero, so its STAM correlation is undefined. **By
  convention an undefined correlation scores 0**, here and wherever else one arises. The arm is still
  fitted, because the bridge's interface-only row and M-b's baseline use it.

**Test-set guard.** `investigation/spike/read_test.py` is the only code that reads the 30 test
animals. It refuses unless this plan and every amendment are committed and unchanged, the split
file's SHA-256 matches the one recorded by an amendment committed before any test animal is read,
and the reading has not been made before. Each reading is appended to `investigation/spike/READS.md`
and committed. **Bars R1 to R5 are read once.** A second reading of a bar needs a committed amendment
saying why, and the first reading is kept beside it. **A gate (G1 to G3) may be retried once,**
within the timebox, under a committed amendment that names the bug found, with the first attempt
kept. That keeps a first-run porting or filter bug from ending the spike for an uninteresting reason.

**The sessions**, each about one part-time day. The hard stop is at the end of session 10
(decision A17); nine are planned, so one is in reserve.

| Session | Task | Output |
|---|---|---|
| 0 | DANDI check (one segmentation file: traces, labels, stimulation times and targets); Creamer's repository pinned; pickles inspected and exported; his held-out list, STAMs and scoring code located; which of AWC, AIB, AIA, AWA, CEP and ADE are among the 154 and carry fitted input filters | **G0**; the facts later sessions depend on, by amendment |
| 1 | Loader and preprocessing on the NWB files; Creamer's mask; the split (and the inference attempt); missingness; split checksum committed, then G1 | **G1** (the pipeline); split and missingness, by amendment |
| 2 | Creamer's scoring ported verbatim; his own script run in the same sandbox; his released model scored through the spike's pipeline | **G2** (the scorer oracle); S_C, by amendment |
| 3 | PyTorch LDS and likelihood; synthetic data from his full model; learning rate on one synthetic set; truth-initialised and default fits on another; timing at full load | **G3** (the fitter oracle); measured time per fit |
| 4 | The wild-type fit on 80 animals; the likelihood check; R1 read once | **R1** |
| 5 | The controls and held-out-perturbation fits launched; held-out neurons scored as fits finish | fits on record |
| 6 | R2 to R5 read once; the slowest mode (M-d) | **R2, R3, R4, R5**; M-d |
| 7 | C.2 scored on Randi's held-out STAMs (M-a); the fitted model on Atanas 2023's head neurons (M-b) | M-a, M-b |
| 8 | The bridge (§ 6): pilot population, then five | B1, B2 or B3 |
| 9 | SPIKE_REPORT.md, with the effort ledger | the report |

## 5. Gates, bars and the verdict map

Every bar below is registered here, with its rule. Gates G0 to G3 run before any fit is scored, so
that a failed R1 can be attributed.

**Gates:**

- **G0, the data (session 0).** The segmentation files hold traces, neuron labels and stimulation
  times and targets for at least 100 animals. If not, the Mi amendment (§ 3) or a stop.
- **G1, the pipeline (session 1).** Three checks on the spike's NWB-derived data, each of which
  should be near exact if the pipeline matches Creamer's:
  - the animal in his `example_recording.pkl`, prepared by the spike from its NWB file, matches that
    file at r ≥ 0.99;
  - STAMs pooled over all 110 animals match his train and test arrays pooled the same way, weighted
    by event counts where his files give them and by animal counts otherwise, at r ≥ 0.95 (this
    check does not depend on the split);
  - only if the split was recovered, the spike's test STAMs match his test array at r ≥ 0.95;
  - **a failure of any check** is a preprocessing failure;
  - the data ceiling (training animals' STAMs against test animals') is computed and reported, with
    a 95% bootstrap interval (2,000 resamples of test animals, `default_rng(9722)`), but it is not a
    gate.
- **G2, the scorer (session 2).** The ported scorer, given his released model and his measured
  STAMs, reproduces the number his own `predict_stams.py` gives in the same sandbox to 10⁻⁶. The
  paper's 0.82 is reported beside it. His released model scored through the spike's own pipeline on
  the spike's 30 test animals gives the reference score S_C. A failure is a scoring failure.
- **G3, the fitter (session 3).** Synthetic data are generated from his full released model on the
  real animals' recorded-neuron sets, stimulation schedules and lengths, with his fitted noise: one
  set from `default_rng(9701)` for choosing the learning rate, and a separate set from
  `default_rng(9702)` for the gate. The oracle metric is the correlation between fitted and true
  in-silico STAMs on stimulations held out of the synthetic training set, whose filters are set to
  the mean of the training neurons' filters, as in R4.
  - A fit initialised at the truth sets what the data allow, c_T.
  - The default-initialised fit, under the real fit's exact budget and convergence rule, must reach
    at least 0.9 c_T.
  - The relabelled arm, fitted to the same synthetic data, must score below the default fit. This is
    the gate's must-fail arm.
  - **A failure** means the fitter, or the problem's identifiability at this size, is at fault.
- **Any gate that fails** stops the spike, unless it is retried once under § 4's rule.

**Bars:**

- **R1, the reproduction (session 4).** Two parts, both required:
  - **R1a, the fitter.** On the 80 training animals, the spike's penalised negative log-likelihood is
    compared with a reference.
    - In the recovered branch, the reference is Creamer's released parameters, which EM fitted to
      the same 80. The spike must be no worse, to 10⁻⁶ relative: gradient descent reaches EM's
      optimum or better.
    - In the unrecovered branch, his parameters were fitted to a different 80, so beating them would
      be too easy. The reference is a second spike fit initialised at his parameters on the spike's
      80 (one extra fit). The default-initialised fit must be no worse, to 10⁻⁴ relative.
  - **R1b, the held-out score.** The lower bound of the 95% paired bootstrap interval of the spike's
    relative correlation minus S_C, resampling the 30 test animals, is at least −0.1 × S_C. In words,
    the gradient fit keeps at least 90% of the EM answer's held-out score, judged on paired noise.
  - **Reported beside R1:** the absolute r, the denominator, and the correlation between the spike's
    weights and his.
  - **In the unrecovered branch**, his model has been fitted to some of the spike's test animals, so
    S_C is inflated and R1b is conservative. An R1b failure there cannot be pinned on the fitter;
    R1a and G3 carry the fitter's part of the attribution.
- **The minimum effect for the wiring bars,** m = 0.25 × the wild type's score. The coupling-free
  arm scores 0 by convention, so this means the specific wiring must carry at least a quarter of the
  model's predictive score. If the wild type's own 95% bootstrap interval includes 0, R2, R3 and R5's
  null part are unreadable, and the report says the wiring question cannot be asked at this size.
- **Each wiring contrast** (wild type minus a null, with a 95% bootstrap interval) is read on four
  levels, named now:
  - **pass:** at least m, with the interval above zero;
  - **below m:** the interval is above zero but the mean is under m;
  - **absent:** the interval includes zero;
  - **reversed:** the interval is below zero, so the null predicts better.
- **R2, the relabelled null (session 6).** Wild type minus relabelled, intervals over test animals.
  There is one relabelling draw, as in the paper.
- **R3, the degree-preserving null (session 6).** The mean over the 8 draws of wild type minus null,
  intervals resampling test animals and draws. The wild type's rank among the 9 is reported; by
  chance it is first one time in nine.
- **R4, held-out perturbations (session 6).** Stimulated neurons are grouped by class (left and right,
  dorsal and ventral partners together; `quantumnematode.connectome.neuron_classes`) into three folds.
  Each fold's stimulations are removed from training, their input filters are replaced by the mean
  of the training neurons' filters, and the model predicts every neuron's response to them in the test
  animals. The floor predicts every held-out stimulation's responses by the mean response pattern of
  the training stimulations. The ceiling on those stimulations is the correlation between the
  training animals' and the test animals' measured STAMs for the same stimulations. Those data exist,
  even though the model never trains on them. R4 passes if the wild type minus the floor is at least
  0.25 × (ceiling − floor), with a 95% bootstrap interval above zero. No null is fitted on the folds
  (24 more fits); that limit is stated.
- **R5, held-out neurons (session 6, no refit).** Creamer's design: a neuron and its partner masked in
  test animals and reconstructed, scored for every arm. The wild type must beat Creamer's
  wrong-neuron control with a 95% interval above zero. Then wild type minus the mean of the 8 nulls
  is read on R3's four levels, with the same m.

**The verdict map,** every branch named before any reading:

**The reproduction:**

| R1a | R1b | Split | Verdict |
|---|---|---|---|
| fail | any | any | **Not reproduced: the fitter does not reach EM's optimum.** G1 to G3 passed, so it is the fitter on real data. The gap and likeliest cause are recorded, nothing is tuned, and the spike stops after M-a for C.2, which needs no fit. |
| pass | fail | recovered | **Not reproduced on held-out data:** the fitter found an optimum of equal likelihood that predicts worse. The spike stops after M-a for C.2. |
| pass | fail | unrecovered | **Fitter verified; held-out reproduction unconfirmed**, because the reference saw some test animals. The spike continues, since every arm shares the verified fitter. |
| pass | pass | any | **Reproduced.** |

**The wiring, once reproduced.** Each contrast's level (pass, below m, absent, reversed) is reported,
and the readings combine as follows:

| R2 (which neuron is which) | R3 (structure beyond each neuron's degree) | Reading |
|---|---|---|
| pass | pass | The specific wiring matters, beyond each neuron's degree. |
| pass | absent or below m | Identities matter, but mostly through each neuron's degree. |
| absent or below m | pass | The structure of the graph matters; which neuron is which does not. This is coherent: relabelling keeps the degree distribution but gives each neuron another's degree, while the degree-preserving null keeps each neuron's degree and moves its edges. |
| absent or below m | absent or below m | The wiring adds little or nothing measurable here. |
| reversed (either) | any | A null predicts better than the wild type: reported as the finding it is, with the draws. |
| unreadable | unreadable | Coupling itself is not detectable at this data size. |

- **The dense model:** if the dense fit at Creamer's ridge scores at or above the wild type (paired
  interval including or above zero), the report adds "the wiring is not needed for prediction at
  this data size", in any branch.
- **An arm unconverged after its extension** makes every row that uses it unreadable.
- **R4 pass:** the fitted model predicts responses to stimulations it never saw, beyond the mean
  response pattern. That is evidence for milestone M-a. **R4 fail:** it does not, which bounds what
  fit-first can claim for M-a from this data.
- **R5 pass:** the connectome helps reconstruct unrecorded neurons, which would replicate Mi et al.'s
  finding on Randi's data. **R5 fail:** it does not, on this model.
- R4 and R5 are reported in every branch where the spike continues.

## 6. The bridging test

The fitted model runs **unchanged**: its weights, filters and noise are frozen. The interface below is
fixed now, from the model and the environment, and uses no behaviour benchmark data. Readout option B
with a control row (decision A21).

- **Controller.** A spike-local class implementing the batched evaluator's `BatchedController`
  protocol (`reset(n)`, `act(senses, noise)` returning `(N, 5)`). The LDS runs two 0.5-s substeps
  per 1-s step, inputs held. Its process noise is drawn from the per-animal generators the evaluator
  passes in, so an animal behaves the same alone or in a batch.

- **Senses in,** through the model's own fitted input filters. The odour signal is the evaluator's
  `senses.derivative`, the food channel's rate of change per second. It is a rate, so holding it over
  both 0.5-s substeps does not double it.

  - AWA takes the odour's rise, max(0, derivative).
  - AWC takes its fall, max(0, −derivative).
  - CEP and ADE take the level of food contact.

  The odour enters as change because the model was fitted to pulses. Food contact enters as a
  level, because CEP and ADE report contact for as long as it lasts. A linear model then sits at a
  shifted fixed point on the lawn; that is a known property of the choice, not a tuning target. If
  session 0 finds AWC absent from the 154 or without a fitted filter, the odour's fall is dropped
  and only AWA carries odour. The lateral (head-sweep) cue is not used, so weathervane steering
  cannot appear.

- **Input scale.** One scale per feature maps the 99th percentile of its absolute value to the
  amplitude of one optogenetic stimulation in the fitted input encoding (session 0 records it). The
  percentile is taken on Scheer's small lawn under the registered constant-hazard negative
  (`BatchedConstantHazards` at 0.15 mm/s and hazards 0.02, 0.01 and 0.01 per s) on the pilot
  population.

- **Locomotion out.** Each output neuron's activity is z-scored against the LDS's own 600-s
  spontaneous run with no input. Unit gains, literature signs, and the registered negative's base
  values:

  - reversal hazard 0.02 × exp(z̄) per s, where z̄ is the mean z of AVA, AVE, AIB and RIM (those present);
  - run speed 0.15 × exp(z̄) mm/s over AVB and RIB;
  - run curvature ω_max × tanh(z_SMDD − z_SMDV) in degrees per second, where ω_max is the
    primitives' maximum turning rate;
  - omega and delta hazards held at 0.01 per s.

  Each value goes into the raw range through the controllers' own `_raw_speed`, `_raw_curvature`
  (which takes degrees per second) and `_raw_hazards` transforms, re-implemented in the spike if they
  cannot be imported.

- **The readout against real activity,** as description: where M-b's Atanas recordings carry
  behaviour, the area under the ROC curve of z̄ for reversal onsets. It lets a B3 be pinned on the
  readout rather than on the model.

- **The benchmark.** v4 through `simulate_v2` and `grade_populations_v4`, on the configs of
  `benchmark_grade.ARMS["connectome_c2"]`, the environment C.2 is graded in. A pilot population of 200 animals
  (1,501,000,000 + i) is graded first and discarded, to check the B3 conditions and the time. Then
  five populations of 200 with presence (1,500,000,000 + 200·g + i, g < 5). Every one of the 24
  statistics is reported with its z.

- **Rows:**

  - the bridged wild type;
  - the **interface-only row**, the coupling-free arm bridged identically, so a pass cannot be the
    readout's doing alone;
  - null draws 1 to 3 bridged;
  - for reference, C.2's recorded answer, labelled in-sample, and the registered negatives already on
    record.

- **Wild type against null and against interface-only.** A difference between two distances to the
  bands would hide the answer: a failing worm has many statistics at the cap of 10, so both distances
  saturate, and two populations can sit at the same distance on opposite sides. Instead:

  - **The distance between the two simulated populations** is computed exactly as W's minimum
    computes the distance between two real halves (`benchmark_v4_controls.w_minimum_v4`'s rule:
    `benchmark_v2_controls._difference` on each of W's fitted families, the small lawn, kinematics
    and activity, scaled by the bands at the populations' size).
  - That distance is set beside two references: W's minimum of 2.597 (the 95th percentile between
    real halves; fit-c2's design on `feat/fit-c2`), and the wild type's own distance between its
    five populations, as the same-parameter floor.
  - They differ if the distance exceeds both references.
  - Each statistic's raw difference is reported beside its same-parameter floor, without the cap.

- **The outcomes, named in advance:**

  - **B1:** some family passes in at least 3 of 5 populations, and the interface-only row and every
    registered negative recorded as failing that family (Logbooks 110 and 124) fail it. B1 licenses
    only: "a head model fitted to neural data, with a literature-sign readout, reproduces family X
    without fitting to behaviour."
  - **B3:** it cannot run sensibly. The conditions are an output at its clip bound in more than 50%
    of steps, the LDS diverging, or, on the lawn assays only, a median time on the lawn below 10% of
    the assay. The report says where.
  - **B2:** otherwise. The report names the statistics that come closest and why.

- **What to expect, stated before running:** B2 or B3. In COMPARISON.md a B2 or B3 counts as the size
  of the interface problem that a closed-loop fit-first route would have to solve, not as evidence
  against fit-first. A B1 counts as M-c evidence only if the interface-only row fails that family.

## 7. What the spike gives Part B

Description, not verdicts; each with a stated baseline.

- **The effort ledger** (`investigation/spike/LEDGER.md`): per session, the planned and actual hours,
  unplanned fixes, gates failed on first run, amendments, and the hours spent ingesting each dataset.
- **M-a for the current approach (session 7).** C.2's network scored on Randi's held-out STAMs, a test
  it was never fitted to. The network is Logbook 131's answer (`hinge4.json`, its `best` vector, a
  discarded probe rather than a registered fit), converted with `c2_encoder_probe.py reproduce`. A
  spike-local copy of the encoder's settling loop adds an input to one neuron's pre-activation for
  one step.
  - **Parity check.** The copy is checked against the package's forward at 100 random nonzero
    sensory inputs, to 10⁻⁶ (the encoded settling uses float32 `tanh`). A zero-input check would be
    vacuous, since the network has no biases and zero input gives zero activity. A unit test checks
    the added-input path: an input equal to a sensory neuron's own drive reproduces the sensory
    forward.
  - **The response** is the change in each recorded neuron's settled activity, from the resting
    state (no sensory input, modulators at their initial values), at one unit of input (0.5 and 2 as
    description).
  - **Scoring** uses R1's metric on R1's pair set, the relative correlation over the pairs R1 scores,
    so the numbers compare across routes. The floor is the encoder at the box's centre.
  - **Limits.** C.2's network has no within-step dynamics, so only the pattern of signs and sizes is
    compared, and the report says so. This is a probe's answer; when C.2's registered fit exists,
    its answer is the proper row.
- **M-b (session 7).** The fitted model's one-step-ahead prediction of Atanas 2023's head neurons
  (DANDI 000776, CC-BY-4.0; NeuroPAL-identified recordings only, resampled to 2 Hz). Its size is
  checked before download.
  - **The metric** is the correlation between each neuron's predicted and actual change from one
    step to the next, not its level, which predicting "no change" would match trivially.
  - **Baselines:** the coupling-free LDS (each neuron's own dynamics only), and the copy-last-value
    predictor, which scores 0 on this metric by convention.
- **M-d (session 6).** The fitted dynamics' slowest time constant, from the eigenvalues of the
  wild-type and dense models, beside Scheer's roaming and dwelling bout means. A slow mode may
  reflect bleaching or drift in the recordings rather than neural dynamics; the report says which
  neurons carry it.
- **What an M-c attempt would need,** item by item, from the bridge's failures.

## 8. Budget

**Engineering:** nine sessions planned, a hard stop at the end of session 10. At the project's recent
pace, about two to three calendar weeks.

**Compute** (estimates; session 3 measures the real time per fit):

| Item | Fits | Estimate |
|---|---|---|
| Fitter oracle: truth-initialised, default, two learning rates | 4 | |
| Wild type, and in the unrecovered branch its warm-started reference | 1 or 2 | |
| Relabelled 1, degree-preserving 8, dense 1 (and 2 ridge values), coupling-free 1 | 13 | |
| Held-out perturbations, three folds | 3 | |
| **All fits** | **21 or 22** | 2 to 3 h each on 8 threads: 42 to 66 fit-hours, about 21 to 33 h of the machine at two fits at a time on 16 cores (about 340 to 530 core-hours) |
| Held-out neurons, M-a, M-d | none | minutes |
| M-b | none | about 1 h |
| Bridge: pilot and five rows on five populations | none | 3 to 6 h |
| **Total** | | **about 25 to 40 h of machine time** |

- **If one fit takes over 6 hours,** the three folds become two and the dense model's extra ridge
  values are dropped. The null draws are not cut.
- **Disk:** about 2 GB for the segmentation files; M-b's subset is sized before download.
- **The machine's cores** (decision A18): the spike runs locally now, while C.2 is paused, and yields.
  - Every fit checkpoints every 100 gradient steps.
  - The spike's runner checks once a minute for a running C.2 process (the fit-c2 scripts). If it
    finds one, it checkpoints and exits, and resumes when C.2 stops.
  - The spike never uses more than 16 threads, and runs nothing while C.2 runs. It stops only its
    own processes, by the PIDs it recorded.
  - Rented CPU remains the fallback if C.2 restarts for long, at about $0.047 per core-hour, so
    under $30 for the fits.

**Seeds,** all outside every band in the phase's seed table on `main` and `feat/fit-c2`:

- `default_rng(9701)`: the synthetic set for the learning rate; `default_rng(9702)`: the synthetic set
  for G3; 9703 is reserved for a G3 retry;
- 9710: the relabelling;
- 9711 to 9718: the degree-preserving draws;
- 9721: the split, if needed;
- 9722: the bootstraps;
- 1,500,000,000 + 200·g + i and 1,501,000,000 + i: the bridge's animals.

The spike does not edit the tracker. These enter the phase's seed table if any spike result is merged.

## 9. Stop rules

- **G0 fails:** the Mi amendment, or stop and record why.
- **G1, G2 or G3 fails:** retry once under § 4's rule if a bug is found; otherwise stop. No fit is
  scored on an unverified pipeline, scorer or fitter.
- **R1 fails:** as § 5's reproduction map says. Where it stops the spike, the score, the gap and the
  likeliest cause are recorded, nothing is tuned, the bridge is not run, and M-a for C.2 still runs.
- **Any wiring reading** (R2, R3, R5), whatever its level, is a finding (§ 5's map). The bridge
  still runs, with the nulls beside the wild type.
- **Timebox:** at the end of session 10, stop wherever the spike stands and write it up.

## 10. Decisions taken, and what needs approval

Taken on 2026-11-14 (DECISIONS.md A15 to A21; A22 records the design choices made in this revision):

- the corrections to Part A applied;
- this plan revised and re-presented before Part B;
- a hard stop at 10 sessions;
- compute run locally now, yielding to C.2;
- Creamer's pickles opened in isolation;
- Mi on Yemini as the only fallback, by amendment, and no OSF data;
- bridge readout B with an interface-only row.

**Needs approval:** this plan, as the spike's registration. *(2026-11-14: the fresh review found it
not ready (REVIEW_PART_A.md § 7). The maintainer approved session 0 as fact-finding only (A24),
the two departures from the brief's stop rule (A25), and the Phase 10 rule (A26). The plan is
revised with session 0's facts and every fix, then re-presented for registration.)*
