# Spike plan: reproduce one connectome-constrained fit, then bridge it to the closed loop

Written 2026-11-14. **Proposed, not run.** Part B runs it after approval. Once approved, this file is
the spike's registration: its bars and rules are not changed after any scored run, and anything else
is reported beside them as description.

## 1. What the spike tests

Two questions, in order:

1. **Can this project reproduce a published connectome-constrained fit to worm perturbation data,
   by gradient descent, and score it on held-out perturbations and held-out neurons against a
   degree-preserving wiring null?** This measures the method's cost and reliability here, on real
   data, before anything is planned around it.
2. **What does that fitted model do, unchanged, in the closed loop on the project's benchmark?** This
   is the first test of the brief's premise that behaviour can be a held-out test of a model fitted
   to neural data.

## 2. The fit to reproduce

**Creamer, Leifer and Pillow, bioRxiv 2024.09.22.614271 (v4, 2026).** It is the only published fit of
a connectome-constrained model to worm perturbation data (LANDSCAPE.md § 1.1). Its code is MIT
(`Nondairy-Creamer/Creamer_LDS_2026`), its headline numbers are stated in the text (checked), and its
fitted weights are already vendored (`data/connectome/creamer_lds_2026_model_weights.csv`).

**The published result to reproduce:**

- A latent linear dynamical system at 2 Hz over 154 head neurons. Each neuron's next activity is a
  weighted sum of all neurons' current activity plus a 45-tap filtered history of its own optogenetic
  input; weights are free only on the connectome mask (chemical and gap junctions summed, from White
  1986 and Witvliet 2020, about 9.4% of pairs); Gaussian process and observation noise; an offset.
- Fitted by expectation maximisation with a Kalman smoother on 80 animals; scored on 30 held-out
  animals.
- Held-out stimulus-triggered response magnitudes (STAMs): model against data r = 0.14, data against
  itself r = 0.17, so a **relative correlation of 0.82**. v3 of the same preprint reported 0.92.
- A relabelled connectome (rows and columns permuted) "performed poorly"; a fully connected model
  "performed no better".
- Masked neurons in held-out animals reconstructed at mean r about 0.25.
- Their README: about 40 hours serially on a desktop, or about 10 hours on 10 CPUs with MPI.

**What changes in the reproduction:** the fitting method. The same model is fitted by gradient
descent on its exact marginal likelihood, computed by a Kalman filter written in PyTorch. That is the
fit-first method on the field's one benchmark result.

## 3. Data

- **Primary: DANDI 001075** (Randi et al. 2023 atlas, version 0.240930.1859, **CC-BY-4.0**, checked).
  113 imaging NWB files (8 to 60 GB each) and 110 segmentation NWB files (13 to 25 MB each).
- **Day-0 check (first task, 1 to 2 hours):** open one segmentation file and confirm it holds the
  extracted traces, neuron labels and stimulation times and targets. If it does, download the 110
  segmentation files (about 2 GB). If the traces sit only in the imaging files, stream just those
  arrays from DANDI with remote HDF5 reads rather than downloading 4 TB.
- **Fallback, needs your approval:** Creamer's reformatted data (OSF `qxhjd`) or Randi's OSF exports,
  both public with no licence stated. They match the paper's pipeline exactly, but nothing derived
  from them could be vendored under the project's CC0-or-CC-BY rule. The spike uses them only if DANDI
  cannot supply the traces, and only locally.
- **Preprocessing:** Creamer's `create_data_set.py` logic, re-implemented on the NWB input: drop each
  recording's first 120 samples, ΔF/F with F₀ the time mean, their neuron list and stimulation
  encoding. A day-1 check compares per-animal trace statistics with their OSF data's, if that fallback
  is approved, or with the paper's stated counts otherwise.
- **The split:** the README fixes none, so 80 training and 30 test animals are drawn with a recorded
  seed. Hyperparameters are chosen on a validation split inside the 80, never on the 30.
- **Nothing large is committed.** Downloads go to a gitignored folder. Committed results are the
  fitted weights (about 50 KB), score JSONs and the report, with CC-BY attribution to Randi et al.

## 4. Method

**Code** lives in `investigation/spike/` on this branch. It imports the package read-only (the
neuron tables, the connectome loaders, `rewire_degree_preserving`, the batched evaluator and the
benchmark grading) and changes no package file. NWB reading uses `pynwb` and `dandi` through
`uv run --with`, so no project dependency is added.

**Framework: PyTorch on CPU, float64.** It is already a dependency, the brain already uses it, and
the model is small. JAX would add a dependency for no gain at this size; Apple's GPU path offers no
float64 in PyTorch, and jax-metal is abandoned (LANDSCAPE.md § 5).

**The likelihood.** Each animal has a fixed set of recorded neurons, so the filter's covariance
reaches a steady state within tens of steps. The spike computes it by 50 differentiable Riccati
iterations per animal per gradient step, then filters with the steady-state gain. That makes a
gradient step cost roughly seconds rather than minutes. Where a neuron's recording breaks mid-session,
the exact time-varying filter is used for that animal.

**The steps, each a session of about one part-time day:**

| Day | Task | Output |
|---|---|---|
| 0 | DANDI check; download or streaming set up; spike environment | data on disk, or a stop with the reason |
| 1 | Loader and preprocessing; connectome mask from Creamer's four NemaNode files (MIT repository); split | dataset summary against the paper's counts |
| 2 | PyTorch LDS and Kalman likelihood; **oracle**: synthetic data from Creamer's released weights on the real animals' recorded-neuron sets and stimulation schedules, fitted by the spike's code | oracle reading; measured time per gradient step |
| 3 | The fit on 80 animals; hyperparameter choice on validation; score on the 30 | **R1** |
| 4 | Controls and held-out tests (below) | **R2**; held-out-stimulation and held-out-neuron readings; degree-preserving null |
| 5 | The bridge (§ 6) | the bridged model graded on benchmark v4 |
| 6 | SPIKE_REPORT.md | the report |

**Hyperparameters,** fixed in advance: Adam then L-BFGS; two learning rates and three L2 penalties
on W, chosen on validation; at most 2,000 gradient steps per fit; parameters initialised as in
Creamer's code. Nothing else is tuned.

## 5. Scores and controls

**Reproduction bars (registered here):**

- **Oracle (day 2).** Fitting synthetic data generated from Creamer's released weights must recover
  the generating model's held-out STAMs at a relative correlation of at least 0.9. If it does not,
  the fitter is wrong or the problem is not identifiable at this data size, and the spike stops.
- **R1, the fit.** The gradient-fitted masked LDS reaches a held-out relative STAM correlation of at
  least **0.72**: the published 0.82 minus 0.10. The tolerance is the paper's own shift between v3
  and v4 (0.92 to 0.82), the size of change its analysis choices alone produced.
- **R2, the connectome claim.** The relabelled-connectome model, fitted identically, scores below
  the masked model on the same metric, with a 95% bootstrap interval over test animals that excludes
  zero.

**Reported beside the bars, each against its null:**

- **The degree-preserving null** the brief asks for: the mask rewired by directed double-edge swaps
  preserving every neuron's in- and out-degree (the project's `rewire_degree_preserving`), three
  draws, each fitted identically. Wild type minus null, with intervals. Creamer's relabelling keeps
  the graph and moves the labels; this keeps the degrees and moves the edges, so the two answer
  different questions.
- **Held-out perturbations**, which the paper did not report: three folds over stimulated neurons.
  Every stimulation of a fold's neurons is removed from training, their input filters are replaced by
  the mean of the training neurons' filters, and the model predicts every neuron's response to them in
  the test animals.
- **Held-out neurons:** Creamer's design, a neuron and its bilateral partner masked in test animals
  and reconstructed, against the paper's mean r of 0.25.
- **A fully connected model,** for reference, as in the paper.
- **May, if time allows:** the same fit on the project's Emmons 2024 mask (chemical and gap union)
  beside Creamer's, which ties the spike to W's wiring question.

## 6. The bridging test

The fitted model runs **unchanged**: its weights, filters and noise are frozen. What has to be added
is an interface, because the model was fitted to optogenetic inputs and calcium, not to senses and
locomotion. The interface is fixed in advance and uses no behaviour benchmark data.

- **Controller.** A spike-local class implementing the batched evaluator's `BatchedController`
  protocol (`reset`, `act`), so the package is not changed. The LDS runs two 0.5-s substeps per 1-s
  environment step, inputs held.
- **Senses in.** Through the model's own fitted input filters, into the head sensory neurons it
  contains, with signs from the literature: AWC driven by the odour's fall (−dC/dt), AWA by its rise,
  CEP and ADE by food contact. One scale maps the 99th percentile of each feature on the small lawn to
  one optogenetic stimulation unit, the amplitude the model was fitted to. That calibration uses the
  environment and the neural data, not worm behaviour.
- **Locomotion out.** The model holds no motor neurons, so a readout from head command neurons is
  needed. Two options, decided at approval:
  - **A (recommended):** a readout fitted on freely moving neural and behaviour recordings (Atanas
    2023, DANDI 000776, CC-BY-4.0): a logistic model of reversal onset on z-scored AVA, AVE, AIB and
    RIM; linear models of speed on AVB and RIB, and of curvature on SMDD minus SMDV. It is fitted to
    recordings, not to the benchmark, so it keeps the brief's rule that behaviour benchmarks are
    held out. It costs about half a day inside day 5. Whether DANDI 000776's NWB files carry the
    behaviour covariates is unchecked; WormWideWeb carries them but states no licence, so day 0
    checks this too.
  - **B:** fixed literature signs with unit gains and the primitives' default base rates, stated in
    full.
  - Either way, the LDS's activity is z-scored against its own spontaneous run, since immobilised
    GCaMP6s and freely moving GCaMP7f have different scales.
- **The benchmark.** v4, graded by `grade_populations_v4` on five populations of 200 animals with
  presence, exactly as C.2 is graded. Every one of the 24 statistics is reported with its z.
- **Rows:** the bridged wild-type fit; the bridged degree-preserving nulls; for reference, C.2's
  recorded answer and the registered negatives (already on record).
- **The result is reported whatever it is.** Named in advance:
  - **B1:** some family passes;
  - **B2:** it runs and fails every family; the report says which statistics come closest and why;
  - **B3:** it cannot run sensibly (for example, outputs pinned by the readout, or the worm leaving the
    lawn at once); the report says where the interface breaks;
  - and separately, whether the wild-type and null bridges differ on any statistic.
- **What to expect, stated before running:** B2 is the likeliest. The model is linear with one fixed
  point, fitted to immobilised animals, covers only the head, and carries no slow state of the kind
  roaming and dwelling need. The value of the run is in where it fails and whether the wiring shows.

## 7. Budget

**Engineering:** six sessions as tabled, with a hard stop at eight. At the project's recent pace,
about one and a half to two calendar weeks.

**Compute** (estimates; day 2 measures the real time per step):

| Item | Estimate |
|---|---|
| One fit, 80 animals, at most 2,000 steps, 8 threads | 2 to 3 hours with the steady-state filter (their EM: about 10 hours on 10 CPUs) |
| Hyperparameter selection, 6 settings on validation | 6 to 18 hours |
| Controls: relabelled 1, degree-preserving 3, held-out folds 3, fully connected 1 | 8 fits, 16 to 24 hours |
| Bridge: v4 grading, wild type and three nulls | 1 to 3 hours |
| **Total** | **about 25 to 50 CPU hours; no GPU needed** |

- **If one fit takes over 6 hours,** the controls shrink to one relabelled, two degree-preserving and
  two held-out folds, and the report says so.
- **Disk:** about 2 to 5 GB if the segmentation files hold the traces.
- **The machine's cores.** The C.2 sequence (readiness pilot, F.0b-ii, registered fit; about 84
  hours by the C.2 fit plan, about 117 by launch record 133) uses 16 workers, the project's limit. The spike must not run beside it. Three ways, decided at
  approval:
  1. **wait** until the C.2 sequence ends (recommended if it ends within a week);
  2. **rent CPU**: a 16-core instance at Modal's rate of about $0.047 per core-hour seen on 2026-11-14
     is about $0.75 an hour, so under $40 for the whole spike;
  3. **rent a GPU**: unnecessary at this size, but an A100 at $1.19 to $2.50 an hour (RunPod, Lambda,
     Modal, 2026-11-14) for about 20 hours would be $25 to $50.

## 8. Stop rules

- **Data.** If day 0 shows that neither DANDI nor an approved fallback gives the traces, stop and
  record why.
- **Oracle.** If the oracle fails at day 2, stop: no real-data fit is scored on an unverified fitter.
- **Reproduction.** If R1 is not met by the end of day 4 under the fixed hyperparameter grid, stop.
  Record the score, the gap, and the likeliest causes (preprocessing, mask, fitter, or the published
  number), and do not tune further. The bridge is not run.
- **Timebox.** At the end of session eight, stop wherever the spike stands and write it up.
- **If R1 passes and R2 fails,** the fit is reproduced and the connectome claim is not. That is a
  finding; the bridge still runs, with the null beside the wild type.

## 9. Decisions needed at approval

1. **Timing against the C.2 sequence:** wait, rent CPU, or rent a GPU (§ 7).
2. **The data fallback:** whether Creamer's or Randi's unlicensed OSF data may be used locally if
   DANDI cannot supply the traces (§ 3).
3. **The bridge's readout:** option A (fitted on Atanas 2023's freely moving recordings) or B
   (literature signs) (§ 6).
