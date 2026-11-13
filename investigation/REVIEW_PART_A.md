# Review of Part A

Three reviewers read Part A on 2026-11-14. Each began with no context from the investigation's
author, and each had a separate scope:

1. **Project facts:** what ORIENTATION, IDENTIFIABILITY, DATA_INVENTORY § 9, SPIKE_PLAN § 6 and
   RECONCILE say about this project's code, logbooks, scores, roadmap and git history.
2. **Literature and data:** the external citations, licences and sizes, opened at their sources.
3. **Design and fairness:** SPIKE_PLAN read as a pre-registration under the phase protocol, and
   whether the documents give the alternatives a fair hearing.

The reviewers were read-only and ran no compute. Their findings are summarised below. Each says
whether I re-checked it myself and what I propose to do. Line numbers refer to the commit reviewed
(5ddd1e70). Nothing has been changed yet; the corrections wait for the maintainer's decision.

## 1. Project facts

**The reviewer's verdict.** The parameter tables, the benchmark history, the scores, the code
interface and the pacing table are right. Every pacing row for Phases 0 to 9 re-derives exactly
from git. The errors are in the forward estimates and in a few counts.

| # | Where | Finding | Severity | Re-checked | Proposed |
|---|---|---|---|---|---|
| P1 | ORIENTATION 211, 276, 286, 290 | W's "450 to 700 hours" rests on "4 null arms with 4 to 6 fits", which is in no file. The tracker registers 5 null families with 8 to 16 draws (its cost line uses 6 arms) and says W at that size is "beyond what it can run as planned". At about 30 h a fit (launch 133), W is about 1,200 to 2,900 h, so "Phase 10 closes 8 to 14 weeks from now" is too short unless W is cut. | error, load-bearing | yes (tracker 187, 997; launch 133 cost table) | Correct. State W at the registered sizing, state the tracker's own options (stopping rule, fewer draws, cheaper fit), and give the close as conditional on that choice. |
| P2 | ORIENTATION 64 | The modulator layer is 120 loading couples plus the ingestion drive, not 121 couples. Under the registered procedure 19 signs are fixed (16 monoamine, pdf-1 and pdf-2 onto pdfr-1, serotonin onto ser-5) and 101 are free. Per-ligand tying gives 57 modulator parameters (75 + 57 + 1 + 4 = 137). | error, small | no | Correct the counts. |
| P3 | ORIENTATION 22 to 24, 182; DATA_INVENTORY 17 | The project does not vendor only CC0 or CC BY data (the Creamer CSV is MIT, the eigenworms BSD-3). The Randi record excluded the atlas for having no licence on OSF and for the GPL package, not for a CC rule. The DANDI correction still stands. | error, small | no | Describe the reasons the record gives. |
| P4 | ORIENTATION 26 to 28, 255 to 261, 270 | Phase 6a's connectome ranked 5th of 6, so it was not "a positive later traced"; the trace came inside 6a. Phase 9 (8 days, 84 commits, 12 PRs) breaks "a phase takes about three weeks". "Artefacts surface a phase late" rests on two cases. | overstated | no | Soften to what the record shows. |
| P5 | ORIENTATION 93 to 96, 273; IDENTIFIABILITY 247; RECONCILE 75 | LB 127 and 128 show wild-type statistics do not pin perturbation responses. LB 129 shows that adding the mutants does constrain them, in-sample. No logbook tests whether C.2's internals are identified. | overstated | partly (LB 128's title) | Restate as "wild-type behaviour statistics do not identify perturbation responses; mutants help in-sample". This weakens one prompt for the investigation, and the documents should say so. |
| P6 | ORIENTATION 275 to 280, 293, 327 to 333 | The effort figures (1.5 to 2 days per change; 2 to 5 weeks, 6 to 12 weeks, 3 to 6 months for the differentiability options) are asserted. Git shows OpenSpec changes landing in hours to about 1.5 days. "Positive controls fail first" ignores F.0b-i on v1 and C.1 on v3, which passed. | unsupported | no | Label the figures as judgement with their basis, or derive them from the archive's timestamps. Part B must not inherit them unexamined. |
| P7 | IDENTIFIABILITY 118 | Flavell's 482 s and 75/520 s are reported, not graded. The graded bout targets are Scheer's bout means and the 5-min roaming survival. | error | no | Correct. |
| P8 | ORIENTATION 21; RECONCILE 74 | "The Creamer prior changed nothing measurable" holds under PPO at the registered minimum. Under the other learner LB 073's interval spans the minimum. | overstated | no | Add the qualifier. |
| P9 | SPIKE_PLAN 147, 154 | The vendored Creamer weights cover 125 neurons, with no AWC, AIB or AIA. The bridge's odour input and its reversal readout may not exist in the model. | error or unverified premise | yes (2,011 rows, 125 neurons, AWC and AIB absent) | Day 0 checks the 154; substitutes named in advance (see D7). |
| P10 | SPIKE_PLAN 123, 164 | `rewire_degree_preserving` takes a typed `Connectome`, not Creamer's untyped mask. `grade_populations_v4` lives in `scripts/analysis/benchmark_grade.py` and grades populations that `simulate_v2` makes for a named arm. | minor | no | Name the adapter and the arm. |
| P11 | various | Smaller items: 18 active days on main, not 19 (20 with `feat/fit-c2`); three revisions of the benchmark, not four; the "D 6.6 to 7.2" range covers the hazard controllers only; v3 added three mechanisms in 5 parameters; the uniform well's bands rest on 2 band dates; 29 graded numbers, not about 28; assays run 3,600 to 5,400 steps; "84 hours by the C.2 fit plan" should be 114 to 127; the model does hold head motor neurons. | minor | no | Correct each. |

## 2. Literature and data

**The reviewer's verdict.** The core holds: Creamer's numbers and controls, Randi's results, the
CC-BY-4.0 licence on DANDI 001075, Dunn 2025, every DANDI and Dryad licence checked, the Lappalainen
quote, Qian, Das and Fiete, Pospisil, arXiv 2411.12091 and WormID. The errors cluster in sources
marked as read in full or in part: Mi, Mehta, Witvliet and Gonçalves.

| # | Where | Finding | Severity | Re-checked | Proposed |
|---|---|---|---|---|---|
| L1 | LANDSCAPE 82 to 88; RECONCILE 36; my last summary to the maintainer | Mi et al.'s "dense 0.008" is a column misread. In Table 2's neuron-holdout row the learned dense models score −0.104 (current-based) and 0.081 (conductance-based); 0.008 is the conductance model's learned sparse value. On held-out worms all nine models score 0.439 to 0.474 (SE 0.006, four seeds). The authors read the connectome-count models as better there; "almost no difference" is my reading of a gap of about 0.01 to 0.02. | error | yes (Table 2 text extracted from the Janelia PDF) | Correct the numbers. Report the authors' held-out-worm claim beside my reading, labelled as such. The headline survives: 0.32 against 0.08 at best without the connectome. |
| L2 | LANDSCAPE 257 to 263; IDENTIFIABILITY 17 to 19; RECONCILE 69 to 71 | Beiran and Litwin-Kumar's student shares the teacher's exact signed weight matrix and fits only per-neuron gains and biases. "Predictions identifiable once about 30 neurons are recorded" therefore does not cover fitting about 2,000 weights from synapse counts. The paper does show errors growing smoothly with weight noise. | overstated (missing caveat, load-bearing) | no | Add the caveat in all three files. It weakens the theoretical support for fit-first. |
| L3 | LANDSCAPE 358 to 360; IDENTIFIABILITY 213; RECONCILE 108, 209 | Mehta 2023's "fails to identify a unique, sparse solution even with extensive data" is from the authors' general limitations section, not a finding on the 18 flies, where the fit found a significant weight-dependent decay term. | misattributed | no | Attribute it correctly and report the 18-fly result separately. |
| L4 | DATA_INVENTORY 21 to 22 | Randi's DANDI files: 110 segmentation files of 1.9 to 42.1 MB (median 15.9), not 13 to 25 MB; 113 imaging files of 4.4 to 60.5 GB. Three sessions have no segmentation file. | error | no | Replace the ranges and note the three gaps. |
| L5 | DATA_INVENTORY 160, 230; RECONCILE 44 | Beets 2023's EC50s are published in Table S6, so they are not "on request only". Whether they come in machine-readable form is unverified. | overstated | no | Correct. |
| L6 | LANDSCAPE 354 to 357; IDENTIFIABILITY 210; RECONCILE 107 | Hironaka and Sumi's equally good opposite-sign networks concern the AIY to AIZ synapse, not the ASER to AIY memory synapse set beside it. | misattributed, minor | no | Correct. |
| L7 | LANDSCAPE 253 to 255 | Gonçalves 2020's genetic-algorithm comparison used the 8-parameter Hodgkin and Huxley model, not the 31-parameter pyloric model. | misattributed, minor | no | Correct. |
| L8 | LANDSCAPE 94, 115; RECONCILE 37 | Two quotations were altered. Witvliet's 43% sentence was reworded inside quotation marks. Morrison and Young's "identifiability has not been established" is stitched together from a general remark about larger nonlinear systems. | minor | no | Restore the exact wording. |
| L9 | various | Smaller items: Creamer's shuffle keeps the whole graph and only relabels it. His ceiling compares 80 training animals with 30 test animals (55/55 halves appear only in a robustness check). "2,000 to 2,200" weights is my arithmetic from 9.4%. r 0.107 is v3's figure (v4 gives 0.108). Randi's published pair count is 1,310. Randi calls 11% inhibition "comparable to" the 20% estimate, not a conflict. Wicks "predicted" signs rather than "resolved" them. The r-elegans quote is from an early pilot. A stale footnote says Mi was not opened. DANDI 000472 and 000565 hold no stimulation data. "Creamer drops each recording's first minute" is in neither source. IDENTIFIABILITY 239 calls r 0.14 "well constrained" against its own table's "constrained with priors". | minor | no | Correct each. Remove 000472 and 000565 from the perturbation table. |

**Not opened by the reviewer:** Atanas 2023's full text (the preprint returned 403), Flavell 2013's
time constants, and several lower-priority sources. Claims resting on them keep their current access
level in LANDSCAPE's source table.

## 3. Design and fairness

**The reviewer's verdict: run after fixes, not a redesign.** The target, data, framework, null family
and bridge concept suit the brief, and the plan is candid that the bridge will probably fail. As
written, R1 can fail for reasons the plan cannot tell apart, or pass on a lucky split. The connectome
claim rests on the weaker null. The brief's own tests have no registered bar. Most fixes cost hours,
because Creamer's repository is reported to ship the released fits, his measured STAMs, his scoring
script and probably his held-out list (seen through fetch summaries, to be confirmed on day 0).

| # | Finding | Severity | Re-checked | Proposed fix |
|---|---|---|---|---|
| D1 | R1 has no oracle. Its 0.10 tolerance is the paper's v3 to v4 shift, not a noise estimate. The plan changes the fitter, the data route and the split at once, so a failed R1 cannot be attributed. | blocker | partly (plan text) | Port Creamer's scoring verbatim and reproduce 0.82 from his released model and STAMs. Use his held-out recordings. Gate the data-only ceiling near 0.17 on day 1. Score his model through the spike's own NWB pipeline. Set the tolerance by bootstrap over test animals. |
| D2 | The oracle's generator cannot be built from the vendored CSV, which has off-diagonal weights only (no diagonal, filters, noise or offsets). Those are in Creamer's pickles, which the project chose not to vendor. "Relative correlation at least 0.9" is undefined on synthetic data and its reachability is unchecked. | major, near blocker | yes (no diagonal entries) | Open the pickles in isolation, inspected first with `pickletools`, and export to a checksummed npz, not vendored. Define the oracle metric as fitted-vs-true in-silico STAM correlation, run a truth-initialised fit first, and use the real fit's exact budget. Needs approval (decision 5). |
| D3 | The tests the brief requires (held-out perturbations, the degree-preserving null) are only reported, with no bar. R2 rests on the paper's relabelling null. | major | yes (plan text) | Register both with floors, minimum effects and a full verdict map that includes the dense model. |
| D4 | The degree-preserving null is weak: with 3 draws the wild type beats all of them by chance one time in four. A directed swap on the merged mask breaks gap symmetry or changes the free-parameter count. | major | no | At least 8 draws. Preserve reciprocity, keep the diagonal, assert equal parameter counts. |
| D5 | Held-out-perturbation folds can put AVAL in test and AVAR in training. | major | no | Fold by bilateral pair or class. Register a floor (the mean response). |
| D6 | The arms may not differ in one thing only: per-arm hyperparameter selection and convergence are unstated. Creamer's own ridge and pruning are not replicated. | major | no | Per-arm selection or one stated grid. A convergence test, with each arm's stopping state reported. Replicate his recipe. |
| D7 | The bridge is underspecified: the omega and delta hazards are unset, the lateral cue is unused, the input calibration depends on an unnamed controller, sustained inputs differ from the 0.5-s pulses fitted, unit conversions are missing, and AWC and AIB may be absent (P9). There is no interface-only control, and no noise floor for wild type against null. | major | partly (P9) | Specify every output and conversion. Add an interface-only row (off-diagonal weights at zero). Compare wild type and null on D against the wild type's own repeat spread. |
| D8 | The plan calls the bridge "the first test of the brief's premise", but the bridge fails whatever fit-first's merit. | major (honesty) | yes (plan line 15 to 17 against 174 to 176) | Say in advance that it cannot support or refute fit-first's closed-loop prospects, and what a pass would license. Label C.2's row in-sample. |
| D9 | The test set has no read guard, and R1 is scored on day 3 while the stop rule fires at day 4, which leaves a day to re-score after debugging. | major | yes (plan text) | Develop on validation only. Read the test set once, through a script that refuses unless the plan and split are committed. Allow a retry only by committed amendment. |
| D10 | The plan's table has seven sessions, not six. Days 0, 1 and 5 will overrun. The steady-state filter assumption is unverified when frames are missing. | major | yes (seven rows) | Budget 9 to 10 sessions. Measure missingness on day 1. Add the oracle, truth-initialised and final fits to the compute table. |
| D11 | The Mi fallback has no degree-preserving null, a trivial R2, a bar on seed standard errors, and no oracle. | minor to major | no | The fallback runs only under its own registered amendment. |
| D12 | Part B needs numbers the spike does not produce. | major for Part B | n/a | Keep an effort ledger (hours per task against plan, unplanned fixes, gates failed on first run). Add three cheap readings: C.2's network scored by in-silico stimulation on Randi's held-out STAMs (the current approach's M-a); the fitted model on Atanas head neurons against a last-value baseline (M-b); and the fitted dynamics' slowest mode against roaming and dwelling timescales (M-d). |

**Fairness.** The reviewer found the IDENTIFIABILITY summary and the RECONCILE verdicts balanced,
and found tilts both ways elsewhere:

- **Toward fit-first:**
  - R2 on the weaker null;
  - a tolerance not set from noise.
- **Against fit-first:**
  - swapping EM for gradient descent adds a failure mode that RECONCILE says is not the lever;
  - an expected-to-fail bridge sits beside C.2's in-sample pass.
- **Against the current approach:**
  - ORIENTATION frames decisive negatives as low yield, where the protocol counts them as the
    product;
  - it reads the benchmark's revisions only as under-determination, though the same objection fits
    neural data (ceiling 0.17, dense no worse than masked);
  - it applies its negative-rate prior to the current approach only.
- **For Part B:** M-c is unreachable by construction for any fitted controller, so Part B needs a
  matching held-out test for the current approach. The hybrid gets no numbers from the spike.

I accept these points. The Part B comparison will apply the same priors to every route.

## 4. What I did not take from the reviews

- **P6, effort from git timestamps.** I accept that my effort figures were asserted and must be
  labelled. I do not accept commit timestamps as the replacement. They bound when work was
  committed, not the reading, review and amendment rounds around it. Part B will rest on the spike's
  effort ledger (D12) and will show both readings.
- **L1, Mi on held-out worms.** I keep my reading that a gap of about 0.01 to 0.02 is small, and
  I will report the authors' contrary reading beside it, labelled.
- **D1 and the fairness point on EM.** The brief asks for a differentiable framework, so the spike
  keeps gradient fitting. Scoring Creamer's released EM model through the spike's own pipeline
  separates a fitter failure from a data or scoring failure. That addresses the point without
  dropping the brief's requirement.
- **Everything else** I accept as stated. The corrections are not yet applied. They wait for the
  maintainer's decision.

## 5. What changes for the conclusions

None of the findings overturns Part A's central reading: perturbation data constrain the head's
effective coupling and little else, and most parameter classes are not constrained by any data
available. Three findings shift the balance, and Part B must carry them:

- **The current approach's remaining cost is larger than Part A said** (P1). W alone is about 1,200
  to 2,900 machine hours at its registered size.
- **The case for fit-first is weaker than Part A said** in two places. The identifiability theory
  assumed known weights (L2), and the logbooks show mutants do constrain perturbation responses in
  behaviour fits (P5).
- **The spike as planned could not have told a fitter failure from a data failure** (D1, D2). It
  also gave the brief's own tests no registered bar (D3).
