# Orientation: the current approach, described honestly

Written 2026-11-14 from `main` at `2c80cfcd`, plus the five unmerged commits on `feat/fit-c2` that
carry launch record 133. "LB n" is Logbook n under `docs/experiments/logbooks/`. Source paths are
under `packages/quantum-nematode/quantumnematode/`. Effort estimates are judgements, labelled as such.

## Summary

- **Parameters.** The current controller, C.2, is the real wiring (302 neurons, 3,709 chemical edges,
  1,095 gap pairs, per-connection signs) run as a memoryless rate network that re-settles from its
  sensory input every second. No synaptic weight is fitted individually. Its 137 free parameters are
  fitted by sep-CMA to real-worm behaviour statistics, and about 130 of them sit at the interfaces:
  sensory gains, the motor readout, noise, and neuromodulator strengths and timescales. No neuron has
  a time constant, a bias or an intrinsic current, and there is no within-lifetime plasticity.
- **Benchmarks.** All grading is behavioural. Benchmark v4 grades 24 statistics in six families on
  public lawn, local-search and chemotaxis data, plus three mutants. It was revised four times in 12
  days, each time after a controller's result. C.2 passes it in 15 of 15 populations in probes, after
  every graded family and both resolved mutants were moved into its objective. Its registered fit is
  in its third readiness pilot. Launch record 133 itself calls the coming verdict in-sample.
- **Neural data.** The only asset derived from recordings is a table of Creamer, Leifer and Pillow's
  fitted weights (125 neurons). Used as a weight prior in Phase 8, it changed nothing measurable. The
  current controller does not read it. No recordings are vendored. The project's licence record says
  the Randi 2023 atlas is excluded by its CC0-or-CC-BY rule, but that check missed the atlas's DANDI
  deposit, which is CC-BY-4.0 (DATA_INVENTORY.md).
- **Roadmap and pacing.** Phases 0 to 9 are closed and Phase 10 is three weeks in. Calendar speed is
  high: recent phases take about three weeks, with 250 to 400 commits each. The slow part is yield.
  From Phase 6a to Phase 9, every phase's headline was a STOP or a SPLIT, or a positive later traced
  to a property of the model. Phase 10 likely closes in 8 to 14 weeks, mostly machine-bound. The
  roadmap has no path to the brief's long-run goal: nothing on it grades the model against neural
  activity, gives neurons dynamics, or fits plasticity.
- **Differentiability.** The brain's one-step map is PyTorch and already supports autograd. It was
  once trained by gradient descent in 3.3 hours (LB 126). The environment, locomotor primitives and
  grading are non-differentiable NumPy. Fit-first does not need them differentiable, because a neural
  fit is open-loop on recorded stimuli. It does need a new dynamical model, a calcium observation
  model and licensed recordings, none of which exist in the repository.
- **This idea was weighed before.** The October 2026 direction review called it option C, a fitted
  whole-brain model in the loop, and deferred it to a later stage of Phase 10. It survives as a MAY
  item ("a Creamer-dynamics controller graded on the benchmark"), not started.

## 1. How neuron and synapse parameters are set today

### 1.1 The connectome controller, C.2

C.2 is the `connectomeppo` architecture under `encoder: compact`, on configs
`configs/scenarios/foraging/*_lawns_encoded_s1_primitives*.yml`. The encoder pins 33 settings
(`brain/arch/connectome_ppo.py:1248-1282`) and exposes a parameter vector that sep-CMA fits to the
benchmark. "sep-CMA" below means fitted to real-worm behaviour statistics; nothing in C.2's line of
results was trained by PPO.

| Class | What the code does | Where the values come from |
|---|---|---|
| Neuron set | All 302 hermaphrodite neurons, pharyngeal included | Emmons 2024 S1 (Cook 2019's matrices with a 2023 gap addition); classification from OpenWorm cect after Cook 2019 and WormAtlas (`connectome/loader.py`, `connectome/neurons.py`) |
| Chemical wiring | An edge exists only where the EM count is above 0; none added | Data: 3,709 connections (`data/connectome/PROVENANCE.md`) |
| Gap wiring | Symmetric, 1,095 pairs | Data: EM counts |
| Synaptic signs | First of four rules: a physiology override (67 edges, 16 of them from López-Cruz 2019), then Fenyves 2020's expression-based polarity (1,676), then a transmitter rule (1,433), else unsigned (533) | Data and literature (`connectome/signs.py:401-486`; LB 091, 122) |
| Signed weight magnitudes | weight = scale × sign × count / √(Σ count² onto the target). Two scales, excitatory and inhibitory | Counts are data; the normaliser is a hand choice; the two scales are sep-CMA. Every unconstrained fit drove the inhibitory scale to 0.005 to 0.09 of its box (LB 127), so the procedure now floors it at 0.09 by hand |
| Unsigned weights | The same normalisation times one signed gain per presynaptic transmitter group (4 gains) | sep-CMA |
| Gap conductances | count / √(sᵢsⱼ), times one scale, additive coupling | Counts are data; the normaliser was chosen in Phase 9 after the previous one produced an artefact (LB 103); the scale is sep-CMA |
| Neuron dynamics | Each 1-s step: inject the senses, then 4 iterations of h ← tanh(h(W + gG) + I_mod). Nothing carries to the next step | Hand-set. Depth 4 is the sensory, interneuron, command, motor path. No time constants, no leak (`connectome_ppo.py:2512-2543`, checked) |
| Intrinsic currents and biases | None. Zero input gives zero activity | Not modelled |
| Sensory input | Four features (concentration, left-right difference, dC/dt, food contact). Each of 11 amphid classes gets the same abstract food signal through its own gains; food contact goes onto CEP, ADE and PDE | 36 gains, sep-CMA; the class choice is from the literature. The temporal derivative is computed by the environment, not by neurons |
| Motor readout | Mean rate of four ventral-cord classes (VB, DB, VA, DA; 39 cells) minus 4 centres, then a 5×4 matrix with biases onto speed, curvature and three bout-start hazards (reversal, omega, delta), with a soft cap | 29 numbers, sep-CMA. The pooling is a hand design; command interneurons are not read |
| Action noise | Gaussian per output; slow Ornstein–Uhlenbeck noise on speed | 5 log-SDs and one timescale, sep-CMA |
| Neuromodulator layer | 55 ligand pools, each a leaky average of its producers' rates, acting on receptor cells through 121 couples; food ingestion drives NSM, scaled by hunger | Placement is data (Bentley 2016; Ripoll-Sánchez 2023). 17 signs fixed from receptor biology, 104 strengths fitted (or tied per ligand). Two timescales, monoamine and neuropeptide, sep-CMA on 10 to 3,000 s. PDF's and ser-5's signs were fixed from receptor coupling after fits reversed both (LB 128) |
| Plasticity | None within a lifetime. The batched evaluator refuses learning rules | Not modelled |

**Counts.** 196 parameters by default, 137 under the registered procedure (per-ligand tying, soft
cap, pool centres, graded box, inhibition floor, cited receptor signs, slow speed noise). Of the 137,
about 7 act on the wiring and about 130 on its interfaces.

**What fixes the locomotion.** Every controller drives the same locomotor primitives (runs, reversal
bouts, omega and delta turns) through hazards. Their kinematics are fixed from data: off food from
published values (Zhao 2003, Broekmans 2016, Pierce-Shimomura 1999, Gray 2005), and on food from
values the project measured on Scheer's held-out dates (LB 105, 108, 132). Those held-out dates are
therefore partly consumed by the body.

### 1.2 The mechanistic baseline, C.1, for contrast

C.1 (`evaluation/controllers.py`, `BatchedMechanistic`) has no neurons. It is a two-state worm with
roaming and dwelling transition rates that depend on intake and satiety, per-state speed and turning,
a weathervane gain, and reversal and omega hazards that depend on dC/dt. It has 20 parameters, 25
with the off-food options of LB 121, and 26 with LB 125's on-food reversal rate. All are sep-CMA
fitted inside hand-set boxes.

### 1.3 Candid reading

- **The wiring contributes structure, not fitted weights.** The 3,709 chemical edges carry two fitted
  scales between them. Whatever information the counts hold passes through a hand-chosen normaliser.
- **The network has no state of its own.** Persistence, such as roaming and dwelling or a committed
  reversal, can only come from the modulator pools, the slow speed noise or the body's bouts.
- **The interfaces carry the fit.** Most of what makes C.2 pass lives in sensory gains, readout and
  modulator strengths, not in the connectome.
- **Behaviour does not identify the internals.** Fitted on wild-type behaviour, nothing chose which
  internal route carries the worm's state, so mutants either did nothing or moved the wrong way
  (LB 127, 128). The remedy was to import constraints from biology (receptor signs, the inhibition
  floor) and to fit the mutants directly (LB 129).
- **Many settings were added after a failed reading.** The soft cap, pool centres, graded box,
  inhibition floor, receptor signs, slow noise, three objective guards, the failure-probability term,
  the curriculum and the start-keeping rule were each added after a fit failed (LB 126 to 131,
  launch 133). Each addition was registered and disclosed, but together they are many forking paths.

## 2. Every benchmark, what it measures, and how the current models score

### 2.1 Earlier validations, Phases 6 to 8

These graded reward-trained (PPO) learners, mostly against sign-only references.

| LB | Measures | Result |
|---|---|---|
| 035 | Chemotaxis: klinokinesis ratio and weathervane slope | Directions reproduced. The turn-rate ratio was 1.15 (connectome) and 1.30 (MLP) against the literature's 1.5 to 3.0 |
| 036 | The same for thermotaxis (MLP, 4 seeds) | Partial: weathervane weak, klinokinesis absent |
| 086 | Body kinematics and posture | Body checks pass by construction; none of about 807,000 turns is omega-shaped |
| 088 | Patch leaving out of roaming | Unreadable: 9 of 16 seeds never roam on a lawn |
| 089 | Connectome dwelling on lawns | No real-data reference |
| 090 | tph-1 and pdfr-1 acute knockouts | tph-1: no effect at the minimum; pdfr-1: unresolved |
| 093 | Thermotaxis through the body | Weathervane present, klinokinesis absent |

Across these, reward-trained learners never reached a worm-sized klinokinesis, did not roam before
leaving, and did not phenocopy knockouts. That record is why Phase 10 moved to fitting.

### 2.2 The Phase 10 benchmark

All versions use 1-s positions, a two-state HMM classifier for roaming and dwelling, bands from a
cluster bootstrap over recording dates, and a Bonferroni bar on each statistic's z-score. A
controller passes the wild type when at least 3 of 5 registered populations of 200 animals pass
every family. Mutants (tph-1, pdfr-1, tax-4, all from Scheer) are graded on their difference from
their own wild type on one or two small-lawn statistics. Only tph-1's and tax-4's directions are
resolved by the data; pdfr-1 is graded by magnitude, so a knockout that does nothing passes it.

| Version | Statistics | Bar z\* | What it added | Validity |
|---|---|---|---|---|
| v1 (LB 109, 110) | 11 | 2.84 | Scheer's small lawn (6), uniform well (3), López-Cruz local search (2) | VALID: held-out real worms pass, six negatives fail |
| v2 (LB 115) | 19 | 3.008 | Kinematics (5), Chen 2025 chemotaxis (3), a tracker measurement model | VALID |
| v3 (LB 120) | 21 | 3.038 | Log early/late ratio, off-food onset rate, direction-flip fraction | VALID |
| v4 (LB 124, current) | 24 | 3.078 | On-lawn activity (pause share, flips, onset rate) | VALID |

The data behind it are thin in places: the uniform well rests on 100 animals over 3 dates, the
chemotaxis family on 6 standard recordings, and tph-1 on 5 dates.

### 2.3 Scores

| Controller | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Hazard controllers and the six negatives | fail (D 6.6 to 7.2) | fail | fail | fail |
| C.1, fitted on that version | 3 of 5, at the threshold (LB 113) | 0 of 5: reverses 52 to 56% of off-food seconds against a real 12 to 17% (LB 119) | 3 of 5, at the threshold, with 5 mechanisms added (LB 121) | v3's answer 0 of 5 (on-lawn onset rate z about −6.5). With a 26th parameter, 4 of 5 from a warm start, 0 of 5 from the box centre; refit deferred (LB 125) |
| C.2, probes | | | | 15 of 15 with every mutant passing, after all six families and the mutants entered the objective and the on-food reversal was changed (LB 131, 132) |
| C.2, registered fit (launch 133) | | | | third readiness pilot running; the first was stopped by its kill rule and the second kept no start |

**What is held out.** Under D37 as first registered, local search and the mutants were held out for
C.2. That did not survive. Fitted only on the four lawn families, C.2 failed local search, chemotaxis,
tph-1 and tax-4 in 5 of 5 populations (LB 126). Those were then moved into the objective (LB 127 to
129; D37's amendment of 2026-11-10). For the registered fit, what remains held out is a
description-level reading of Scheer's held-out recording dates and a tax-4 leave-one-out that gates
nothing. C.1 also fits local search and chemotaxis.

### 2.4 How the benchmark and the controllers co-evolved

| Revision | Came after | Change |
|---|---|---|
| v2 | C.1 passing v1 "partly through routes no real worm takes" (LB 115) | kinematics, chemotaxis, measurement model |
| v3 | C.1b's reversal-dominated near-miss (LB 119) | onset rate, flips, log early/late |
| v4 | C.2's answer that did not run, and C.1's v3 pass that rarely reverses (LB 123, 124) | the activity family |
| Primitives (LB 132) | C.2 stuck at 8 or 9 of 15 | on-food reversals that follow the worm's speed |
| Objective (LB 127 to 129) | held-out families and mutants failing (LB 126) | everything graded moved into the fit |

Each revision was registered and positive-controlled, and each made the benchmark harder to pass in a
non-worm way. The pattern is still adversarial: statistics were chosen by probes on the controllers
they then caught, and held-out data were read during planning. Read plainly, the benchmark keeps
finding ways a fitted controller can match the statistics without behaving like a worm. That is
itself evidence that behaviour statistics under-determine the controller.

## 3. Neural-data assets

- **`data/connectome/creamer_lds_2026_model_weights.csv`** holds the signed weights of Creamer,
  Leifer and Pillow's connectome-constrained linear dynamical system (a preprint fitted to the Randi
  2023 atlas): 2,011 entries over 125 neurons, as coefficients of a 2 Hz dynamics matrix on calcium.
  It covers 1,049 of the 3,709 chemical edges and no edge onto a body motor neuron. In Phase 8 it
  served as a starting prior under PPO: it did not move the wiring effect (LB 072, 073). C.2's encoder
  refuses any weight prior other than random, so the current phase does not read it.
- **Nothing else** derived from neural activity is in the repository: no recordings, no loader, no
  observation model.
- **The licence record is out of date.** The project vendors only CC0 or CC BY data
  (`data/neuromodulators/PROVENANCE.md`, checked). Its 2026-09-23 check of the Randi atlas
  (`data/connectome/PROVENANCE.md`) found the OSF deposit with no licence, `funatlas.h5` under
  GPL-3.0 and Creamer's reformatted recordings with none, and concluded the atlas could not be
  vendored. It did not check DANDI 001075, the atlas's raw NWB deposit, which is CC-BY-4.0 (checked
  on the host 2026-11-14; DATA_INVENTORY.md). The recordings themselves are usable under the rule; the
  convenient processed forms are not.

## 4. The roadmap

**Done.** Phases 0 to 9. Phase 6b (NEAT topology search) is deferred with no schedule, and Phase 7's
cross-species half (7b) is deferred.

**Current.** Phase 10, *fidelity on the plate* (roadmap v5.0, 2026-10-23). The project's question was
re-framed then, from "which architecture learns nematode behaviour best" to "how faithfully does a
simulated worm reproduce standard behaviour, graded against real-worm data, and what does the
connectome contribute". Blocks and status:

| Block | Status |
|---|---|
| H, housekeeping | done |
| E + T, 1-s worm, primitives, assays, batched evaluator | done (LB 104, 108) |
| B, benchmark | v4 VALID (LB 124) |
| N, data-derived neuromodulator layer | done (LB 106) |
| F.0, parameter recovery | C.1's passed (LB 112, 117); C.2's (F.0b-ii) is part of launch 133 |
| C.1 | passes v3; fails v4; refit deferred to C.4 |
| C.2 | registered fit in its readiness pilot (launch 133) |
| C.3, the same architecture trained by reward | not started |
| C.4, the fidelity table | not started |
| W, the wiring question (wild type against five null families, several fits each) | not started; sized at 450 to 700 machine hours |
| Synthesis | not started |

**After.** Phase 11 opens the body and posture (D41). Nothing beyond it is planned. The roadmap's
"Future" level lists biophysical neurons, a metabolic model and 3D, with no phase attached.

**Mapped against the brief's milestones** (the comparison itself is Part B's):

- M-a (held-out perturbation responses) and M-b (freely moving whole-brain activity on held-out
  worms) appear nowhere on the roadmap. An earlier note on research question 3 (roadmap, 2026-06-04)
  recorded neural-tuning comparison à la Atanas 2023 as the defensible validation for the
  connectome arm, and deferred it.
- M-c (pass the behaviour benchmarks with no fitting to them) conflicts with the method: the
  benchmarks are the fit's objective.
- M-d (a neuromodulator-driven state switch) is close in behavioural terms: C.2's probes reproduce
  roaming and dwelling with the right tph-1 and tax-4 directions, but fitted, with the mutants inside
  the objective.
- M-e (experience-dependent behaviour) has no destination since Phase 7 found no biologically
  plausible rule that learns this substrate.

## 5. Pacing

### 5.1 Per phase

Boundaries for Phases 0 to 4 are my reading of commit subjects and logbook dates, accurate to about
a week. Later boundaries are the roadmap's close dates. Commits exclude merges.

| Phase | Dates | Weeks | Active days | Commits | PRs | Logbooks | Main output |
|---|---|---|---|---|---|---|---|
| 0 Foundation | 2025-04-16 to 2025-12-29 | 37 | 86 | 1,016 | 20 | 001 to 004 | six architectures, PPO baseline, CMA-ES for quantum circuits, first QPU run |
| 1 Sensory and threat | to 2026-01-26 | 4 | 16 | 198 | 9 | 005 to 007 | thermotaxis, mechanosensation, predators, health |
| 2 Architecture analysis | to 2026-03-08 | 6 | 30 | 310 | 22 | 008 | 300-session quantum campaign: no quantum advantage |
| 3 Temporal sensing | to 2026-04-04 | 4 | 23 | 289 | 35 | 009, 010 | STAM, LSTM/GRU, aerotaxis |
| 4 Multi-agent | to 2026-04-25 | 3 | 13 | 175 | 13 | 011 | pheromones; no genuine multi-agent complexity |
| 5 Evolution | to 2026-05-23 | 4 | 27 | 408 | 52 | 012 to 021 | Lamarckian inheritance positive; Baldwin, co-evolution, transgenerational memory STOP |
| 6a Connectome substrate | to 2026-07-07 | 6.5 | 37 | 422 | 87 | 022 to 037 | connectome imported; connectome ranks 5th of 6, a degree-statistics property (LB 034) |
| (pause) | to 2026-08-26 | 7 | 6 | 71 | 18 | none | README, release v0.5.0 |
| 7 Plasticity | to 2026-09-19 | 3.5 | 18 | 362 | 91 | 038 to 069 (25) | SPLIT: no plausible rule learns the substrate; a PPO wiring advantage, later conditioned and then dissolved |
| 8 Ground, then embody | to 2026-10-14 | 3.5 | 21 | 287 | 57 | 070 to 094 (25) | measured prior (no move), body, lawns; modulators do not phenocopy; a gap-placement effect |
| 9 Screen, then mechanism | to 2026-10-22 | 1 | 8 | 84 | 12 | 095 to 103 (9) | STOP: the gap-placement effect was the gap model's degree normaliser |
| 10 Fidelity on the plate | 2026-10-23 to now | 3 so far | 19 | 262 | 61 | 104 to 132 (29) | benchmark v1 to v4, primitives, batched evaluator, data-derived modulator layer, C.1 and C.2 fitting |

### 5.2 What the record shows

- **Throughput is not the bottleneck.** Since August 2026 a phase takes about three weeks, with 250 to
  400 commits, 50 to 90 PRs and up to 29 logbooks. Phase 10 has had 19 active days on `main` out of 23.
- **Yield is.** Phase 5 had one real positive (Lamarckian inheritance) beside three STOPs. Of the four
  phases after it, two closed STOP or SPLIT on their primary question (7, 9). The positive headlines of
  the other two were later traced to properties of the model: Phase 6a's connectome ranking to degree
  statistics (LB 034), and the wiring advantage carried from Phase 7 into Phase 8 first to an operating
  point (LB 071), then, in its surviving gap-placement part, to the gap model's normaliser (LB 103).
  The research question of Phases 0 to 9 was then retired (v5.0).
- **Phase 10 repeats the pattern inside one phase.** The benchmark went through four versions in 12
  days. C.1's verdict changed with every version (pass, fail, pass, fail). C.2 needed about 40 probe
  fits and a dozen encoder options before it passed, and its registered fit's pilot has been amended
  twice.

### 5.3 Recurring bottlenecks

1. **Artefacts surface a phase late.** The control that exposes a result's artefact has repeatedly
   been the next phase's first task, so a phase's output is provisional until the next one closes.
2. **The instrument keeps moving.** In Phase 10 each controller result prompted a benchmark revision.
3. **Behaviour under-determines the model.** Lawn statistics do not choose internal routes (LB 127,
   128). Each fix imports a constraint from biology or moves a held-out test into the objective.
4. **Machine time.** One C.2 fit is about 28 hours on the whole machine. Each positive control is
   another full fit. W is sized at 450 to 700 hours.
5. **Positive controls fail first.** F.0a (NPE), F.0b-i on v2's first attempt, the slow-noise control
   (LB 123) and Phase 7's rule positive control each failed on first run and needed a redesign.
6. **Rigour costs time, deliberately.** Registration, spec review, preflight and oracles take about
   1.5 to 2 days of engineering per change. They are also how every artefact above was caught.

### 5.4 Time remaining under the current approach

**To close Phase 10.** Remaining machine time is about 600 to 850 hours, or 25 to 35 days of the
machine at full use. That covers the readiness pilot, F.0b-ii and the registered C.2 fit (launch 133
projects about 117 hours), C.1's refit, C.3 and W (450 to 700 hours). C.1's refit and C.3 have not
been costed. Assumptions:

- one machine;
- W at the C.2 fit plan's 4 null arms with 4 to 6 fits each;
- no further benchmark version;
- registered steps amended at Phase 10's observed rate, about twice each before they run;
- 5 to 8 more OpenSpec changes at 1.5 to 2 days each, overlapping compute.

On those, Phase 10 closes 8 to 14 weeks from now, between early January and late February 2027. Six
weeks is the floor if W moves to rented capacity or shrinks.

**To the brief's long-run goal.** This cannot be read off the roadmap. After Phase 11's body, nothing
is planned that adds neural dynamics, grades against neural activity, or fits plasticity and local
learning. A rough bound, as an assumption: if each of those three layers takes one phase of 4 to 8
weeks at the Phase 10 rate, and closes negative as often as Phases 6a to 9 did, reaching the goal
takes more than a year of calendar time, probably closer to two. Its endpoint would still not include
agreement with neural recordings, because no milestone on the current path grades it.

## 6. Could the simulator be made differentiable, and at what cost?

### 6.1 What exists

| Component | Framework | Gradients today |
|---|---|---|
| Wiring composition and the settling map | PyTorch | yes; the encoded settling switches to `torch.tanh` when a gradient is taken |
| Modulator current from concentrations | PyTorch | yes, with respect to the strengths |
| Modulator pool update across steps, timescales | NumPy, detached; timescales are buffers | no |
| Leaky dynamics with BPTT chunks (one τ for all neurons; refused under the encoder) | PyTorch | yes, within a chunk |
| Environment, locomotor primitives, measurement model, grading | NumPy | no: Bernoulli bout starts, categorical kinds, gamma durations, clamps, HMM decoding, counts and z-caps |
| The fitter | `cma`, black-box | not applicable |

The one-step map has been fitted by gradient descent once: LB 126 trained C.2 to imitate C.1's
actions from 720,000 state-action pairs in about 3.3 hours. It reached R² about 0.3 on speed and the
reversal hazard, then left the lawn in closed loop (D_fit 8.0). The logbook's diagnosis: a memoryless
student learns the teacher's average action given the senses, not its roaming or dwelling state.
That shows both that open-loop gradient fitting here is cheap and that this architecture lacks the
state a neural fit would need.

### 6.2 The options

The efforts are judgements from the code's structure, for one engineer who knows the codebase.

| Option | What it is | Effort | Main risk |
|---|---|---|---|
| (a) A differentiable neural model only, fitted open-loop to recorded stimuli and activity | New: a continuous-time rate model with per-neuron or per-class time constants, biases and gains, possibly per-edge weights on the connectome mask; modulator pools inside autograd; a calcium observation model; per-cell stimulus and optogenetic inputs; data ingestion; a training loop; held-out evaluation. Reused: loaders, signs, neuron tables, the modulator layer and its knockouts, the degree-preserving rewiring, the wiring composition | 1,000 to 2,500 new lines; about 2 to 5 weeks; small compute | licensed data; the unit bridge from a calcium-scale model to the closed-loop readout |
| (b) A PyTorch adapter that runs the existing brain and evaluator differentiably | Brain side small (100 to 300 lines). Evaluator side large: relax the discrete bouts (Gumbel-softmax, reparameterised durations) or use score-function gradients; soften clamps; replace grading with surrogates | about 6 to 12 weeks | gradients through 2,400 to 5,400 stochastic steps; surrogates are not the registered benchmark; parity and byte-identity checks to re-earn |
| (c) A rewrite, for example in JAX | Port brain, evaluator, primitives and grading surrogates, about 6,000 to 8,000 lines | about 3 to 6 months | loses the byte-identity link to about 30 logbooks' records; (b)'s obstacles remain |

The key point for fit-first: **it needs (a), not (b) or (c).** A neural fit runs open-loop on
recorded stimuli and recorded activity, so the environment never enters the gradient. The closed loop
is used only to test behaviour, and it can stay the existing non-differentiable NumPy evaluator, as
it is today. JAX is not a dependency; PyTorch is.

A middle path the audit noted, untested: the primitives' bout starts are competing Poisson hazards,
so the exact log-likelihood of a real track's runs and bouts, given senses reconstructed along it, is
differentiable. That would fit the controller to recorded behaviour by gradient, open-loop. LB 126's
imitation is its nearest precedent, and shows the memoryless brain's limit.

### 6.3 What fitting to neural data would need that does not exist

- recordings ingested from a source the project's licence rule accepts (none are vendored; the
  Randi atlas and Dunn 2025 are CC-BY-4.0 on DANDI, but only as large NWB files);
- a calcium observation model: indicator kinetics, a per-neuron scale, offset and nonlinearity,
  noise, bleaching, the sampling rate;
- neural time constants and persistent state (the settling model resets every second);
- a treatment of unrecorded neurons (Creamer's table covers 125 of 302 and no body motor neuron);
- cell-specific stimulus channels in place of one food signal shared by 11 classes;
- for freely moving data, a map from recorded behaviour to the model's motor side (the readout pools
  ventral-cord classes that head imaging rarely records);
- a unit bridge if fitted weights are reused in the rate model.
