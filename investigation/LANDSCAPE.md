# Landscape: an independent review

Written 2026-11-14. It reports what others have done; it gives no verdict on fit-first.

## How this was compiled

- **Sources.** Three research agents searched and read independently: worm neural-data fitting;
  analogues in other organisms, identifiability and tooling; and neuromodulation, plasticity and the
  calcium link. Each was told to cite only what it opened. I then re-opened the sources that carry the
  most weight in this review and checked the quoted claims myself. Those are marked **checked**.
- **Access levels** are given per source in the table at the end, as **F** (full text: methods and
  results read), **A** (abstract or summary page only), **S** (second-hand: a report about the work)
  or **R** (a code repository's README). Figures were never visible, so numbers that appear only in a
  figure are missing.
- **Coverage limits.** Nature, PMC, Cell and eLife pages mostly would not load, so journal versions
  were read through Europe PMC, bioRxiv or arXiv copies where they existed. PDFs could not be parsed,
  so some classics stay at abstract level. Each agent's web-search budget ran out near the end of its
  sweep, so work from late 2025 to November 2026 is under-covered.

## 1. Fitting connectome-constrained models to worm neural data

### 1.1 Fits to perturbation data (stimulate one neuron, record the rest)

**Creamer, Leifer and Pillow (bioRxiv 2024.09.22.614271, v1 2024 to v4 2026; F, checked).** The only
fit to the worm's perturbation data.

- **Data:** Randi et al.'s atlas: 110 immobilised animals, nuclear GCaMP6s with NeuroPAL identities,
  one head neuron stimulated optogenetically every 30 s. The model holds 154 head neurons.
- **Model:** a latent linear dynamical system at 2 Hz. Each neuron's next activity is a weighted sum
  of all neurons' activity plus a filtered history of its own stimulation. Weights are allowed only on
  connectome pairs (chemical and gap summed, about 9.4% of pairs). Calcium is treated as the state;
  there is no indicator model.
- **Fitting:** expectation maximisation with a Kalman smoother, not gradients. Unrecorded neurons are
  missing values in the latent state.
- **Validation:** 80 animals train, 30 held out. The score compares in-silico single-neuron
  stimulation with held-out animals' stimulus-triggered responses, divided by the data's own
  train-test agreement.
- **Numbers (v4):** model against data 0.14, data against itself 0.17, so "82% as well as the
  reproducibility". v3 reported 92%. Masked neurons in held-out animals are reconstructed at mean
  r about 0.25.
- **Controls:** a shuffled connectome (rows and columns relabelled) "performed poorly". A fully
  connected model "performed no better" on perturbation responses, and slightly better on correlation
  structure. Its edge on putative extrasynaptic pairs went from p = 0.011 (v3) to p = 0.1 (v4).
  Raw synapse counts correlate poorly with the fitted weights.
- **Stated limits:** linear and time-invariant (no bistability or limit cycles), no neuromodulation
  or state dependence, single-neuron stimulation only, calcium as a proxy. On synthetic data with
  strong extrasynaptic signalling, the constrained model fails.
- **Not reported:** a held-out-stimulation design (all stimulations of one neuron left out). The
  reviewer found none.

**Randi, Sharma, Dvali and Leifer 2023, Nature 623:406 (F via arXiv, checked).** The perturbation
atlas itself.

- 23,427 neuron pairs measured; at least 1,314 (6%) functionally connected under stringent criteria.
- Anatomy-based simulations agreed poorly with measurement: "fairly poor agreement", with R² < 0 over
  all pairs. Fitting weights and signs to the data improved agreement but left it poor.
- In unc-31 mutants (no dense-core vesicle release), 53 connections were lost, which the authors treat
  as a lower bound on extrasynaptic signalling. "Extrasynaptic signaling not visible from anatomy
  contributes to this difference."
- Responses vary between trials in presence, amplitude, kinetics and even sign.

**Toyoshima et al. 2024, PLoS Comput Biol (F).** 24 restrained animals under salt switches, a
connectome-defined input set per neuron, fitted per animal. Its virtual optogenetics matched Randi's
responses at r = 0.25 to 0.49, rising with the number of trials behind each reference pair. A
deterministic version decayed to flat activity, and realistic dynamics needed added noise. Sensory
routes differed between animals.

**Dvali, Seguin, Betzel and Leifer 2025, PRX Life (A).** The signalling network's modules and rich
club differ from the anatomical ones; structural hubs need not be influential.

### 1.2 Connectome-constrained fits to spontaneous or behaving activity

- **Mi et al., ICLR 2022, "Connectome-constrained latent variable models of whole-brain neural
  activity" (F, checked, from the Janelia-hosted PDF; code TuragaLab/wormvae, R).** The one
  gradient-fitted connectome-constrained worm model with a held-out test.
  - *Data:* Yemini et al. 2021: 21 immobilised NeuroPAL worms, 170 head neurons, 5 min at 4 Hz,
    presented with 2-butanone, 2,3-pentanedione and NaCl.
  - *Model:* passive point neurons with latent voltages for all 300 neurons, chemical (current- or
    conductance-based) and electrical synapses, a calcium observation model, fitted as a variational
    autoencoder by gradient, with a black-box model of sensory tuning. Nine variants, from learned
    dense through connectome sparsity to connectome counts.
  - *Held-out neurons* (each of 107 bilateral pairs withheld in turn, 4 seeds): conductance-based with
    connectome counts r = 0.319 ± 0.016; with connectome sparsity 0.262; learned dense 0.008;
    current-based learned dense −0.104. A learned sparse (L1) model did worse, so the gain comes from
    the topology, not the parameter count. Predictions were better for interneurons and motor neurons
    than for sensory neurons.
  - *Held-out worms* (15 train, 6 test): every variant scored r = 0.44 to 0.47 (connectome counts 0.474,
    current-based dense 0.465). On whole held-out animals, the connectome made almost no difference.
    The authors read the dense model as "an equally accurate but mechanistically incorrect" fit.
  - *No wiring null* beyond the unconstrained variants.
- **Morrison and Young 2025, PLoS Comput Biol (F).** 15 premotor neurons with cubic intrinsic terms;
  gap weights fixed from White 1986; chemical weights and signs regressed on Atanas 2023's freely
  moving data, with 112 other recorded neurons fed in as inputs. No held-out split. It reproduces
  dwell times qualitatively; in-silico ablations partly disagree with the literature; the authors write
  that "identifiability has not been established".
- **BAAIWorm (Zhao et al. 2024, Nat Comput Sci; F).** 136 multicompartment neurons, channel
  parameters fitted to patch-clamp recordings of five neurons and shared across groups; weights and
  signs fitted by approximate backpropagation through time to one 65-neuron correlation matrix (MSE
  0.076). No held-out data; the authors call the parameters "one of many solutions". The body's
  readout is trained, and the network's contribution is not separated from it.
- **Warrington, Spencer and Wood 2019, arXiv (F).** Sequential Monte Carlo to impute voltages from
  calcium in a 302-neuron model; recovered two coupling parameters on synthetic data only.
- **nih-niddk-mbs/r-elegans (R, checked).** A "differentiable, connectome-constrained C. elegans
  simulation in JAX", MIT, active to November 2026. Its goal is one shared model fitted to physiology,
  neural activity, movement and perturbation data, tested on held-out animals. Its README says it is
  "not yet an achieved biological model", and reports that in its latest fits "calcium agreement was
  essentially unchanged". Self-reported and unreviewed.

### 1.3 Anatomy against function, and variability between animals

- **Uzel, Kato and Zimmer 2022, Curr Biol (A).** Few local motifs predict activity correlations;
  non-local features (triplet motifs, shared inputs) do better; connection strength adds nothing.
- **Seyedolmohadesin et al. 2025, bioRxiv (F).** Functional connectivity is very similar between sexes
  (R² = 0.8), with no clear relation to their synaptic differences.
- **Witvliet et al. 2021, Nature (F of preprint).** "About 43% of all connections and 16% of all
  synapses are not conserved" between isogenic animals. Any model's mask is not the recorded animal's
  wiring.
- **Brennan and Proekt 2019, eLife (F of preprint).** Some identified neurons behave qualitatively
  differently across worms.
- **Atanas et al. 2023, Cell (F of preprint).** Several classes have tunings that differ across
  animals; about 20% of encoding neurons changed encoding within a recording.

### 1.4 Models without a connectome, as performance references

- **Linderman et al. 2019, bioRxiv (F).** Hierarchical recurrent switching linear dynamics on Kato
  2015 and Nichols 2017; discrete states match forward, reverse, turns and quiescence; long
  simulations "eventually become unstable".
- **Brennan and Proekt 2019 (F of preprint).** A two-variable loop-and-phase model predicts motor
  command timing up to 30 s ahead in a held-out worm.
- **Simeon et al. 2024, bioRxiv (F).** Six architectures trained on eight pooled open datasets (about
  284 worms) for next-step prediction. The best (a CTRNN) is about 9% below a copy-last-value baseline;
  transformers and feedforward models were worse than that baseline. Long-horizon generation was not
  achieved. Validation was within worm, across time only.
- **Simeon et al. 2024/2025, arXiv 2411.12091 (F).** A homogenised resource: 12 imaging studies (about
  919 worms) and 10 connectome files on Hugging Face. No models.
- **Sprague et al. 2024, bioRxiv (A).** WormID: 118 whole-brain datasets from five labs in NWB, with an
  atlas for automated neuron identification.
- **Atanas et al. 2023 (F of preprint).** CePNEM, a Bayesian per-neuron encoding model of velocity,
  head curvature and feeding in freely moving animals: 58.6% of neurons encode behaviour, against 2.1%
  in GFP controls.
- **Kato 2015, Kaplan 2020, Costa 2019, and four others (A).** Global cyclic brain states mapped to
  motor commands; nested timescales; locally linear models.

### 1.5 Connectome-only models, nothing fitted to recordings

Varshney et al. 2011 (A), Kunert, Shlizerman and Kutz 2014 (F) and Kunert-Graf et al. 2017 (A) used
uniform conductances and transmitter-rule signs, validated qualitatively against ablations. modWorm
(Kim et al. 2025, A) couples connectome, dynamics and biomechanics; what it fits could not be read.

### 1.6 The behaviour-first counterpoint

- **Wicks, Roehrig and Rankin 1996, J Neurosci (A).** Enumerated every sign configuration of the
  tap-withdrawal circuit and scored each against ablated animals' behaviour; predicted the polarity of
  7 of 9 cell classes.
- **Izquierdo and Lockery 2010 (A); Izquierdo and Beer 2013 (A).** Klinotaxis circuits evolved to
  match behaviour; they reproduced two observations they were not fitted to, and the ensemble reached
  the same behaviour by different mechanisms.

### 1.7 Proposals

**Haspel et al. 2023/2024, arXiv 2308.06578 (A).** 37 authors propose measuring each neuron's
state-dependent input-output function by optophysiology across many animals, pooled by identity, then
training models on it, at a scale of "hundreds of thousands of experiments". Its data and timeline
estimates could not be read.

### 1.8 What worked and what failed, for the worm

- **Worked.** A connectome mask is enough to reach a fully connected model's accuracy on held-out
  animals' perturbation responses, and it beats a relabelled-connectome null (Creamer). Connectome
  constraints, fitted by gradient, predict withheld neurons at r = 0.32 against about 0 unconstrained
  (Mi). Low-dimensional models predict motor-command timing in held-out worms (Brennan and Proekt).
- **Failed or weak.** Anatomy alone predicts signal propagation poorly (Randi; Uzel; Seyedolmohadesin;
  Dvali). The best perturbation fit agrees with data at r = 0.14 because the data agree with
  themselves at only 0.17 (Creamer). On whole held-out worms, connectome-constrained and dense models
  score alike (r = 0.44 to 0.47; Mi), and on perturbation responses a fully connected model does as well
  as the masked one (Creamer): the connectome shows on withheld neurons and against relabelling, not
  on held-out animals' overall fit. BAAIWorm and Morrison and Young have no held-out validation. Pooled connectome-free models barely beat copying the last value
  (Simeon). The one open differentiable JAX effort reports no calcium gain yet (r-elegans).
- **Not found.** No published work fits a connectome model to worm neural data and then scores it on
  held-out behaviour statistics.

## 2. Analogues in other organisms

### 2.1 The fly visual system and whole-brain fly models

- **flyvis (Lappalainen et al. 2024, Nature; F, checked).** Task-first, not fit-first: "We used only
  connectome and task constraints to construct our DMN, without any measurements of neural activity."
  64 cell types, 45,669 neurons, **734 free parameters** tied by cell type (a time constant and
  resting potential per type, one strength per type pair scaling synapse counts; signs fixed from
  transmitters). Trained on optic flow by backpropagation through time on clips of about 40 steps from
  a steady state. Compared afterwards with recordings from 26 studies: contrast preference right for
  32 of 32 types; of 19 types with asymmetric inputs, only 12 predicted direction-selective. Models
  with random parameters got direction selectivity wrong. Its 50-model ensemble disagreed on some
  types, and sparse recordings could select among the members.
- **Zhou and Hasler 2026, bioRxiv (F).** Untrained connectome-constrained flyvis networks against
  shuffled ones: no significant difference after correction. The connectome's contribution appears
  after training.
- **Shiu et al. 2024, Nature (F).** An unfitted spiking model of 127,400 FlyWire neurons with one
  tuned constant. 91% of 164 testable predictions held; the feeding motor neuron was driven in 100% of
  runs on the real wiring and 1 in 100 on shuffled weights. Errors concentrated where gap junctions,
  peptides, background activity or sign assignments matter.
- **Minnick et al. 2026, arXiv 2610.33881 (F).** Shiu's untuned parameters on the brain and nerve cord
  in a body reached 95% of odour sources, against 18% for a degree- and sign-preserving rewiring, and
  2% with sensory feedback removed. Failures: a gap-junction path that could not conduct, steering that
  needs background activity, a mis-signed neuron.
- **Li, Ping, Zhang and Wang 2026, bioRxiv (F; existence and abstract checked).** The closest fit-first
  analogue found. About 15 million weight magnitudes and per-neuron time constants on FlyWire, fitted
  by full backpropagation through time with a GCaMP observation kernel to 73 brain-region signals from
  one fly, on one workstation GPU. Held-out r = 0.423 against a train-test ceiling of 0.473; a matched
  GRU reached 0.288. No shuffled-connectome control; no experimental validation.
- **Cowley et al. 2024, Nature (F).** Knockout training: a model unit is silenced when the matching
  neuron type is silenced in the behaving fly. It aligned units with neuron types, but only for types
  whose silencing changed behaviour.
- **Pospisil et al. 2024, Nature (F).** Treats the brain as a linear system and shows that stochastic
  optogenetic stimulation is what identifies causal effects; a connectome prior on the effects cuts
  error by an order of magnitude. The identifiable object is the Jacobian of the dynamics, not the
  biophysical parameters.
- **Digital sphinx (Brunton et al. 2026, eLife; S).** The worm connectome trained by reinforcement
  learning to walk a fly body: realistic walking, biologically meaningless wiring.
- **flybench (Cho 2026; R)** and **Jin et al. 2026 (A).** A pre-registered behavioural benchmark with
  wiring nulls for fly whole-brain models; and a connectome-topology controller trained by RL.

### 2.2 Other circuits

- **Zarin et al. 2019, eLife (F, partial).** A larval fly premotor circuit fitted by gradient descent
  to motor-timing targets on its EM wiring, as an ensemble. The fit exposed a missing input rather than
  supplying it.
- **Vishwanathan et al. 2024, Nat Neurosci (F).** A zebrafish oculomotor integrator with two or three
  fitted scalars predicts population statistics; a proximity-based wiring gave different results.
  Single cells could not be compared because the imaged fish were not the reconstructed one.
- **ZAPBench (Lueckmann et al. 2025; F).** 71,721 zebrafish neurons over 2 h: models beat naive
  baselines, but a stimulus-only baseline is competitive at short context. The connectome is not yet
  used.
- **Wang et al. 2025 and Ding et al. 2025, Nature (F).** In mouse cortex, prediction of activity is
  driven by data volume, not by structure, and function predicts connectivity weakly pairwise
  (r = 0.032).

### 2.3 What the analogues say about method

- The best-known "connectome predicts activity" result (flyvis) used recordings only to check and to
  select among ensemble members.
- Fits directly to activity on a real connectome are rare, recent and mostly lack a wiring null
  (Li 2026) or held-out data.
- Unfitted connectome models do well on feedforward sensorimotor paths and fail where gap junctions,
  background activity, peptides or signs matter.
- Wiring nulls are standard in the stronger papers, and an optimised controller can reproduce
  behaviour with the wrong wiring.

## 3. Identifiability and degeneracy

- **Degeneracy is the norm at conductance level.** Prinz, Bucher and Marder 2004 (A): "virtually
  indistinguishable network activity can arise from widely disparate sets of underlying mechanisms".
  Gonçalves et al. 2020 (F): in a 31-parameter pyloric model, many parameters are "only weakly
  constrained", lying on connected sets with stiff and sloppy directions; a genetic algorithm found
  one small region and missed the rest. Deistler et al. 2022 (F): extra objectives narrow the set
  without removing it.
- **Prediction becomes identifiable before parameters do.** Beiran and Litwin-Kumar 2024/2025
  (F of preprint, checked): "a connectome is often insufficient to constrain the dynamics of networks
  that perform a specific task", but "recordings from a small subset of neurons can remove this
  degeneracy": for a 300-neuron network, error on unrecorded neurons falls sharply once more than about
  30 are recorded. Yet "the unknown parameters are never precisely recovered". The number needed scales
  with the dynamics' dimensionality and grows for chaotic networks. Unmodelled peptides are named as a
  threat.
- **Connectivity from spontaneous activity is biased.** Das and Fiete 2020 (F of preprint): in strongly
  coupled networks the error is bias that "do[es] not shrink with data volume"; partial observation
  adds more. Qian et al. 2024 (F): with about 5% of neurons observed, fitted models invent attractors,
  across four fitting methods.
- **Perturbations break confounding, but only for what they perturb.** Pospisil 2024; Lepperød et al.
  2023 (F): stimulation that is not cell-specific still confounds; Timme 2007 (A): each distinct drive
  adds N constraints; Wagenmaker et al. 2024 (A): actively chosen stimulation halves the data needed.
- **Calcium caps temporal resolution.** Berens et al. 2018 (F): the best spike inference from calcium
  reaches correlation 0.44 at 40 ms bins and about 0.73 at 333 ms. Sutera et al. 2014 (F): connectivity
  from simulated fluorescence reached high AUROC but low precision and no direction.

## 4. What fit-first does not address, and how others have handled it

### 4.1 Calcium against voltage

- **Most worm neurons are graded.** Goodman et al. 1998 (F): ASER gives graded, nonlinear responses
  with no classical action potentials. Exceptions: AWA fires calcium-mediated spikes (Liu et al. 2018,
  F of preprint); AVL and DVB fire long spikes once per defecation cycle (Jiang et al. 2022, F); RMD
  shows plateau events (Mellem 2008 and Lockery and Goodman 2009, S).
- **Direct voltage-to-calcium evidence exists for one neuron.** In AWA, simultaneous patch clamp and
  GCaMP6f showed graded depolarisation appearing in fluorescence after about 1 s, while each spike
  sharpened the trace's slope at once (Liu 2018).
- **Fast voltage events can be invisible in calcium.** Rhodopsin voltage imaging in RIM recorded brief
  events of about 8% ΔF/F lasting about 120 ms that separately recorded RIM calcium lacked (Azimi
  Hashemi et al. 2019, F). Signals were small, uncalibrated and never simultaneous with calcium.
- **Indicators are slow in worm neurons.** In the touch neuron ALM, fluorescence decay half-times
  were 2.7 ± 2.0 s for GCaMP6f and 18 ± 8 s for jGCaMP7s (Ding et al. 2023, F, checked). These mix the
  neuron's own time course with the indicator's. jGCaMP8's mammalian half-decays are about 84 to
  200 ms (Zhang et al. 2023, F), but no whole-brain worm dataset opened here uses it.
- **Whole-brain datasets use slow sensors at about 2 Hz:** Randi 2023 nuclear GCaMP6s at 2 volumes/s;
  Atanas 2023 and Dag 2023 nuclear GCaMP7f at about 1.7 Hz; Ji 2021 GCaMP6m at about 2 Hz.
- **How modellers handled it.** Almost every fit to real worm data treats ΔF/F, or its z-score, as the
  state: Randi's kernels, Creamer, CePNEM, Linderman's rSLDS, Morrison and Young. Explicit observation
  models appear in Warrington 2019 (synthetic data only), Mi et al.'s latent-voltage model (fitted
  to Yemini 2021's recordings), and, in the fly, Li 2026's GCaMP kernel.

### 4.2 Intrinsic currents and expression priors

- **Few neurons have recordings.** No authoritative count was found. BAAIWorm states, uncited, that
  "less than 20%" of neurons have been recorded, and fits 5 neurons while every other neuron copies
  its group's parameters. Neurons with patch-clamp data in the sources opened: ASER, RIM, AIY, AFD,
  AWA, AWCon, RMD, AVA, AVL, DVB, VA5, VB6, VD5.
- **Expression picks the channel set, not the conductances.** CeNGEN (Taylor et al. 2021, F) profiles
  all 118 classes; at its threshold 2, true-positive rate 0.81 and false-discovery rate 0.14 against
  ground-truth genes. Nicoletti et al. 2024 (F) used it to choose each neuron's currents, then fixed
  gating kinetics from heterologous or other-species data and fitted only maximal conductances, leak
  reversal and capacitance. Nicoletti et al. 2019 (F) took some kinetics from rat and fly.
- **Fits remain non-unique.** Naudin et al. 2022 (F): fitting traces and steady-state I-V curves
  conflicts, giving a Pareto set; single-objective fits produced spurious N-shaped curves.
- **Not found:** any paper quantifying how far expression priors narrow conductance posteriors.
- **Expression also predicts signs.** Fenyves et al. 2020 (F) predicted polarity for about 73% of
  synapses from transmitter and receptor expression; against 12 interneuronal connections of known
  polarity, one was discordant, against 6 of 12 for a transmitter-only rule. The project already uses
  these predictions.

### 4.3 Neuromodulation

- **Release has not been measured.** Serotonin and PDF release were not measured in Flavell et al.
  2013 (F), Ji et al. 2021 (F) or Dag et al. 2023 (F). Their roles come from genetics, ablation,
  optogenetics and calcium. Flavell's diffusion distance is "plausible but untested".
- **What is measured is behavioural and correlational.** Dwelling bouts fit one exponential
  (τ ≈ 482 s), roaming two (75 s and 520 s) (Flavell 2013). NSM activity rises about 23 s before a
  roaming-to-dwelling transition, and 10 recorded neurons classify the state at over 95% (Ji 2021).
  Knock-in reporters place six serotonin receptors in 8 to 24 classes each (Dag 2023).
- **The maps are expression-based.** Bentley et al. 2016 (F) and Ripoll-Sánchez et al. 2023 (F) rest
  on expression, in vitro potencies and assumed diffusion ranges; edge weights count possible
  pairings, not strengths. The peptide network is more than ten times denser than the synaptic one
  (0.344 against 0.025 at short range). Beets et al. 2023 (F) validated 459 peptide-receptor pairs in
  vitro, with EC50s from about 0.1 pM to 22 µM. The project's data-derived layer is built from these.
- **No release kinetics.** A 2025 review (Beets and Watteyne, F) gives no worm peptide release
  kinetics or diffusion distances, and in vivo peptide sensors are described for mouse and fly only.
- **Randi's unc-31 data** are the nearest whole-brain measurement of peptidergic signalling, read out
  in calcium: RID drove URX, ADL and AWB with no wired path, and that response was reduced or absent
  without dense-core vesicles.
- **No fitted model with modulator dynamics in the worm.** Elsewhere, Kringelbach et al. 2020 (F)
  coupled a human whole-brain model to serotonin release with Michaelis-Menten kinetics fixed from the
  literature and two or three fitted gains; two of the gains lay on a ridge and could not be separated.

### 4.4 Plasticity and learning

- **Paradigms with neural correlates** (all F): PA14 aversive learning, where sensory responses were
  unchanged and the change lies downstream (Ha et al. 2010); butanone conditioning, where AWC's
  dynamic range shifts non-associatively and the associative change appears in the output (Cho et al.
  2016); salt memory, set by basal glutamate release from ASER and a receptor-dependent sign flip in
  AIB (Hiroki et al. 2022); thermotaxis, with AFD adapting at τ ≈ 45 s for warming and about 14 min for
  cooling (Hawk et al. 2018); tap habituation, reviewed by Bozorgmehr et al. 2013.
- **Imaging after learning compares separate groups.** Liang et al. 2025 (F): 2,445 recordings from
  856 worms covering about 78% of head classes; PA14 responses changed in about 35% of sensory types
  after training; naive and trained worms were different animals. Pritz et al. 2023 (F) imaged selected
  neurons, again in separate groups, with high variability between animals.
- **Worm learning models impose plasticity rather than infer it.** Hironaka and Sumi 2025 (F) fitted a
  five-neuron klinotaxis circuit to a behavioural chemotaxis index, with memory as an imposed sign
  switch at one synapse. A network with the opposite sign fitted equally well, so behaviour did not
  discriminate. No worm learning model with a local rule fitted to neural data was found.
- **Rule inference is unidentified even elsewhere.** Mehta et al. 2023 (F, checked) fitted plasticity
  rules by gradient through whole trajectories; on behaviour from 18 flies, the model "fails to
  identify a unique, sparse solution even with extensive data". The fly mushroom-body model of Jiang
  and Litwin-Kumar 2021 (F) takes its dopamine-gated rule from experiment. Lim et al. 2015, Confavreux
  et al. 2020 and Ashwood et al. 2020 were seen only second-hand.

### 4.5 Variability between animals

- Wiring: about 43% of connections, carrying about 16% of synapses, vary between isogenic animals
  (Witvliet 2021).
- Encoding: some classes are consistent across animals (AVA, AIM), others not (OLL, OLQ, IL1, RIH,
  URB); generalisation across animals tracks that (Atanas 2023).
- Averaging a neuron across animals destroyed behavioural information in Kato's data; aligning animals
  on a shared low-dimensional manifold instead predicted new animals' switch times at r = 0.74
  (Brennan and Proekt 2019, F).
- How it was handled: one pooled parameter set (Creamer; Morrison and Young, who averaged per-animal
  fits), per-animal fits (Toyoshima; CePNEM), or hierarchical models with per-worm parameters around
  a shared mean, which needed 8 states instead of 14 and generalised better (Linderman 2019).

## 5. Differentiable tooling and cost

| Tool | What it simulates | Gradients | Status | On Apple Silicon | Licence |
|---|---|---|---|---|---|
| Jaxley (Deistler et al. 2025, Nat Methods; F) | multicompartment and point neurons; ionotropic and rate synapses; gap junctions not confirmed | autodiff through implicit Euler with multilevel checkpointing; a gradient costs 3 to 20 forward runs | published; used on data up to about 1 s; long horizons named as the open problem | JAX CPU | Apache-2.0 |
| flyvis (PyTorch) | graded point neurons on a connectome | BPTT on short clips | published, pretrained ensembles | not stated | MIT |
| BrainPy and BrainX (JAX) | rate and spiking networks | BPTT with checkpointing; an online method | used for Li 2026's 138,639-neuron fit | JAX CPU | BrainPy's licence is inconsistent (Apache on PyPI, GPLv3 badge); BrainState Apache-2.0 |
| Diffrax (JAX) | ODE and SDE solvers, implicit included | checkpointed adjoint by default | v0.7.2, February 2026 | JAX CPU | Apache-2.0 |
| Dynamax (JAX) | HMMs and linear or nonlinear state-space models | EM, SGD, HMC | JOSS 2024 | JAX CPU | MIT |
| wormvae (PyTorch) | worm connectome-constrained latent model | variational | ICLR 2022; looks dormant | CPU | not visible |

- **Apple Silicon.** JAX runs on M-series CPUs. A JAX maintainer advised in January 2026 to "consider
  the jax-metal project dead" (issue #26968, F). PyTorch's MPS backend has no float64.
- **Named practical problems.** Gradients through long horizons explode or vanish, worst for chaotic
  dynamics; single shooting flattens oscillatory fits, which multiple shooting fixes; gap junctions add
  stiffness that favours implicit steps; a calcium observation model is needed; one best fit
  misrepresents a degenerate solution set. flyvis used short clips from steady states; Jaxley used
  truncated backpropagation; Li 2026 used full-segment backpropagation at the imaging interval.
- **Compute.** No published wall-clock figure for fitting a roughly 300-neuron connectome model by
  gradient was found. A 300-neuron state is about 13,000 times smaller than Jaxley's benchmark network,
  so cost is set by horizon, restarts and ensembles rather than per-step arithmetic (the reviewer's
  inference). Cloud prices seen on 2026-11-14: A100 $1.19 to $2.50 per hour and H100 $1.99 to $4.89
  per hour across RunPod, Lambda and Modal (Lambda's page says its prices take effect 2026-11-16).

## 6. What remains open

Each item is named by at least one source above.

01. **Nonlinearity and state.** The only perturbation fit is linear with one fixed point (Creamer);
    state dependence and plasticity of propagation are uncharacterised (Randi).
02. **Extrasynaptic signalling.** At least 53 connections depend on dense-core vesicles (Randi); whether
    a connectome mask can represent them is unresolved (Creamer v3 against v4).
03. **One model across animals.** Data reproduce themselves at 0.17 (Creamer); 43% of connections
    differ between isogenic animals (Witvliet); encodings and routes differ (Atanas; Toyoshima).
04. **Parameters against predictions.** Parameters are degenerate (Prinz; Gonçalves) and not recovered
    even when prediction succeeds (Beiran and Litwin-Kumar); fitted worm models say so themselves
    (BAAIWorm; Morrison and Young).
05. **Signs.** Measured signs conflict with predicted ones (Randi, AFD to AIY) and regressed signs
    conflict across studies (Morrison and Young).
06. **Calcium against voltage.** Calcium is low-pass and lagged (Simeon; Berens); few worm fits model
    the indicator.
07. **Coverage.** Perturbation data cover head neurons of immobilised animals only (Randi); about half
    the neurons go unobserved in many datasets (Toyoshima).
08. **Long-horizon generation.** Not achieved by pooled models (Simeon) and unstable in switching
    models (Linderman).
09. **From neural fit to behaviour.** Not yet done with held-out behaviour statistics in the worm.
10. **Pairwise perturbation only.** Group stimulation and nonlinear interactions are untested (Creamer;
    Randi).
11. **Data volume.** Creamer's relative score is "roughly independent of data set size"; Simeon's loss
    falls log-linearly with data; Haspel proposes hundreds of thousands of experiments.

## Sources and access levels

F: full text; A: abstract; S: second-hand; R: repository README. "Checked" means I re-opened the
source and confirmed the claim used here.

| Source | Access | Checked |
|---|---|---|
| Creamer, Leifer and Pillow, bioRxiv 2024.09.22.614271 v3, v4 | F | yes (v4) |
| Randi et al. 2023, Nature 623:406 (arXiv 2208.04790) | F (arXiv) | yes |
| Toyoshima et al. 2024, PLoS Comput Biol, 10.1371/journal.pcbi.1011848 | F | |
| Dvali et al. 2025, PRX Life 3:033021 | A | |
| Mi et al., ICLR 2022 (Janelia-hosted PDF); TuragaLab/wormvae | F; R | yes |
| Morrison and Young 2025, PLoS Comput Biol 21:e1013818 | F | |
| Zhao et al. 2024 (BAAIWorm), Nat Comput Sci 4:978 | F | |
| Warrington, Spencer and Wood 2019, arXiv 1907.11075 | F | |
| nih-niddk-mbs/r-elegans | R | yes |
| Uzel, Kato and Zimmer 2022, Curr Biol 32:3443 | A | |
| Seyedolmohadesin et al. 2025, bioRxiv 2025.05.15.654129 | F | |
| Witvliet et al. 2021, Nature 596:257 | F (preprint) | |
| Currier and Clandinin 2025, Cell (fly, not worm) | A | |
| Kato et al. 2015, Cell 163:656 | A | |
| Linderman et al. 2019, bioRxiv 621540 | F | |
| Brennan and Proekt 2019, eLife 8:e46814 | F (preprint) | |
| Costa, Ahamed and Stephens 2019, PNAS 116:1501 | A | |
| Simeon et al. 2024, bioRxiv 2024.02.13.580186 | F | |
| Simeon et al., arXiv 2411.12091 | F | |
| Sprague et al. 2024, bioRxiv 2024.04.28.591397 | A | |
| Atanas et al. 2023, Cell 186:4134 | F (preprint) | |
| Kaplan et al. 2020, Neuron 105:562 | A | |
| Yezerets et al. 2024; Morrison, Fieseler and Kutz 2020; Kinger et al. 2026; Dyballa et al. 2024 | A | |
| Varshney et al. 2011; Kunert-Graf et al. 2017; Kim et al. 2025 (modWorm); Sunil et al. 2026 | A | |
| Kunert, Shlizerman and Kutz 2014, arXiv 1310.6689 | F | |
| Wicks, Roehrig and Rankin 1996; Izquierdo and Lockery 2010; Izquierdo and Beer 2013 | A | |
| Haspel et al., arXiv 2308.06578 | A | |
| Lappalainen et al. 2024, Nature 634:1132 (flyvis) | F | yes |
| Zhou and Hasler 2026, bioRxiv 2026.06.10.731214 | F | |
| Shiu et al. 2024, Nature 634:210 | F | |
| Pospisil et al. 2024, Nature 634:201 | F | |
| Cowley et al. 2024, Nature | F | |
| Beiran and Litwin-Kumar, bioRxiv 2024.02.22.581667; Nat Neurosci 28:2561 (2025) | F (preprint) | yes |
| Litwin-Kumar and Turaga 2019, Curr Opin Neurobiol 58:94 | A | |
| Zarin et al. 2019, eLife 8:e51781 | F (partial) | |
| Vishwanathan et al. 2024, Nat Neurosci 27:2443 | F | |
| Lueckmann et al. 2025 (ZAPBench), arXiv 2503.02618; Immer et al. 2025 | F; A | |
| Wang et al. 2025, Nature 640:470; Ding et al. 2025, Nature 640:459 | F | |
| Li, Ping, Zhang and Wang 2026, bioRxiv 2026.08.21.745055 | F | yes (abstract) |
| Minnick et al. 2026, arXiv 2610.33881 | F | |
| Brunton et al. 2026, eLife RP111516 (digital sphinx) | S (The Transmitter) | |
| Cho 2026, flybench | R | |
| Jin et al. 2026, arXiv 2602.17997 | A | |
| Prinz, Bucher and Marder 2004; Marder and Goaillard 2006 | A | |
| Gonçalves et al. 2020, eLife 9:e56261 | F (partial) | |
| Deistler, Macke and Gonçalves 2022, PNAS 119:e2207632119 | F (partial) | |
| Das and Fiete 2020, Nat Neurosci | F (preprint) | |
| Qian et al. 2024, bioRxiv 2024.05.24.595741 | F | |
| Lepperød et al. 2023, PLoS Comput Biol | F (partial) | |
| Timme 2007; Wagenmaker et al. 2024; Orlandi et al. 2015 | A | |
| Sutera et al. 2014, arXiv 1406.7865; Berens et al. 2018, PLoS Comput Biol 14:e1006157 | F | |
| Deistler et al. 2025 (Jaxley), Nat Methods 22:2637 | F (partial) | |
| Diffrax, Dynamax, BrainPy, BrainState repositories and documentation | R | |
| jax-ml/jax issue #26968; PyTorch MPS documentation | F | |
| RunPod, Lambda, Modal pricing pages, 2026-11-14 | F | |
| Goodman et al. 1998, Neuron 20:763 | F | |
| Liu, Kidd, Dobosiewicz and Bargmann 2018, Cell 175:57 | F (preprint) | |
| Jiang et al. 2022, Nat Commun (AVL, DVB) | F (partial) | |
| Mellem et al. 2008; Lockery and Goodman 2009, Nat Neurosci | S | |
| Azimi Hashemi et al. 2019, PNAS 116:17051 | F (partial) | |
| Chen et al. 2013, Nature 499:295; Zhang et al. 2023, Nature 615:884 | F (partial) | |
| Dana et al. 2019, Nat Methods 16:649 | S (via Zhang 2023) | |
| Ding et al. 2023, bioRxiv 2023.03.06.531342 | F | yes |
| Nicoletti et al. 2019, PLoS One 14:e0218738; Nicoletti et al. 2024, PLoS One 19:e0298105 | F (partial) | |
| Naudin et al. 2022, PLoS One 17:e0268380 | F (partial) | |
| Taylor et al. 2021 (CeNGEN), Cell 184:4329 | F (partial) | |
| Fenyves et al. 2020, PLoS Comput Biol 16:e1007974 | F (partial) | |
| Flavell et al. 2013, Cell 154:1023 | F | |
| Ji et al. 2021, eLife 10:e62889 | F (partial) | |
| Dag et al. 2023, Cell 186:2574 | F (preprint, partial) | |
| Bentley et al. 2016, PLoS Comput Biol 12:e1005283 | F (partial) | |
| Ripoll-Sánchez et al. 2023, Neuron 111:3570 | F (author manuscript, partial) | |
| Beets et al. 2023, Cell Rep 42:113058; Beets and Watteyne 2025, Curr Opin Neurobiol | F (partial) | |
| Kringelbach et al. 2020, PNAS 117:9566 | F (partial) | |
| Shim, Prinz and Smolinski 2012, BMC Neurosci | A | |
| Ha et al. 2010, Neuron 68:1173 | F (partial) | |
| Liang et al. 2025, bioRxiv 2025.10.31.685731 | F (partial) | |
| Pritz et al. 2023, eLife 12:e74434 | F (partial) | |
| Cho et al. 2016, eLife | F (partial) | |
| Hiroki et al. 2022, Nat Commun 13:2928 | F (partial) | |
| Hawk et al. 2018, Neuron 97:356 | F (preprint, partial) | |
| Bozorgmehr et al. 2013, Front Physiol 4:88 | F (partial) | |
| Hironaka and Sumi 2025, eLife 104456 | F (partial) | |
| Sanders and Cohen 2012; Hasani et al. 2017 | A | |
| Mehta et al. 2023, bioRxiv 2023.12.11.571103 | F | yes |
| Lim et al. 2015; Confavreux et al. 2020; Ashwood et al. 2020 | S (via Mehta 2023) | |
| Jiang and Litwin-Kumar 2021, PLoS Comput Biol 17:e1009205 | F (partial) | |
| Eschbach et al. 2019 preprint (2020 Nat Neurosci) | A | |

**Looked for and not found or not opened:** "Reimers et al. 2026" (no match found); the flybench
preprint; Haspel et al.'s full text; the full texts of Uzel 2022, Kato 2015, Mi 2022 and modWorm;
any 2024 to 2026 graph-network or foundation model of worm activity with held-out-animal results;
any nonlinear refit of the Randi atlas; flyvis's training compute; a count of worm neurons with
patch-clamp recordings; any measurement of neuropeptide release or diffusion time courses in the worm
(Laurent et al. 2018 and Ding et al. 2019 could not be opened); any worm model with neuromodulator
dynamics fitted to data; whole-brain imaging of the same animals before and after learning; Zhang, Lu
and Bargmann 2005 and Kunitomo et al. 2013 (search snippets only, so not used).
