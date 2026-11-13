# Literature review: analogues, identifiability, and differentiable tooling (for the fit-first investigation)

Compiled 2026-11-14 from web search and web fetch only. No local repository was read.

**Scope.** Three parts: (1) what connectome-plus-activity modelling achieved in organisms other than *C. elegans* and what it needed; (2) whether circuit parameters can be identified from activity data; (3) what differentiable simulation costs and where it breaks. Worm-specific neural-fitting papers belong to another reviewer; here they appear only as codebases (Part 3). This report gives **no verdict on fit-first**. The "implications for method" lines say what each source implies for how such fitting would have to be done.

**Access levels.**

- **full text**: I read the article body. For long papers the fetch tool returns about 100,000 characters per call, and I read one to three such windows. Where a later window was not read, the entry says "partial".
- **abstract**: I read only the abstract, a bibliographic record, or a summary page.
- **second-hand**: I read only a report about the work, and the entry names where.

**How I reached publisher pages.** Nature pages redirected to a login, PMC showed a browser check, and eLife pages failed to load. I read open-access journal versions through the Europe PMC full-text XML service instead, which serves the same published text. PDFs could not be parsed by the fetch tool. The brief limited me to web search and fetch, so I did not parse them locally, and some sources stay at abstract level for that reason.

**My own inferences** are marked *(my inference)*. Everything else is reported from the source.

______________________________________________________________________

## Part 1. Analogues in other organisms

### 1.1 Lappalainen et al. 2024, *Nature* 634: "Connectome-constrained networks predict neural activity across the fly visual system" (flyvis)

- **Citation.** Lappalainen JK et al. (affiliations include Tübingen and HHMI Janelia, per a search summary). *Nature* 634:1132–1140 (2024), doi 10.1038/s41586-024-07939-3, PMCID PMC11525180. The volume and pages come from a search-result summary. The DOI and PMCID come from Europe PMC. I did not check the full author list this session.
- **Access.** Full text of the published version (Europe PMC XML, three windows, partial: about 300k of 442k characters). Also the bioRxiv v1 full text (10.1101/2023.03.11.532232, two windows).
- **What was done.**
  - **Not fitted to recordings.** The model was trained on a task, optic-flow estimation from Sintel video, by backpropagation through time (BPTT). It was then compared with published recordings from 26 studies. The paper says it was built "without any measurements of neural activity".
  - **Size.** 64 modelled cell types, tiled over a hexagonal lattice of 721 columns, give 45,669 neurons and 1,513,231 connections. Connectivity comes from the FIB-25 and FIB-19 reconstructions (1,801 neurons).
  - **What the connectome supplies.** Synapse counts between cell types, and signs inferred from neurotransmitter and receptor expression.
- **Free parameters: 734, all tied by cell type.**
  - Two per cell type: a membrane time constant and a resting potential.
  - One per type-to-type connection: a unitary synapse strength that multiplies the connectome's synapse count.
  - The preprint itemises this as 65 + 65 + 604, with 2,355 synapse-count values and 604 signs taken from the connectome.
  - For comparison, the preprint computes that a fully unconstrained RNN of this size would have 2,196,421,955 parameters, and its baseline was 5 unconstrained CNNs of 414,666 parameters each.
- **Training details.**
  - Euler integration. The paper found no gain from steps smaller than 20 ms, and evaluation simulations used 5 ms steps.
  - Time constants clamped at or above Δt.
  - 19-frame clips (about 792 ms), each started from a steady state reached after 500 ms of grey.
  - Adam, learning rate 5×10⁻⁵ decayed to 5×10⁻⁶ in ten steps, batch size 4. Models generalised after about 250,000 iterations.
  - Software: PyTorch.
- **How predictions scored.**
  - **ON/OFF contrast preference.** The ensemble median was right for all 32 cell types with experimentally known preference, and the best single model for 30 of 32. The preprint reported 31 of 31 and 29 of 31.
  - **Direction selectivity.** T4 was predicted ON-motion selective and T5 OFF-motion selective, as known. Of 19 cell types with asymmetric multicolumnar inputs, only 12 were predicted to be motion selective.
  - **Task performance versus realism.** Across models, task error correlated with the realism of the direction-selectivity index (DSI): r = 0.60, P = 2.6×10⁻⁶.
  - **Correlation with recorded tuning curves.** Fig. 2d plots predicted motion-tuning correlations with measured curves, taking the maximum over six speeds. I could not extract the values from the text.
- **Connectome versus task.**
  - Random-parameter models with only the decoder trained got contrast preference right but direction selectivity and preferred direction wrong. In the preprint, task-trained models beat random ones on DSI with P = 2×10⁻¹¹.
  - Models with only cell-type-level connectivity "predicted neural activity poorly".
  - The authors credit the connectome with the parameter reduction ("Knowledge of the connectome played a critical role").
- **Ensemble variability.**
  - 50 models, all trained from the same initial values; the ten best were analysed.
  - Task performance was "similar (but not identical)", and the ensemble found "a variety of parameter configurations".
  - Some cell types' responses clustered: T4c into 3 clusters with mean task errors 5.297, 5.316 and 5.357; TmY3 into 4 clusters (preprint).
  - Sparse recordings can select among the ensemble. Choosing models by measured T4c direction selectivity "is sufficient to correctly constrain the tuning of both Mi4 and Mi9 neurons".
- **Sparsity (synthetic networks).** With connectivity alone, the median correlation between true and model tuning was 0.85 at 10% connectivity and 0.38 at 80%. With a noisy estimate of connection strength as well, it was above 0.9 at every density.
- **Compute.** Not stated in any part I read. The flyvis repository and documentation also give no training time or hardware.
- **What failed or was left out.** Tm4 was mispredicted. Unrectified cells (R1–R8, L1, L2) are not captured by the contrast index. Electrical synapses, nonlinear chemical synapses, neuromodulation and the large tangential neurons were omitted. Unitary synapse strength stays unknown, and only relative strengths come from counts.
- **Implications for method.**
  - The flagship "connectome predicts activity" result is task-first, not fit-first. Recordings were used only for validation and, at most, to select among ensemble members.
  - Its parameterisation (time constant and resting potential per type, one scale per type pair, signs fixed) is a worked example of tying parameters by class.
  - The ensemble clustering shows that the task leaves some cell types' responses undetermined.
  - The sparsity result suggests that a sparse connectome constrains function more than a dense one.

### 1.2 Zhou & Hasler 2026, bioRxiv 10.64898/2026.06.10.731214 (v6, 25 June 2026): "Representational geometry as a fidelity metric for connectome-constrained networks"

- **Access.** Full text (first window).
- **What was done.** The 50 pretrained flyvis models were compared with sign-preserving shuffled networks (n = 50), filtered for stability, using representational similarity analysis and CKA. The fetched summary calls the flyvis models "leaky integrate-and-fire networks". Lappalainen describes threshold-linear graded units, so either the paper or the extraction is wrong on this point.
- **Results.**
  - Trained connectome-constrained models versus random ones: RSA r = 0.686 on ON edges and 0.846 on ON+OFF edges.
  - Against a reconstruction of T4/T5 tuning: 0.930 for connectome-constrained and 0.603 for random.
  - The 24-condition biological comparison was not significant (r = 0.049).
  - Untrained connectome-constrained versus shuffled: r of about 0.2–0.26, not surviving correction.
  - Only 5 of 50 random networks passed the stability filter on the first try.
- **Implications for method.**
  - The connectome's contribution appears mainly after training, and untrained structure alone is weak.
  - The stability filter on the null networks can itself shape the result.
  - The biological reference data were the binding constraint: 8 of 65 cell types, on mismatched stimuli.

### 1.3 Shiu et al. 2024, *Nature* 634: "A *Drosophila* computational brain model reveals sensorimotor processing"

- **Citation.** Shiu PK et al. *Nature* 634:210–219 (2024), doi 10.1038/s41586-024-07763-9, PMCID PMC11446845. Volume and pages are from a search summary.
- **Access.** Full text (Europe PMC XML, first window, partial).
- **What was done.**
  - Leaky integrate-and-fire model of all 127,400 proofread FlyWire neurons (materialisation v630), with alpha synapses, in Brian2.
  - Biophysical constants were taken from the literature. The one free parameter, W_syn = 0.275 mV per synapse, was set so that 100 Hz of sugar-receptor activation gives about 80% of maximal MN9 firing.
  - Signs: a neuron is inhibitory if more than half its presynaptic sites are predicted GABA or glutamate. Treating glutamate as excitatory raised the optogenetic false-positive rate from 1% to 16%.
- **Data and compute.**
  - No fitting to activity beyond setting W_syn.
  - Each experiment ran 30 simulations of 1,000 ms. Sugar-neuron activation took about 5 minutes per trial per CPU thread.
- **What worked.**
  - **Connectome versus shuffled.** With the real connectome, MN9 activation was robust in 100% of simulations; with shuffled weights, in 1 of 100.
  - **Optogenetic screen of 106 cell types.** Of 11 predicted to activate MN9, 10 drove proboscis extension. Of 95 predicted not to, 4 showed some extension.
  - **Overall.** 91% of 164 testable predictions held, or 84% excluding the split-GAL4 screen.
  - **Robustness.** Predictions were 90.2% and 95% consistent at W_syn −30% and +30%.
- **What failed.**
  - The model has no gap junctions, graded neurons, neuromodulation, peptides or internal state.
  - With zero basal firing, inhibition onto silent neurons has no effect, which the authors link to misses on inhibitory types (Phantom, Tentacular).
  - A suspected peptidergic cell (Usnea) was mispredicted.
  - The authors treat absolute rates as unreliable and interpret only relative differences.
- **Implications for method.**
  - An unfitted, one-knob model can predict necessity and sufficiency on mostly feedforward sensorimotor paths.
  - A shuffled-weight null is cheap and decisive.
  - Errors concentrate where the model leaves out mechanisms (inhibition needing background activity, peptides, sign errors). Those are where fitting would have to add detail.

### 1.4 Pospisil et al. 2024, *Nature* 634: "The fly connectome reveals a path to the effectome"

- **Citation.** Pospisil DA et al. *Nature* 634:201–209 (2024), doi 10.1038/s41586-024-07982-0, PMCID PMC11446844.
- **Access.** Full text (first window, partial).
- **What was done.**
  - The brain is approximated as a VAR(1) linear system, and its matrix W of causal effects is the "effectome".
  - Stochastic optogenetic stimulation serves as an instrumental variable (IV). Passive recordings cannot separate causation from shared input, and in simulation least squares stayed biased where IV converged.
  - **IV–Bayes estimator.** A Gaussian prior is placed on each weight, with mean proportional to signed synapse count. Unconnected pairs are pulled strongly toward zero. The estimator stays consistent if the prior is wrong, and in linear simulations it had at least an order of magnitude lower error than raw IV.
  - **Validation.** Linear simulations of 121,327 neurons. On a conductance-based model, IV converged to the Jacobian of the dynamics (r² = 0.97 in the example shown).
- **What it needs.**
  - Stimulating one neuron while observing the whole brain identifies that neuron's outgoing weights.
  - The full experiment needs technologies not yet combined: sparse expression, voltage imaging, holographic stimulation and connectome identification. The authors propose piecemeal experiments across genetically identical flies.
- **Dominant circuits.**
  - The connectome's eigenvalues decay slowly: the 1,000th is about one tenth of the largest.
  - The top 10 eigenvectors involve about 50 neurons, eigenvectors 10–100 about 500, and 100–1,000 about 1,500.
- **Limitations.** Only linear dynamics, at a fixed 1 ms interaction timescale. No peptide signalling. The eigencircuits are hypotheses.
- **Implications for method.**
  - Perturbation data are what make causal connectivity identifiable.
  - The connectome is best used as a prior on effects, not as fixed weights.
  - The identifiable object is a local, state-dependent linearisation (the Jacobian), not the biophysical parameters.

### 1.5 Cowley et al. 2024, *Nature*: "Mapping model units to visual neurons reveals population code for social behaviour"

- **Citation.** Cowley BR et al., *Nature* (published online 22 May 2024), doi 10.1038/s41586-024-07451-8, PMCID PMC11136655.
- **Access.** Full text (first window, partial).
- **What was done.**
  - Architecture: a deep network running from a vision module, through a 23-unit bottleneck (one unit per lobula columnar (LC) neuron type), to a decision network.
  - **Knockout training.** When the model trains on data from a male with one LC type genetically silenced, the matching unit is set to zero.
  - The model "never had access to real LC responses during training".
- **Data.** Courtship behaviour with each of 23 LC types silenced (more than 12 sessions per type; 459 evaluation sessions; 27,000 test frames per type, at 30 Hz). Ten random initialisations were trained.
- **Results.**
  - Held-out behaviour: R² about 0.5 for forward velocity and about 0.15 for lateral and angular velocity.
  - Calcium imaging of 5 LC types: noise-corrected R² about 0.35, rising to about 0.65 with a fitted linear mapping.
  - The knockout network predicted neural responses better than dropout-trained, no-knockout and untrained networks (p < 0.05, n = 27). Under a linear mapping the trained networks were similar, so only knockout training aligns units with individual neuron types.
  - In simulations, knockout training recovered the one-to-one mapping only for units whose silencing changed behaviour.
- **Limitations.** The network is deterministic. Unit sign and scale are not identifiable. Only 23 of 57 LC types were silenced. Recordings came from passively viewing flies.
- **Implications for method.** Perturbation (knockout) data identify only the components whose perturbation has a measurable effect. Units whose silencing changes nothing stay unidentified.

### 1.6 Beiran & Litwin-Kumar: "Prediction of neural activity in connectome-constrained recurrent networks"

- **Citation.** bioRxiv 10.1101/2024.02.22.581667 (v1; v2 with figures dated 28 May 2024). Journal version: *Nature Neuroscience* 28(12):2561–2574 (2025), doi 10.1038/s41593-025-02080-4, PMCID PMC12648571 (from the Europe PMC record).
- **Access.**
  - Preprint: full text of v1 and v2 (first windows).
  - Journal version: bibliographic record only, because its full-text request returned HTTP 500.
  - COSYNE 2025 abstract: read.
- **What was done.**
  - Teacher–student setup. The student has the teacher's connectivity J, and only single-neuron gains and biases (about 2N unknowns) are fitted, by BPTT in PyTorch with Adam (1,400–7,000 epochs).
  - Comparison: students that instead fit their weights.
- **Results.**
  - "A connectome is often insufficient to constrain the dynamics of networks that perform a specific task."
  - Recording a subset removes this degeneracy. For a 300-neuron task-trained teacher, error on unrecorded neurons falls sharply above about M = 30 recorded neurons. In the linear case the threshold is the connectivity rank (D = 60 in the example).
  - The required number scales with the dimensionality of the dynamics, not with N. It grows with N for chaotic networks.
  - Students with trained connectivity "cannot predict unrecorded activity" even when most neurons are recorded.
  - **Single-neuron parameters "are never precisely recovered"**, even when unrecorded activity is predicted well.
  - Moderate mismatch in the activation function or connectivity is tolerated ("some degree of mismatch is tolerable"). Systems dominated by unmodelled processes such as peptides, or by spike synchrony, may fail.
  - Neurons to record can be ranked. In the nonlinear example, the best 10 chosen this way gave 60% lower error than random choice.
  - The COSYNE 2025 abstract applies the framework to the larval *Drosophila* motor–premotor circuit and the adult head-direction system.
- **Perturbations.** The fetched preprint text has no perturbation experiments. Inputs are assumed known.
- **Implications for method.**
  - This is the closest theory to fit-first.
  - The scoring target should be held-out neurons and held-out conditions, not recovered parameters.
  - How many recorded neurons are enough depends on the dimensionality of the dynamics.
  - Recording choice can be optimised.
  - Unmodelled peptides and gap junctions are named threats.

### 1.7 Litwin-Kumar & Turaga 2019, *Curr Opin Neurobiol* 58:94–100: "Constraining computational models using electron microscopy wiring diagrams"

- **Citation.** doi 10.1016/j.conb.2019.07.007.
- **Access.** Abstract only (Janelia page summary and the NSF PAR record). The NSF PDF could not be parsed.
- **Content.** A review of approaches that build wiring-constrained models. It notes there is no standard toolset for bringing wiring constraints into models, and calls for new techniques to scale to larger datasets.
- **Implications for method.** Framing only; I have no specifics.

### 1.8 Zarin et al. 2019, *eLife* 8:e51781: "A multilayer circuit architecture for the generation of distinct locomotor behaviors in *Drosophila*" (larval)

- **Citation.** Zarin AA, Mark B, Cardona A, Litwin-Kumar A, Doe CQ. PMCID PMC6994239.
- **Access.** Full text (first window, partial; the optogenetic results were in the unread part).
- **What was done.**
  - EM reconstruction of all 60 motor neurons and 236 premotor neurons in one segment.
  - A threshold-linear recurrent model of two segments, constrained to the reconstruction (premotor-to-premotor density about 7%), with signs fixed where the neurotransmitter is known.
  - Trained by gradient descent to produce the timing of motor-neuron activity seen in calcium imaging for forward and backward crawling, with tonic command inputs.
  - An ensemble was analysed: "there is a space of models".
- **What worked.** The connectome was "sufficient to generate appropriately timed MN activity". Predicted premotor sequences were checked by dual-colour imaging.
- **What failed.** The premotor-to-motor connectome alone "is insufficient to capture the selectivity of A18b to backward locomotion". A penalty term was needed, which implies missing descending input.
- **Implications for method.** This is an early case of fitting to activity targets with an ensemble on a real connectome. It shows the fit exposing missing inputs rather than fixing them.

### 1.9 Vishwanathan et al. 2024, *Nature Neuroscience* 27(12):2443–2454: "Predicting modular functions and neural coding of behavior from a synaptic wiring diagram" (larval zebrafish)

- **Citation.** doi 10.1038/s41593-024-01784-3, PMCID PMC11614741.
- **Access.** Full text (first window, partial).
- **What was done.**
  - A linear rate model of the oculomotor integrator from EM.
  - Weights are proportional to synapse count, normalised by each neuron's total input.
  - Signs were assigned from prior studies. Contralateral interactions and interactions with the other module were omitted.
  - Fitted: one global scale β (set to give persistent activity), a second scale matching mean eye-position sensitivity, and the intrinsic time constant (τ ≈ 1 s).
- **Validation.** Population-level only, because the imaged fish differ from the reconstructed one.
  - Eye-position sensitivity distributions "matched the model predictions quite well".
  - A predicted bimodality was not clearly seen.
  - No correlation coefficients were reported.
  - A model built from potential synapses (axon–dendrite proximity) instead of the EM connectome gave "considerably different" distributions.
- **Implications for method.**
  - With two or three fitted scalars, a connectome predicts population statistics.
  - The specificity of the EM wiring mattered relative to a proximity null.
  - Single-cell comparisons were out of reach when activity and connectome came from different animals.

### 1.10 Lueckmann et al. 2025, ICLR: "ZAPBench: A Benchmark for Whole-Brain Activity Prediction in Zebrafish" (arXiv 2503.02618); and Immer et al. 2025 (arXiv 2503.00073)

- **Access.** ZAPBench: full text (arXiv HTML, first window). Immer et al.: abstract.
- **Data.** 71,721 segmented neurons recorded for about 2 hours at 914 ms per volume (7,879 timepoints), across 9 stimulus conditions. Each condition is split 70/10/20, and the taxis condition is held out entirely.
- **Task.** Forecast H = 32 steps from a context of 4 or 256 steps, scored by mean absolute error.
- **Results.**
  - Models "largely outperform naive baselines", but at short context the stimulus-only baseline is competitive: only 53 of 120 runs beat it at 32 steps.
  - Long context helps at longer horizons.
  - Multivariate time-series models did not clearly beat univariate ones, so cross-neuron information may be underused.
- **Connectome.** Not yet used. The EM volume was in proofreading, with about 45,000 cells preliminarily registered to the activity data.
- **Compute and software.** Built in JAX. The U-Net has about 18M weights and trained for 250k–500k steps; hardware was not stated. Immer et al. report that forecasting directly from volumetric video beats trace-based methods on ZAPBench.
- **Implications for method.**
  - Even with 70k neurons and 2 h of data, forecasting whole-brain activity beyond trivial baselines is hard.
  - A stimulus-only baseline belongs in any activity-prediction score.
  - The benchmark for adding a connectome to this kind of forecasting is not yet run.

### 1.11 Wang et al. 2025, *Nature* 640:470–477: "Foundation model of neural activity predicts response to new stimulus types" (mouse visual cortex)

- **Citation.** doi 10.1038/s41586-025-08829-y, PMCID PMC11981942.
- **Access.** Full text (first window, partial).
- **What was done.** A foundation core was trained on 8 mice: more than 900 minutes of natural-video responses from about 66,000 neurons in 6 areas. Across the study, about 135,000 neurons were recorded in 14 mice.
- **Results.**
  - CCnorm improved 25–46% over the previous dynamic model.
  - To reach a median CCnorm of 0.65, individual models needed more than 60 minutes of data and foundation models less than 30.
  - In the MICrONS mouse, individual models reached 0.48–0.65 and foundation models 0.58–0.76.
  - From readout weights, cell type was predicted at 32% balanced accuracy (chance 9%) and area at 68% (chance 25%).
  - The authors say "the main driver of increased performance is the much larger dataset".
- **Compute.** Not in the part I read.
- **Limitations.** Passive viewing only. No anatomy in training.
- **Implications for method.** Here prediction of activity is driven by data volume, not by structure. Structure-from-function predictions are modest.

### 1.12 Ding et al. 2025, *Nature* 640:459–469: "Functional connectomics reveals general wiring rule in mouse visual cortex" (MICrONS)

- **Citation.** doi 10.1038/s41586-025-08840-3, PMCID PMC11981947.
- **Access.** Full text (first window, partial).
- **Data.** 148 functionally characterised presynaptic neurons, 4,811 characterised postsynaptic partners, and 6,608 connected pairs, from a single mouse.
- **Results.**
  - Like-to-like connectivity holds across areas and layers.
  - The effects are small: signal correlation versus cleft volume r = 0.032.
  - The digital twin supplied in-silico tuning, and only its feature component, not its receptive-field location, predicted synapses beyond what axon–dendrite proximity explains.
- **Implications for method.** In cortex, function predicts connectivity statistically but weakly at the pairwise level. This sets expectations for any structure–function mapping in a dense circuit.

### 1.13 Li, Ping, Zhang & Wang 2026, bioRxiv 10.64898/2026.08.21.745055 (v1): "Connectome-constrained modeling identifies neurons and synapses that sustain spontaneous activity in *Drosophila*"

- **Access.** Full text (two windows, about 194k characters).
- **Why it matters.** This is the closest analogue to fit-first that I found. It is by the BrainPy author group.
- **Data and model.**
  - Data: whole-brain GCaMP6s resting-state imaging at 1.2 Hz from Turner et al., averaged into 73 neuropils. The analyses use "the single example fly"; the first half trains and the second half is held out.
  - Model: FlyWire v783 (138,639 neurons, 15,091,983 chemical synapses), a ReLU rate model with noise.
  - Fixed: topology and each neuron's sign.
  - Fitted: about 1.5×10⁷ synaptic-weight magnitudes, per-neuron time constants, and two global parameters.
  - Readout: neuropil signals pooled by synapse count.
  - Observation model: GCaMP6s convolution with τCa = 1.5 s. The data were deconvolved by Tikhonov-regularised non-negative least squares (FISTA).
- **Fitting.**
  - BPTT over the full training segment "without truncation", with gradient checkpointing. An online method, BrainTrace, gave comparable fits.
  - Adam, learning rate 10⁻³, batch 64, Δt = 1/1.2 s.
  - Built on BrainX 2026.3.12 (a JAX runtime). One NVIDIA RTX 6000 Pro. Training time was not reported.
- **Results.**
  - Held-out r = 0.423, against a train–test ceiling of 0.473 and a parameter-matched GRU at 0.288.
  - Three initialisation priors converged within ±5% of each other.
  - In-silico silencing identifies a 14-neuropil core and inhibitory hubs.
- **What is missing.**
  - No shuffled-connectome control is described; the controls are a GRU and MLPs.
  - No experimental validation: the mechanisms "remain model-derived hypotheses".
  - FlyWire lacks gap junctions, graded transmission and neuromodulation.
  - Code is promised upon acceptance.
- **Implications for method** *(my inference)*.
  - About 15 million free parameters against 73 coarse signals from one fly is a heavily underdetermined fit. Agreement across initialisation priors on the loss does not show the parameters are identified (compare Part 2).
  - Without a wiring null, beating a GRU does not isolate what the connectome contributes.
  - The study shows that gradient fitting at whole-brain scale is practical on one workstation GPU when the time step equals the imaging interval.

### 1.14 Minnick et al. 2026, arXiv 2610.33881 (24 October 2026): "Sensorimotor behaviour from an untrained brain-and-nerve-cord connectome"

- **Access.** Full text (arXiv HTML, first window).
- **What was done.**
  - A spiking LIF network of 172,433 neurons from BANC, using Shiu et al.'s parameters unchanged ("retuned none of them").
  - Brian 2 at a 0.1 ms step, coupled to NeuroMechFly/flygym. Each run used a 2-core, 12 GiB pod.
- **Results.**
  - In the body, the fly reached 95% of odour sources. A degree- and sign-preserving rewired network reached 18%.
  - Replaying the motor commands without sensory feedback reached 2.0%.
  - Pathways conduct when they supply about 2% or more of their target's input. Seven pathways at 0.03% or less did not conduct.
- **What failed.**
  - The giant fibre to TTMn path did not conduct for lack of gap junctions. One connection was "added by hand".
  - Steering by suppression needs background activity.
  - APL was predicted cholinergic when it is GABAergic.
- **Method.** Pathway tests were pre-registered, each with a rewired twin and an undriven control.
- **Implications for method.** The degree- and sign-preserving rewiring null is the standard comparison. Missing gap junctions and missing spontaneous activity are where unfitted models fail.

### 1.15 Brunton, Abe, Hu & Tuthill 2026: "The digital sphinx: Can a worm brain control a fly body?"

- **Citation.** eLife reviewed preprint RP111516 (v1 dated 17 August 2026, "not revised", per a search-result summary); earlier a bioRxiv preprint.
- **Access.** Second-hand. I read *The Transmitter*, 2 April 2026, and search-result summaries. The eLife pages did not load.
- **Content.** The adult *C. elegans* connectome was attached to a biophysical fly body and trained by deep RL to walk. The walking was realistic but, by the authors' account, biologically meaningless: behavioural fidelity without biological fidelity. In *The Transmitter*, Cowley and Turaga note that random networks might also produce walking.
- **Implications for method.** Behaviour reproduced by an optimised controller does not validate the wiring. Wiring nulls are needed whenever the controller is trained.

### 1.16 Cho 2026, flybench (GitHub brandoncho369/flybench; Zenodo doi 10.5281/zenodo.23678526)

- **Access.** Repository README only. I did not open the Zenodo preprint.
- **Content.**
  - A behavioural benchmark for whole-brain fly LIF models on FlyWire v783 and MaleCNS. Its v1.0 task set (frozen 2026-10-22) has 37 tasks, while the repository's About blurb says 31.
  - Shuffled-wiring controls (rewired, random, sign-flip) flag tasks the null also passes. The best row's specificity is +0.31.
  - Every check must hold on every seed. Synapse counts are jittered ±25%. A public versus hold-out gap metric guards against overfitting to the public tasks.
  - A spike-frequency adaptation model beats the reference LIF (0.65 versus 0.51 on the hard tier). No model passes closed-loop escape.
  - MIT licence.
- **Implications for method.** This is a parallel effort to the project's own benchmark design, with the same ingredients: pre-registration, nulls, seeds and hold-out.

### 1.17 Jin, Zhu, Zhang & Sui 2026, arXiv 2602.17997 (v3, 14 June 2026): "Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly"

- **Access.** Abstract.
- **Content.** A graph controller with the adult fly's connectome topology, trained by deep RL to control a simulated body. It reportedly learns with fewer samples than baselines.
- **Implications for method.** The connectome is used as an architectural prior for control, which is subject to the same caution as 1.15.

### Part 1: cross-cutting points that bear on method

1. **The best-known success was task-first.** flyvis used no recordings for fitting. Its parameters were tied by cell type (734 in all), and recordings were used for validation and, at most, to select among ensemble members (1.1).
2. **Fitting directly to activity on a real connectome is rare.**
   - Zarin 2019 fitted to timing targets, with an ensemble.
   - Vishwanathan 2024 fitted two or three scalars.
   - Li et al. 2026 fitted about 15 million parameters to 73 neuropil traces from one fly, with no wiring null and no experimental validation.
   - Cowley 2024 fitted to behaviour under perturbation.
3. **Unfitted connectome models do well on feedforward sensorimotor paths** (Shiu 2024, Minnick 2026). They fail where gap junctions, background activity, peptides or sign assignments matter. Those are where added detail would be tested.
4. **Wiring nulls are standard in the stronger papers.** Shiu used shuffled weights (1 of 100 versus 100%). Minnick used degree- and sign-preserving rewiring (18% versus 95%). flyvis used random-parameter models; Vishwanathan used a potential-synapse null; flybench uses shuffled controls; Zhou & Hasler used shuffled networks. The digital-sphinx critique is that optimised controllers can reproduce behaviour with the wrong wiring.
5. **Perturbation data identify what they perturb** (Cowley; Pospisil). Pospisil's route to a causal map needs whole-brain imaging with stimulation and treats the connectome as a prior.
6. **Volume of activity data drives predictive accuracy** in mouse cortex (Wang 2025). Whole-brain forecasting in zebrafish beats a stimulus-only baseline only modestly (ZAPBench). Function–structure links in cortex are weak pairwise (r = 0.032; Ding 2025).

______________________________________________________________________

## Part 2. Identifiability and degeneracy

### 2.1 Prinz, Bucher & Marder 2004, *Nature Neuroscience* 7(12):1345–1352: "Similar network activity from disparate circuit parameters"

- **Citation.** doi 10.1038/nn1352.
- **Access.** Abstract (Europe PMC record). The PDF at NJIT was not parseable.
- **Content.** More than 20 million versions of a three-cell pyloric network model were simulated, varying synaptic strengths and intrinsic properties. "Virtually indistinguishable network activity can arise from widely disparate sets of underlying mechanisms."
- **Implication.** At conductance-level detail, the same output does not pin down the parameters.

### 2.2 Marder & Goaillard 2006, *Nature Reviews Neuroscience* 7(7):563–574: "Variability, compensation and homeostasis in neuron and network function"

- **Citation.** doi 10.1038/nrn1949.
- **Access.** Abstract (Europe PMC).
- **Content.** Reviews how precisely intrinsic and synaptic currents must be tuned, and how compensation allows multiple solutions for similar behaviour.
- **Implication.** Animal-to-animal variability in parameters is expected, so pooling across animals to fit one parameter set may be ill-posed *(my inference)*.

### 2.3 Gonçalves, Lueckmann, Deistler et al. 2020, *eLife* 9:e56261: "Training deep neural density estimators to identify mechanistic models of neural dynamics"

- **Citation.** PMCID PMC7581433.
- **Access.** Full text (first window, partial; the Methods with simulation counts were not reached).
- **What was done.** Simulation-based inference (SNPE) on a pyloric model with 31 parameters (24 membrane and 7 synaptic conductances), using 18 summary features.
- **Results.**
  - "Many parameters seem only weakly constrained and only weakly correlated." Some marginals span almost the whole prior.
  - The high-posterior region is connected: a path joins two disparate parameter sets. There are sloppy and stiff directions, with perturbations orthogonal to the path quickly breaking the rhythm.
  - Sampling each parameter independently from its marginal rarely gives pyloric activity, so parameters must be tuned relative to each other.
  - Conditional correlations are strong within neurons and weak across them, for example I_A with I_H, which matches experiment.
  - A genetic algorithm (IBEA) fitted about as well but concentrated in small regions and missed the posterior.
- **Implication.** A point estimate from gradient descent or evolution strategies would land on one member of a connected set of solutions. Characterising the set needs posterior or ensemble methods.

### 2.4 Deistler, Macke & Gonçalves 2022, *PNAS* 119(44):e2207632119: "Energy-efficient network activity from disparate circuit parameters"

- **Citation.** PMCID PMC9636970.
- **Access.** Full text (first window, partial).
- **Results.**
  - Of 1 million posterior samples, 35,939 (about 3.5%) matched the data. Getting the same number from the prior would have taken about 600 billion simulations.
  - About 99% of prior samples gave no spikes or bursts.
  - Among matching circuits, energy use varied ninefold.
  - The authors judge that "posterior uncertainty is likely dominated by parameter degeneracy".
  - Energy efficiency and temperature robustness narrowed the parameters, but substantial variability remained.
- **Implication.** Adding further objectives narrows degeneracy without removing it.

### 2.5 Beiran & Litwin-Kumar 2024/2025 (see 1.6)

Directly relevant here:

- A connectome-constrained student predicted unrecorded neurons once enough neurons were recorded, but its unknown single-neuron parameters "are never precisely recovered".
- What became identifiable was the prediction, not the parameters.

### 2.6 Das & Fiete 2020, *Nature Neuroscience*: "Systematic errors in connectivity inferred from activity in strongly recurrent networks"

- **Citation.** doi 10.1038/s41593-020-0699-2.
- **Access.** Full text of bioRxiv v1 (10.1101/512053, "…strongly coupled recurrent circuits"), first window. I did not open the published version.
- **What was done.** Tested GLMs, logistic regression (with and without an l1 penalty), inverse-Ising methods and naive mean-field estimates on simulated networks.
- **Results.**
  - In strongly coupled networks, error is dominated by bias: spurious "side-band" connections between unconnected neurons.
  - Correlation between inferred and true weights was 0.99 at weak coupling and 0.32 at strong coupling.
  - The data needed for a fixed error diverge at a finite coupling, so there are "biases that do not shrink with data volume".
  - Partial observation adds similar bias, mild at weak coupling and large at strong coupling.
  - Data taken while the network moves out of equilibrium improve inference; data taken after a pattern forms hurt it.
  - The authors recommend perturbations and noise-driven recordings.
- **Implication.** More spontaneous recording does not fix connectivity inference in recurrent, attractor-like regimes. Perturbed, non-stationary data do help.

### 2.7 Qian, Zavatone-Veth, Ruben & Pehlevan 2024, NeurIPS: "Partial observation can induce mechanistic mismatches in data-constrained models of neural dynamics"

- **Citation.** bioRxiv 10.1101/2024.05.24.595741.
- **Access.** Full text of the preprint (first window).
- **Results.**
  - With about 5% of neurons observed, a line attractor and a functionally feedforward integrator both fit well and both look like line attractors.
  - Students overestimate eigenvalues and create spurious attractors and limit cycles. This held across MAP, CORNN, BPTT and FORCE fitting.
  - Choosing latent dimension by explained variance biases models toward attractors.
  - Direct perturbation is named as the strongest validation.
  - The analysis does not cover unobserved inputs.
- **Implication.** Fits to partially observed recordings can give the right activity and the wrong mechanism. Validation should use held-out perturbations.

### 2.8 Pospisil et al. 2024 (see 1.4)

- Perturbation as an instrumental variable is what breaks confounding by unobserved common input.
- With a connectome prior, the data needed fall by an order of magnitude or more.
- The identified object is the Jacobian of the dynamics.

### 2.9 Lepperød, Stöber, Hafting, Fyhn & Kording 2023, *PLoS Computational Biology*: "Inferring causal connectivity from pairwise recordings and optogenetics"

- **Citation.** PMCID PMC10656035.
- **Access.** Full text (first window, partial).
- **Results.**
  - Under wide-field stimulation, naive estimators (OLS and cross-correlation histograms) "falsely conclude" connections that do not exist.
  - IV with difference-in-differences reduces the bias but needs about 10⁶ trials.
  - The method works best in sparse systems. "Stimulating the presynaptic neuron is the gold standard."
  - Simulation only.
- **Implication.** Even perturbation data carry confounds when the stimulation is not cell-specific.

### 2.10 Timme 2007, *Physical Review Letters* 98:224101: "Revealing network connectivity from response dynamics"

- **Citation.** arXiv cond-mat/0610188.
- **Access.** Abstract (arXiv), plus a search-result summary of the counting argument.
- **Content.**
  - Each driving condition gives N constraints on N² couplings, so roughly N distinct drives identify the network.
  - Sparsity gives good reconstructions even when the problem is formally underdetermined.
- **Implication.** Diverse perturbations are what add the independent constraints on connectivity.

### 2.11 Wagenmaker et al. 2024, NeurIPS: "Active learning of neural population dynamics using two-photon holographic optogenetics"

- **Citation.** arXiv 2412.02529, v4 dated 8 May 2025.
- **Access.** Abstract.
- **Content.** Low-rank linear dynamical models fitted to photostimulation responses in mouse motor cortex. Choosing stimulation targets actively cut the data needed for a given predictive power by up to about half.
- **Implication.** The value of perturbation data depends on design, and adaptive designs pay.

### 2.12 Calcium imaging as the observation channel

- **Orlandi et al. 2015**, "First Connectomics Challenge: From Imaging to Connectivity", PMLR 46. Access: abstract. Connectivity was inferred from simulated fluorescence of 1,000 neurons; 143 entrants and 16 finalist teams.
- **Sutera et al. 2014**, the winning entry, "Simple connectome inference from partial correlation statistics in calcium imaging", arXiv 1406.7865. Access: full text.
  - Named difficulties: slow fluorescence decay, a low sampling rate relative to neural dynamics, light scattering, and high global activity masking pairwise interactions.
  - Partial correlation after filtering reached AUROC 0.943 but AUPRC only 0.403. It cannot give the direction of connections.
  - The tuned method "clearly overfits the simulator".
- **Berens et al. 2018**, *PLoS Computational Biology* 14(5):e1006157, the spikefinder benchmark. PMCID PMC5997358. Access: full text (first window).
  - On 92 recordings with simultaneous electrophysiology, the best spike-inference correlation was 0.44 at 40 ms bins, rising to about 0.73 at 333 ms. Performance varied substantially between datasets.
  - Generative approaches need accurate models of indicator dynamics, saturation and noise.
- **Implication** *(my inference)*. A calcium observation model is not optional. At the native temporal resolution, even the best deconvolution recovers a minority of the variance. Fitting should either forward-model fluorescence (as Li et al. 2026 did with a GCaMP kernel) or score at coarse time bins.

### Part 2: synthesis

- **Parameters.** In small circuits with conductance-level detail, parameters are degenerate on connected manifolds with stiff and sloppy directions (2.1–2.4). In connectome-constrained rate networks, single-neuron parameters are not recovered even when activity is predicted (2.5).
- **Connectivity from activity.** Inference is biased in strongly coupled regimes and under partial observation, and the bias does not shrink with more spontaneous data (2.6, 2.7). Common input and unobserved neurons require interventions (2.8, 2.9).
- **Perturbations.** They add independent constraints (2.10). Their value can be raised by design (2.11), but they identify only what they perturb (1.5).
- **Calcium.** Slow indicators and low frame rates cap temporal resolution (2.12).
- **What becomes identifiable.** Across these sources, predictions under perturbation and local effect maps (the effectome or Jacobian) are recoverable. Raw synaptic weights, gap conductances and time constants mostly are not.

______________________________________________________________________

## Part 3. Differentiable simulation tooling and its cost

### 3.1 Jaxley: Deistler et al. 2025, *Nature Methods* 22(12):2637–2657

- **Citation.** "Jaxley: differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics", doi 10.1038/s41592-025-02895-w, PMCID PMC12695658. Preprint: bioRxiv 10.1101/2024.08.21.608979.
- **Access.**
  - Full text of the journal version (first window, partial; the later Methods were not read).
  - GitHub page and documentation home page.
- **Numerics.** Implicit Euler for voltage, exponential Euler for gates, and Stone's algorithm for the tridiagonal branch systems.
- **Gradients and memory.**
  - Backprop through the solver with multilevel checkpointing. Without it, storing the forward pass "can easily correspond to terabytes of data" for large systems.
  - A gradient costs about 3–20 times a simulation run.
- **Stability measures.**
  - Inverse-sigmoid parameter transforms.
  - A Polyak-style step normalised by the gradient norm (β 0.8–0.99).
  - Separate optimisers per parameter type.
  - **Truncated BPTT**, used only for the retina model (reset every 50 ms).
  - Differentiable summary losses: window means and SDs, and soft dynamic time warping.
- **Benchmarks.**
  - Matched NEURON at sub-ms and sub-mV resolution.
  - About two orders of magnitude faster than NEURON on GPU for large or parallel workloads, and at least as fast on CPU.
  - A network of 2,000 detailed neurons (1 million synapses, 3.92 million states): 200 ms (8,000 steps of 0.025 ms) in 21 s on one A100; gradients for 3.2 million parameters in 144 s.
- **Applications.**
  - Single-neuron fits with 19 parameters (median 9 gradient steps). A 1,390-parameter model converged in about 100 iterations, where a genetic algorithm stayed about two orders of magnitude higher in loss.
  - A hybrid retina model fitted to two-photon calcium data: positive correlation at 146 of 147 sites, mean 0.25, maximum 0.51.
  - Working-memory RNNs with 109 and 542 parameters.
  - A detailed network of about 106,000 parameters on MNIST (94.2%).
- **Stated limits.**
  - "Optimizing models on tasks that require backpropagating gradients along even longer simulations is challenging." The authors note they have fitted data only up to about 1 s.
  - Non-convex losses.
  - Degeneracy means one best fit may not represent the solution space.
  - Spiking nonlinearities may be harder to train.
- **Text inconsistency.** The Results give 287 synaptic and 320 cell parameters for the retina model; the Methods give 250 synaptic and 556 in total.
- **Software.** Apache-2.0, 218 stars. CPU, GPU or TPU through JAX: `pip install jaxley` gives the CPU build, and GPU means CUDA JAX. Apple Silicon is not mentioned.
- **Gap junctions.** The visible synapse modules are `ionotropic`, `tanh_conductance` and `tanh_rate`. I found no gap-junction module; one could presumably be written as a custom synapse *(my inference, unverified)*.

### 3.2 flyvis: Lappalainen et al. 2024 (see 1.1)

- **Software.** PyTorch, MIT licence, 215 stars. Tested on Linux with Python 3.9–3.12; macOS, MPS and GPU requirements are not stated.
- **Numerics.** Euler integration with Δt of 20 ms in training (5 ms for evaluation), and time constants clamped at or above Δt.
- **Keeping gradients tractable.** Training clips are short, about 792 ms, which is roughly 40 steps at 20 ms *(my arithmetic)*, and each starts from a pre-computed steady state.
- **Optimisation and scale.** Adam with a decaying learning rate, batch 4, about 250k iterations, and an ensemble of 50.
- **Compute.** Not stated in anything I opened.

### 3.3 BrainPy, BrainState and the BrainX ecosystem (Chaoming Wang's group)

- **Access.** BrainPy PyPI page; BrainState GitHub page. The BrainState PyPI page failed to load.
- **BrainPy.** Version 2.9.0, released 24 October 2026. A JAX framework for brain-dynamics programming with JIT compilation. The licence is inconsistent: the PyPI metadata says Apache, the README badge says GPLv3.
- **BrainState.** A state-based transformation system on JAX. Apache-2.0, 23 stars. Part of BrainX (`pip install BrainX`).
- **In practice.** Li et al. 2026 (1.13) used BrainX 2026.3.12 for a 138,639-neuron fitted rate model. They used full-segment BPTT with checkpointing on one RTX 6000 Pro, and an online-learning alternative, BrainTrace, gave comparable fits.

### 3.4 Diffrax (Kidger)

- **Access.** GitHub page, adjoint documentation, PyPI.
- **Package.** Version 0.7.2, released 18 February 2026. Apache-2.0, 2.1k stars. JAX solvers for ODEs, SDEs and CDEs, including implicit solvers.
- **Adjoints.**
  - **RecursiveCheckpointAdjoint** is the default and recommended choice. It uses binomial checkpointing, with memory of roughly the number of checkpoints times the state size and O(n log n) backprop time by default.
  - **BacksolveAdjoint** is "not recommended" because its gradients are approximate.
  - **ForwardMode** is recommended for least-squares-type inference with many outputs.
  - **ImplicitAdjoint** is for steady states.

### 3.5 Dynamax (Linderman, Murphy et al.)

- **Access.** GitHub page.
- **Package.** JAX state-space models: HMMs, linear Gaussian, nonlinear Gaussian and generalised Gaussian SSMs. Fitting by EM, SGD (optax) and HMC/SMC (Blackjax). MIT licence, about 1.0k stars. JOSS paper doi 10.21105/joss.07069, labelled 2024.
- **Relevance** *(my inference)*. Latent linear or switching models with a calcium emission model, as baselines against a mechanistic fit. It does not provide connectome-constrained biophysics.

### 3.6 C. elegans-specific differentiable code

- **TuragaLab/wormvae.** Access: GitHub README.
  - PyTorch implementation of Mi et al., "Connectome-constrained Latent Variable Model of Whole-Brain Neural Activity" (ICLR 2022).
  - Synapses can be conductance- or current-based. Weights can be constrained by connectome synapse counts, by a binarised sparsity mask, or left unconstrained.
  - Training holds out neurons and selects worms. Data: the *C. elegans* connectome plus NeuroPAL fluorescence.
  - Gap junctions are not mentioned in the README. Licence not visible. 19 stars, 26 commits.
  - Runs on Linux or macOS, on CPU or NVIDIA GPU.
  - I could not open the paper itself: the PDF was unparseable and the ICLR page returned HTTP 500.
- **Simeon, Kashyap, Kording & Boyden**, "Homogenized *C. elegans* Neural Activity and Connectivity Data", arXiv 2411.12091 (v4, 19 May 2025). Access: abstract.
  - 12 activity datasets (about 900 worms, about 250 labelled neurons) and 9 connectome annotations, standardised and hosted on Hugging Face.
  - Data only, no models.
- **MetaWorm / BAAIWorm.** Access: second-hand (search-result snippets). An integrative multicompartment brain–body model. I did not confirm that it is differentiable or how it was fitted.

### 3.7 Practical issues the literature names

| Issue | What the sources say | Handling in flyvis / Jaxley / others |
|---|---|---|
| **Exploding or vanishing gradients over long horizons** | Mikhaeil, Monfared & Durstewitz 2022 (NeurIPS, arXiv 2110.07238; abstract): RNNs with stable fixed points or cycles have bounded gradients, chaotic ones always diverge, and training should be tuned to the Lyapunov spectrum. Metz et al. 2021 (arXiv 2111.05803; abstract): chaos-driven gradient failure in RNNs, physics simulators and learned optimisers, traced to the Jacobian spectrum. | flyvis: short clips of about 40 steps from steady state. Jaxley: truncated BPTT (50 ms) for the retina, and fits only up to about 1 s of data. Li et al. 2026: full segment, which at Δt = 0.83 s is about 500 steps for 7 min *(my arithmetic)*. |
| **Chaotic or oscillatory targets** | Hess et al. 2023 (ICML, arXiv 2306.04406; abstract): generalised teacher forcing provably bounds gradients when training on chaotic systems. Turan & Jäschke 2021 (*IEEE Control Systems Letters*, arXiv 2109.06786; abstract): single shooting gives "flattened" trajectories on oscillatory data, which multiple shooting fixes. | Neither flyvis nor Jaxley reports teacher forcing or multiple shooting in the parts I read. |
| **Memory** | Jaxley: storing the forward pass can reach terabytes for large systems. | Jaxley: multilevel checkpointing. Diffrax: binomial checkpointing by default. Li et al. 2026: gradient checkpointing. |
| **Stiffness** | Jaxley uses implicit Euler for voltage. Diffrax provides implicit solvers. | flyvis: explicit Euler, with τ clamped at or above Δt to keep it stable. Strong gap-junction coupling adds fast modes that explicit steps must resolve *(my inference)*. |
| **Ill-conditioning and parameter scales** | Jaxley names instabilities from parameters on different scales. | Jaxley: sigmoid transforms, a normalised (Polyak) step and per-type optimisers. flyvis: parameters tied by cell type. |
| **Non-differentiable targets** | Jaxley replaces spike features with differentiable losses (window statistics, soft DTW). | Relevant if a fit scores behaviour statistics rather than traces *(my inference)*. |
| **Calcium observation** | Spikefinder ceiling of about 0.44 at 40 ms (2.12). | Li et al. 2026: GCaMP6s kernel forward model plus FISTA deconvolution. Jaxley's retina model was fitted to calcium data. wormvae infers voltage from fluorescence with a VAE. |
| **Degeneracy** | Jaxley motivates Bayesian sampling because one best fit may not represent the solution space. Part 2. | flyvis: ensemble of 50. Zarin 2019: ensemble. Li et al. 2026: 5 replicates. |

### 3.8 Apple Silicon (as of 2026-11-14)

- **JAX on CPU (M-series).** Supported. The JAX installation docs list "macOS, Apple ARM-based" jaxlib wheels and mark Mac aarch64 CPU "yes".
- **JAX on Metal: effectively dead.**
  - Apple's page still lists `jax-metal` 0.1.0 as **experimental**, without float64 or complex types, and failing some JAX tests.
  - jax-ml/jax issue #26968 (opened 6 March 2025, closed 20 January 2026): a contributor said jax-metal is incompatible with JAX 0.5.1 and later. A JAX collaborator (jakevdp) wrote that the last release was over a year old, removed jax-metal from the install docs, and advised users to "consider the jax-metal project dead". Access: full text of the issue page.
  - The JAX install page is itself inconsistent. Its table says Apple GPU is "experimental", while its Mac GPU section says JAX "is not supported on Mac/OSX GPU".
  - An Apple Developer Forums thread (February 2026, no replies) reports the same closures. It is second-hand and unverified.
  - Third-party Metal and MLX backends exist on PyPI (jax-mps, jax-metallib, metaljax, applejax). I saw them only in search snippets, all labelled alpha, beta or experimental.
- **PyTorch MPS.**
  - The 2.15 docs say MPS "does not support torch.float64 (torch.double) or torch.complex128", because Metal Shading Language has no double type. `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help with this.
  - Tunguz, "State of PyTorch Hardware Acceleration: May 2026" (v0.2, 15 May 2026), rates MPS maturity Medium and torch.compile on MPS Low: "eager MPS is a real tool; compiled MPS is still an experiment."
- **Implication** *(my inference)*. On a Mac, a JAX fit runs on CPU only; a PyTorch fit can use MPS only in float32. For a roughly 300-neuron model, CPU may be adequate (see 3.10).

### 3.9 Cloud GPU prices (all seen 2026-11-14)

| Provider | GPU | Price per GPU-hour | Notes |
|---|---|---|---|
| RunPod (Secure / Community Cloud) | A100 PCIe | $2.09 / $1.19 | Community figures are "from" prices |
| RunPod | A100 SXM | $2.19 / $1.39 | |
| RunPod | H100 PCIe | $2.89 / $1.99 | |
| RunPod | H100 SXM | $4.39 / $2.69 | |
| RunPod | H100 NVL | $3.19 / $2.59 | |
| RunPod | H200 | $5.29 / $3.59 | |
| RunPod | L40S | $1.09 / $0.79 | |
| RunPod | RTX 4090 | $0.89 / $0.34 | |
| Lambda (1x on-demand) | A100 SXM 40 GB | $1.99 | Page says "These prices take effect November 16, 2026"; current prices only after login. The fetch tool matched the 1x table by tab order, so treat as approximate. |
| Lambda | A100 PCIe 40 GB | $1.99 | same |
| Lambda | H100 PCIe 80 GB | $3.79 | same |
| Lambda | H100 SXM 80 GB | $4.89 | same |
| Lambda | GH200 96 GB | $2.59 | same |
| Lambda | B200 | $7.99 | same |
| Lambda | 8x HGX A100 80 GB | $3.19 per GPU | only A100 80 GB listing |
| Modal (serverless, per second) | A100 40 GB | $0.000583/s (about $2.10/h) | hourly rates are my conversion |
| Modal | A100 80 GB | $0.000694/s (about $2.50/h) | |
| Modal | H100 | $0.001097/s (about $3.95/h) | |
| Modal | H200 | about $4.54/h | |
| Modal | CPU | $0.0000131 per physical core per second (about $0.047/core-h) | |
| Google Cloud | n/a | not obtained | the page was truncated before the GPU tables |

### 3.10 What fitting a roughly 300-neuron model would cost (*my inference, built from the figures above; not a measured result*)

- **Per-step compute is negligible.** The state is about 300 voltages, plus synaptic and calcium states if modelled. That is about 13,000 times smaller than Jaxley's 3.92-million-state benchmark network (3.92×10⁶ / 300 ≈ 1.3×10⁴), and about 150 times smaller than flyvis's 45,669 neurons. A GPU's advantage comes from batching (ensembles, restarts, conditions), not from a single trajectory, whose sequential steps are latency-bound.
- **Memory is small.** Storing 300 float32 states for 10⁵ steps is about 120 MB per trajectory before checkpointing (300 × 4 B × 10⁵). Checkpointing as in Diffrax or Jaxley reduces it further.
- **Gradient cost.** At 3–20 times a forward pass (Jaxley's range), the cost is set by horizon length × number of restarts or ensemble members × hyperparameter sweeps.
- **The binding costs are the optimisation pathologies, not compute.**
  - Long horizons: minutes of recording at sub-second steps.
  - Possibly chaotic or attractor dynamics.
  - Stiffness from gap junctions, which favours implicit steps.
  - The calcium observation model.
- **Priced examples.** At the prices above, 100 A100-class GPU-hours cost about $120–$250 (my arithmetic from the $1.19–$2.50/h range across RunPod, Lambda and Modal). The same work on local M-series CPU cores under JAX has no rental cost and keeps float64. Whether that is fast enough depends on horizon length and the number of restarts, which I found no published figure for.
- **Not established.** I found no published wall-clock figure for fitting a roughly 300-neuron connectome model by gradient descent.

### 3.11 Tool summary

| Tool | Language | What it simulates | Gradient method | Maturity (evidence) | Apple Silicon | Licence |
|---|---|---|---|---|---|---|
| Jaxley | Python / JAX | Multicompartment HH-type neurons and networks; ionotropic, tanh-rate and tanh-conductance synapses; gap junctions not confirmed | Autodiff through implicit/exponential Euler; multilevel checkpointing; optional truncated BPTT | *Nature Methods* 2025; 218 stars | CPU via JAX (macOS arm64 wheels); GPU path is CUDA; Metal not mentioned | Apache-2.0 |
| flyvis | Python / PyTorch | Graded (threshold-linear) point neurons on the fly visual connectome, about 45k neurons | BPTT, Adam | *Nature* 2024; 215 stars; pretrained ensembles | Not stated; tested on Linux. MPS would need float32 throughout *(my inference)* | MIT |
| BrainPy | Python / JAX | Spiking and rate brain-dynamics models | JAX autodiff | *eLife* 2023 paper; v2.9.0 (24 Oct 2026) | JAX CPU (inference); Metal not viable | Inconsistent: Apache (PyPI) vs GPLv3 (badge) |
| BrainState / BrainX | Python / JAX | State-based model building; used for a 138k-neuron fitted rate model | BPTT with checkpointing; online BrainTrace (per Li et al. 2026) | 23 stars; used in a 2026 preprint | JAX CPU (inference) | Apache-2.0 |
| Diffrax | Python / JAX | General ODE/SDE/CDE solvers, including implicit | RecursiveCheckpointAdjoint (default), ForwardMode, ImplicitAdjoint | v0.7.2 (18 Feb 2026); 2.1k stars | JAX CPU | Apache-2.0 |
| Dynamax | Python / JAX | HMMs, linear/nonlinear Gaussian SSMs | EM, SGD, HMC/SMC | JOSS (2024 label); about 1.0k stars | JAX CPU | MIT |
| wormvae | Python / PyTorch | *C. elegans* connectome-constrained conductance/current model with VAE inference from fluorescence | Variational inference with backprop | ICLR 2022; 19 stars; looks dormant | Linux/macOS CPU, or NVIDIA GPU | Not visible |
| JAX (platform) | n/a | n/a | n/a | n/a | CPU supported; jax-metal "dead" per a JAX collaborator (Jan 2026) | not checked this session |
| PyTorch MPS (platform) | n/a | n/a | n/a | Eager usable; compile "early prototype" per the Tunguz scorecard | No float64 or complex128 | not checked this session |

______________________________________________________________________

## Sources I could not find or open

- **Reimers et al. 2026.** Not found. Two targeted searches turned up no paper by a Reimers on fly connectome modelling. The closest 2026 fitted-to-activity fly paper I found is Li, Ping, Zhang & Wang (1.13); I cannot say whether that is what was meant.
- **flybench preprint** (Cho 2026, Zenodo 10.5281/zenodo.23678526). Not opened; I read only the GitHub README. My search found no arXiv version.
- **Beiran & Litwin-Kumar, *Nature Neuroscience* 2025 journal text.** The Europe PMC full-text request returned HTTP 500. I used the bioRxiv v1/v2 text and the COSYNE 2025 abstract. The journal version may differ, for example by adding the *Drosophila* applications.
- **Litwin-Kumar & Turaga 2019 full text.** The NSF PAR PDF was not parseable. Abstract-level only.
- **Prinz, Bucher & Marder 2004 and Marder & Goaillard 2006.** Full texts not opened (PDF not parseable; not in PMC). Abstract-level only.
- **Das & Fiete 2020, published *Nature Neuroscience* version.** Not opened; only bioRxiv v1.
- **Digital sphinx (Brunton et al. 2026).** The eLife reviewed-preprint and review pages did not load. Second-hand only, from *The Transmitter*.
- **Gonçalves et al. 2020 and Jaxley 2025 Methods sections** (simulation counts; full Jaxley Methods). Not reached, because only the first window was read.
- **flyvis compute and training time.** Not stated in the paper sections, README or docs that I opened.
- **Wang et al. 2025 compute.** Not in the window I read.
- **Jaxley gap-junction support.** Not confirmed either way.
- **Mi et al. 2022 ICLR paper (wormvae).** PDF unparseable and the ICLR page returned HTTP 500. Repository README only.
- **MetaWorm / BAAIWorm.** Search snippets only. I did not confirm its fitting method or whether it is differentiable.
- **BrainState PyPI page.** Failed to load, so no version or date.
- **Google Cloud GPU prices.** Page truncated. AWS and Azure were not tried.
- **Mikhaeil 2022, Metz 2021, Hess 2023, Turan & Jäschke 2021, Timme 2007, Wagenmaker 2024, Orlandi 2015.** Abstract only. The specific remedies (sparse teacher forcing, Lyapunov-based forcing intervals) are not confirmed from text I read.
- **Search budget.** The session's web-search limit was reached near the end. I could not search for follow-ups that fit flyvis-type models to recordings, for Apple MLX-based scientific JAX use, or for a measured wall-clock cost of fitting a roughly 300-neuron connectome model.
