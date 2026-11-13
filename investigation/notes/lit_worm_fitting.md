# Fitting connectome-constrained models to *C. elegans* neural activity: what has been done

Literature review, compiled 2026-11-14, using web search and web fetch only.

## How this was done, and its limits

- **Access.** Every entry gives the access level I actually reached. "Full text" means I read the methods and results through an HTML rendering: bioRxiv, ar5iv/arXiv HTML, PLOS, or Europe PMC XML. Figures themselves (images) were never visible, so a number that lives only in a figure panel is missing here unless the text quotes it. Several long pages were cut off after the first 100,000 characters; I say so where it happened, and usually only the reference list was lost.
- **Blocked or unreadable.** PMC pages (pmc.ncbi.nlm.nih.gov) showed a reCAPTCHA, nature.com redirected to a login, Cell Press returned 403, and every arXiv or journal PDF came back as unparseable binary. In each case I used a bioRxiv, arXiv HTML or Europe PMC copy instead and named the copy.
- **Search budget ran out.** The shared web-search budget was exhausted near the end of the sweep. The 2025–2026 coverage is therefore incomplete, especially for graph-neural-network or "foundation" models of worm activity (none found before the cut-off) and for Leifer-lab follow-ups. See section 3.
- **No verdict.** No verdict on fit-first is given, as the brief asked. Where I add a comment of my own it is marked as mine.

______________________________________________________________________

## 1. Sources

### A. Fits to perturbation data: stimulate one neuron, record the rest

#### A1. Creamer, Leifer & Pillow. "Bridging the gap between the connectome and whole-brain activity in *C. elegans*." bioRxiv 10.1101/2024.09.22.614271

- **Versions I opened:**
  - v1: posted 23 Sep 2024; abstract only.
  - v3: posted 26 Sep 2025; full text.
  - v4: full text. No date is printed on the page. Figure asset paths contain "early/2026/05/18", which is my inference for its posting date.
  - v3 is probably the PMC12485687 copy, but I could not open that page (captcha).
  - Code: github.com/Nondairy-Creamer/Creamer_LDS_2026. README read; MIT licence.
- **Access level:** full text of v3 and v4, read in several passes (v4 to about 100k of 110k characters; the unread rest is mostly references). Supplementary figures: titles only.
- **Data:**
  - The Randi et al. 2023 perturbation atlas: whole-brain calcium imaging (GCaMP6s, nuclear; NeuroPAL identities) of immobilised worms. A randomly chosen head neuron is stimulated by two-photon optogenetics every 30 s over 20–40 min recordings.
  - 110 animals. The model has 156 neurons in v3 and 154 in v4, all in the head. Signals are ΔF/F, with F0 the time-mean of F, at 2 Hz.
- **Model class:**
  - A latent linear dynamical system with no dimensionality reduction: each neuron's next activity is a weighted sum of all neurons' current activity plus a linearly filtered history of its own optogenetic input.
  - Input filters are diagonal and 45 time points (22.5 s) long.
  - Observations are latent activity plus an offset plus diagonal Gaussian noise. Process noise is also diagonal Gaussian.
  - Calcium is treated as the state itself, with no indicator kinetics model and no calcium-to-voltage transform.
- **Connectome and fitted parameters:**
  - The weight matrix W is masked to anatomically connected pairs, chemical and electrical synapses summed. About 9.4% of possible pairwise weights are free; the rest are fixed at 0.
  - The mask is "the sum of the matrix of synapse counts from three adult connectomes" plus one L4 connectome. v4 says the connectome was taken from nemanode.org on 1 Nov 2025, in four files: white_1986_jsh, white_1986_n2u, witvliet_2020_7, witvliet_2020_8. Which file is L4 is my inference, not stated on the page.
  - v4 does not say whether the matrix was binarised for the main model. Gap-junction symmetry is not discussed.
  - Fitted: W (on the mask), the input filters B, the noise covariances Q and R, and the offset d.
- **Fitting method:**
  - Expectation maximisation, with latent states from a Kalman filter-smoother. Not gradient-based.
  - Unrecorded neurons in a given animal are missing values in the latent state.
  - It appears to be one model shared across animals: neuron identity lets the model "integrate information from the same neuron across different animals". That it is shared is my inference; the text does not say so outright.
- **Validation design:**
  - Split by animal: 80 animals train, 30 held out. A robustness check uses five 55/55 animal splits (Supp. Fig. 4).
  - Main metric: stimulus-triggered average magnitude (STAM, area under the stimulus-triggered average over 30 s), compared between the model's in-silico single-neuron stimulations and the held-out animals.
  - The "relative correlation" divides the model–data correlation by a noise ceiling, the train–test correlation of the data with themselves.
  - Held-out neurons: a neuron and its bilateral partner are masked in held-out animals and reconstructed from the others.
  - I found no held-out-stimulation design (for example, leaving out every stimulation of one neuron).
- **Numbers:**

| | v3 (2025) | v4 (2026) |
|---|---|---|
| Model vs measured STAM correlation | 0.15 | 0.14 |
| Train–test (noise ceiling) | 0.16 | 0.17 |
| Relative correlation (abstract) | 92% | 82% |
| Held-out neuron reconstruction, mean r | 0.30 (AVER 0.92 best; median neuron URYVL 0.30) | 0.25 (AVER 0.93; URYVL 0.23) |
| Fully connected better on 56 putative extrasynaptic pairs | p = 0.011, "two-sided paired correlation test" | p = 0.1, Wilcoxon signed-rank (not significant) |
| Electrical synapse count vs |model weight| correlation | 0.107 | 0.108 |

- The v1 abstract also says 92%. Search-engine summaries that put "82%" in the v1 abstract disagree with what I read on the v1 page.
- **Controls:**
  - **Fully connected (no mask):** "no better" than the constrained model on STAMs; "slightly outperformed" it on the correlation structure; higher test log-likelihood, which the authors attribute to extra parameters rather than overfitting. No numbers in the text.
  - **Shuffled connectome:** the binary connection matrix with its rows and columns permuted, which keeps the topology and sparsity but relabels neurons. It "performed poorly" on both causal and correlative measures; the abstract says "much lower performance". The values are in figures I could not see.
  - **Direct (monosynaptic-only) prediction:** "significantly worse".
  - **Held-out neurons:** beat a control that compares predictions with the wrong neurons. No number given.
  - **Raw synapse counts:** much weaker than the model at predicting |STAM|. Supp. Fig. 2 title: synaptic counts "do not correlate to the measured STAMs when exponentiated".
  - **Synthetic data:** when extrasynaptic signalling is strong, the constrained model fails (Supp. Fig. 3).
  - **Initialisation:** results are insensitive to it (Supp. Fig. 4).
  - A random-mask control is not reported.
- **Dataset size (Fig. 1I):** the relative correlation is "roughly independent of data set size". The raw correlation rises with stimulation events per pair.
- **unc-31:** neither v3 nor v4 mentions the unc-31 data. Extrasynaptic signalling is probed only through the 56 putative extrasynaptic pairs taken from Randi et al.
- **What worked:**
  - A connectome mask with about 9.4% density reaches the fully connected model on held-out-animal STAMs.
  - It beats a relabelled-connectome null.
  - It explains responses between unconnected neurons through multi-hop paths.
  - It reconstructs masked neurons above chance.
- **What failed or is weak:**
  - The absolute agreement is low: correlation 0.14–0.15, because the data reproduce themselves only at 0.16–0.17.
  - The mask gives no gain over full connectivity, and full connectivity is slightly better on correlation structure.
  - Chemical synapse counts correlate poorly with the fitted weights.
  - The model underestimates response magnitudes (no number given).
- **Stated limitations:**
  - Linear and time-invariant with a single fixed point, so it cannot represent bistability or limit cycles.
  - No learning, neuromodulation or state dependence.
  - Only pairwise (single-neuron) stimulation; nonlinear group interactions are not captured.
  - Calcium is a proxy, and voltage may propagate faster.
  - Large animal-to-animal variability may mask the role of extrasynaptic signalling.
  - Neuromodulatory signalling is excluded from the STAM measure.

#### A2. Randi, Sharma, Dvali & Leifer. "Neural signal propagation atlas of *Caenorhabditis elegans*." *Nature* 623:406–414 (2023), DOI 10.1038/s41586-023-06683-4; arXiv 2208.04790

- **Access level:** full text of the arXiv preprint through ar5iv. The page does not show a version number; arXiv v3 is dated May 2023 and carries the comment "Nature (2023)". I read about 100k of 147k characters plus a second pass at the methods. I could not open the Nature page (login redirect) or PMC10632145 (captcha). The data are on DANDI as dandiset 001075; I confirmed the record exists, with Leifer as contact.
- **Data:**
  - 113 wild-type animals (the arXiv text) and 18 unc-31 animals. Recordings up to 40 min, one mostly random neuron stimulated every 30 s for 0.3 or 0.5 s.
  - 23,427 neuron pairs measured, 66% of possible head pairs, each pair up to 59 times; 186 of 188 head neurons recorded.
  - Methods: nuclear GCaMP6s; the GUR-3/PRDX-2 opsin under a dexamethasone-inducible QF-GR system; NeuroPAL identities; two-photon temporal-focusing stimulation.
  - Search snippets give the published count as 23,433 pairs; I did not open the published text.
- **Model or anatomy comparison:**
  - Anatomy-based predictions come from differential-equation simulations of electrical and chemical synapses, with a nonlinear synaptic activation and parameters "commonly used in *C. elegans* modeling efforts".
  - Weights are a scaled average of three adult connectomes and one L4 connectome.
  - Polarities come from receptor gene expression (CeNGEN and other sources); ambiguous cases are treated as excitatory.
  - The predicted response is the peak response to current injected in silico.
- **Fitted parameters:** only in one analysis (Fig. 3d), which lets weights and signs change "in the most favorable way" to fit the functional data.
- **Results on anatomy versus function:**
  - Only 1,314 of 23,427 pairs (about 6%) were functionally connected at q < 0.05.
  - Functionally connected pairs had a mean minimal anatomical path length of 2.1 hops.
  - Anatomy-predicted connections were enriched among functional connections (KS test, p < 0.0001), and single-hop anatomical pairs were more likely to be functionally connected.
  - "Overall, however, there was fairly poor agreement between anatomical predictions and measurement"; over all pairs R² < 0 (Fig. 3d).
  - Fitting weights and signs improved agreement, but it stayed poor. I found no number in the text.
  - About 11% of q < 0.05 connections were inhibitory, which the authors call a likely lower bound against roughly 20% expected from polarity predictions.
  - AFD→AIY is predicted inhibitory from glutamate machinery but measured excitatory; the authors suggest peptidergic signalling.
- **unc-31 (dense-core vesicle release defective):**
  - 53 of the 1,314 wild-type functional connections were lost in unc-31 animals. The authors call these "purely extrasynaptic-dependent" candidates and a conservative lower bound.
  - unc-31 animals had a smaller proportion of high-confidence connections.
  - Their propagation agreed better with anatomy than wild type did, "while agreement was still poor".
- **Spontaneous activity:**
  - Correlations from anatomical synapse counts agreed poorly with spontaneous correlations.
  - A connectome-constrained biophysical simulation did "better but still poor".
  - Kernel (atlas)-based simulations beat both, especially when only the top 6 neurons were driven. No numbers appear in the text.
- **Reproducibility across animals:**
  - Some pairs respond in only some trials, and amplitude, kinetics and even sign vary between trials.
  - Kernels within a pair are more stereotyped than kernels across pairs (Extended Data Fig. 6e–f; values not in the text).
  - I found no overall fraction of pairs that respond consistently.
- **Stated limitations:**
  - Results are a lower bound: some neurons were too dim to identify, and RID's own activity was not measured.
  - Nuclear calcium only, so compartmental dynamics are missed.
  - Immobilised animals and single-neuron stimulation only; nonlinearities, state and plasticity are not characterised.
  - The measurements are effective, not direct, connections.
  - The unc-31 inference assumes no other signalling defects.

#### A3. Dvali, Seguin, Betzel & Leifer. "Diverging network architecture of the *C. elegans* connectome and signaling network." *PRX Life* 3:033021 (published 10 Sep 2025), DOI 10.1103/6wgv-b9m6

- **Access level:** abstract. The PDF was fetched but could not be parsed.
- **Data:** the Leifer signal-propagation network compared with the EM wiring.
- **Model:** network analysis (modules, rich club). No dynamical model is fitted.
- **Findings:**
  - Signalling modules do not line up with the anatomical modules.
  - The pharynx is a separate community in both networks.
  - Signalling modules are enriched for cell types and functions.
  - The signalling network's rich club differs from the anatomical one; AVEL/R is the only shared member.
  - The authors conclude that the two networks are related but distinct, and that structural hubs need not hold positions of influence in the signalling network.
- **Limitations:** not available at abstract level.

#### A4. Toyoshima, Sato, Nagata, …, Iino. "Ensemble dynamics and information flow deduction from whole-brain imaging data." *PLoS Comput Biol* (15 Mar 2024), DOI 10.1371/journal.pcbi.1011848

- **Access level:** full text on the PLOS site, about 100k of 149k characters. The methods were cut off partway.
- **Data:**
  - 24 whole-brain recordings, about 4 volumes/s for about 25 min, using the nuclear YC2.60 indicator.
  - Animals were held in a microfluidic chip under periodic NaCl switches, so not freely moving.
  - About 147 cells annotated per sample; 177 of 196 head cells covered in total.
- **Model classes:**
  - (i) Ensemble motifs from time-delay embedding plus reconstruction ICA (14 motifs).
  - (ii) A synapse-based model in which the connectome defines each neuron's inputs (EM wiring as digitised by Oshio et al.; chemical synapses directed, electrical undirected).
  - Each target neuron gets a gradient kernel dimension reduction followed by a Gaussian mixture model, fitted separately per animal ("Each target neuron i in each sample j had its GMM model").
  - Using "indirect links" (the presynaptic neurons of missing neurons) worked better than direct links.
- **Validation:**
  - Within animal, across time: one third of the data trains and two thirds test. In 58.6% of models the target neuron is predicted significantly better than a permutation null.
  - Free-running simulations reproduced pair correlations in some samples but not all; globally synchronised samples were hard.
  - A deterministic predictor failed ("the neuronal activities either became constant or changed slowly"). Realistic dynamics needed added noise.
- **Comparison with perturbation data:**
  - Virtual optogenetics with the model of one sample (sample 14) against Randi et al.'s measured responses.
  - Pearson r = 0.250, 0.342, 0.433 and 0.486 for pairs that Randi et al. tested at least 10, 15, 20 or 25 times (n = 246, 100, 56 and 31 pairs).
  - The figure legend gives the thresholds as 5, 10, 15 and 20, an inconsistency in the source.
  - The correlation rises with the reference data's trial count.
- **Connectome controls:** no all-to-all or shuffled-connectome comparison in the text I read. The authors note that the method could assume all-to-all links but did not test it.
- **Findings:**
  - The inferred functional network resembled the gap-junction network more than the chemical-synapse network.
  - Sensory routes differ between animals.
  - The relationship between motifs 8 and 9 was positive in one animal and negative in another.
- **Stated limitations:**
  - About half the neurons were unobserved or unannotated.
  - The weights may not be unique.
  - Extrasynaptic signalling is not modelled.
  - Animals were restrained.
  - Results vary across samples.

### B. Connectome-constrained models fitted to spontaneous or behaving activity

#### B1. Mi, Xu, Prakhya, Lin, Shavit, Samuel & Turaga. "Connectome-constrained latent variable model of whole-brain neural activity." ICLR 2022

- **Access level:** abstract, from the Janelia publication page. That page lists the year as 2021; an OpenReview listing (seen only in a search snippet) and the conference name say ICLR 2022. I could not open the paper itself (PDF unparseable; the ICLR page returned 500).
- **Model:** a connectome-constrained latent variable model (CC-LVM): a biophysical simulation of latent voltages across the nervous system, linked to observed calcium. It is fitted as a variational autoencoder, with an inference network for the voltage posterior and the simulation as prior. The fitting is gradient-based.
- **Results (abstract):**
  - On an experimental whole-brain dataset (not named in the abstract), the connectome constraints predicted withheld neurons' activity "substantially better" than models without them.
  - Conductance-based synapses predicted better than current-based ones.
  - No numbers at abstract level.
- **Validation:** held-out neurons. Held-out animals or stimulations are unknown at this access level.

#### B2. Morrison & Young. "A data-driven biology-based network model reproduces *C. elegans* premotor neural dynamics." *PLoS Comput Biol* 21(12):e1013818 (29 Dec 2025), DOI 10.1371/journal.pcbi.1013818. Preprint arXiv 2501.00278, "A data-driven biophysical network model…"

- **Access level:** full text of arXiv v1 (HTML), plus the published PLOS version (about 100k of 147k characters; labelled "Uncorrected Proof").
- **Data:**
  - Atanas et al. 2023 freely moving NeuroPAL recordings: 22 datasets with enough core neurons, each with 64–106 labelled neurons. Six datasets were used to tune the timescale.
  - Voltage-clamp I–V curves of RMD set the intrinsic cubic.
- **Model class:**
  - ODEs for the GCaMP z-score of 15 core premotor neurons: AVB, RIB, RID, AVA, RIM, AVE, AIB and AVD (left/right).
  - Each neuron has a cubic intrinsic term with its own bias, linear gap-junction coupling, and ReLU-rectified chemical input.
  - 112 "signal" neurons are not simulated: their recorded activity is fed in as input. PVC and DVA are excluded because they are absent from the imaging data.
- **Connectome and fitted parameters:**
  - Gap-junction weights are fixed from White et al. 1986 (symmetric), with one global scale β = 10 chosen by a sweep.
  - Chemical synapses may be nonzero only where the connectome has a synapse. Their signs and magnitudes, and the per-neuron biases, are fitted.
- **Fitting method:**
  - Linear least-squares regression on the rearranged equation (with τ = 1), with weights averaged across the 22 datasets.
  - τ was then tuned to 0.2 to match reaction times.
  - A 1.4× gain on synaptic input compensates for unrecorded presynaptic neurons.
  - Missing neurons are replaced by a partner correlated at least 70% (30 such pairs).
- **Validation design:**
  - No held-out split: the 22 fitting datasets are also the evaluation pool.
  - Comparisons are mostly qualitative: time series, pair correlations, activity distributions and dwell times. Mean dwell times are 0.2–0.33 min in both data and model, slightly longer in the model.
  - In-silico ablations are checked against the literature:
    - RIB: forward fraction falls from 0.46 to 0.23 and pause fraction rises from 0.13 to 0.44. The literature reports reduced speed without a significant change in reversal frequency.
    - AIB: forward duration rises from 0.30 to 0.38 and reversal duration falls from 0.39 to 0.14. Extended forward runs match the literature; the experimentally reported change in reversal frequency does not occur in the model.
    - Heat shock: reversals increase, but the model shows no persistent post-stimulus state, unlike the data.
- **Controls:** gap-junction-only and synapse-only variants (Fig. 4). No shuffled connectome.
- **Comparison with the signal-propagation map:**
  - The model "shares some similarities with the map … but differs from it in several ways".
  - Excitation within the forward and reversal clusters is stronger and less varied than measured.
  - Forward neurons consistently inhibit reversal neurons in the model, whereas the experiment shows both signs.
  - The authors attribute the differences to pathways the model leaves out.
- **Signs:**
  - Forward→reversal inhibition matches one polarity atlas "fairly well".
  - RIM→AVA and RIM→AVB conflict between sources.
- **Stated limitations:**
  - No feedback from core to signal neurons or from motor neurons.
  - Indirect pathways outside the core are omitted.
  - Missing neurons are filled with proxies, and the 1.4× factor is ad hoc.
  - Synaptic signs are uncertain.
  - Other states (turns, quiescence) are not modelled.
  - "Identifiability has not been established."

#### B3. Zhao, Wang, Jiang, Ma, Ma, He, Du, Ma & Huang. "An integrative data-driven model simulating *C. elegans* brain, body and environment interactions" (BAAIWorm). *Nat Comput Sci* 4:978–990 (Dec 2024), DOI 10.1038/s43588-024-00738-w. Preprint "MetaWorm", bioRxiv 10.1101/2024.02.22.581686

- **Access level:**
  - Full text of the bioRxiv preprint (about 100k of 156k characters).
  - The published version through Europe PMC XML (PMC11659166), about 100k of 173k characters; the methods were truncated.
- **Data:**
  - Voltage- and current-clamp recordings of five representative neurons (AWC or AWA, the text is inconsistent; AIY, AVA, RIM, VD5).
  - The Pearson correlation matrix of 65 identified head neurons from a prior whole-brain calcium study, cited only as ref. 24 and not named in the text.
- **Model class:**
  - Multicompartment neurons with realistic morphology, Hodgkin–Huxley soma (14 channel types) and passive neurites.
  - 136 neurons in the published version: 15 sensory, 80 motor and the rest interneurons.
  - The body is in 3-D fluid, with 96 muscles.
- **Connectome and fitted parameters:**
  - Synapse and gap-junction counts come from EM adjacency matrices.
  - Channel parameters were fitted to electrophysiology for 5 neurons and shared within 5 functional groups; fewer than 20% of neurons have recordings.
  - Connection weights and polarities were optimised, starting from random weights with an excitatory/inhibitory ratio of 2.33.
- **Fitting method:**
  - Gradient descent with an unpublished approximate-gradient backpropagation through time.
  - The objective is the MSE between simulated and experimental correlation matrices.
  - Note: the model correlates membrane voltages while the target is calcium; this mismatch is not discussed in the preprint.
- **Validation:**
  - MSE 0.076. PC1 separates forward from backward neurons.
  - No held-out data: the fit targets and the checks use the same 65 neurons.
  - The authors say this validates the optimiser rather than the parameters, which are "one of many solutions".
- **Behaviour:**
  - A linear 80×96 motor-neuron→muscle readout drives the body; the preprint says it is trained on 10 s of simulated movement.
  - The network is used as a reservoir, and the preprint does not separate the network's contribution from the readout's.
  - The zigzag chemotaxis trajectory is compared visually and is "not as regular" as in experiments.
- **Controls:** neurite removal, shuffled connection locations, shuffled weights, and removal of all synapses or all gap junctions. Results are ranked by distance between correlation matrices, with no numbers or statistics in the text; seed 64 is shown and others are in the supplement, and outcomes depend on the seed.
- **Stated limitations:**
  - 136 of 302 neurons, and only zigzag locomotion.
  - Dale's law is not enforced; no plasticity.
  - Simplified sensing and hydrodynamics.
  - Generalised channel parameters.
  - Training is expensive.

#### B4. Warrington, Spencer & Wood. "The virtual patch clamp: imputing *C. elegans* membrane potentials from calcium imaging." arXiv 1907.11075 (24 Jul 2019)

- **Access level:** full text (about 100k of 197k characters).
- **Model:**
  - A whole-connectome simulator of all 302 neurons ("Simple C. elegans", after Kunert and Wicks, with calcium dynamics added) coupled to the WormSim body.
  - Hill-type calcium-to-fluorescence observation.
  - The latent state has 994 dimensions; the authors describe the model as non-differentiable.
- **Method:**
  - Sequential Monte Carlo for imputation.
  - "Particle marginal variational optimisation": SMC evidence plus a score-function gradient with ADAM, for parameters.
- **Data:** synthetic only, conditioned on the 49 neurons whose identities match the Kato et al. dataset. No real recordings were fitted.
- **Results:**
  - Recovers 2 coupling parameters across 20 initialisations (500 gradient steps, up to 17 h on 10 nodes).
  - Imputes unobserved voltages in simulation.
- **Limitations:**
  - No real data.
  - All other neural parameters are fixed and assumed known.
  - Some motor neurons reach unphysiological voltages.

#### B5. nih-niddk-mbs/r-elegans (GitHub). "Differentiable, connectome-constrained *C. elegans* simulation in JAX"

- **Access level:** README (about 100k of 115k characters). This is a code repository, not a publication, and its results are self-reported.
- **What it is:**
  - A 302-neuron graded-potential network with conductance models for AWCON and RMD, a resistive-force body and a 95-muscle adapter.
  - Wiring from Cook et al. 2019; signs from Wang et al. 2024 transmitter assignments.
- **Stated goal:** "fit one shared model to physiology, neural activity, movement, and perturbation data, then test it on held-out animals". The README says it is "not yet an achieved biological model or a completed joint-data trainer".
- **Results (self-reported):**
  - Many calcium-fitting attempts are negative:
    - E026: initial fits "have not established predictive accuracy".
    - E052: objective fell 5.3% while "nominal calcium error worsened slightly".
    - E053: loss fell 11.1%, but "calcium agreement was essentially unchanged" and bend fluctuations were 11–34× too small.
  - Date-group holdouts exist for the paired data, but the E050 pilot used no held-out targets.
  - The most recent dated entries are from November 2026.
- **Datasets:** no named neural datasets visible (an unnamed ~569 MB archive).

### C. Anatomy versus function, and variability of the substrate

#### C1. Uzel, Kato & Zimmer. "A set of hub neurons and non-local connectivity features support global brain dynamics in *C. elegans*." *Curr Biol* 32(16):3443–3459.e8 (2022), DOI 10.1016/j.cub.2022.06.039

- **Access level:** abstract, through Europe PMC. Cell Press returned 403.
- **Findings (abstract):**
  - Few local connectivity motifs predicted functional relationships (activity correlations).
  - Non-local features predicted them better: triplet motifs and shared input patterns.
  - Connection strength and the number of shared inputs did not improve prediction, so topology is the key factor.
  - Hub neurons were central to the informative features, and inhibiting several hubs specifically disrupted brain-wide correlations.
- **Not available at abstract level:** methods and numbers.

#### C2. Seyedolmohadesin, Fu, Torkashvand, …, Cook, Schroeder, Yemini & Venkatachalam. "Whole-brain chemosensory responses of both *C. elegans* sexes." bioRxiv 10.1101/2025.05.15.654129 (May 2025)

- **Access level:** full text (about 100k of 189k characters).
- **Data:**
  - 22 hermaphrodites and 25 males, NeuroPAL with GCaMP6s, partially immobilised with levamisole.
  - 8 chemical cues plus 2 controls, each repeated three times.
- **Analysis:**
  - Functional dimorphism (difference in mean correlation coefficients between the sexes) against synaptic dimorphism, from three hermaphrodite connectomes and one male connectome.
- **Findings:**
  - Hermaphrodite and male functional connectivity were highly similar (R² = 0.8).
  - There was no clear relationship between synaptic and functional dimorphism (Fig. 4D; no coefficient given).
  - The authors' framing of the prior literature: "synaptic and functional connectivity are nearly entirely uncorrelated" (introduction) and "little to no simple correlation between the two" (discussion).
- **Limitations:**
  - The synaptic metric ignores sign.
  - Only one male connectome exists.
  - Head neurons only; partial immobilisation.

#### C3. Witvliet, Mulcahy, Mitchell, …, Samuel & Zhen. "Connectomes across development reveal principles of brain maturation." *Nature* 596:257–261 (2021), DOI 10.1038/s41586-021-03778-8. Preprint bioRxiv 10.1101/2020.04.30.066209

- **Access level:** full text of preprint v2 (about 100k of 181k characters). Journal metadata from Europe PMC.
- **Data:** 8 isogenic N2 hermaphrodites reconstructed by EM from L1 to adult.
- **Finding:**
  - "About 43% of all connections and 16% of all synapses are not conserved between animals."
  - Modulatory neurons are the most variable in their outputs, and motor neurons the least.
- **Relevance:** a model's mask, from any one animal or a sum of them, is not the recorded animal's wiring.

#### C4. Currier & Clandinin. "Infrequent strong connections constrain connectomic predictions of neuronal function." *Cell* (2025), DOI 10.1016/j.cell.2025.05.007 per search snippets; bioRxiv 10.1101/2025.03.06.641774

- **Access level:** abstract (bioRxiv).
- **Not the worm.** This is *Drosophila* visual system work: 91 cell types recorded.
- **Findings:**
  - Connectome predictions are accurate for some properties (orientation tuning) and "surprisingly poor" for others (receptive field size).
  - Strong inputs are functionally homogeneous.
  - "physiology is a stronger predictor of wiring than wiring is of physiology".
- **Relevance:** included only because the brief asked. It says nothing directly about *C. elegans*.

### D. Fits to worm neural data without a connectome: performance references

#### D1. Kato, Kaplan, Schrödel, Skora, Lindsay, Yemini, Lockery & Zimmer. "Global brain dynamics embed the motor command sequence of *Caenorhabditis elegans*." *Cell* 163(3):656–669 (2015), DOI 10.1016/j.cell.2015.09.034

- **Access level:** abstract (Europe PMC).
- **Finding:**
  - Brain-wide activity forms a shared cyclic network state, and each phase maps onto a motor command (forward, reverse, turn), forming run-and-turn sequences.
  - Neuron counts and preparation details are not given in the abstract. The Brennan & Proekt preprint (D3) gives 107–131 neurons per worm for five Kato worms.

#### D2. Linderman, Nichols, Blei, Zimmer & Paninski. "Hierarchical recurrent state space models reveal discrete and continuous dynamics of neural activity in *C. elegans*." bioRxiv 10.1101/621540 (2019)

- **Access level:** full text (about 100k of 194k characters).
- **Data:**
  - Kato 2015: 5 immobilised worms, about 18 min each, 59 neurons identified in at least one worm.
  - Nichols 2017: 44 worms in 4 groups (N2 and npr-1, prelethargus and lethargus), 22–42 identified neurons per worm, with O2 varied.
- **Model:**
  - A hierarchical recurrent switching linear dynamical system (rSLDS): 10 continuous latent dimensions; 8 discrete states for Kato and 12 for Nichols, chosen by cross-validation.
  - Per-worm parameters are tied to a global mean, O2 enters as an input, and the noise is heavy-tailed.
- **Fitting:** factor analysis, with unobserved neurons treated as missing at random, then EM. A black-box variational alternative is also described. No connectome.
- **Validation:**
  - Held-out log-likelihood.
  - The hierarchical model beat the non-hierarchical one, which needed 14 states and scored lower held-out likelihood.
  - I did not see held-out-worm tests or numbers in the portion I read.
- **Findings:**
  - Discrete states match forward, reverse, turns and quiescence.
  - Transitions depend on the continuous state and on O2.
  - Recurrent simulations were better than non-recurrent ones, which diverged.
- **Limitations:**
  - Only a subset of neurons is observed, and immobilisation limits the repertoire.
  - Long simulations "eventually become unstable", even for the recurrent model.

#### D3. Brennan & Proekt. "A quantitative model of conserved macroscopic dynamics predicts future motor commands." *eLife* 8:e46814 (2019), DOI 10.7554/eLife.46814. Preprint: "Universality of macroscopic neuronal dynamics in *Caenorhabditis elegans*," arXiv 1711.08533

- **Access level:**
  - eLife: abstract (Europe PMC); the eLife page failed to load.
  - Preprint: full text (ar5iv, about 100k of 111k characters).
- **Data:** Kato 2015, 5 worms with 107–131 neurons each. Only 15 neurons were identified in all five, and those were used across worms.
- **Model:**
  - Two macroscopic variables: which loop (flux) the state is on, and the phase along it.
  - Built with an asymmetric diffusion map on delay-embedded activity, with Louvain clustering. No connectome.
- **Validation:**
  - Leave-one-worm-out: a manifold built from four worms predicts the fifth.
  - The eLife abstract says the model predicts the timing of motor commands up to 30 s ahead in individuals not used to build it, and reproduces dwell-time statistics, command sequences and single-neuron activity.
  - The preprint gives no single held-out accuracy figure, only p-values and entropy comparisons.
- **Variability:**
  - Some identified neurons differ qualitatively between worms: AVAL, RMED and SMDVR, with ANOVA p = 2.1×10⁻⁷, 3.7×10⁻¹⁷ and 1.0×10⁻⁹.
  - A neuron can be consistent across worms in one behaviour and not in another.
  - The model works "despite consistent inter-individual differences in neuronal activation" (eLife abstract).
- **Limitations:** 5 worms, 15 shared neurons, many tunable settings.

#### D4. Costa, Ahamed & Stephens. "Adaptive, locally linear models of complex dynamics." *PNAS* 116(5):1501–1510 (2019), DOI 10.1073/pnas.1813476116

- **Access level:** abstract (Europe PMC).
- **What it does:**
  - Fits local linear models in adaptively chosen windows.
  - Applied mainly to worm posture: fine-grained states near an instability boundary, and a bifurcation in the forward-to-backward switch.
  - In worm whole-brain imaging, lower oxygen damps global dynamics away from that boundary.
  - No connectome.

#### D5. Simeon, Venâncio, Boyden, Yang & Nayebi. "Scaling properties for artificial neural network models of a small nervous system." bioRxiv 10.1101/2024.02.13.580186 (14 Feb 2024); also SoutheastCon 2024 per search snippets

- **Access level:** full text (bioRxiv v1). The author list on the page differs from search snippets, which add Skuhersky.
- **Data:**
  - Eight open datasets: Kato 2015, Nichols 2017, Skora 2018, Kaplan 2020, Yemini 2021, Uzel 2022, the Leifer atlas and Flavell 2023. 284 worms (the per-source counts sum to 287).
  - Z-scored ΔF/F, causally smoothed and resampled to Δt ≈ 0.667 s, with a mask over the 302 neurons.
- **Model:**
  - Six architecture classes (Linear, Feedforward, LSTM, Transformer, CTRNN, Attention), each a single-layer core between an embedding and a linear readout.
  - No connectome.
- **Training:** next-step prediction with masked MSE, AdamW.
- **Validation:** temporal only, within each worm: the first half trains and the second half validates. No held-out-worm split.
- **Numbers (validation loss):**

| Model | Loss | Parameters |
|---|---|---|
| Copy-last baseline | 0.035413 | |
| Linear | 0.035328 | |
| Feedforward | 0.036377 | |
| Transformer | 0.036321 | 55.5M |
| Attention | 0.072782 | |
| LSTM | 0.034167 | |
| CTRNN | 0.032193 | 222k |

- The best model is about 9% below the copy-last baseline. That percentage is my arithmetic; the paper gives only the raw losses.
- Loss falls roughly log-linearly with data; at about 0.58M parameters the slope is −0.65 for CTRNN and −0.40 for Feedforward.
- **What failed:** "long-horizon predictive and generative models" were not achieved; autoregressive generation is shown only qualitatively.
- **Limitations:**
  - Data-limited.
  - Uneven neuron coverage: some neurons appear in only one worm.
  - No behavioural context.

#### D6. Simeon, Astrom, Kashyap, Kording & Boyden. "Homogenized *C. elegans* neural activity and connectivity data." arXiv 2411.12091 (v1 18 Nov 2024; v4 19 May 2025)

- **Access level:** full text (arXiv v4 HTML).
- **What it is:**
  - A data resource of 12 calcium-imaging studies, about 919 worms by Table 1, including Leifer2023 (110) and Lin2023 (577, about 8 labelled neurons per worm).
  - 10 connectome files: Cook2019, White1986 subsets, Witvliet2020 datasets 7 and 8, OpenWorm2023, Chklovskii2011 and Randi2023 functional connectivity.
  - Released on HuggingFace. No models trained.
- **Stated limitations:**
  - The connectome "does not come from the same animals used for neural activity measurements".
  - Synaptic strengths vary across individuals and the lifespan.
  - Calcium is low-pass and lagged.
  - Wireless (neuropeptide) connections are missing.
  - Z-scoring over the whole series is non-causal.

#### D7. Sprague, …, Kato. "Unifying community-wide whole-brain imaging datasets enables robust automated neuron identification and reveals determinants of neuron positioning in *C. elegans*." bioRxiv 10.1101/2024.04.28.591397 (30 Apr 2024)

- **Access level:** abstract.
- **What it is:** WormID, 118 whole-brain activity datasets from 5 labs in a standard (NWB-based) format, plus a statistical atlas for automated neuron identification across labs. No dynamics model.

#### D8. Atanas, Kim, Wang, …, Mansinghka & Flavell. "Brain-wide representations of behavior spanning multiple timescales and states in *C. elegans*." *Cell* 186(19):4134–4151.e31 (2023), DOI 10.1016/j.cell.2023.07.035. Preprint bioRxiv 10.1101/2022.11.11.516186

- **Access level:**
  - Cell version: abstract (Europe PMC); the Europe PMC full-text XML returned 500.
  - Preprint v1: full text, about 200k of 265k characters.
  - Numbers below are the preprint's; the Cell version may have more animals (the homogenised dataset lists 40 Flavell2023 worms).
- **Data:**
  - Freely moving recordings: 14 GCaMP animals (143 ± 12 head neurons each) and 12 NeuroPAL animals (96 ± 14 identified neurons).
- **Model:**
  - CePNEM, a Bayesian per-neuron encoding model of velocity, head curvature and feeding, with direction interactions and an exponential-history timescale (half-decay 0.9–31.7 s).
  - Fitted in Gen (particle filtering plus MCMC). Not a dynamical or connectome model.
- **Controls:**
  - 2.1% of neurons "encode" behaviour in pan-neuronal GFP controls and 2.7% with another animal's behaviour, against 58.6% in GCaMP datasets.
- **Variability:**
  - Several classes (AIN, OLQ, IL1, RIH, URY) had tunings that "differed across animals and/or were highly multiplexed".
  - About 20% of encoding neurons changed their encoding during a recording (14% in the NeuroPAL strain).
- **Limitations:**
  - Association, not causation.
  - The extent to which representations are constrained by wiring versus neuromodulation "remains to be seen".

#### D9. Kaplan, Salazar Thula, Khoss & Zimmer. "Nested neuronal dynamics orchestrate a behavioral hierarchy across timescales." *Neuron* 105(3):562–576.e9 (2020), DOI 10.1016/j.neuron.2019.10.037

- **Access level:** abstract.
- **Finding:** slow brain-wide dynamics gate faster motor-neuron oscillations, nested across three timescales. No model fit to report at this level.

#### D10. Other models without a connectome, abstract only

- Yezerets, Mudrik & Charles. "Decomposed linear dynamical systems (dLDS) models reveal context-dependent dynamic connectivity in *C. elegans*." bioRxiv 10.1101/2024.05.31.596903 (1 Jun 2024). Reusable linear operators and one aligned latent space across worms; compares anatomical, functional and "dynamical" connectivity. No numbers in the abstract.
- Morrison, Fieseler & Kutz. "Nonlinear control in the nematode *C. elegans*." arXiv 2001.08332 (2020). A global nonlinear control model with 4 parameters that reproduces low-dimensional features and the state distribution of whole-brain data; linear or locally linear models cannot capture the multiple fixed points with a single model.
- Kinger, Bertram, Dyballa, Yemini & Zucker. "Inferring active neural circuits using diffusion scores." arXiv 2605.02852 (v1 4 May 2026; v2 8 Sep 2026). Score models recover lag-specific directed interactions from worm whole-brain data, with "improved alignment with independent connectomes". The dataset is not named in the abstract.
- Dyballa, Lang, Haslund-Gourley, Yemini & Zucker. "Learning dynamic representations of the functional connectome in neurobiological networks." ICLR 2024, arXiv 2402.14102. Unsupervised dynamic affinities, with results "experimentally validated". Comparison with anatomy not stated in the abstract.

### E. Connectome-only dynamical models, not fitted to neural recordings

#### E1. Varshney, Chen, Paniagua, Hall & Chklovskii. "Structural properties of the *Caenorhabditis elegans* neuronal network." *PLoS Comput Biol* 7(2):e1001066 (2011), DOI 10.1371/journal.pcbi.1001066

- **Access level:** abstract.
- **What it is:**
  - Self-consistent chemical and gap-junction networks built from White et al. plus new micrographs.
  - Uses linear systems theory to model how activity spreads after stimulation.
  - Says the wiring "can guide predictions" for perturbation experiments. Nothing fitted.

#### E2. Kunert, Shlizerman & Kutz. "Low-dimensional functionality of complex network dynamics: neuro-sensory integration in the *Caenorhabditis elegans* connectome." arXiv 1310.6689 (v2 Feb 2014)

- **Access level:** full text of the arXiv preprint. Journal version not confirmed.
- **Model:**
  - 279 somatic neurons from Varshney 2011, with Wicks-type graded single-compartment dynamics.
  - Uniform conductances: g = 100 pS for each synapse and gap junction.
  - Signs: GABAergic neurons inhibitory, all others excitatory.
- **Fitted:** nothing; parameters are taken from the literature.
- **Validation:** qualitative only.
  - PLM stimulation gives two-mode motor dynamics compared with eigenworms.
  - Ablations of AVB and AVA are compared with Chalfie 1985.
  - Robust to ±20% parameter perturbations.
- **Limitations:** connection parameters are uncertain, and the model "does not produce directly the behavioral response".

#### E3. Kunert-Graf, Shlizerman, Walker & Kutz. "Multistability and long-timescale transients encoded by network structure in a model of *C. elegans* connectome dynamics." *Front Comput Neurosci* 11:53 (2017), DOI 10.3389/fncom.2017.00053

- **Access level:** abstract.
- **Findings:**
  - With identical neurons that differ only in connectivity, the model shows multistability: fixed points and limit cycles depending on input.
  - Input to sensory neurons is more likely than input to motor neurons to cause a bifurcation.
  - Transients last much longer than intrinsic timescales. Nothing fitted to recordings.

#### E4. Kim, Florman, Santos, Alkema & Shlizerman. "Modular integration of neural connectomics, dynamics and biomechanics for identification of behavioral sensorimotor pathways in *Caenorhabditis elegans*." arXiv 2504.18073 (v1 25 Apr 2025; v2 3 Jun 2025). Also bioRxiv 724328

- **Access level:** abstract. The PDF was unparseable and the HTML returned 404.
- **What it is:**
  - modWorm, a modular neuromechanical framework built on the complete connectome.
  - Validated by current injection into locomotion neurons and by external forces applied to the body; reproduces forward and backward movement, avoidance and turning.
  - In-silico ablations propose new sensorimotor circuits, and "empirically based model variations and optimizations" are tested.
- **Not available at abstract level:** what is fitted, to which data, and how.

#### E5. Sunil, Benali & Moutuou. "Thermodynamic connectivity reveals functional specialization and multiplex organization of extrasynaptic signaling." arXiv 2604.02057 (2 Apr 2026)

- **Access level:** abstract.
- **What it is:** functional connectivity derived from the synaptic and neuropeptidergic connectomes by equilibrium statistical physics. No fit to recordings, as far as the abstract shows.

### F. The behaviour-first counterpoint

#### F1. Wicks, Roehrig & Rankin. "A dynamic network simulation of the nematode tap withdrawal circuit: predictions concerning synaptic function using behavioral criteria." *J Neurosci* 16(12):4017–4031 (1996), DOI 10.1523/jneurosci.16-12-04017.1996

- **Access level:** abstract (Europe PMC).
- **Method:**
  - An anatomy- and physiology-constrained network of the tap-withdrawal circuit.
  - Every array of synaptic signs was enumerated exhaustively, and each was scored against the behaviour of animals with ablated neurons.
- **Result:** four related experiments predicted the polarity of 7 of 9 cell classes. The total number of sign configurations is not in the abstract.

#### F2. Izquierdo & Lockery. "Evolution and analysis of minimal neural circuits for klinotaxis in *Caenorhabditis elegans*." *J Neurosci* 30(39):12908–12917 (2010), DOI 10.1523/jneurosci.2606-10.2010

- **Access level:** abstract.
- **Method:** an evolutionary algorithm, with inputs and outputs constrained to the klinotaxis circuit.
- **Results:**
  - The networks reproduced two experimental observations they were not fitted to.
  - A dynamical analysis of 77 evolved networks proposed a new orientation mechanism.

#### F3. Izquierdo & Beer. "Connecting a connectome to behavior: an ensemble of neuroanatomical models of *C. elegans* klinotaxis." *PLoS Comput Biol* 9(2):e1002890 (2013), DOI 10.1371/journal.pcbi.1002890

- **Access level:** abstract.
- **Method:**
  - The circuit was taken from connectome pathways (chemosensory to neck motor neurons) and pruned.
  - Unknown electrophysiological parameters were set by evolution to match behaviour.
- **Result:** an ensemble of models that fit the behaviour with different mechanisms. The authors propose experiments to tell the alternatives apart.

### G. Proposals

#### G1. Haspel, Baker, Beets, …, Kording (37 authors). "The time is ripe to reverse engineer an entire nervous system: simulating behavior from neural interactions." arXiv 2308.06578 (v1 12 Aug 2023; v5 18 Sep 2024), first titled "To reverse engineer an entire nervous system"; an opinion paper

- **Access level:** abstract. The arXiv abstract page is v5; a mirror page shows an earlier abstract. Full text unavailable: the HTML returned 404, ar5iv failed, and the PDF was unparseable.
- **Proposal:**
  - Measure each neuron's state-dependent input–output function by optophysiology across many animals, pooling data because neuron identities are "largely conserved".
  - Then use machine-learning model training to simulate the worm's states and behaviours.
  - The earlier abstract says the approach can "scale to hundreds of thousands of experiments, across a large population of animals".
- **Not available at abstract level:** data needs, timelines, costs and connectome constraints.

______________________________________________________________________

## 2. What remains open

Each item is a problem one of the sources names, with its source.

01. **Nonlinearity, multistability and state dependence.**
    - The only fitted model of perturbation data is linear, with one fixed point, no bistability or limit cycles, no state dependence, and response magnitudes that are too small (Creamer et al. v3/v4, limitations).
    - Nonlinearities, state and plasticity in signal propagation are not characterised (Randi et al., limitations).
    - Linear or locally linear models cannot capture multiple fixed points with one model (Morrison, Fieseler & Kutz 2020, abstract).
    - The anatomy-only model shows multistability (Kunert-Graf et al. 2017).
02. **Extrasynaptic and neuropeptide signalling.**
    - At least 53 functional connections depend on dense-core vesicles, as a lower bound (Randi et al.).
    - Whether a connectome mask can represent extrasynaptic signalling is unresolved:
      - The fully connected model did better on the putative extrasynaptic pairs, p = 0.011 in v3 but p = 0.1 in v4.
      - On synthetic data, the constrained model fails when extrasynaptic signalling is strong (Creamer et al.).
    - Not modelled by Toyoshima et al. 2024 or BAAIWorm. "Wireless" connections are missing from the connectome data (Simeon et al. homogenised data). Wiring against neuromodulation "remains to be seen" (Atanas et al. preprint).
03. **One model across animals.**
    - "Significant animal-to-animal variability" may mask effects, and the train–test reproducibility of STAMs is only 0.16–0.17 (Creamer et al.).
    - Responses vary in presence, amplitude, kinetics and sign between trials (Randi et al.).
    - 43% of connections are not conserved between isogenic animals (Witvliet et al.).
    - Neurons differ qualitatively across worms (Brennan & Proekt preprint).
    - Toyoshima et al. fit each animal separately and find sensory routes and motif relationships that differ between animals.
    - Encodings of the same class differ across animals (Atanas et al. preprint).
    - The connectome comes from different animals than the activity (Simeon et al. homogenised data).
04. **Identifiability and degeneracy of fitted parameters.**
    - The parameters are "one of many solutions" (BAAIWorm).
    - The weights may not be unique (Toyoshima et al.).
    - "Identifiability has not been established" (Morrison & Young).
    - Behaviour-fitted ensembles reach the same behaviour by different mechanisms (Izquierdo & Beer 2013).
    - Ablation behaviour resolved the signs of 7 of 9 classes, not all (Wicks et al. 1996).
05. **Synaptic signs.**
    - Regressed signs conflict across studies (Morrison & Young).
    - AFD→AIY is measured excitatory though predicted inhibitory, and the inhibitory fraction is likely underestimated (Randi et al.).
    - Dale's law is not enforced (BAAIWorm).
    - The synaptic metric ignores sign (Seyedolmohadesin et al.).
06. **Calcium versus voltage.**
    - Calcium is a proxy and voltage may propagate faster (Creamer et al.).
    - Only nuclear calcium is imaged (Randi et al.; Atanas et al.).
    - "No formula" relates voltage to calcium (Morrison & Young).
    - Calcium is low-pass and lagged (Simeon et al. homogenised data).
    - Latent voltage with a calcium observation model is the CC-LVM approach (Mi et al.). Converting voltage to calcium for comparison was handled in only some works; BAAIWorm fits voltage correlations to a calcium target.
07. **Coverage and unobserved neurons.**
    - Models are of head neurons only (Creamer et al.; Seyedolmohadesin et al.).
    - About half the neurons are unobserved (Toyoshima et al.).
    - PVC and DVA are absent, many neurons are rarely labelled, proxies stand in for missing traces, and a 1.4× gain compensates for missing input (Morrison & Young).
    - Coverage is uneven, with some neurons in only one worm (Simeon et al. 2024).
    - The CC-LVM's held-out-neuron gains are claimed in its abstract only (Mi et al.).
08. **Immobilised versus behaving preparations.** Perturbation data exist only for immobilised animals (Randi et al.). Restrained animals (Toyoshima et al.) and immobilisation limit the repertoire (Linderman et al.).
09. **Long-horizon generation.**
    - Long-horizon generative models were not achieved (Simeon et al. 2024).
    - rSLDS simulations eventually become unstable (Linderman et al.).
    - A deterministic synapse-based model decays to flat activity, so noise is needed (Toyoshima et al.).
10. **How much data, and does more help?**
    - Relative correlation is "roughly independent of data set size" (Creamer et al., Fig. 1I).
    - Loss falls log-linearly with data, and more recordings are "fruitful" (Simeon et al. 2024).
    - "Hundreds of thousands of experiments" are proposed (Haspel et al., abstract).
11. **Only pairwise perturbations.** Only single-neuron stimulation has been done, so nonlinear group interactions are untested (Creamer et al.; Randi et al.).
12. **From neural fit to behaviour.**
    - Other locomotor states are not modelled, and ablation predictions partly disagree with the literature (Morrison & Young).
    - The behaviour readout is trained, and the network's contribution is not separated from it (BAAIWorm preprint).
    - The model "does not produce directly the behavioral response" (Kunert et al.).
    - Biological fidelity, autonomous locomotion and food-seeking are unvalidated (r-elegans README).
    - I found no published work that fits a connectome model to neural data and then scores it on held-out behaviour statistics.
13. **Stronger nulls.**
    - Shuffled-connectome results are given only qualitatively in the text (Creamer et al.), and a random-mask null at matched density is not reported.
    - BAAIWorm's shuffle controls have no statistics.
    - Toyoshima et al. did not test all-to-all links.

______________________________________________________________________

## 3. Sources I could not find or could not open

- **PMC12485687**, the PMC copy of Creamer et al.: reCAPTCHA. I used the bioRxiv v1, v3 and v4 pages instead.
- **Creamer_LDS_2024**, the older repository: not checked.
- **Randi et al., *Nature* published version** (nature.com login redirect; PMC10632145 captcha). I used the arXiv preprint through ar5iv. Extended Data values (kernel stereotypy, unc-31 proportions) are in figures I could not see.
- **Leifer-lab follow-ups 2024–2026** on extrasynaptic or neuropeptide propagation, or on stimulation in freely moving worms: none found apart from Dvali et al. 2025 (abstract only). The TWISP strain paper (Sharma, Randi, Kumar, Dvali & Leifer, *Genetics* 2024) appeared only in search results; I did not open it, so it is not an entry.
- **Uzel et al. 2022 full text:** Cell Press 403. Abstract only.
- **Atanas et al. 2023, *Cell* full text:** the Europe PMC XML returned 500. I used the 2022 preprint.
- **Brennan & Proekt 2019, eLife full text:** the page failed to load. I used the abstract plus the arXiv preprint.
- **Kim et al. 2025 (modWorm) full text:** PDF unparseable, HTML 404. Abstract only, so what is fitted to what remains unknown.
- **Haspel et al. full text:** HTML 404, ar5iv failed, PDF unparseable. Abstract only, so its data and timeline estimates are missing.
- **Dvali et al. 2025, *PRX Life* full text:** PDF unparseable. Abstract only.
- **Mi et al. CC-LVM full text:** PDF unparseable, ICLR page 500. Abstract only. The dataset it used and its numbers are unknown, and the year is inconsistent between sources (2021 against ICLR 2022).
- **Kunert et al. 2014:** the *Physical Review E* version is not confirmed. I read the arXiv preprint.
- **Kato 2015 and Kaplan 2020:** abstracts only.
- **Currier & Clandinin 2025:** abstract only, and it concerns *Drosophila*, not the worm.
- **Not found before the search budget ran out:**
  - Any 2024–2026 graph-neural-network or "foundation model" trained on pooled worm whole-brain data with held-out-animal results. The Simeon et al. homogenised dataset says it is meant for this but reports no models.
  - Any nonlinear refit of the Randi atlas.
  - Beiran & Litwin-Kumar's work on prediction in connectome-constrained recurrent networks, which is not worm-specific.
  - Ripoll-Sánchez et al.'s neuropeptide connectome.
  - Fieseler, Zimmer & Kutz 2020.
  - Jaxley (Deistler et al., *Nat Methods* 2025): search snippets show it is a differentiable biophysical simulator. I could not verify any *C. elegans* application, so it is not an entry.
- **Search budget exhausted.** The web-search budget ran out during the final 2025–2026 sweep, so recent preprints, especially from Sep 2025 to Nov 2026, are probably under-covered.
