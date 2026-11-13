# Literature review: the layers a fit-first worm model would have to fit through

*Compiled 2026-11-14 with web search and web fetch only. This part covers calcium versus voltage, intrinsic currents, neuromodulation, plasticity and variability between animals. It gives no verdict on fit-first.*

## How to read this file

- **Access levels.** "Full text" means I opened the article body, sometimes only the first 100–200k characters of a long page (marked "partial"). "Abstract" means I saw only the abstract or metadata. "Second-hand" means I did not open the source and report it from another source I did open, which is named.
- **Extraction caveat.** WebFetch returns a small model's extraction from each page, not the raw text. Every number below came through that step. Check any number that a decision will rest on against the paper.
- **Search budget.** The shared web-search budget ran out during question 4. After that I opened only URLs I already had or could build from known identifiers. The 2025–2026 literature is covered unevenly, especially for questions 4 and 5.
- **Discrepancies** between sources, or between a paper's abstract and its body, are flagged where I saw them.

______________________________________________________________________

## Q1. Calcium, voltage and indicators in *C. elegans*

### 1.1 Goodman, Hall, Avery & Lockery 1998, *Neuron* 20:763–772, "Active currents regulate sensitivity and dynamic range in C. elegans neurons"

- **Access:** full text (PMC4444786).
- **Measured:** in situ patch clamp of the identified sensory neuron ASER across L1–L4 (22–23 recordings) and of 42 unidentified L1 neurons.
- **Numbers:**
  - ASER input resistance was 2.0–8.4 GΩ (n = 9).
  - The cell is nearly isopotential. Capacitance scales with body length (slope 1.07 ± 0.08), and electrotonic length is about 0.37 (amphid neurite) and 0.21 (ring neurite).
  - Injected current gives graded, nonlinear responses with no classical Na⁺ action potentials. A depolarisation-activated K⁺ current lowers membrane resistance above about −30 mV, which limits sensitivity to strong inputs.
  - ASER has an estimated maximum of about 30 Ca²⁺ channels.
  - The voltage dependence of membrane current was similar in every neuron examined.
- **Limitations stated:**
  - The preparation was dissected, so the in vivo ionic milieu is unknown.
  - Seal and pipette capacitance may have shunted the cell, and Ca²⁺ spikes were not ruled out.
  - Series resistance was not compensated, and responses washed out over the recording.

### 1.2 Lockery & Goodman 2009, *Nat Neurosci* 12:377–378, "The quest for action potentials in C. elegans neurons hits a plateau"; Mellem, Brockie, Madsen & Maricq 2008, *Nat Neurosci* 11:865–867 (RMD)

- **Access:** second-hand. The pages were paywalled or behind a browser check, so I have only citation metadata and descriptions from search results and citing pages.
- **What I could establish:**
  - Mellem et al. reported regenerative, Ca²⁺-dependent, long-lived events in RMD, evoked by small depolarising steps, and none in AVA.
  - Lockery and Goodman argued that these are plateau potentials rather than action potentials in the rate- or timing-code sense. A companion letter, "First report of action potentials in a C. elegans neuron is premature", appeared in the same issue.
- I could not verify any numbers from either paper.

### 1.3 Liu, Kidd, Dobosiewicz & Bargmann 2018, *Cell* 175:57–70, "C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials"

- **Access:** full text of the bioRxiv preprint (10.1101/359935), partial. The Cell version was not opened, and its numbers may differ.
- **Survey:** current clamp and I–V recordings of four neuron types (RIM n = 3, AIY n = 7, AFD n = 3, AWA n = 16). The authors classed them as follows:
  - RIM: transient outward rectifier.
  - AIY: outward rectifier.
  - AFD: bistable, with potential confined near −80 or −20 mV.
  - AWA: bistable and spiking.
- **AWA spikes** (preprint text, n = 19):
  - Threshold about −34.0 ± 0.6 mV.
  - Amplitude 25.8 ± 0.8 mV above threshold.
  - Half-width 18.0 ± 0.8 ms, with an 8.4 mV afterhyperpolarisation.
  - Bursts up to 10.5 Hz (mean 4.2 Hz).
  - Spikes are initiated by EGL-19 (CaV1) and terminated by SHK-1 (Kv1).
  - A search-engine summary of the same preprint gave different figures (111 mV amplitude, −36 mV threshold, 117 V/s). I did not see those in the text I opened, so treat them as unverified.
- **Simultaneous patch clamp and GCaMP6f in AWA** (this is direct voltage-to-calcium evidence):
  - Graded depolarisation to about −40 mV produced a smooth, roughly linear rise in fluorescence after a delay of about 1 s.
  - Repolarisation produced decreases with no appreciable delay.
  - Each spike produced an immediate increase in the slope of the fluorescence trace. Spike-associated signals were larger in *shk-1* mutants and in the axon.
  - The analysis used a 1-s decay kernel. No millisecond rise or decay values were given.
- **Model:** a Hodgkin–Huxley AWA model with these currents:
  - leak and EGL-19, with voltage-dependent self-inactivation;
  - SHK-1, with conductance and time constant fitted to voltage clamp;
  - SLO-type K⁺, tuned to the I–V curve;
  - inward-rectifier K⁺, setting rest near −75 mV;
  - two stochastic K⁺ channels (Kr and Kra), tuned by bifurcation analysis.
- **Limitation stated:** spike inference from calcium in intact animals is "an indirect approach". Thresholds were tuned against the simultaneous recordings.

### 1.4 Jiang, Su, Zhang, Li, Tao & Liu 2022, *Nat Commun*, "C. elegans enteric motor neurons fire synchronized action potentials underlying the defecation motor program" (AVL, DVB)

- **Access:** full text (Europe PMC, PMC9120479), partial.

- **Numbers (current clamp, dissected):**

  | Cell | Rest | Threshold | Peak | Half-width | n |
  |---|---|---|---|---|---|
  | DVB | −49.0 mV | −23.4 mV | 26.3 mV | 359 ms | 5 |
  | AVL | −45.7 mV | −23.3 mV | 9.8 mV | 394 ms | 17 |

  - AVL fires a compound spike: a UNC-2 (CaV2) Ca²⁺ spike followed by an EXP-2-mediated K⁺ "negative spike".
  - In vivo GCaMP6f calcium spikes recur once per defecation cycle (about 45 s). AVL leads DVB by 215 ± 176 ms.
  - A Hodgkin–Huxley AVL model was tuned to voltage and current clamp, with parameters in the supplement.

- **Limitations stated:**

  - Electrophysiology and calcium imaging were done in separate preparations.
  - Calcium is called an indirect proxy, and GCaMP bleaching affected spike amplitudes.
  - Several experiments have small n.

### 1.5 Azimi Hashemi et al. 2019, *PNAS* 116:17051–17060, "Rhodopsin-based voltage imaging tools for use in muscles and neurons of Caenorhabditis elegans"

- **Access:** full text (PMC6708366), partial.
- **Sensors:** Arch(D95N), Archon, QuasAr and the eFRET sensors MacQ-mCitrine and QuasAr-mOrange. They were tested in body-wall and pharyngeal muscle, and **QuasAr alone in neurons** (RIM, and cholinergic motor neurons in the retrovesicular ganglion).
- **Sensitivity:**
  - About 128% ΔF/F per 100 mV for Arch(D95N) with retinal in pharynx.
  - About 78% for Arch(D95N) and QuasAr in body-wall muscle.
  - The abstract quotes 45–78% for the eFRET sensors, which does not match the body-text values (about 10–34%). The paper does not explain the difference.
- **Neuronal signals:**
  - RIM showed rapid fluctuations up to about 4% ΔF/F, brief events of about 8% lasting about 120 ms, and slow 5–15% fluctuations.
  - Cholinergic motor neurons showed about 4% fluctuations, read as graded potentials.
- **Voltage versus calcium:** not imaged simultaneously. RIM calcium recorded separately showed the slow fluctuations but **lacked the brief events**.
- **Limitations stated:**
  - Neuronal signals are small and were not calibrated in neurons.
  - Recordings were in immobilised animals.
  - Some sensors need high illumination and bleach.
  - Signals varied between animals.

### 1.6 Chen et al. 2013, *Nature* 499:295–300, "Ultrasensitive fluorescent proteins for imaging neuronal activity" (GCaMP6)

- **Access:** full text (Europe PMC, PMC3777791). The absolute kinetic values are in figures and supplementary tables that the text extraction did not reach.
- **In the text:**
  - GCaMP6f rises about 2× faster and decays about 1.7× faster than GCaMP5G in culture.
  - GCaMP6s has about 3× higher apparent Ca²⁺ affinity than GCaMP5G.
  - In mouse V1, single spikes were resolvable at intervals of about 100–150 ms (6s), 75–100 ms (6m) and 50–75 ms (6f).
- **Not obtained:** absolute half-rise and half-decay times, Kd and Hill coefficients.

### 1.7 Dana et al. 2019, *Nat Methods* 16:649–657 (jGCaMP7)

- **Access:** not opened. I found only bibliographic records, and no kinetics.
- Second-hand jGCaMP7f values from Zhang et al. 2023 (below):
  - half-decay median 110 ms in mouse V1 at 30 Hz imaging;
  - half-rise 128 ± 11 ms and half-decay 277 ± 29 ms in *Drosophila* L2 neurons.

### 1.8 Zhang et al. 2023, *Nature* 615:884–891, "Fast and sensitive GCaMP calcium indicators for imaging neural populations" (jGCaMP8)

- **Access:** full text (Europe PMC, PMC10060165), partial. GCaMP6s/6f values sit in Extended Data Table 3, which I did not reach.
- **Kinetics:**
  - Abstract: half-rise about 2 ms. Text: 6.6 ± 1.0 ms for jGCaMP8f in the cultured-neuron screen.
  - In vivo 0–95% rise: 5 ms (8f), 5 ms (8m), 10 ms (8s), against 30 ms for jGCaMP7f.
  - Mouse V1 half-decay medians: 84 ms (8f), 84 ms (8m), 110 ms (7f).
  - jGCaMP8s half-decay is about 200 ms in mouse.
  - Fly L2 half-decay: 192 ± 26 ms (8f), 137 ± 21 ms (8m), 198 ± 21 ms (8s).
- **Linearity:**
  - Hill coefficients are 1.9–2.2 for jGCaMP8, against 3.1 for jGCaMP7f.
  - jGCaMP8 is more linear across 1–5 APs. GCaMP6s is highly nonlinear.
- **Relevance:** all of these are mammalian or fly measurements, mostly at the soma, triggered by spikes. They are not graded-potential responses in worm neurons.

### 1.9 Ding, Peng, Moon, Lee, Patel & Lu 2023, bioRxiv 10.1101/2023.03.06.531342, "An expanded GCaMP reporter toolkit for functional imaging in C. elegans"

- **Access:** full text.
- **Worm-specific measurement:** GCaMP6f, 6s, jGCaMP7f and 7s, each in cytosolic, nuclear and membrane-targeted forms, in mechanosensory neurons (mainly ALM). Stimulus was mechanical, 3 s at 2.5 Hz.
- **Decay half-times after stimulus:**
  - jGCaMP7s 18 ± 8 s (slowest).
  - GCaMP6f 2.7 ± 2.0 s (fastest).
  - jGCaMP7f about half of 7s.
  - Membrane-targeted 7s about 31 s in the cell body and 16.4 s in the neurite.
- **Caveat (mine):** these decays mix the neuron's own response time course with indicator kinetics. They are not pure indicator constants. Even so, they are one to two orders of magnitude longer than the mammalian in vivo half-decays above.
- jGCaMP8 was not tested.

### 1.10 Indicators used in whole-brain and multi-neuron worm datasets (as stated in the papers I opened)

| Dataset | Indicator | Imaging rate | Animals | Access |
|---|---|---|---|---|
| Nguyen et al. 2016 (arXiv 1501.03463), freely moving | GCaMP6s | 5 head-volumes/s | 78 neurons recorded | abstract |
| Randi et al. 2023 *Nature*, perturbation atlas | nuclear GCaMP6s | 2 volumes/s | 113 WT-background + 18 *unc-31* | full text |
| Atanas et al. 2023 *Cell* | NLS-GCaMP7f | 1.7 Hz (1.4 Hz on one rig) | 14 brain-wide, 40 NeuroPAL atlas | full text, partial |
| Dag et al. 2023 *Cell*, preprint version on PMC | NLS-GCaMP7f |: | 7 + 6 | full text, partial |
| Ji et al. 2021 *eLife*, 10 neurons, freely moving | GCaMP6m | about 2 Hz | 17 WT | full text, partial |
| Liang et al. 2025 bioRxiv, about 78% of 94 head classes across 7 sparse strains | cytoplasmic GCaMP6 | 5 frames/s | 856 worms | full text, partial |

- Atanas et al. state that GCaMP7f half-decay is under 1 s. They attribute their fitted encoding timescales of 0.9–31.7 s to neural encoding, not to the sensor.
- I did not open Kato et al. 2015 or Uzel et al. 2022 and cannot state their indicators.
- None of the opened whole-brain datasets used jGCaMP8.

### 1.11 How modellers have treated calcium

- **Randi et al. 2023** (full text; methods partial):
  - Kernels are sums of up to two terms, each a convolution of decaying exponentials. They map the stimulated neuron's observed ΔF/F to the responder's.
  - No deconvolution and no explicit indicator model.
  - The responses accepted for fitting depend on thresholds originally set to match human perception of a response.
- **Creamer, Leifer & Pillow 2024**, bioRxiv 10.1101/2024.09.22.614271, "Bridging the gap between the connectome and whole-brain activity in C. elegans" (full text):
  - Latent linear dynamical system. Its weights are nonzero only on connectome edges (9.4% of pairs).
  - Gaussian observation of ΔF/F with an offset. No GCaMP kinetics or deconvolution.
  - A Kalman smoother handles unrecorded neurons. Fitted by EM.
- **Atanas et al. 2023, CePNEM** (full text, partial): an exponentially weighted moving average of behaviour plus a Gaussian-process residual. No GCaMP observation model.
- **Linderman et al. 2019**, bioRxiv 10.1101/621540 (full text, partial): fitted to standardised ΔF/F directly, each neuron a noisy linear readout of a 10-D latent. The paper states that no smoothing or derivative is needed.
- **Morrison & Young 2024**, arXiv 2501.00278 (full text):
  - The state variable is the GCaMP z-score itself. The authors write that there is no known formula relating calcium to voltage.
  - The fixed points of a cubic intrinsic term are read off peaks in the data distribution.
- **Warrington, Spencer & Wood 2019**, "The Virtual Patch Clamp", arXiv 1907.11075 (full text, partial):
  - An explicit observation model: a calcium ODE low-pass filtering voltage (after Rahmati et al. 2016) plus a Hill-type fluorescence map with noise.
  - Inference by sequential Monte Carlo.
  - **Synthetic data only**, 49 observed neurons and traces of about 5 s, with two parameters learned.
- **Mi et al. 2022, ICLR**, "Connectome-constrained latent variable model of whole-brain neural activity":
  - Access: bibliographic metadata, plus an abstract description from search results (second-hand).
  - A VAE over latent voltages of the whole nervous system with an observation model to calcium.
  - Applied to recordings of about 170 neurons across multiple worms under chemosensory stimuli.
  - Conductance-based synapses predicted held-out neurons markedly better than current-based ones.
  - I could not read the calcium model's details.
- **Zhao et al., MetaWorm preprint** (bioRxiv 10.1101/2024.02.22.581686; published 2024 in *Nat Comput Sci* as BAAIWorm, version not opened). Full text of the preprint, partial:
  - Weights and polarities were fitted by gradient descent so that the correlation matrix of *simulated membrane potentials* matches a 65-neuron *calcium* correlation matrix.
  - No calcium model.

**Q1, cross-cutting points (mine, from the sources above):**

- Direct voltage-to-calcium evidence in identified worm neurons is limited to the papers opened here:
  - simultaneous patch clamp and GCaMP6f in AWA (Liu 2018);
  - separate voltage and calcium recordings in AVL/DVB (Jiang 2022) and in RIM (Azimi Hashemi 2019).
- In AWA, graded depolarisation shows up in calcium after about 1 s. In RIM, brief voltage events seen with QuasAr were absent from calcium.
- The only worm voltage imaging found is rhodopsin-based. In neurons the signals were about 4–8% ΔF/F and uncalibrated, and none were simultaneous with calcium.
- Every whole-brain model I opened that was fitted to real data treats ΔF/F or its z-score as the observation, without an indicator model. The one explicit calcium observation model (Warrington 2019) was tested only on synthetic data.

______________________________________________________________________

## Q2. Intrinsic currents and priors

### 2.1 Nicoletti, Loppini, Chiodo, Folli, Ruocco & Filippi 2019, *PLoS One* 14:e0218738, "Biophysical modeling of C. elegans neurons: single ion currents and whole-cell dynamics of AWCon and RMD"

- **Access:** full text (Europe PMC, PMC6602206), partial.
- **Model:** 14 channel models (several K⁺ channels, EGL-19, UNC-2, CCA-1, SLO-1, SLO-2, KCNL-1/4, NCA, leak). Channels were chosen by expression in WormBase and the literature.
- **Kinetics sources:**
  - Ten of the 14 cite *C. elegans* data. Several of these are heterologous expressions of worm genes. Native worm recordings are cited only for SHL-1 and EGL-19.
  - UNC-2 kinetics come from rat/mammal data and CCA-1 from *Drosophila*.
  - KCNL-1/4 and NCA have no published patch-clamp data.
- **Fitting:**
  - About 3 parameters per gating function, fitted by least squares.
  - BK channels: 8 parameters by genetic algorithm plus SQP.
  - SK channels not fitted: K_Ca was set to 0.33 µM with a Hill coefficient of 1.
  - Conductances were set to match reported peak and steady-state currents.
- **Assumed from other organisms or other studies:**
  - single-channel conductance 40 pS;
  - nanodomain and buffer parameters;
  - basal Ca²⁺ 0.05 µM and the Ca²⁺ removal time constant;
  - cell volumes from NeuroMorpho.
- **Limitations stated:**
  - Fits may sit at local minima.
  - Worm electrophysiology is hard because of small somata and cell viability.
  - Validation rests on two patch-clamp datasets.

### 2.2 Nicoletti, Chiodo, Loppini, Liu, Folli, Ruocco & Filippi 2024, *PLoS One* 19:e0298105, "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families"

- **Access:** full text (Europe PMC, PMC10980225), partial.
- **Neurons:** AIY, RIM, AVAL/AVAR, VA5, VB6 and VD5, drawn from 17 current types.
- **Use of CeNGEN as a prior.** For AIY, AVA and RIM, channels were chosen from CeNGEN at threshold 1 (AIY) or 2 (AVA, RIM), cross-checked with WormBase and the literature. For example, AIY has 6 currents and AVA has 5.
- **What was free after that:**
  - Gating kinetics were **fixed**.
  - Only maximal conductances, the leak reversal and the capacitance were fitted, by least squares or NSGA-II.
  - Conductances fitted near zero were dropped.
  - E_K was fixed at −80 mV and E_Ca at +60 mV.
- **Data fitted:**
  - AIY and RIM: current and voltage clamp from Liu et al.
  - AVA: current clamp.
  - Motor neurons: published I–V curves, and the fit was **hand-tuned**.
- **Limitations stated:**
  - Single compartment, for lack of data on where channels sit in the cell.
  - In silico knockouts are not comparable to mutants, which may compensate.
  - Mutant recordings exist only for the motor neurons.
  - Some fit errors remain.
  - The authors say action potentials have been observed only in AWA and AVL.
- No count of electrophysiologically characterised neurons is given; the paper says only a few mechanisms and sub-networks have been explored.

### 2.3 Naudin, Jiménez Laredo, Liu & Corson 2022, *PLoS One* 17:e0268380, "Systematic generation of biophysically detailed models with generalization capability for non-spiking neurons"

- **Access:** full text (Europe PMC, PMC9106219), partial. The parameter tables S1 and S2 were not reached.
- **Neurons and fitting:**
  - RIM, AIY and AFD, with four-current models.
  - Multi-objective differential evolution against current-clamp traces and steady-state I–V curves (RIM n = 3, AIY n = 7, AFD n = 3 cells). Data mostly from Liu et al. 2018.
  - Held-out traces at 30 and 35 pA were used for selection and test.
- **Finding:** single-objective fits produced spurious N-shaped I–V curves for AFD and RIM. The two objectives conflict, which yields a Pareto set of solutions.
- **What stays loose:**
  - Many parameters are constrained only to plausible ranges.
  - The authors say microscopic ionic data are largely unavailable, so many parameters remain hypotheses.
  - Steady-state curves are cell averages while traces come from single cells.
  - Generalisation was checked on one held-out trace per split.

### 2.4 Conductance models of AWA (Liu 2018, see 1.3) and AVL (Jiang 2022, see 1.4)

- Both are Hodgkin–Huxley models fitted to the authors' own patch-clamp data.
- In AWA, two stochastic K⁺ channels were added and tuned by bifurcation analysis. That means some components were introduced to reproduce behaviour rather than measured.

### 2.5 How many worm neurons have published patch-clamp recordings?

- **No authoritative count found.**
- The MetaWorm preprint (Zhao et al. 2024, full text, partial) states, without a citation in the text I saw, that "Less than 20% of the C. elegans neurons have been recorded".
- The identified neurons with recordings in the sources I opened are:
  - ASER and 42 unidentified neurons (Goodman 1998);
  - RIM, AIY, AFD and AWA (Liu 2018);
  - RMD and AVA (Mellem 2008, second-hand);
  - AVL and DVB (Jiang 2022);
  - VA5, VB6 and VD5, and AVA, among the data sources of Nicoletti 2024;
  - AWCon (Nicoletti 2019 validation).
- This list is not exhaustive.
- **Parameter sharing in practice:** MetaWorm fitted only 5 neurons (AWC, AIY, AVA, RIM, VD5) to recordings. Every other neuron copies the parameters of its functional group (sensory, inter, command, head motor, body motor).

### 2.6 Taylor et al. 2021, *Cell* 184:4329–4347, "Molecular topography of an entire nervous system" (CeNGEN)

- **Access:** full text (PMC8710130), partial.
- **Coverage:** 70,296 neurons profiled at L4, covering all 118 anatomical classes, resolved into 128 transcriptional types.
- **Thresholds:** four expression thresholds were calibrated against 160 ground-truth genes. At threshold 2 the true positive rate is 0.81 and the false discovery rate 0.14.
- **Per-class counts:** about 23 neuropeptide genes (range 6–62), about 36 neuropeptide receptors, about 20 ionotropic receptors, and 1–18 TWK K⁺ channels.
- **Limitations stated:**
  - Some classes are under-represented because of dissociation bias.
  - Most non-polyadenylated RNAs are missing.
  - Alternative splicing is rarely detected.
- mRNA is not protein, and subcellular localisation is not addressed; these are my additions, and Fenyves 2020 says the same.

### 2.7 Fenyves, Szilágyi, Vassy, Sőti & Csermely 2020, *PLoS Comput Biol* 16:e1007974, "Synaptic polarity and sign-balance prediction using gene expression data in the C. elegans chemical synapse neuronal connectome network"

- **Access:** full text (PMC7785220), partial.
- **Expression to receptor map:** presynaptic transmitter matched to postsynaptic receptor genes, using WormBase, WormAtlas and binarised CeNGEN. Cation channels count as excitatory and anion channels as inhibitory.
- **Coverage:** polarity was predicted for about 73% of 20,589 synapses, with excitatory to inhibitory near 4:1 (counts give about 3.5:1).
- **Validation:**
  - 70% agreement (456 of 652 synapses) with an earlier locomotor-circuit prediction.
  - Against 12 interneuronal connections with experimental polarity, 1 was discordant, against 6 of 12 for a transmitter-only rule.
- **Limitations:** subunit expression is not the same as functional receptors, and localisation, monoamines, peptides and metabotropic receptors are excluded.
- **Related:** Randi et al. 2023 also assigned polarities in its biophysical model from gene-expression analysis, setting ambiguous cases to excitatory.

### 2.8 Did any paper turn expression into conductances, and how much did that narrow parameters?

- **Into conductances.** Nicoletti 2024 used CeNGEN to choose *which* currents each neuron has. Kinetics and maximal conductances still came from heterologous or other-organism data and from fits.
- **Into receptor maps.** Fenyves 2020 (polarity), Ripoll-Sánchez 2023 (peptide edges; see 3.5) and Dag 2023 (serotonin receptors; see 3.3) did this.
- **Narrowing not quantified.** I found no paper that measured how much expression priors narrowed a posterior over conductances.
- The closest evidence is that expression fixes the channel set, after which multi-objective fits still return a Pareto set (Naudin 2022). In other systems, maximal conductances are degenerate even when the channel set is fixed (Gonçalves 2020; see 3.10).

______________________________________________________________________

## Q3. Neuromodulation: what is measured and how it is modelled

### 3.1 Flavell, Pokala, Macosko, Albrecht, Larsch & Bargmann 2013, *Cell* 154:1023–1035, "Serotonin and the neuropeptide PDF initiate and extend opposing behavioral states in C. elegans"

- **Access:** full text (PMC3942133).
- **Behavioural states:**
  - Two-state HMM on 350 wild-type animals.
  - Dwelling durations fit a single exponential (τ ≈ 482 s).
  - Roaming fits two exponentials (τ ≈ 75 s and 520 s; about 85% of bouts are short).
  - In *mod-1* mutants, 69% of dwelling bouts are short (τ ≈ 63 s).
- **Sources and receptors:**
  - Serotonin comes from NSM and HSN. MOD-1 (5-HT-gated Cl⁻) acts in AIY, ASI and RIF.
  - PDF comes from AVB, PVP and SIAV. PDFR-1 (Gαs) acts in AIY, RIM and RIA.
  - The two systems overlap mainly at AIY.
- **Optogenetics:** for example, 1 min of ChR2 in serotonergic neurons moved 70% of roaming animals to dwelling, against 19% of controls.
- **Release not measured.** Neither serotonin nor PDF release was measured. Roles were inferred from genetics, ablation, optogenetics and calcium in NSM and AIY.
- **Diffusion not tested.** The authors call the diffusion distance plausible but untested.
- **Unexplained:** the origin of NSM's spontaneous calcium peaks, and why roaming persists after cAMP stimulation ends.

### 3.2 Ji, Madan, Fischer, Ma, Stolarski, Flavell et al. 2021, *eLife* 10:e62889, "A neural circuit for flexible control of persistent behavioral states"

- **Access:** full text (PMC8660023), partial.
- **Imaging:** 10 neurons (NSM, AVB, AIY, RIB, AVA, AIA, among others) with GCaMP6m at about 2 Hz, in 17 freely moving wild-type animals.
- **Mutants:** *tph-1*, *mod-1*, *pdfr-1* and the double mutant.
- **Mechanism:** mutual inhibition between NSM and MOD-1 neurons, with PDFR-1 neurons downstream inhibiting NSM.
- **Findings:**
  - NSM activity rises about 23 s before roaming-to-dwelling transitions.
  - A logistic classifier on all 10 neurons reached over 95% state accuracy.
  - NSM bouts are shortened in *tph-1* and *mod-1* and more persistent in *pdfr-1*.
- **Release still not measured.** 5-HT and PDF release and concentration were not measured. The authors suggest lingering extracellular 5-HT as a possible cause of variable lags, untested.
- **No biophysical model.** The analysis is statistical only (PCA, HMM, CNN).

### 3.3 Dag, Nwabudike, Kang, Gomes, Kim, Atanas, Flavell et al. 2023, *Cell* 186:2574–2592, "Dissecting the functional organization of the C. elegans serotonergic system at whole-brain scale"

- **Access:** full text of the preprint version on PMC (PMC9882198), partial.
- **Receptor map from knock-ins:**
  - Six CRISPR T2A-mNeonGreen knock-in reporters identified with NeuroPAL.
  - Each receptor is in 8–24 of the 118 classes, and 52 classes (44%) express at least one.
  - Compared with Taylor 2021, with differences noted but not quantified.
- **Behavioural genetics:**
  - 64 strains lacking every combination of six receptors.
  - SER-4 mediates a transient slowing at the onset of release. MOD-1 mediates sustained slowing.
  - A linear model with receptor-presence terms plus 18 interaction terms was cross-validated across genotypes.
- **Whole-brain imaging (NLS-GCaMP7f) during NSM activity:**
  - 33% of neurons were associated with NSM activity and 21% with its derivative.
  - Temporal kernels ranged up to about ±10 s.
  - Receptor-expressing neurons were somewhat more likely to correlate with NSM (p = 0.054).
- **Release not measured.** NSM was driven optogenetically; no serotonin sensor was used.
- **Limitations stated:** promoter rescues may not match native expression, and unidentified receptors may exist.

### 3.4 Bentley et al. 2016, *PLoS Comput Biol* 12:e1005283, "The multilayer connectome of Caenorhabditis elegans"

- **Access:** full text (PMC5215746), partial.
- **Monoamine layer:** 1,940 directed edges (2,260 with provisional *dop-5/6*): serotonin 490, dopamine 1,168, octopamine 54, tyramine 228.
- **Neuropeptide layer:** a partial network of 12 receptors, giving 7,035 connections among 239 neurons.
- **Assumptions:**
  - Every receptor-expressing neuron is a responder.
  - No distance restriction.
  - One edge per pair.
- **Unknown:** valence, strength, timing and signalling range ("uncharacterized for most neuropeptide systems").
- **Limitations stated:** reporter false negatives are likely, and the networks are static.

### 3.5 Ripoll-Sánchez et al. 2023, *Neuron* 111:3570–3589, "The neuropeptidergic connectome of C. elegans"

- **Access:** full text (Europe PMC author manuscript EMS192830), partial.
- **Inputs:**
  - CeNGEN at threshold 4, checked against 17 precursor and 9 GPCR reporters.
  - Beets 2023 pairs at EC50 ≤ 500 nM, giving 92 neuropeptide–receptor gene couples.
  - Four spatial models: long range, mid range (same body region), short range (shared process bundle), and contact.
- **Density:** short range 0.344 and mid range 0.443, against 0.025 for the synaptic and 0.024 for the monoamine networks. That is more than 10-fold denser.
- **Limitations stated:**
  - Release sites and diffusion range are not measured; expression is the proxy.
  - Pairings come from in vitro assays.
  - The network is incomplete.
  - Edge weights count possible pairings, not signal strength.
  - Release dynamics and timing are not modelled.

### 3.6 Beets et al. 2023, *Cell Rep* 42:113058, "System-wide mapping of peptide-GPCR interactions in C. elegans"

- **Access:** full text (PMC7615250), partial.
- **Screen:** about 55,384 peptide–receptor combinations (344 peptides against 161 receptors) gave 776 candidate hits.
- **Validated pairs:** 459 in the Results text, against 461 in the abstract.
- **EC50:** about 0.1 pM to 22 µM, with 73% of couples sub-micromolar. My extraction of Ripoll-Sánchez gave "22 mM", probably a typo or extraction error.
- **In vivo:** the authors state that over 10 of the new couples have been validated in vivo by other groups.
- **Limitations stated:**
  - Heterologous expression may fail for some receptors or need co-factors.
  - The Gα16 readout is weaker for Gi-coupled receptors.
  - EC50s may differ in vivo.
  - The map is incomplete.

### 3.7 Beets & Watteyne 2025, *Curr Opin Neurobiol*, "Mapping and decoding neuropeptide signaling networks in nervous system function"

- **Access:** full text (Europe PMC, PMC12178255), partial.
- **Relevance:** this recent review gives **no *C. elegans* release kinetics or diffusion distances**.
- **Sensors:** in vivo peptide-release and GPCR sensors are described mainly in mouse and fly, with none described for the worm.
- **Mouse figure:** peptides act more than 100 µm from release sites within seconds.
- **Open problems listed:**
  - compartmental release, diffusion and degradation;
  - how neurons integrate many peptide inputs;
  - which interactions are accessible, and when;
  - cAMP and DAG sensing at brain scale.
- **Modelling:** existing functional analyses treat the connectomes as static.

### 3.8 Measured extrasynaptic effects in the perturbation atlas (Randi et al. 2023; see 1.10)

- *unc-31* (dense-core vesicle release-defective) animals had fewer functional connections, and their data agreed better with anatomy than wild-type data did.
- 53 candidate purely extrasynaptic pairs were found.
- RID stimulation evoked URX, ADL and AWB responses that were reduced or absent in *unc-31*, with no wired path.
- This is the nearest thing to a whole-brain measurement of peptidergic signalling among the sources I opened. It reads out calcium, not peptide.

### 3.9 Any measurement of neuropeptide release or diffusion time course in the worm?

- Among the sources I opened, none.
- Hiroki et al. 2022 (see 4.5) measured basal *glutamate* release from ASER with *eat-4*::pHluorin and FRAP; that is not a peptide.
- I could not open Laurent et al. 2018 (PNAS, pHluorin-tagged peptides in PQR) or Ding et al. 2019 (eLife, neuropeptide release reporter; the organism was not confirmed). Both are in the final list.

### 3.10 Models with neuromodulator dynamics fitted to data

- **Worm.** I found none. Among the worm models I opened:
  - Morrison & Young 2024: no neuromodulators.
  - Creamer et al. 2024: lists neuromodulation and extrasynaptic signalling as not modelled.
  - MetaWorm: none.
  - Dag 2023: a statistical genotype-to-behaviour model, not dynamics.
- **Kringelbach et al. 2020**, *PNAS* 117:9566–9576, "Dynamic coupling of whole-brain neuronal and neurotransmitter systems" (human; full text PMC7196827, partial):
  - **Structure.** A dynamic mean-field model of 90 regions, coupled to Michaelis–Menten serotonin release and reuptake driven by the raphe, with a 5-HT2A PET density map (210 people) scaling the serotonergic current. Fitted to fMRI from 9 participants (psilocybin against placebo).
  - **Free parameters:** global coupling G (fitted to placebo) and the two feedback gains W_ES and W_IS (fitted by grid search).
  - **Fixed from the literature:** V_max = 1,300 nM/s, K_m = 170, and the sigmoid and neuronal parameters.
  - **Identifiability:** W_ES and W_IS lie on a ridge with an invariant ratio, so they are not separately identified.
  - **Cohorts:** structural, functional and receptor data come from different people.
- **Gonçalves et al. 2020**, *eLife* 9:e56261, "Training deep neural density estimators to identify mechanistic models of neural dynamics" (full text PMC7581433, partial):
  - Pyloric STG model with 31 parameters (24 membrane and 7 synaptic conductances) fitted to one *Cancer borealis* recording with 18 summary features.
  - Many marginals cover nearly the whole prior. Tight tuning appears only in conditional distributions, and widely separated parameter sets both reproduce the rhythm.
  - Neuromodulation was not modelled in the part I read.
  - This is evidence on degeneracy with a fixed channel set, not on neuromodulator fitting.
- **Shim, Prinz & Smolinski 2012**, *BMC Neurosci* 13(Suppl 1):P37 (abstract only):
  - AB-neuron model in a 12-D space (21.6 million combinations), with neuromodulation represented as presence or absence of a proctolin current.
  - Conductance correlations persist with changed slopes.
- **Not found:** an STG paper that fits neuromodulator kinetics themselves to data.

______________________________________________________________________

## Q4. Plasticity and learning: what is measured and how it is modelled

### 4.1 Ha et al. 2010, *Neuron* 68:1173–1186, "Functional organization of a neural network for aversive olfactory learning in Caenorhabditis elegans" (PA14)

- **Access:** full text (PMC3038580), partial.
- **Paradigm:** about 6 h on PA14. Learning appears within about 2 h, is complete by about 4 h, and fades over about 2 h on OP50.
- **Circuits:**
  - Naive preference needs AWB and AWC, plus AIY, AIB and AIZ.
  - The learned shift needs ADF (serotonin; TPH-1, MOD-1), RIA, SMD and RIM.
- **Calcium:** G-CaMP in AWC and AWB, n ≥ 5 per condition. Sensory responses were **unchanged by training**, so the change is inferred to lie downstream.
- **No computational model.**

### 4.2 Liang, Moon, Moza, Lee, Eleftheriadis, Chen, Ge, Chen, Lu & Zhang 2025, bioRxiv 10.1101/2025.10.31.685731, "Aversive learning induces context-gated global reorganization of neural dynamics in Caenorhabditis elegans"

- **Access:** full text, partial.
- **Design:** the closest to brain-wide imaging before and after learning that I found.
  - Seven sparse-label GCaMP6 strains together cover about 78% of 94 head neuron classes.
  - 2,445 recordings from 856 worms.
  - Naive and trained worms are **separate groups**. Paired imaging of the same animal before and after training is not described.
- **Changes after learning:**
  - In a discrimination task, PA14-evoked responses changed in about 35% of sensory, 27% of inter and 6% of motor neuron types.
  - In a detection task only 5 types changed.
- **Models:** a linear dynamical system and double-exponential pairwise kernels. The authors call the kernels "an abstraction and simplification".
- **Unconstrained or limited:** missing neurons were imputed, variability between recordings was handled by bootstrap, and coverage is head neurons only.

### 4.3 Pritz et al. 2023, *eLife* 12:e74434, "Principles for coding associative memories in a compact neural network"

- **Access:** full text (PMC10159626), partial.
- **Paradigm:** butanone paired with food (appetitive) or starvation (aversive), for short-term (about 1–1.5 h) or long-term (about 12 h spaced) memory.
- **Imaging:** about 6–18 restrained animals per group, chosen sensory neurons, interneurons and command neurons. Not whole-brain; separate trained, mock and naive groups.
- **Findings:**
  - Sensory neurons are modulated mainly in short-term memory; interneurons in both.
  - Single-neuron decoding reached F1 ≤ about 50%; all chemosensory neurons together gave about 90% decoding accuracy.
- **Linear model:** five sensory types explained up to 88% of AIY variance, with different best combinations for each paradigm.
- **Variability:** high animal-to-animal variability. About half of naive or aversively trained animals had ASER responses, suggesting bimodality. Behavioural assays varied from day to day.

### 4.4 Cho, Brueggemann, L'Etoile & Bargmann 2016, *eLife*, "Parallel encoding of sensory history and behavioral preference during Caenorhabditis elegans olfactory learning"

- **Access:** full text (eLife API, article 14000), partial.
- **Paradigm:** butanone paired with food or with 90 min starvation.
- **AWC^ON imaging** (GCaMP5A, then nuclear GCaMP6s):
  - Both conditionings shifted the dynamic range, raising the threshold about ten-fold and speeding recovery. This is read as non-associative adaptation.
  - The associative, bidirectional change appeared in the behavioural output to ChR2-driven AWC activation.
  - Nuclear EGL-4 translocation depended on AIA.
- **No model.**
- **Limitations stated:**
  - The two AIA silencing methods disagree.
  - Short-term suppression recovers within 10 min.

### 4.5 Hiroki et al. 2022, *Nat Commun* 13:2928, "Molecular encoding and synaptic decoding of context during salt chemotaxis in C. elegans"

- **Access:** full text (PMC9142520), partial.
- **Measured:**
  - Cultivation salt sets basal glutamate release from ASER through PKC-1 and UNC-64 (Ser65), measured with *eat-4*::pHluorin and FRAP.
  - ASER's sensory calcium response is unchanged.
  - AIB's response sign flips depending on which of two receptors dominates: the sensitive inhibitory AVR-14 or the less sensitive excitatory GLR-1.
- **Model:** conceptual only.
- **Limitations stated:** the desensitisation difference was not significant, and the role of neuropeptide release was not separated out.

### 4.6 Hawk et al. 2018, *Neuron* 97:356–367, "Integration of plasticity mechanisms within a single sensory neuron of C. elegans actuates a memory" (thermotaxis)

- **Access:** full text of the bioRxiv preprint 10.1101/157958, partial.
- **AFD adaptation:** AFD's threshold adapts toward the holding temperature with τ ≈ 45 s for warming and about 14 min for cooling.
- **Presynaptic plasticity:** AFD-to-AIY plasticity was inferred from how often AIY responded (GCaMP6s, synaptic zones). Presynaptic release was not imaged directly.
- **Model:** a simple exponential model of the AFD threshold.
- **Test:** an ectopic gap junction bypasses the gate.
- **Limitations:** a single postsynaptic readout and immobilised imaging.

### 4.7 Bozorgmehr, Ardiel, McEwan & Rankin 2013, *Front Physiol* 4:88, "Mechanisms of plasticity in a Caenorhabditis elegans mechanosensory circuit" (tap habituation; review)

- **Access:** full text, partial.
- **Mechanisms:**
  - Short-term habituation involves *eat-4*, *dop-1* (food-dependent), *mps-1* and *tom-1*.
  - Long-term memory after spaced training needs GLR-1 and CREB, and training shrinks GLR-1 clusters.
  - Intermediate memory after massed training needs *flp-20* and involves SNB-1.
  - Context effects need NMR-1 in RIM.
- **Models:** the review describes no computational model. Wicks, Roehrig & Rankin 1996, a network simulation in which only synaptic sign was varied, came up only in search results; I did not open it.

### 4.8 Computational models of worm learning with a plasticity mechanism

- **Hironaka & Sumi 2025**, *eLife* 104456, "A neural network model that generates salt concentration memory-dependent chemotaxis in Caenorhabditis elegans" (full text, Europe PMC PMC12393885, partial):
  - Minimal connectome-derived klinotaxis circuit (ASEL, ASER, AIY, AIZ, SMB), fitted by evolutionary algorithm to a **behavioural** chemotaxis index (0.877 and 0.855).
  - Memory is an **imposed sign switch** of the ASER-to-AIY synapse, not a learning rule. It is motivated by analogy with Hiroki 2022, with the MGL-1 sensitivity extrapolated from MGL-2.
  - Calcium was used for validation only. The constrained model reproduces AIZ activity in sync with ASER; the unconstrained model gives the opposite sign at similar fitness.
  - In other words, behaviour alone does not discriminate between the networks.
- **Sanders & Cohen 2012**, *BMC Neurosci* 13(Suppl 1):P162 (abstract):
  - A salt chemotaxis associative-learning model. Synaptic plasticity alone "fails to explain all results", which points to cellular plasticity.
  - No rule details are given.
- **Hasani, Fuchs, Beneder & Grosu 2017**, arXiv 1703.06264 (abstract):
  - A multi-scale sensory-neuron and dynamic-synapse model said to reproduce tap habituation.
  - No fit statistics in the abstract.
- **Not found:** a worm learning model with a local plasticity rule fitted to neural data, or validated by predicting held-out neural data after learning.

### 4.9 Inferring plasticity rules elsewhere

- **Mehta, Tyulmankov, Rajagopalan, Turner, Fitzgerald & Funke 2023**, bioRxiv 10.1101/2023.12.11.571103, "Model-based inference of synaptic plasticity rules" (full text):
  - **Method.** The rule is parameterised as a Taylor series or an MLP and fitted by gradient descent through whole activity and behaviour trajectories.
  - **Validation.** It recovers Oja-type and reward-modulated rules in simulation.
  - **Fly application.** Behaviour only: 18 flies in a probabilistic odour-reward Y-arena. The fit infers a negative weight-dependent term ("active forgetting"), which improves fit significantly (p = 5·10⁻⁵).
  - **Limitations stated:**
    - The model is "sloppy" and "fails to identify a unique, sparse solution even with extensive data".
    - No metaplasticity.
    - Unknown initial weights and a possible need for the exact connectome.
    - Behavioural data may not be adequate.
- **Lim et al. 2015**, *Nat Neurosci*, "Inferring learning rules from distributions of firing rates in cortical neurons":
  - Access: second-hand, from Mehta et al. 2023; the paywalled page would not open.
  - Inferred the rule and the transfer function from firing-rate distributions before and after familiarity learning, assuming the distribution and a first-order form of the rule.
- **Also second-hand, from Mehta et al. 2023:**
  - Confavreux et al. 2020 (NeurIPS) meta-learned rules for a target network function. The target task may not be defined biologically.
  - Ashwood et al. 2020 fitted learning-rule parameters to rodent choices in a Bayesian framework.
  - Rajagopalan et al. 2023 fitted rules for the same fly task, depending only on presynaptic activity and reward, as logistic regression.
  - Chen et al. 2023 fitted rules with a Gaussian process or a Taylor expansion from weight or activity changes.
- **Jiang & Litwin-Kumar 2021**, *PLoS Comput Biol* 17:e1009205, "Models of heterogeneous dopamine signaling in an insect learning and memory center" (full text, partial):
  - **Plasticity.** Kenyon-cell-to-MBON synapses follow a fixed, dopamine-gated, timing-dependent rule with eligibility traces, *taken from experiment*, not inferred.
  - **Training.** The recurrent output circuitry is trained by gradient descent on tasks.
  - **Validation.** Qualitative agreement with recorded DAN responses (valence, reward prediction error, novelty, movement), not fits to recordings.
  - **Unconstrained:**
    - The feedback neurons are not mapped anatomically.
    - One rule is used for all compartments, since compartment-specific parameters are undescribed.
    - The DAN penalty and the tasks are the authors' choices.
    - Model neurons do not correspond to real cells.
- **Eschbach et al. 2019 preprint** (bioRxiv 10.1101/649731, "Multilevel feedback architecture for adaptive regulation of learning in the insect brain"; published 2020 in *Nat Neurosci* as "Recurrent architecture for adaptive regulation of learning in the insect brain", version not opened):
  - Access: abstract.
  - EM connectome of the circuitry upstream of the larval mushroom-body modulatory neurons, including 61 feedback neuron pairs, some tested functionally.
  - A circuit model constrained by connectome and functional data.
  - Whether a plasticity rule was fitted to data could not be determined from the abstract.

______________________________________________________________________

## Q5. Variability across animals

### 5.1 Witvliet et al. 2021, *Nature* 596:257–261, "Connectomes across development reveal principles of brain maturation"

- **Access:** full text (PMC8756380), partial.
- **Reconstructions:** 8 isogenic animals from L1 to adult, two of them adults.
- **Variability:**
  - About 43% of cell–cell connections are variable between animals (about 50% in a conservative comparison), about 43% stable and about 14% developmentally dynamic.
  - Variable connections carry about 16% of chemical synapses; stable connections about 72%.
  - No synapse-count threshold cleanly separates variable from stable connections.
- **Limitation stated:** most ages have one connectome, so variability at a given age was not measured directly.

### 5.2 Atanas et al. 2023 (see 1.10)

- **Encoding variability by class:**
  - Each neuron class has a variability index for how its encoding differs across animals.
  - Some classes are consistent (AVA, AIM). Others vary substantially (OLL, OLQ, IL1, RIH, URB).
- **Generalisation across animals** inversely tracked that index.
- **Behaviour matters per animal:** another animal's behaviour explained activity in 2.7% of neurons, against 58.6% for the correct animal's.
- **Within one recording:** about 31% of encoding neurons changed their encoding.
- **Handling:** per-animal fits with a Bayesian posterior for each neuron and recording.

### 5.3 Randi et al. 2023 (see 1.10)

- Kernels varied across trials and animals. Kernels from the same pair resembled each other more than kernels from other pairs.
- Data were pooled across animals per pair, with up to 59 observations of a pair, and q-values.
- The authors raise plasticity and neuromodulation as possible sources of the variability.
- The strain is not behaviourally wild type.

### 5.4 Creamer, Leifer & Pillow 2024 (see 1.11)

- **Handling:** one parameter set pooled over 80 training animals, with neurons aligned by identity and 30 held out.
- **Ceiling:** train–test correlation of stimulus-triggered average responses was 0.16, and the model reached 0.15.
- **Held-out neurons:** reconstruction had mean r = 0.30.
- **Limitation stated:** large animal-to-animal variability in responses limits the fit.

### 5.5 Brennan & Proekt 2019, *eLife* 8:e46814, "A quantitative model of conserved macroscopic dynamics predicts future motor commands"

- **Access:** full text (eLife API), partial.
- **Data:** Kato 2015, 5 immobilised animals with only 15 neurons identified in all; validated on Nichols 2017 (11 animals).
- **Problem with averaging:** individual neurons differ consistently between animals (for example RIML during backing), and averaging neuron activity across animals destroys behavioural information. Decoding backing onset across animals was at chance.
- **Handling:** averaging by position on a shared low-dimensional manifold instead.
- **Generalisation:**
  - Predicted against observed switch times on new animals correlated at r = 0.74.
  - Leave-one-animal-out state decoding was 81%, against 83% for the manifold built from all animals.

### 5.6 Linderman, Nichols, Blei, Zimmer & Paninski 2019 (see 1.11)

- **Hierarchical model:** each worm's dynamics are drawn around a global mean, with shared discrete states and per-worm observation variances.
- **Effect:** the hierarchical model needed 8 states; the non-hierarchical one needed 14 and did worse on validation and test data.
- **Applied to** Kato 2015 (5 worms) and Nichols 2017 (44 worms; strain and stage decodable).

### 5.7 Morrison & Young 2024 (see 1.11)

- Regression coefficients were fitted per dataset (22 animals from Atanas 2023), then **averaged** into one parameter set.
- Differences between animals enter only through each animal's recorded input neurons.
- Missing neurons were replaced by correlated proxies, and synaptic input was scaled by 1.4×.

### 5.8 Other variability notes from the sources above

- Pritz 2023: ASER responses bimodal across animals, and behavioural results variable from day to day.
- Azimi Hashemi 2019: voltage-imaging signals varied between animals.
- Liang 2025: variability between recordings handled by bootstrap but not characterised.
- Flavell 2013 and Ji 2021: state statistics pooled across animals.

______________________________________________________________________

## Sources I could not find or open

- **Lockery & Goodman 2009** (*Nat Neurosci* 12:377) and **Mellem et al. 2008** (*Nat Neurosci* 11:865): paywall, browser check and Europe PMC errors. Reported second-hand only.
- **Liu et al. 2018, *Cell* final version:** only the bioRxiv preprint was opened. The spike figures in a search summary (111 mV, −36 mV, 117 V/s) did not match the preprint text I read.
- **Chen et al. 2013:** absolute GCaMP6 kinetics, Kd and Hill coefficients are in figures and supplementary tables I could not extract.
- **Dana et al. 2019 (jGCaMP7):** not opened. Only second-hand jGCaMP7f values via Zhang 2023.
- **Zhang et al. 2023:** the Extended Data Table 3 values for GCaMP6s, 6f and 7f in culture were not reached.
- **Kato et al. 2015** (*Cell*) and **Uzel, Kato & Zimmer 2022** (*Curr Biol*): not opened. Their indicators and variability analyses are not verified here.
- **Mi et al. 2022 (ICLR):** only metadata and a search-result description of the abstract. The calcium observation model was not read.
- **BAAIWorm, Zhao et al. 2024 (*Nat Comput Sci* final version):** redirected to a login. The MetaWorm preprint was opened instead.
- **Naudin et al. later papers** (for example a 2023 RIM, AIY and AFD sequel, or *Neural Computation* work): not found or opened. **No dedicated RIM or AVA model beyond Nicoletti 2024 and Naudin 2022** was opened.
- **An authoritative count of worm neurons with published patch-clamp recordings:** not found. Only MetaWorm's uncited "less than 20%".
- **Any paper that quantified how much CeNGEN priors narrow conductance posteriors:** not found.
- **Neuropeptide release and diffusion time courses in *C. elegans*:** Laurent et al. 2018 (*PNAS*) and Ding et al. 2019 (*eLife*) could not be opened. No worm release kinetics were obtained from any source.
- **A worm whole-brain or circuit model with neuromodulator dynamics fitted to data:** not found.
- **A stomatogastric-ganglion model fitting neuromodulator kinetics to data:** not found. Only a conference abstract (Shim et al. 2012) and Gonçalves 2020's degeneracy result.
- **Zhang, Lu & Bargmann 2005** (*Nature*, PA14 learning) and **Kunitomo et al. 2013** (*Nat Commun*, salt): not opened. Only search-result abstracts, so not used as sources.
- **Wicks, Roehrig & Rankin 1996** (tap-withdrawal network model) and a Rankin-lab honours thesis on inferring synaptic dynamics in habituation: not opened.
- **Lim et al. 2015, Confavreux et al. 2020, Ashwood et al. 2020, Rajagopalan et al. 2023, Chen et al. 2023:** not opened directly. Second-hand via Mehta et al. 2023.
- **Eschbach et al. 2020, *Nat Neurosci* final version:** not opened. Only the 2019 preprint abstract, whose title differs.
- **Whole-brain imaging of the same animals before and after learning:** not found. Liang 2025 and Pritz 2023 compare separate groups.
- **Coverage:** the web-search budget was exhausted during question 4, so later literature (2025–2026) on variability, rule inference and worm voltage imaging is under-sampled.
