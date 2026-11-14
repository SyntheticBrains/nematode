# Session 0b: fact-finding for the Mi et al. 2022 target

Run on 2026-11-14, 11:50 to about 12:12 AEDT, under decision A27: fact-finding only. No training or
fitting was run. The only model code executed was one forward and backward pass per timing
(task E), and one 4-neuron numerical check of a synapse module, each after reading every module it
imports. Every script is in `investigation/spike/session0b/`, and every download is in
`investigation/spike/_downloads/`, which is gitignored. Confidence words: **confirmed** (checked
directly, with evidence given), **likely** (strong indirect evidence) and **unknown** (with what
would settle it).

Pinned sources:

- **Code.** `TuragaLab/wormvae` at commit `15b58bff08bdf4eb04709434d766fca6835d7d79` (2022-04-25,
  26 commits, branch `master` only). Cloned to `_downloads/wormvae_repo/repo`. Line numbers below are
  at this commit.
- **Paper.** The Janelia PDF, SHA-256
  `3a0c7ca46d022e2a7b9cdeda6b4eab7c17134e48eeaf44888faa614a94b8e11c`, 36 pages, headed "Published as a
  conference paper at ICLR 2022". Its text layer was extracted with pypdf to
  `_downloads/mi_paper_pdf/mi2022_text.txt`. The pages could not be rendered, because poppler is not
  installed, so every table value below comes from the text layer.
- **DANDI 000541**, version 0.241009.1457, CC-BY-4.0. 21 NWB files, 30.5 GB in all. One file was
  downloaded in full. From the other 20, only the small datasets were read, by HTTP byte ranges.
- **Zenodo 3906530**, "NeuroPAL Data" (2020-06-24), MIT licence. Its zip's central directory was
  listed by byte ranges, and three members were extracted: `LICENSE.md`, `README.md` and
  `Head_Activity_OH16230.mat`.

## Facts that decide the new plan

1. **The code has no licence.** The repository has no LICENSE file, and the GitHub API reports
   `license: null`, so its code may not be copied or adapted. A reproduction has to be written from
   the paper and from facts about the code's behaviour (A.1).
2. **The data Mi trained on are available under the MIT licence, and they are not the DANDI traces.**
   - The code reads `data/worm_activity/all_traces_Heads_new.mat`, which is in the unlicensed
     repository.
   - Its `traces` and `norm_traces` arrays are bitwise identical, in all 3,969 neuron-worm slots, to
     `Head_Activity_OH16230.mat` in Zenodo 3906530 (MIT, copyright "Oliver Hobert Lab, Columbia
     University"). The only differences are names: `DB01` against `DB1`, and the file paths.
   - The 21 worms are the 21 DANDI 000541 subjects, in the same order and with the same stimulus
     orders.
   - DANDI's CC-BY traces are a different extraction. The two NWB trace series are raw fluorescence,
     and they are bitwise identical to each other in 14 of 21 files. Against the `.mat` ΔF/F0 trace
     of the same label they correlate at a pooled median r of 0.69 (n = 3,550; only 16% reach
     r ≥ 0.9). Affine maps, lags of -3 to 3 frames, smoothing and baseline removal do not reconcile
     them.
   - So the licensed input is the Zenodo file (124 MB), not DANDI (B).
3. **No scoring code exists anywhere in the repository or its history.** Nothing computes the
   held-out-neuron or held-out-worm correlations. Also unrecorded:
   - which worm or worms each neuron-holdout model was trained on (the code's default, `[1]`, and
     the paper's 680-sample SE suggest one worm);
   - which 6 worms were held out;
   - which frames were scored;
   - whether scoring used the training-mode or eval-mode forward pass.
     `--random_init_index` sets no seed (A.5, C).
4. **The code departs from the paper in ways that change the model.**
   - (a) The chemical mask has 3,464 directed entries, of which 1,270 are reversed copies of real
     synapses: the parser tests for type `'SP'`, but the file spells it `'Sp'`. So 1,503 pairs are
     reciprocal, against 233 in the data. The paper's "3464 chemical synapses" is this buggy count.
   - (b) In the conductance model the code multiplies by g of the postsynaptic voltage, where the
     paper's eq. 3 uses the presynaptic one. Its eval-mode branch computes a third formula.
   - (c) g is `leaky_relu`, where the paper says softplus.
   - (d) Neurons not recorded in a worm enter training as a constant 2.0 marked as observed, not as
     missing (419 of 3,969 slots).
   - (e) Only 5 of the 9 variants exist. The current-based count variant raises `AttributeError`.
     There is no L1 ("learned sparse") term, no total-count term, and no count2 retraining.
   - A reproduction must choose between the paper's model and the code's (A.2 to A.4, D).
5. **Cost on this CPU (float32).**
   - One forward and backward pass on one worm-window (300 neurons, 30 frames, 1,200 simulation
     steps) costs 0.75 s on 1 thread and 0.59 s on 8 threads. Eight concurrent 1-thread processes take
     1.74 s each, 3.4 times the throughput of one. Float64 takes 2.3 times longer.
   - A neuron-holdout run on one worm (7,200 passes) is about 1.5 h. A worm-holdout run on 15 worms
     (108,000 passes) is about 18 to 23 h.
   - A 20-run panel is about 9 h of wall-clock for single-worm runs on 8 processes, and about 130 h
     for 15-worm runs.
   - The paper's full neuron-holdout protocol (3,424 models) would be about 1,500 h (E).
6. **The wiring the model sees.**
   - 300 neurons. Chemical: 3,464 directed entries (3.86% of off-diagonal pairs). Gap: a symmetric
     1,031-entry mask (514 undirected pairs plus 3 self-entries).
   - 21 neurons have no partner: the 20 pharyngeal neurons and VC06. Twenty of them are among the
     189 recorded neurons.
   - Counts are fixed magnitudes with two fitted scales (count1), or free magnitudes on the mask
     (sparsity). A degree-preserving rewiring is well defined on either mask, but the reciprocity
     to preserve differs by a factor of 6.5 between the code's mask and the true one (D).
7. **The bridge.** The only sensory input is 3 binary stimulus channels (butanone, pentanedione and
   NaCl), mapped by a learned Linear(3, 300) and softplus onto 79 "sensory" neurons. The model is
   never run free in the code: every prior step is computed from the inference network's posterior
   sample (F).
8. **Possible blocker: the connectome file's licence is unknown.** `white_neuron_connect.csv` has
   the WormAtlas NeuronConnect layout (likely Varshney et al. 2011), and it ships inside the
   unlicensed repository. It may have to be rebuilt from a licensed source, which would change the
   mask unless the same source is found under a licence (D, G).

## A. The code

### A.1 Repository and licence

| Fact | Evidence | Confidence |
|---|---|---|
| Commit `15b58bf…` (2022-04-25) is HEAD. There are 26 commits, from 2020-04-14 to 2022-04-25, and one branch. | `git log` | confirmed |
| No LICENSE file. GitHub reports `"license": None`, and `/repos/TuragaLab/wormvae/license` returns 404. Without a licence, default copyright applies, so the code may not be reused. | `ls`; `api.github.com/repos/TuragaLab/wormvae` | confirmed |
| Structure: `experiment/main.py` (entry), `model/` (VAE, data loader, connectome parsers, loss, train), `cnctm/nn/` (the two leaky-integrator synapse models, the neuron-wise affine layers), `cnctm/utils/` (`ConnectomeConstructor`, an unused lottery-ticket `prune.py`), `data/worm_activity/` (the `.mat`, a readme, a notebook) and `data/worm_connectivity/` (7 CSVs). 1,425 lines of Python. | `wc -l` | confirmed |
| Dependencies: Python 3.6, PyTorch 1.8.0, SciPy 1.5.2, pandas 1.1.5, NumPy 1.16.4. | `environment.yml` | confirmed |
| Hardware: CUDA is hard-wired. `torch.set_default_tensor_type('torch.cuda.FloatTensor')` runs at import (`main.py` 11, `train.py` 9), and `device = 'cuda'` is set (`main.py` 48). Training ran in float32, on one Quadro RTX 8000 per model. | code; paper § 5 | confirmed |
| On current libraries the code needs three shims: `np.int` (`data_loader.py` 87) and `np.NaN` (`connectome_preprocess.py` 140) were removed from NumPy, and the CUDA default-type call has to be neutralised for CPU use. | harness in `session0b/time_forward_backward.py.txt` | confirmed |

### A.2 The model

| Fact | Evidence | Confidence |
|---|---|---|
| Latent voltage for all N = 300 neurons, simulated by Euler steps at dt = 0.25/40 = 6.25 ms (upsample factor 40). Each training window is 30 imaging frames = 7.5 s = 1,200 steps. | `main.py` 39-45; paper A.5 | confirmed |
| Neurons: the columns of `chem.csv` that appear in `sex_shared_cells.csv` (292 rows) or `herm_specific_cells.csv` (149 rows). This gives exactly 300, with indices equal to positions. Types come from `white_neuron_types.csv`: 79 sensory, 121 motor, 80 inter and 20 pharyngeal. | `connectome_preprocess.py` 43-70; `session0b/connectome_facts.json` | confirmed |
| Dynamics: v(t+dt) = v + (dt/τ)(chem_in + elec_in + bias + input − v), with x the posterior sample shifted by one step. | `leaky_integrator_*.py` 109-113 / 89-93 | confirmed |
| Current-based: chem_in_i = Σ_j W_ij g(x_j), with W = T ⊙ M · 10. M is initialised uniform on ±0.01 and is free in sign. | `leaky_integrator_current.py` 38-45, 71, 78 | confirmed |
| Conductance-based, training mode: chem_in_i = g(x_i) Σ_j W_ij (E_ij − x_i), with g applied to the **postsynaptic** voltage. The paper's eq. 3 is Σ_j (E_ji − v_i) W_ji g(v_j). Eval mode computes g(x_i) Σ_j W_ij (E_ij − x_j), which matches neither. A numerical check on 4 neurons matches both formulas to 0.0, and the paper's to no better than 0.007. | `leaky_integrator_conductance.py` 94-108; `session0b/check_conductance_formula.json` | confirmed |
| The synaptic g is `F.leaky_relu` (`main.py` 96). The paper says softplus (§ 2.1). | code | confirmed |
| Gap: elec_in_i = Σ_j W_e,ij (x_j − x_i), with W_e = T_e ⊙ (M_e + M_eᵀ) · α_e. | `leaky_integrator_*.py` 88/72, 97-99 | confirmed |
| Voltage to calcium is softplus(v) alone. `NeuronWiseAffine.forward` computes `scale*input+shift` and then returns `nonlinearity(input)`, so its scale and shift are dead parameters. Calcium is an exponential filter with one shared τ_Ca. Fluorescence is a per-neuron affine map (`NeuronWiseLinear`). The initial calcium comes from the first 10 frames of the target. | `linear.py` 20-24; `cnctm_vae.py` 176-189, 202-225 | confirmed |
| Constraint variants as coded. `weight` (count1, conductance only): magnitudes are the fixed counts, and α_c and α_e (initialised at 0.01) are trained. `sparsity`: the mask is fixed and the magnitudes are free (conductance: initialised from the counts, α fixed at 0.01; current: initialised uniform on ±0.01, α_c = 10). `unconstrained`: an all-ones mask with uniform-initialised magnitudes. The reversal potentials E are a dense N×N parameter in every conductance variant. | `leaky_integrator_conductance.py` 42-70; `leaky_integrator_current.py` 38-54 | confirmed |
| The current-based `weight` variant is not implemented: running it raises `AttributeError: 'leaky_integrator_current' object has no attribute 'sparsity_c'`. Also missing: "learned sparse" (no L1 term in `loss.py`), "learned total count" (no total-count term), and count2 (no code loads count1's parameters). So 5 of the paper's 9 variants exist. | `loss.py`; `session0b/timing.jsonl` | confirmed |
| After each optimiser step, clamps hold the following at or above their floors: the magnitudes (non-negative; the chemical ones only in conductance models), α (count1), the fluorescence scale (≥ 1e-8), τ (≥ dt) and τ_Ca (≥ dt). | `train.py` 84-93 | confirmed |
| `signs_c` is a trainable parameter in both synapse modules and is used by neither forward pass. | `leaky_integrator_*.py` 37/36 | confirmed |
| Trainable parameters: 45.0 to 45.2 million, almost all in the inference network's convolutions. | `timing.jsonl`, `trainable_params` | confirmed |

### A.3 Inference and objective

| Fact | Evidence | Confidence |
|---|---|---|
| Inference network: conv1 (2N to 2N channels, kernel 11) on [fluorescence, missing mask] → ReLU → ×4 linear upsample → conv2 (kernel 21) → ReLU → ×10 upsample. The stimulus branch is conv3 (N to 2N, kernel 21) → ReLU. The two are concatenated, and conv4 (kernel 41) gives μ while 0.1·softplus(conv5) + 1e-8 gives σ. In eval mode σ is 0. | `cnctm_vae.py` 45-89; `main.py` 60-74 | confirmed |
| ELBO. The reconstruction term is the mean over observed entries of −log N(f; f̂, σ_f) on the frames, divided by (1 − missing fraction). KL: a learned initial prior at t = 0, then, at each step, N(prior mean, σ_v) against the factorised posterior, with σ_v and σ_f per neuron. The KL is averaged over the 1,200 steps. The weights are 1 and 1. | `loss.py` 42-81 | confirmed |
| Optimiser: Adam with lr 3e-4, StepLR (step 50, γ 0.5), 300 epochs, and element-wise gradient clipping at 1.0 applied after each worm's backward pass. Checkpoints every 100 epochs. | `main.py` 49-56; `train.py` 31-112 | confirmed |
| Batching: the DataLoader yields one window at a time (`batch_size = 1`). Each step loops over the training worms, accumulates their gradients and takes one Adam step. One epoch is 24 windows (32,001 upsampled steps // 1,200 − 2), covering frames 30 to 750 of the 800 kept, with crop padding around each window. | `data_loader.py` 147-160; `train.py` 37-81; `timing.jsonl` `windows_per_epoch` | confirmed |
| Seeds: `--random_init_index` appears only in the run name (`main.py` 18, 21, 29). No `manual_seed` call exists, so the paper's "4 random initialisations" were unseeded. | grep | confirmed |

### A.4 The data loader

| Fact | Evidence | Confidence |
|---|---|---|
| Reads `all_traces_Heads_new.mat` (MAT 5.0, created 2020-05-12) with `scipy.io.loadmat`, and builds an array of 21 worms × 189 neurons × 960 frames from `traces`, which is ΔF/F0, not the recommended `norm_traces`. It drops the first 160 frames (40 s), keeping 800 frames (200 s at 4 Hz), and adds 2. | `data_loader.py` 25-70 | confirmed |
| 189 neurons: 80 L/R classes (each giving L and R) and 29 single classes, 109 classes in all. Every name exists in the connectome. | `session0b/mat_facts.json` | confirmed |
| **Unrecorded neurons are not marked missing.** An empty slot loads with shape (1, 0), so the `data.shape[0] < 1` test fails and the row stays 0, which becomes 2.0 after the offset. This affects 419 of 3,969 slots (12 to 43 per worm). Traces shorter than 960 frames are padded the same way (1,364 slots end in 2.0). The only NaN is the last frame of some traces (0.11% of entries). | `data_loader.py` 51-60; `session0b/mat_empty_check.json`, `mat_nan_pattern.json` | confirmed |
| So the recorded neurons per worm are 146 to 179 (mean 169.0, total 3,550), matching the paper's "about 170". | `mat_nan_pattern.json` | confirmed |
| Stimulus: 3 binary channels, on where 60 < t < 70, 120 < t < 130 and 180 < t < 190 s (39 frames each after the crop), assigned per worm by `stims`. They are linearly interpolated to the simulation rate. `stim_times` is stored as uint8, so the true onsets of 60.5, 120.5 and 180.5 s (Zenodo, NWB) are truncated to 60, 120 and 180. | `data_loader.py` 81-91, 116-134; `compare_zenodo_wormvae.json` | confirmed |
| Neuron holdout as coded: names in `holdout_list` are left as NaN, so they are masked in both the loss and the inference input. `--neuron_holdout` and `--train_worm` arrive from argparse as strings. A string `train_worm` raises `IndexError` when it indexes the array, and the holdout test becomes a substring test (for SAAV, SIAV and URAV it also drops AVL). The README's own command line therefore fails as written unless the code is edited. | `main.py` 14-15, 103-109; `connectome_preprocess.py` 141-144; check in this session | confirmed |
| Worm holdout: not implemented beyond `--train_worm`. No evaluation pass on held-out worms exists. | grep over all commits | confirmed |

### A.5 Scoring

| Fact | Evidence | Confidence |
|---|---|---|
| No code computes a correlation, or scores held-out neurons or held-out worms, at HEAD or in any earlier commit. A grep for corrcoef, pearson, holdout and held_out over every commit finds only the arguments. | `git grep` over `git rev-list --all` | confirmed |
| How predictions were produced for scoring (eval mode with σ = 0, or a posterior sample; whole trace or windows) | not in the code; the paper says only "the reconstructed fluorescence traces" (A.4) | unknown |

## B. The data

### B.1 DANDI 000541

| Fact | Evidence | Confidence |
|---|---|---|
| "NeuroPAL Microfluidic Chip Images and GCaMP activity". Licence `spdx:CC-BY-4.0`. Published versions 0.240611.1947, 0.240625.0452 and 0.241009.1457 (the latest), each with 21 assets totalling 30,525,162,874 bytes. | DANDI API; `_downloads/dandi000541_listing/` | confirmed |
| 21 files of 1.42 to 1.46 GB, one per subject, named `sub-2019MMDD-NN`. The sizes are in `session0b/dandi000541_assets.tsv`. | API | confirmed |
| Each file holds: the raw GCaMP video (`acquisition/CalciumImageSeries`, about 962 × 256 × 128 × 21 uint16, 1.32 GB, contiguous); the NeuroPAL structural volume (110 MB); per-frame voxel masks; two ROI traces of frames × ROIs (`SignalRawFluor`, `SignalRawFluordNMF`); `NeuronIDs/labels`; and `intervals/chemical_stimuli`. | `_downloads/session0b_derived/nwb_tree_sub-20190924-01.txt`; `session0b/nwb_small_sub-20190924-01.json` | confirmed |
| The traces are raw fluorescence (0.006 to 4.2), at 4.0 Hz, 936 to 962 frames, with 166 to 188 ROIs of which 164 to 184 are labelled, and no duplicate labels. The two series are bitwise identical in 14 files. In the other 7 they differ and contain 0.5 to 1.7% NaN. | `session0b/nwb_remote_summary.json` | confirmed |
| Stimuli: each file has 3 intervals at 60.5 to 70.5, 120.5 to 130.5 and 180.5 to 190.5 s: 160 mM NaCl, 10⁻⁴ 2-butanone and 10⁻⁴ 2,3-pentanedione in randomised order. Strain OH16230, young adult. | same | confirmed |
| The 21 subjects are Mi's 21 worms. The `.mat` `files` entries (`…/Heads/0924_01/…` to `…/1104_10/…`) map one to one, in order, onto the DANDI subject IDs, and all 21 stimulus orders agree. | `session0b/mat_files.json`; `nwb_remote_summary.json` | confirmed |
| Every one of the 3,550 non-empty `.mat` traces has an NWB ROI with the same label in its worm (after `DB01` → `DB1`). | `session0b/compare_all/` | confirmed |
| The NWB traces are not the `.mat` traces. Pooled same-label r is 0.69 at median (5th to 95th percentile 0.37 to 0.97), and the per-worm median runs from 0.47 to 0.85. Only 16.5% reach r ≥ 0.9, and 2.0% reach r ≥ 0.99. First differences correlate at a median of 0.32 (worm 0). On worm 0, an affine fit, lags of -3 to 3 frames, Gaussian smoothing (σ 1 to 4) and a 61-frame running-median baseline leave the median at 0.84 to 0.86. | `compare_mat_nwb*.json`, `compare_all/summary.json` | confirmed |
| Size actually needed: the trace datasets of all 21 files total 55 MB, read by byte ranges in 2.5 min. Nothing over 1.5 GB was downloaded. | `_downloads/dandi000541_traces/` | confirmed |

### B.2 The file the code expects, and its licensed source

| Fact | Evidence | Confidence |
|---|---|---|
| The code expects `data/worm_activity/all_traces_Heads_new.mat` (52 MB), committed to the unlicensed repository on 2022-04-24. Its `files` paths point to a personal Dropbox folder of traces, created 2020-05-12. | `git log`; mat header | confirmed |
| Zenodo 3906530 ("NeuroPAL Data", 2020-06-24) is MIT-licensed (record metadata `mit-license`, with a `LICENSE.md` in the zip). It is a 1.12 GB zip of 45 files, including `Whole-Brain Calcium Activity/Young Adults/Head_Activity_OH16230.mat` (123.8 MB, SHA-256 `3d89eeba…d930`). | Zenodo API; `session0b/zenodo3906530_zip_listing.tsv` | confirmed |
| That file's `traces` and `norm_traces` equal wormvae's in all 3,969 slots (maximum absolute difference 0.0), and `fps`, `stims`, `is_AWCL_OFF`, `samples` and the neuron list (after zero-padding) agree. It adds `positions` and `max_norm_traces`. Its `stim_times` are 60.5, 120.5 and 180.5 s. | `session0b/compare_zenodo_wormvae.json` | confirmed |
| Its README: 21 OH16230 heads recorded for 4 minutes, downsampled to 4 Hz, with traces as ΔF/F0 and histogram-normalised (recommended). Neurons within 5 µm of each other with at least 95% correlation were removed from the NeuroPAL paper's analyses but are kept in the file, and the README recommends removing them. | `_downloads/zenodo3906530_members/README.md` lines 9, 74, 97-98 | confirmed |
| Recording length: 960 frames = 240 s, of which the code uses 200 s and trains on 180 s. The paper's "5 min" does not match. | `.mat`; README | confirmed |

## C. The paper's protocol and numbers

| Fact | Evidence | Confidence |
|---|---|---|
| Neuron holdout: a bilateral pair is withheld from training and testing, and the model's predicted fluorescence for the pair is correlated with the measured trace (Pearson, eq. 23). This was done for "all 107 measured neuron pairs". The data have 109 classes (80 L/R and 29 single), so how 107 was counted is unknown. | § 3.2; A.4 | confirmed (text); the 107 is unknown |
| Neuron-holdout models: 8 constraints × 107 pairs × 4 initialisations = 3,424 models. SE "based on 4 × 170 = 680 samples" per model type. | A.4 | confirmed |
| Training set for neuron holdout: not stated. A single worm is suggested by three things: 680 = 4 × 170, where about 170 is one worm's neurons; Table 3's caption ("models fitted on a single worm"); and the code's default `train_worm = [1]`. Which worm is unknown. | A.4; Table 3; `main.py` 15 | likely |
| Worm holdout: train on 15 worms and test on 6 withheld; 4 initialisations per variant; SE over 6 × 170 × 4 = 4,080 samples. Which 6 worms is not stated. | § 3.3; A.4 | confirmed; the split is unknown |
| Hyperparameters as in A.3. The paper adds a simulation step of 6.25 ms, a 7.5 s window of 30 frames, and loss weights of 1 and 1. | A.5 | confirmed |
| Count1's fitted scales are α_c = 0.002 and α_e = 0.065 (Table 1). | Table 1; § 2.6 | confirmed |

**Table 2 (text layer).** Variant order: current-based (learned dense, learned sparse, connectome
sparsity), then conductance-based (learned dense, learned sparse, learned total count, connectome
sparsity, connectome count1, connectome count2).

| Holdout, metric | cur. dense | cur. sparse | cur. c.sparsity | cond. dense | cond. sparse | cond. total count | cond. c.sparsity | cond. count1 | cond. count2 |
|---|---|---|---|---|---|---|---|---|---|
| neuron, corr | -0.104 ± 0.013 | 0.001 ± 0.010 | 0.087 ± 0.021 | 0.081 ± 0.012 | 0.008 ± 0.011 | -0.033 ± 0.011 | 0.262 ± 0.016 | 0.318 ± 0.016 | 0.319 ± 0.016 |
| worm, corr | 0.465 ± 0.006 | 0.439 ± 0.007 | 0.440 ± 0.006 | 0.457 ± 0.006 | 0.445 ± 0.006 | 0.470 ± 0.006 | 0.463 ± 0.006 | 0.474 ± 0.006 | 0.468 ± 0.006 |
| worm, ELBO | -0.772 ± 0.037 | -0.826 ± 0.036 | -0.803 ± 0.036 | -0.849 ± 0.033 | -0.862 ± 0.035 | -0.931 ± 0.033 | -0.898 ± 0.032 | -0.854 ± 0.036 | -0.697 ± 0.066 |
| worm, recon | 0.745 ± 0.037 | 0.765 ± 0.037 | 0.754 ± 0.035 | 0.771 ± 0.033 | 0.780 ± 0.035 | 0.805 ± 0.032 | 0.082 ± 0.031 | 0.744 ± 0.035 | 0.636 ± 0.065 |
| worm, KLD | 0.028 ± 0.001 | 0.061 ± 0.003 | 0.049 ± 0.002 | 0.078 ± 0.003 | 0.082 ± 0.002 | 0.126 ± 0.005 | 0.077 ± 0.003 | 0.110 ± 0.006 | 0.061 ± 0.002 |

Seeds: 4 initialisations per cell, unseeded (A.3). The conductance connectome-sparsity recon value
of 0.082 is inconsistent with its ELBO (−0.898) and KLD (0.077), and is likely a typo for 0.82.
The paper's own reading is that worm-holdout correlations do not separate the variants (A.8).
Confidence: confirmed from the text layer. The rendered table was not checked visually.

## D. The wiring

| Fact | Evidence | Confidence |
|---|---|---|
| Weights come from `white_neuron_connect.csv` (6,417 rows: Neuron 1, Neuron 2, Type, Nbr). Its types are S 950, Sp 1,625, R 773, Rp 1,885, EJ 1,031 and NMJ 153. That is the WormAtlas NeuronConnect layout, most likely Varshney et al. 2011, which the paper cites as [27], beside "White 1986" [30]. `chem.csv` and the `elec_*.csv` files set only the neuron order: their values are never used, and `chem.csv` has 3,707 non-zero entries totalling 20,905, a different matrix. | `connectome_preprocess.py` 22-41, 88-107; `connectome_facts.json` | confirmed (use); likely (source) |
| Licence of the CSVs: none stated in the repository. WormAtlas's terms were not checked. | | unknown |
| Chemical mask: 3,464 directed entries, 3.86% of the 89,700 off-diagonal pairs, with no self-entries. A graph built from the rows' documented meanings (S/Sp: 1 sends to 2; R/Rp: 1 receives from 2) has 2,194 directed pairs. The code's mask contains all 2,194, plus 1,270 reversed entries that come only from `Sp` rows, because the code tests for `'SP'` and so treats `Sp` as a receive row. | `connectome_direction_check.json` | confirmed |
| Reciprocity: the code's mask has 1,503 reciprocal pairs (3,006 of 3,464 entries). The true graph has 233. | same | confirmed |
| Chemical counts: each entry holds the `Nbr` of the last row written, not the S + Sp sum. Overwrites with a different value happen 836 times. In the code's matrix the counts total 8,341 (maximum 31); the true send counts total 6,394. | `connectome_facts.json`; `connectome_direction_check.json` | confirmed |
| Gap mask: 1,031 entries, symmetric, 514 undirected pairs plus 3 self-entries. The magnitudes are symmetric too. Because W_e uses M_e + M_eᵀ, every gap weight is 2 × count × α_e. | `connectome_facts.json` | confirmed |
| Chemical and gap entries overlap on 313 directed pairs. Their union is 4,179 directed off-diagonal entries (4.66%). | same | confirmed |
| 21 neurons have no partner of either kind: I1L/R, I2L/R, I3 to I6, M1, M2L/R, M3L/R, M4, M5, MCL/R, MI, NSML/R and VC06. All but VC06 are among the 189 recorded neurons. | same; `mat_facts.json` | confirmed |
| Free versus fixed. Count1: the magnitudes are the counts, fixed, with 2 fitted scales, plus a fitted reversal potential per entry (dense N×N, effective on the mask). Connectome sparsity: each mask entry gets a free non-negative magnitude (conductance) or a free signed one (current). Dense: all N² entries are free. | A.2 | confirmed |
| Degree-preserving rewiring. Chemical: directed edge swaps that preserve every in- and out-degree are well defined on either mask, and they can be restricted to preserve the reciprocal-pair count (1,503 in the code's mask, 233 in the true one). Gap: undirected double-edge swaps preserving degree are well defined on the 514 pairs. The 3 gap self-entries carry no current (x_i − x_i = 0) and can be left fixed. The 21 isolated nodes stay isolated. For count1, whether counts move with their edges is a design choice that has to be stated. | structure above | confirmed (well defined); the choice is open |

## E. Cost

All timings are on the Apple M5 Max (18 cores, 128 GB RAM, otherwise idle), with torch 2.15.0 on
CPU. Each is one forward pass, ELBO and backward pass (no optimiser step) on worm index 1, window 0,
built exactly as `main.py` builds it, with the median of 3 to 5 repetitions after a warm-up. Every
module on the path was read first: `model/__init__.py`, `cnctm_vae.py`, `connectome_preprocess.py`,
`connectome_parser.py`, `data_loader.py`, `loss.py`, `train.py`, `cnctm/nn/*` and
`cnctm/utils/data.py`. The repository was put on `sys.path` from `session0b/_empty_cwd`. Script:
`session0b/time_forward_backward.py.txt`; results: `session0b/timing.jsonl` and `timing_concurrent8.jsonl`.
Total compute was about 2 minutes.

| Variant | 1 thread | 4 threads | 8 threads | Peak RSS |
|---|---|---|---|---|
| conductance count1, float32 | 0.749 / 0.756 s | 0.624 s | 0.586 s | 3.0 GiB |
| conductance sparsity, float32 | 0.741 s | | 0.582 s | 3.0 GiB |
| conductance dense, float32 | 0.745 s | | | 3.0 GiB |
| current sparsity, float32 | 0.631 s | | 0.550 s | 2.2 GiB |
| current dense, float32 | 0.632 s | | | 2.2 GiB |
| conductance count1, float64 | 1.727 s | | 1.310 s | 5.7 GiB |
| current sparsity, float64 | 1.537 s | | | 4.1 GiB |
| 8 concurrent 1-thread processes, conductance count1, float32 | 1.51 to 1.88 s each (mean 1.74) | | | |

Threads give little: 8 threads are 1.28 times faster than 1. Eight processes give 3.4 times the
throughput of one, about 4.6 passes per second.

Extrapolation at the paper's settings (24 windows × 300 epochs). This excludes the Adam step and
data slicing, which were not measured.

| Run | Passes | 1 process, 1 thread | 1 process, 8 threads | Throughput on 8 processes |
|---|---|---|---|---|
| Neuron holdout, 1 training worm | 7,200 | 1.5 h | 1.2 h | 3.5 h per run, 8 at a time |
| Worm holdout, 15 training worms | 108,000 | 22.5 h | 17.6 h | 52 h per run, 8 at a time |
| Panel of 20 single-worm runs | 144,000 | 30 h | 23 h | about 8.7 h wall-clock |
| Panel of 20 fifteen-worm runs | 2.16 M | 450 h | 350 h | about 130 h wall-clock |
| The paper's full neuron-holdout protocol (3,424 models) | 24.7 M | 5,140 h | | about 1,490 h |

Float64 multiplies every figure by about 2.3. Confidence: confirmed for the per-pass timings. The
run figures are extrapolations, likely to within about 10 to 20%, given the unmeasured optimiser
step.

## F. The bridge

| Fact | Evidence | Confidence |
|---|---|---|
| Sensory input: 3 binary channels (butanone, pentanedione, NaCl) on during each 10 s pulse, mapped by a learned `Linear(3, 300)` and softplus, then masked to the 79 neurons typed "sensory" in `white_neuron_types.csv` (including AWA, AWB, AWC, ASE, ASH, ADL, ASI, ASJ, ASK, AFD, BAG, URX, SAA, the IL2s, the OLQs and the ALMs). That file types the dopaminergic CEPs and ADE as interneurons, so they get no stimulus input. No other modality exists. | `cnctm_vae.py` 10-26, 164; `connectome_facts.json` | confirmed |
| Motor and premotor coverage: 121 neurons are typed "motor" (head RMD, RME, SMD, SMB and URA, the IL1s, ventral-cord AS/DA/DB/DD/VA/VB/VC/VD, and others). Recorded premotor interneurons include AVA, AVB, AVD, AVE, RIM, RIB, AIB and AIY, and the recorded motor-typed neurons include RMD, RME, SMD, SMB, URA, the IL1s, AS01, DA01, DB01, DB02, DD01, VA01, VB01, VB02, VD01 and VD02. | `mat_files.json`; `white_neuron_types.csv` | confirmed |
| The generative model is an Euler simulation at 6.25 ms, driven by the stimulus input, so a 1 s step is 160 simulation steps. The released code never runs it free: the prior at each step is computed from the posterior sample of the previous step, which needs fluorescence data as input (`leaky_integrator_*.py` 91-92/75-76). A free-running simulation from a sensory stream would need new code, and how it behaves is unknown. | code | confirmed (code); unknown (behaviour) |
| The model has no body, position or proprioceptive input. Voltages are in arbitrary units. | code | confirmed |

## G. What blocks a reproduction on licensed data within a small budget

| Item | Status | Confidence |
|---|---|---|
| Code licence | None, so the code cannot be reused. A clean-room implementation from the paper and these facts is needed, and it must choose between paper and code wherever they differ (A.2, A.4, D). | confirmed |
| Training data | Not a blocker: Zenodo 3906530 (MIT) is identical to the code's input. Attribution: the MIT notice of the Hobert Lab. DANDI 000541 (CC-BY) is the same animals but a different extraction, so a DANDI-based reproduction would not be on Mi's input. | confirmed |
| Connectome data | Provenance is likely Varshney/WormAtlas, and the licence is unknown. A licensed rebuild of the same NeuronConnect matrix would reproduce the code's mask only if the `Sp` handling is copied. | unknown |
| Scoring protocol | No code. Unrecorded: the neuron-holdout training worm(s), the 6 test worms, the scored frames, eval mode or sampling, and the 107-pair count. A reproduction has to fix these by registration and cannot match the paper's numbers exactly. | confirmed |
| Variant coverage | 5 of 9 variants exist in code. The headline comparison (conductance count1 0.318 against connectome sparsity 0.262 against dense 0.081) is covered by coded variants. | confirmed |
| Compute | One neuron-holdout run takes about 1.5 h on 1 CPU process. A small panel (about 20 runs on one worm) takes about 9 h on 8 processes. The paper's 3,424-model protocol (about 1,500 h) and 15-worm runs (18 to 23 h each) are outside a small budget unless the pairs or worms are subsampled. | likely |
