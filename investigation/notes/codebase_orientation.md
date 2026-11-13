# Codebase orientation: how the current approach sets parameters, what it is graded on, the neural-data assets, and what differentiability would cost

Read-only audit, 2026-11-14. Nothing was run except small JSON reads with `python3 -I` and
`git show`. Paths are relative to the investigation worktree
`/Users/chriszaharia/projects/personal/nematode/.claude/worktrees/fit-first-investigation` unless
they are prefixed `feat/fit-c2:`. Those were read from the main checkout's branch with
`git -C /Users/chriszaharia/projects/personal/nematode show feat/fit-c2:<path>`.

Reference conventions:

- **"LB n"** is Logbook n, at `docs/experiments/logbooks/NNN-*.md`.
- **"LB n:a-b"** gives lines a to b of that logbook.
- **Source paths** are abbreviated:
  - `cppo` is `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`;
  - `enc` is `packages/quantum-nematode/quantumnematode/connectome/encoder.py`;
  - `ctl` is `packages/quantum-nematode/quantumnematode/evaluation/controllers.py`;
  - `bat` is `packages/quantum-nematode/quantumnematode/evaluation/batched.py`;
  - `prim` is `packages/quantum-nematode/quantumnematode/env/primitives.py`.

Two forked sub-audits produced Part 2. I spot-checked their load-bearing claims against the
logbooks and JSON records, and they matched: LB 129:131 (the CEP nudge), LB 131:182-186 (15 of 15),
launch 133's in-sample verdict, and `bands_v4.json`'s fit families. Their unabridged notes are in
the same scratchpad directory, in `q2a_benchmarks.md` and `q2b_scores.md`.

______________________________________________________________________

## 0. The short version

- **The connectome controller (C.2) is the real wiring with a static, memoryless rate network.** It
  has very few free parameters on the wiring, and many on its interfaces and its neuromodulation.
  - The wiring is taken from data and is fixed: 302 neurons, synapse counts, per-connection signs and
    gap-junction counts.
  - The brain is a function of 137 parameters under the registered procedure (196 by default). Each
    parameter is fitted by **sep-CMA** to real-worm behaviour statistics:
    - 3 wiring scales and 4 unsigned-group gains;
    - 36 sensory gains;
    - a 25-number motor readout onto five locomotor-primitive outputs;
    - 5 noise levels;
    - about 57 to 121 neuromodulator strengths;
    - 2 or 3 timescales;
    - 4 pool centres.
  - **No synaptic weight is fitted per synapse.** No neuron has a time constant, a bias or any
    intrinsic current. The network re-settles from its sensory input every 1-s step, through 4
    iterations of `tanh`.
  - Its only memory is 55 neuromodulator pools, which are leaky averages of their producer cells'
    rates, plus a slow noise process on the speed output.
- **The benchmark is behavioural only.** Four versions, v1 to v4, were each validated.
  - The current version, v4, grades 24 statistics in 6 families: lawn roaming and dwelling, uniform
    well, local search, kinematics, chemotaxis and on-lawn activity. The data are Scheer 2023,
    López-Cruz 2019 and Chen 2025.
  - Mutants are graded separately on one or two small-lawn statistics.
- **Neither controller has a clean held-out pass.**
  - The mechanistic baseline, C.1, passes v1 and v3 at exactly 3 of 5 populations, and fails v2 and
    v4.
  - C.2 passes v4 at 15 of 15 populations only in probes, and only after three things:
    - every family, the held-out ones included, was moved into its objective;
    - its mutants were moved into the objective too;
    - the body's on-food reversal was changed.
  - C.2's registered fit (launch 133) is in its third pilot, with no result.
  - Every benchmark revision since v1 followed a controller's result.
- **Neural-data assets: one derived table.**
  - `data/connectome/creamer_lds_2026_model_weights.csv` holds Creamer–Leifer–Pillow's fitted LDS
    weights: 2,011 entries over 125 neurons, a preprint fitted to Randi 2023. It covers 1,049 of
    3,709 chemical edges and none onto a body motor neuron.
  - It was used as a weight prior under PPO in Phase 8. It neither moved the wiring effect nor
    resolved it (LB 072, 073).
  - **The current phase does not read it**: the encoder refuses `weight_prior` away from `random`.
  - No raw neural recordings are vendored. The Randi atlas is cited but not vendored, because its
    licence is GPL-3 or unstated.
- **Differentiability.**
  - The brain is PyTorch, and its one-step map is autograd-capable. C.2 was once trained by gradient
    descent, open-loop, to imitate C.1 (LB 126). It reached R² about 0.3, and failed closed-loop.
  - The modulator pools, the timescales and the vector write path sit outside autograd.
  - The environment, the primitives (discrete Bernoulli, categorical and gamma draws), the
    measurement model and the grading (an HMM decode, counts, caps and resampling) are NumPy and
    non-differentiable.

______________________________________________________________________

## 1. How neuron and synapse parameters are set today

### 1.1 The connectome brain under the compact encoder (C.2, the current phase)

The current phase's connectome controller is `connectomeppo` with `encoder: compact`. The configs are
`configs/scenarios/foraging/*_lawns_encoded_s1_primitives*.yml`, which set
`connectome_source: emmons_2024_hermaphrodite` (line 16) and `forward_pass_depth: 4` (line 20).
The encoder **fixes** 33 settings (`_ENCODER_FIXED`, `cppo:1248-1282`), among them:

- settling dynamics;
- klinotaxis sensing;
- primitives actions;
- the data modulator layer;
- per-connection signs;
- `weight_prior: random`;
- strength gap normalisation;
- additive gaps;
- no plastic gaps;
- no predator or thermal projections;
- no activity traces.

The table gives each parameter class, how it is set and where its value comes from. **"sep-CMA"**
means fitted to real-worm behaviour statistics by the Phase 10 fit, not trained by PPO.

| Class | What the code does | Source of the values | Where |
|---|---|---|---|
| **Neuron set** | All 302 hermaphrodite neurons, pharyngeal included, sorted by name. Muscles and glia are filtered out | Cook 2019 SI 5 or Emmons 2024 S1; the classification is cect `Cells.py`, after Cook 2019 and WormAtlas | `connectome/loader.py:1-48`; `connectome/neurons.py:1-30`; `data/connectome/PROVENANCE.md:8-77,168-186`; `cppo:3963-3970` |
| **Chemical wiring (mask)** | Strict mask: an edge exists only where the EM count is above 0. No edge is added | Emmons 2024 (identical chemical matrix to Cook 2019: 3,709 connections) | `cppo:1864-1910`; `PROVENANCE.md:52-59` |
| **Gap wiring** | Symmetric, with both sheets folded by max. Emmons gives 1,095 pairs | Emmons 2024 EM counts | `loader.py:169-186`; `PROVENANCE.md:55-59` |
| **Synaptic signs** | `per_connection_signs()` takes the first of four steps: physiology override (67 edges, 16 of them López-Cruz 2019), then Fenyves 2020 expression polarity (1,676), then the per-neuron transmitter rule (1,433), then no sign (533) | Data and literature; ACh and Glu are excitatory and GABA inhibitory under the rule | `connectome/signs.py:1-25,401-486`; `data/connectome/sign_overrides_physiology.csv`; `PROVENANCE.md:188-256`; counts in LB 091:44-55 |
| **Weight magnitudes, signed edges** | weight = scale\_±1 × sign × count / sqrt(Σ count² onto the post neuron). The two scales (excitatory, inhibitory) are fitted; the counts are data; the normalisation rule is hand-chosen | Counts are data. The 2 scales are sep-CMA, on a box of [0, 1.5], or [0, 0.75] graded, with the inhibitory floor at 0.09 in the procedure | `enc:429-453, 302-320`; box `enc:88, 102, 197-215` |
| **Weight magnitudes, unsigned edges (533)** | Count-normalised as above, times a signed gain shared by the presynaptic transmitter group: dopamine, serotonin, octopamine or none | 4 gains, sep-CMA, on [-2, 2] | `enc:66-73, 440-453` |
| **Gap conductances** | count / sqrt(s_i s_j), with s a neuron's summed counts, so the spectral radius is 1, times one fitted scale. Additive coupling (`+ G h`), not ohmic | Counts are data. The normaliser was chosen in Phase 9 (LB 103). 1 scale, sep-CMA | `cppo:1955-1986` (normalisation `1968-1975`); `enc:496-498`; `cppo:2523` |
| **Neuron dynamics** | Per 1-s environment step: h₀ = sensory injection, then 4 iterations of h ← tanh(h·(W_chem + g·G_gap) + I_mod). Nothing carries to the next step | **Hand-set.** K = 4 is justified as the sensory → interneuron → command → motor depth (`cppo:24-28, 502-508`). tanh is hand-chosen. There are no time constants and no leak | `cppo:2512-2543`; `_settling_tanh` `cppo:1587-1597` |
| **Intrinsic currents and biases** | None. There is no per-neuron bias, gain, threshold, time constant, bistability, plateau or spike. Zero input gives zero activity (tanh is odd). The only extra current is the neuromodulator "held current" |: | `cppo:2525-2531` |
| **Sensory input** | 4 features per step: concentration, lateral (right − left), dC/dt and food contact. Each of 11 amphid classes (AWC, AWA, ASE, AWB, ASK, ADF, ASI, ASJ, ASG, ADL, ASH) gets a concentration gain and a dC/dt gain shared L/R, plus an antisymmetric lateral gain. Food contact goes onto CEP, ADE and PDE. **Every class sees the same abstract food signal**; there is no modality-specific stimulus | 36 gains, sep-CMA, on [-5, 5] or [-2, 2] graded. The class choice is hand-made (literature) | `enc:43-62, 220-227, 345-370`; `cppo:4422-4436` |
| **Sensory preprocessing (environment side)** | The odour field is a Fick kernel around lawns. The head-sweep lateral samples sit ±0.5 mm. dC/dt comes from a 30-sample short-term-memory weighted difference, then a fold-change readout derivative / (c + ε). The brain then applies tanh(lateral·50) and tanh(dC/dt·50) | **Hand-set** environment constants: `gradient_decay_constant 8.0`, `stam_decay_rate 0.1`, `adaptive_chemosensor_alpha 0.1`, `epsilon 0.1`, scales of 50 | `bat:575-636`; `utils/config_loader.py:936-956`; encoded config lines 68-81 |
| **Motor readout** | The mean rate of each of four ventral-cord motor classes (VB, DB, VA, DA; 39 cells), minus 4 fitted centres. A 5×4 matrix plus bias gives speed, curvature and three bout-start hazards (reversal, omega, delta). A soft cap 2·tanh(x/2) follows, then the tanh squash to [-1, 1]. Command interneurons and head motor neurons are not read | 20 weights, 5 biases and 4 centres, sep-CMA on [-3, 3] and [-1, 1]. The pooling is hand-chosen | `cppo:2029-2066, 2534-2539`; `enc:79-80, 228-235, 258-261` |
| **Action noise** | A Gaussian on each pre-squash output. With `encoder_speed_noise: ou`, the speed's noise is an Ornstein–Uhlenbeck process | 5 log-SDs, sep-CMA. The OU timescale is sep-CMA on 0.25–300 s; it is a buffer, held under PPO | `enc:96-99, 236, 249-257, 480-484`; `ctl:244-272` |
| **Neuromodulator layer** | 55 ligand pools. Each pool's concentration ← d·c + (1−d)·(mean positive settled rate of its producers), with d = exp(−dt/τ). Each couple adds strength × concentration × sign onto its receptor cells, held through the settling. One drive: food ingestion onto NSM, scaled by hunger | Placement is data (Bentley 2016, Ripoll-Sánchez 2023). 17 fixed signs (16 monoamine receptors plus the drive) and 104 fitted signed strengths. Magnitudes are sep-CMA (121 per couple, or tied per ligand). 2 timescales (monoamine, neuropeptide; 10–3000 s), sep-CMA | `connectome/neuromodulator_layer.py:1-20,121-198`; `cppo:2425-2476, 2554-2561, 4373-4403`; `data/neuromodulators/PROVENANCE.md:32-127` |
| **Cited receptor signs (procedure option)** | `modulator_receptor_signs: [pdfr-1, ser-5]` signs those couples from `receptor_couplings.json`. It was added after fits reversed both signs (LB 128) | Literature (G-protein coupling) | `enc:166-175`; `PROVENANCE.md:105-127` |
| **Plasticity** | **None** within a lifetime. The batched evaluator refuses any learning rule other than PPO, activity traces and node noise. Brains are frozen during evaluation |: | `ctl:74-89` |

**Parameter counts.**

- 196 parameters by default: 7 wiring, 36 sensory, 25 readout, 5 noise, 121 modulator and 2
  timescales (`enc:178-262`; LB 122:55-64).
- 132 under per-ligand tying.
- **137 under the registered procedure** (`scripts/analysis/c2_encoder_probe.py:1761-1775`;
  feat/fit-c2:`scripts/analysis/c2_fit.py:5-8`). The procedure adds per-ligand tying, the soft cap,
  pool centres, the graded box, an inhibition floor of 0.09, the PDFR-1 and SER-5 signs and OU speed
  noise.
- The modulator rows reconcile:
  - `layer.json` lists 121 couples and 1 drive;
  - flp-24 has no producer, so its couple is dropped, leaving 120 couples and 1 drive, which is 121
    rows;
  - 17 rows are fixed (16 monoamine couples plus the drive), and 104 are fitted;
  - `PROVENANCE.md:71-72` counts 16 fixed and 105 fitted, by couple.

**How the vector is fitted.** The fit is sep-CMA in the unit box, with population 48 and 200 animals
per candidate after stage 1. The staged procedure, the objective and the guards are in
feat/fit-c2:`scripts/analysis/c2_fit.py:9-41`:

- **The objective:**
  - the definedness-first ranking score;
  - plus the mutants at half the weight;
  - plus saturation, responsiveness and lawn guards;
  - plus, in the final stage, each statistic's probability of failing.
- **The stages:**
  - stage 1 fits 77 parameters;
  - 2a fits all 137 on D_fit;
  - 2b uses six families plus the mutants;
  - a noise step follows;
  - then the final stage.
- `write_encoder` maps the unit box onto the tensors under `torch.no_grad()` (`enc:568-621`).
- A fresh encoded brain is written at the box's centre, so it does not depend on its seed
  (`cppo:4150-4161`).

**What PPO does in this phase.**

- Under the encoder, PPO's learnable set is the encoder tensors, the readout, the bias, the log-SDs
  and the modulator strengths (`cppo:3865-3870`). The timescales stay held.
- **No C.2 result in LB 122–133 was trained by PPO.** C.3 ("the same architecture trained by
  reward") is a planned block (`docs/roadmap.md:1699`); I found no C.3 logbook.

**Options that exist in code but that the encoder refuses** (from Phases 6–9):

- random or measured weight priors per synapse (`cppo:288-301, 431-436, 1886-1893`);
- count-scaled draws;
- leaky dynamics with one membrane τ, in environment steps, for every neuron, gap junctions as an
  implicit Laplacian, and BPTT chunks (`cppo:509-534, 2579-2609, 2685-2697, 3784-3828`). LB 080 found
  it learned "far slower";
- ohmic gaps (`cppo:2563-2577`);
- plastic gap junctions (`cppo:479-483, 1993-1998`);
- three-factor and e-prop plasticity (`cppo:4203-4265`);
- a neuromuscular body drive (`cppo:2611-2648`).

None of these is in use for C.2.

### 1.2 The mechanistic baseline (C.1, `BatchedMechanistic`) for contrast

C.1 is a 20-parameter phenomenological two-state worm (`ctl:496-517, 537-722`). It has no neurons.
Each second:

- the dwell → roam and roam → dwell rates are exp(intercept + intake and satiety slopes);
- each state has its own speed and curvature noise;
- the weathervane is κ·tanh(lateral);
- the reversal hazard is r_s·exp(−β·tanh(dC/dt))·exp(edge·loss)·search;
- the omega hazard has the same form;
- delta turns run at the minimum hazard;
- an exponential off-food search boost applies.

**Optional extras, each off by default:**

- `kappa_off_food`, `v_off_food`, `sigma_off_food` and the two `speed_log_sd_*` take it to 25
  parameters (LB 121);
- `r_dwell_on_food` takes it to 26 (LB 125).

**Where its values come from:**

- every parameter is sep-CMA-fitted to the bands, from the box centre;
- the boxes are hand-set (`scripts/analysis/npe_recovery.py:81-105`), for example κ on
  [−50, 50] °/s and the hazards on [1e-3, 0.5] /s;
- LB 113:78-80 notes two parameters at their box limits.

**The contrast with C.2:**

- C.1 holds an explicit behavioural state across steps. C.2's network does not, and its persistence
  must come from the modulator pools.
- C.1's parameters map one-to-one onto mechanisms. C.2's are wiring scales and interface gains.

### 1.3 Fixed from data in the body (shared by every controller)

The locomotor primitives' kinematics are fixed from data (`data/locomotion/primitives.json`; `prim:1-24`):

- **Off food, from the literature:**
  - v_max 0.2 mm/s and a reversal speed of 0.14 mm/s (Zhao 2003);
  - reversal duration, a gamma with mean 2.69 s and SD 1.75;
  - omega and delta angles and durations, from Broekmans 2016, some read from figures by eye at
    about ±10%;
  - turn-rate cap 50 °/s (Pierce-Shimomura 1999);
  - hazard range 1e-4 to 1 /s (Gray 2005, read from a figure).
- **On food, measured by the project** on Scheer's *held-out* PD1074 dates:
  - reversal speed 0.069 mm/s;
  - reversal duration 2.047 ± 0.899 s;
  - the speed-following knots (LB 105, 108, 132).

So the held-out wild-type dates are partly consumed by the body (launch 133 notes this at
feat/fit-c2:`.../133-c2-fit/launch.md:171`).

### 1.4 Candid weaknesses of the parameterisation

- **Wiring scales, not weights.** C.2's 3,709 chemical edges carry two fitted numbers between them,
  plus four group gains. Whatever the per-synapse magnitudes are, they are counts under a
  hand-chosen normaliser. Under that normaliser, LB 127 found every fit drives the inhibitory scale to
  0.005–0.09, so signed inhibition is close to off. The procedure now floors it at 0.09 by hand
  (LB 127:81-97; LB 129; `enc:200-207`).
- **No neuronal dynamics.** The network is a 4-hop static map re-evaluated each second, with no time
  constants and no intrinsic currents:
  - persistent states (roaming and dwelling, reversal commitment) can only come from the modulator
    pools, the slow speed noise or the body's bouts;
  - temporal differentiation of the odour signal is done by the environment (`bat:629-636`), not by
    neurons.
- **Interfaces dominate the count.** About 130 of the 137 parameters sit at the interfaces: sensory
  gains, the readout, noise, modulator strengths and timescales. Some of the readout's function is
  hand-designed: pooled ventral-cord classes map onto five abstract primitive outputs.
- **A single abstract stimulus.** All 11 amphid classes receive the same concentration, lateral and
  derivative signal, so the model cannot represent cell-specific sensory tuning, such as AWA to
  diacetyl, ASE to salt or ASH to aversive cues, except through gain signs.
- **Several hand-set constants** live in the environment and brain: K = 4, tanh, the normaliser, box
  ranges, scales of 50, STAM and fold-change constants, the soft cap, the inhibition floor and pool
  centres. Most were added after a failing fit (LB 126–131, see 2.6).
- **Identifiability is weak.**
  - LB 128 found that wild-type data do not identify perturbation responses: mutants trade against
    the wild type.
  - LB 121:102 found single C.1 parameters missing truth by up to 0.86–0.88 of their range while
    behaviour matched.
  - F.0b-ii in launch 133 uses one truth and "cannot tell recovering it from fitting the wild type"
    (feat/fit-c2:`.../launch.md:124`).

______________________________________________________________________

## 2. Every benchmark, what it measures, and how the models score

### 2.1 Older validations (Phases 6–8; reward-trained controllers, mostly sign-only references)

| LB | Measures | Reference | Controllers | Result |
|---|---|---|---|---|
| 035 | Klinokinesis (down/up turn-rate ratio, turn magnitude ratio) and weathervane slope; point worm, 5-s step | `data/chemotaxis/behavioural_bias_signatures.json`: one magnitude (Pierce-Shimomura 1999, ratio 1.5–3.0), the rest sign-only | MLP-PPO, connectome-PPO, derivative control; 8 seeds | "Both reproduced", but the turn-rate ratio is 1.30 (MLP) and 1.15 (connectome), below 1.5–3.0. Only the sharpest turns reach 2.78. Weathervane present by sign (035:134-173) |
| 036 | The same four statistics on thermotaxis | `data/thermotaxis/behavioural_bias_signatures.json`, sign-only | MLP-PPO, **n = 4** | PARTIAL: weathervane weak, klinokinesis absent (036:70-93) |
| 086 | Body: frequency, wavelength, eigenworms, speed, omega postures, plus 035's curves | Fang-Yen 2010, Ramot 2008, `data/posture/` | Connectome WT, null, MLP | Body checks pass **by construction** (the generator's own settings). Speed partial. **0 of about 807k turns are omega-shaped.** Klinokinesis ratio 1.03–1.24 (086:24-29, 87-96) |
| 088 | Patch leaving: roaming enrichment before exits | `data/roaming_dwelling/scheer2023_leaving.json` (+0.288) | MLP-PPO, 16 seeds | **Unreadable**: 7 of 16 seeds take both states. Exits 7.7/h against the real 0.21 (088:3-16, 83-106) |
| 089 | Connectome on lawns: dwelling share against the untrained floor | **None against real data** (089:141-143) | Connectome-PPO, 19 seeds | "dwells" +0.514, but 4 of 19 seeds have both states (089:98-100) |
| 090 | Acute tph-1 and pdfr-1 effect on on-lawn speed | Directions from Ji 2021 and Flavell 2013 (`data/roaming_dwelling/ji2021_readings.json`) | Connectome-PPO with hand-placed modulators, 32 seeds | tph-1: no effect at the minimum. pdfr-1: unresolved (090:11-14) |
| 093 | 036's statistics through the body | Thermotaxis signatures, sign-only | Connectome WT and gap-only null | Weathervane present, **klinokinesis absent** (093:5-10) |

Across these, reward-trained learners reproduce the weathervane's direction but never a worm-sized
klinokinesis. They do not roam before leaving, and their knockouts do not phenocopy. That is why
Phase 10 moved to fitting (`docs/roadmap.md:1648-1658`). Several measurement rules were added after
data were seen, each registered before its scored panel (035:97-103, 036:128-131, 086:148-149,
090:106-114).

### 2.2 The Phase 10 benchmark, v1 to v4

**The data** (`data/benchmark/PROVENANCE.md`):

- **Scheer & Bargmann 2023 (CC0):**
  - PD1074 at OD 2: 1,586 animals on 65 dates, of which 41 dates are in the bands and 24 held out;
  - the uniform lawn: **100 animals on 3 dates**, 2 of them band dates;
  - tph-1: 139 animals on 5 dates; pdfr-1: 395 on 16; tax-4: 499 on 19.
  - The classifier was calibrated across all wild-type dates, so "held out" means held out of the
    bands only.
- **López-Cruz 2019 (CC BY):** reorientation counts per 2-min bin from 27 batches, 17 in the bands
  and 10 held out.
- **Chen 2025 (CC BY):** naive N2 butanone chemotaxis, **6 standard recordings**, as fragments.
- **Roberts 2016 and Calhoun 2014:** off-food tracks for detector tuning and the reported reversal
  share (17.3%).

**The machinery** (`packages/quantum-nematode/quantumnematode/benchmark/grading.py:1-22`):

- 1-s positions are cut into 10-s windows;
- a two-state Gaussian HMM labels each window roaming or dwelling
  (`validation/roaming_dwelling.py:258-300`; `data/benchmark/classifier.json`);
- per-animal tallies give the statistics;
- bands come from a cluster bootstrap over recording dates at n animals;
- z = (s − centre)/scale, capped at ±10, and an undefined statistic sits at the cap;
- a family passes when every |z| ≤ z\*, a Bonferroni bar;
- D is the weighted RMS of the capped z's;
- a controller passes the wild type when ≥ 3 of 5 registered 200-animal populations pass every family
  (LB 113 launch).

**Mutants** are graded in scripts on their difference from their own wild type, on the small-lawn
roaming fraction and exits:

- the mutant bar is 2.24;
- only tph-1 (roams more) and tax-4 (roams less) have resolved directions;
- pdfr-1 is graded by magnitude only;
- mutants never enter D (`supporting/113-c1-fit/encodings.md`).

| Version (LB) | Statistics | Families | z\* | Presence rules | Fitted (D_fit) | Validity controls |
|---|---|---|---|---|---|---|
| **v1** (109, 110) | 11 | small lawn (6: roaming fraction, roaming and dwelling bout means, exits per on-lawn hour, pre-exit roaming enrichment, time on lawn); uniform well (3); local search (2: γ, linear early/late) | 2.84 | Local search present (early/late > 1 at its lower bound), added after a no-search population passed 12–13% | small lawn + uniform well | VALID: 14 of 15 real positive units pass. tph-1's leaving reference is unreliable (z 6.37; 110:80). Six negatives fail. W's minimum 1.42 |
| **v2** (115) | 19 | v1 + **kinematics** (5: dwell and roam window speed median and p90, roaming-bout survival at 5 min) + **chemotaxis** (3: drift, log quintile ratio, weathervane slope) | 3.0078 | + chemotaxis presence (drift and log ratio > 0 at 2.5%) | small lawn + uniform + kinematics | VALID: 26 unread N2 dates pass; Chen leave-one-out passes; negatives fail. Adds a measurement model for simulated tracks (`benchmark/measurement.py:1-18`). The on-lawn reversal rate was demoted to reported because no oracle reached it |
| **v3** (120) | 21 | local search on **log** early/late; chemotaxis + **onset rate** + **direction-flip fraction** | 3.0381 | same | unchanged | VALID; catches C.1b's reversal-dominated answers 5 of 5. Reversal share reported, not graded |
| **v4** (124, current default) | 24 | + **activity** on the small lawn (pause share, on-lawn flip fraction, on-lawn onset rate) | 3.0781 | same | small lawn + uniform + kinematics + activity (`bands_v4.json` `fit_families`, checked) | VALID; catches C.2's stage-2 answer 5 of 5. Every negative fails 5 of 5. W's minimum 2.597 (fitted) and 2.047 (held out) |

**Negatives and hazard controllers.**

- **v1 preflight** (LB 109:143-164):
  - ConstantHazards D 6.63 and PirouetteHazards D 7.22;
  - both fail all three families, on roaming fraction, exits, time on the lawn, the uniform roaming
    fraction and γ;
  - **four statistics pass for a Poisson worm**: the dwelling bout mean, pre-exit enrichment, the
    uniform roaming bout mean and early/late.
- **v1 controls** (LB 110:84-103): every negative fails exits and time on the lawn.
  - The open-loop HMM worm passes roaming fraction and dwelling bout on both lawns.
  - The correlated walk passes local search.
  - The untrained MLP passes the uniform well.
  - The untrained connectome passes local search.
- **v2:** the four open-loop negatives fail chemotaxis **only through presence**; none of their
  chemotaxis z's is outside the bar. The weathervane band (0.364 ± 0.883 °/s) includes zero.
- **v4 activity:** the Poisson worm fails only the pause share, and the untrained MLP passes activity
  in 1 of 5 populations. On v4 the untrained connectome still passes the uniform well, the blind worm
  passes kinematics, and C.1's v1 answer passes local search (LB 124:115-128; `read.json`).

### 2.3 C.1's scores

| LB | Version | Fit | Verdict | D | Failing statistics |
|---|---|---|---|---|---|
| 111 | (v1) | NPE, 20 parameters | Informative gate failed (5.32 against ≤ 1.42). The noise floor of 4.08 sat above the bar |: | NPE dropped |
| 112 | v1 | sep-CMA recovery, 4 truths | Pass, median gain 0.991 |: |: |
| 113 | **v1** | 20 parameters, 8 box-centre restarts; **local search fitted** | **3 of 5, pass at threshold** | 0.95–1.26 | small-lawn roaming bout (z 3.55, 4.68) in the failing 2 |
| 115 | v2 | v1 answer | 0 of 5 | 6.1–6.2 | roamed at about 2.6× real speed |
| 119 | **v2** | 20 parameters, ranking score, 8 restarts | **0 of 5** | 1.71–1.87 | **chemotaxis presence in all 5**. Addendum: reversing 52–56% of seconds off food against a real 12–17%; the first diagnosis was retracted |
| 121 | **v3** | **25 parameters** (mechanisms "chosen by looking"), 12 restarts | **3 of 5, pass at threshold** | 1.03–1.14 | small-lawn roaming bout; uniform roaming bout short |
| 124 | v4 | v3 answer | **0 of 5** |: | on-lawn onset rate 0.78–0.85/min against 2.28 (z −6.4 to −6.7) |
| 125 | v4 | +`r_dwell_on_food` (probes) | warm 4 of 5; box centre 0 of 5 in each of 3 runs; refit deferred to C.4 | 1.28–1.44 | one roaming bout |

**C.1's held-out status:**

- Local search and chemotaxis were **fitted, not held out**, for C.1.
- Mutants were C.1's declared prediction test, but have **never been graded**, and pdfr-1 is not
  encodable. The D37 amendment of 2026-11-10 moves tph-1 and tax-4 into C.1's refit too.
- C.1 under-reverses off food (3–5% against 17.3%) and nothing grades that from below (LB 121:132).

### 2.4 C.2's scores (everything before launch 133 is description: one restart, discarded seeds)

| LB | Objective | Wild type | D or D_fit | What fails, including held-out families |
|---|---|---|---|---|
| 122 | v3 D_fit; staged 18 → 196 parameters, 500 generations | not graded on 5 populations | score 3.43; D_fit 3.5–3.6 | speed p90s (+6.5, −5.3), uniform roaming bout, roaming fraction. Fitted before the López-Cruz sign change |
| 123 | + OU speed noise |: |: | **the answer does not run**: speed pinned at 0, curvature at the maximum. Mutants all fail; held-out off-food onset and flip rates fail; the registered slow-noise control fails |
| 126 | v4 D_fit, from an **imitation of C.1** + guard + population 48 | **0 of 5** | D_fit 1.29–1.40 (fitted families 5 of 5); D 3.5–3.7 | **local search fails 5 of 5, chemotaxis 5 of 5, tph-1 5 of 5, tax-4 5 of 5 (wrong way)**; pdfr-1 passes 4 of 5 |
| 127 | **all six families now fitted** | 5 of 6 families on 1 population | D 1.77 | flips and presence; tph-1 fails; pdfr-1 goes the wrong way; mgl-1 inert; inhibition driven to about 0 |
| 128 | ± mutants in the objective |: | D 2.48–3.37 | mutants trade against the wild type; the PDF and ser-5 signs come out reversed |
| 129 | six families + **mutants at half the objective**, inhibition ≥ 0.09, signed PDF and ser-5 | 0 of 5 → 0 of 5 → **3 of 5 after CEP's gain was set by hand** (chosen among ~10 nudges on 1 population; 129:131) | 1.63–1.68 | "The mutants were fitted, not predicted" (129:132) |
| 130 | procedure moved into the package | same 3 of 5 |: |: |
| 131 | clean start (no imitation), curriculum + 3 penalty variants | 0 → 2 → 3 → 3 of 5 (8–9 of 15) |: | frontier between onset rate and dwelling p90 |
| 131 (refit) | + **speed-following on-food reversal patched into the evaluator** | **5 of 5; 15 of 15** | 1.37–1.51 | none. Closest: flips +2.85 |
| 132 | that reversal as `follows_run_speed` | **10 of 10** fresh populations |: |: |

**Launch 133** (feat/fit-c2:`docs/experiments/logbooks/supporting/133-c2-fit/launch.md`):

- **The fit:** the 137-parameter procedure on the following-reversal arm, staged and fully budgeted,
  with up to six starts kept by a lawn rule (mean time on the lawn ≥ 0.95 at 2a generations 40, 50
  and 60).
- **The verdict:** the wild type in ≥ 3 of 5, plus tph-1 and tax-4 each in ≥ 3 of 5. It is stated
  as **"in-sample: every graded statistic and both resolved mutants are fitted"** (lines 132-140,
  checked).
- **Held out:** only a description-level read of Scheer's held-out dates, and a tax-4 leave-one-out
  that gates nothing.
- **Status:**
  - pilot 1 was killed (D_fit 3.13 > 2.5);
  - pilot 2 kept no start of 6 (15.8 h);
  - Part A was amended twice. The 0.95 keep bar was "read off the first six rows" of earlier runs
    (line 284);
  - **pilot 3 is running, with nothing committed.** Off-repo logs suggest start 0 passed generation
    60\. This is unverified, since the forked audit did not open `state.pkl`.
- The projected cost is about 117 h.

### 2.5 Revisions that followed a controller's result

| Revision | It followed | What changed |
|---|---|---|
| v2 | C.1's v1 pass, made "partly through routes no real worm takes" (LB 115:45-50) | measurement model, kinematics, chemotaxis. The record says B was to be "reopened once… so it would not need reopening again"; **it was reopened twice more** |
| v2, internal | an odour-blind worm passing the chemotaxis bands; no oracle reaching the reversal rate | the presence test; the reversal rate demoted to reported |
| v3 | C.1b's reversal-dominated near-miss (LB 119) | log early/late, onset rate, flips. "The candidate fix is in the benchmark" (LB 119:193) |
| v4 | C.2's stage-2 answer that does not run (LB 123), and C.1's v3 pass, which rarely reverses | the activity family, chosen by a probe that scored those controllers (LB 124:66-78). C.1's v3 pass then fails v4 at 0 of 5 |
| Body (LB 132) | C.2 stuck at 8–9 of 15 | speed-following on-food reversal, measured on held-out cohorts. It turned 3 of 5 into 15 of 15 after a refit |
| Objective (LB 127–129; D37 amendment 2026-11-10) | the held-out families and mutants failing (LB 126) | **local search, chemotaxis and the mutants moved into the fit.** No graded family is now held out for C.2 or C.1's refit |

Each revision was registered and validated (held-out real worms pass, negatives fail). The pattern is
still adversarial design: the statistics were picked by probes on the controllers they then catch,
and the held-out data were read in planning (v3's brief read every held-out López-Cruz batch and all
the Chen recordings; LB 120 brief).

### 2.6 Candid flags on the scores

- **Passes sit at the threshold.** C.1 passes at exactly 3 of 5 (LB 113, 121). C.2's first 3 of 5 needed a
  hand nudge.
- **Forking paths.**
  - About 40 adaptively chosen C.2 probe fits or continuations.
  - Each encoder option was added after a failed reading: the soft cap, centres, graded box,
    inhibition floor, receptor signs, OU noise, three guards, the failure-probability term, the
    curriculum and the keep rule.
- **C.2's 15 of 15 is in-sample.** Its final stage directly minimises each statistic's probability of
  failing the bar (LB 131:121-129).
- **The mutant tests are thin.** Each mutant rests on one or two small-lawn statistics. pdfr-1 is
  magnitude-only, so a null passes it. tph-1 has 3 band dates, and its leaving reference is
  unreliable.
- **Thin cells.** The uniform well rests on 2 band dates. Chemotaxis rests on 6 recordings, so in
  practice the presence test carries the family.
- **Both controllers under-reverse off food**, at 3–7% against 17.3%, and that share is not graded.

______________________________________________________________________

## 3. Neural-data assets in the repository

### 3.1 `data/connectome/creamer_lds_2026_model_weights.csv` (the only neural-recording-derived asset)

**What it is** (`data/connectome/PROVENANCE.md:135-166`):

- the signed fitted weights of Creamer, Leifer & Pillow's connectome-constrained linear dynamical
  system;
- a **preprint** (bioRxiv 2024.09.22.614271), fitted to the Randi 2023 optogenetic whole-brain
  calcium atlas;
- 2,011 off-diagonal entries over 125 neurons (the fitted model has 154);
- coefficients of a **2 Hz dynamics matrix on calcium signals**, in no unit the rate model shares;
- fitted on a mask that unions **chemical and gap** edges from White 1986 and Witvliet 2020, untyped
  by connection;
- MIT licence, pinned by SHA-256.

**Consumer:** `packages/quantum-nematode/quantumnematode/connectome/measured_weights.py`:

- digest-checked read (lines 55-91);
- a name-keyed `coverage()` join (lines 134-150).

**Coverage on Cook 2019** (`openspec/changes/archive/2026-09-23-add-measured-weight-prior/design.md:36-46`):

| Edges | Count |
|---|---|
| Chemical edges in Cook 2019 | 3,709 |
| Head scope | 1,386 (23 of them self-loops) |
| Covered | **1,049**: 28.3% of all, 77.0% of coverable head scope; 635 positive, 414 negative |
| On a Cook gap junction only | 265, not applied |
| On no Cook connection | 697, not applied |
| Onto the 39 body motor neurons | **0** |

**How it was consumed** (`cppo:288-301`, `1886-1893`, `805-855`):

- `weight_prior`:
  - `measured` places each covered value × 1/sqrt(in-degree) × one constant that matches the random
    draw's RMS on covered edges;
  - `measured_signs` takes the sign only;
  - `measured_shuffled` permutes the values among covered edges.
- `measured_weight_scale` is the multiplier.
- Uncovered edges keep the random draw.
- These were starting points for PPO (or a frozen matrix under the reading learner), not fixed
  weights.

**What LB 072 found** (the pilot):

- At every scale from 0.25× to 4×, and sign-only, the pathway stays learnable under PPO and under
  the reading learner. The Lee 2026 failure did not appear.
- Under PPO the prior barely moves the plateau (74–77%).
- Under the reading learner, small multipliers *reduce* learning (36.6% against 53.6% at 0.25×).
- Multiplier 1.0 was chosen by a rule blind to the contrast.

**What LB 073 found** (the 2×3 contrast):

- **Under PPO:** the measured prior does not move the wild-type-versus-null wiring effect (+0.013
  [−0.001, +0.027], `no_move`). Whether the fitted placement matters is unresolved (+0.020, upper
  bound 0.040 against a minimum of 0.037).
- **Under the reading learner:** unresolved. The null's lead narrows from −0.109 to −0.026, but the
  interval spans the minimum.
- The block-V wiring advantage is present under measured weights, reported beside the verdict.
- Caveats in its "does not establish": head scope only, no motor layer, and a preprint fitted on
  another connectome (073:176-180).

**Status in the current phase.** The table is **not read by C.2**: `_ENCODER_FIXED` requires
`weight_prior: random` (`cppo:1264`), and the encoder never reads `w_chem` (`cppo:2506-2510`).
"A Creamer-dynamics controller graded on the benchmark" is listed as an optional (MAY) Phase 10
deliverable (`docs/roadmap.md:1742`), and not started.

### 3.2 What is cited but not vendored

From `data/connectome/PROVENANCE.md:258-277`:

- **The Randi et al. 2023 signal-propagation atlas.** The OSF `e2syt` deposit states no licence, the
  `funatlas.h5` in `wormneuroatlas` is GPL-3.0, and `leiferlab/worm-functional-connectivity` states
  no licence.
- **Creamer's model pickles** (they execute code on load).
- **OSF `qxhjd`**, Randi's recordings reformatted, with no licence.

The project's data rule accepts CC0 or CC BY only (`data/neuromodulators/PROVENANCE.md:49-52`).
That excludes the Randi atlas as distributed, and MIT-only data.

### 3.3 Nothing else derived from neural activity

- I searched `data/`, `docs/`, `packages/`, `scripts/` and `openspec/` for Atanas, Randi, calcium,
  GCaMP, NeuroPAL, whole-brain, `funatlas` and Kato. I found no vendored recording, no loader and no
  observation model. Hits were citations and roadmap prose (for example `docs/roadmap.md:795, 1283`)
  and a validation idea in `docs/nematode_biology.md:691`.
- The other `data/` assets are anatomical or behavioural:
  - **Expression and anatomy:** the Fenyves sign predictions, the Wang 2024 transmitter atlas, the
    Bentley and Ripoll-Sánchez neuromodulator placement, and the receptor couplings.
  - **Behaviour:** `data/benchmark/*`, `data/roaming_dwelling/*`, `data/chemotaxis/*`,
    `data/thermotaxis/*` and `data/posture/*`.
- The 67 physiology sign overrides (`sign_overrides_physiology.csv`) are qualitative signs quoted
  from electrophysiology and imaging papers, not activity data.

______________________________________________________________________

## 4. Could the simulator be made differentiable, and at what cost?

### 4.1 What is PyTorch and what is NumPy

| Component | Framework | Autograd today? | Where |
|---|---|---|---|
| Wiring composition (`compose_chemical`, `compose_injection`), batched over candidates | torch | **yes** | `enc:302-342` |
| Settling map (4 tanh iterations) and readout | torch. NumPy `tanh` only when no gradient is needed | **yes**. `_settling_tanh` switches to `torch.tanh` when a gradient is taken | `cppo:1587-1597, 2512-2543` |
| Modulator current from concentrations | torch | yes, with respect to the strengths | `cppo:2554-2557` |
| Modulator concentration update across steps | **NumPy, detached** ("not a path credit flows back through") | **no** | `cppo:4390-4403`; `ctl:231-238` |
| Modulator timescales and OU speed-noise timescale | torch **buffers** → NumPy decay | **no** | `enc:479-484, 519-541` |
| Unit-box vector → tensors (`write_encoder`) | Python `math` under `no_grad` | **no** (easy to re-express) | `enc:568-621` |
| Policy sampling (tanh-Gaussian, OU) | torch | reparameterisable in principle, but the batched controller runs under `no_grad` and returns NumPy | `ctl:92-108, 220-272` |
| Leaky dynamics and BPTT chunks (refused under the encoder) | torch | **yes, through time within a chunk** | `cppo:2685-2697, 3784-3828` |
| Environment: odour field, STAM, fold-change, movement, eating, satiety | **NumPy** | no | `bat:487-792` |
| Locomotor primitives: bout start, kind, duration, angle | **NumPy RNG** in a Python loop over idle animals | no | `bat:712-762`; `prim:62-90, 361-454` |
| Measurement model | NumPy | no | `benchmark/measurement.py` |
| Grading: HMM decode, windows, tallies, bands, z, presence | NumPy | no | `benchmark/grading.py`, `chemotaxis.py`, `readouts.py`; `validation/roaming_dwelling.py` |
| C.1 (`BatchedMechanistic`) | NumPy | no | `ctl:537-722` |
| The fitter | `cma` (sep-CMA), black-box |: | feat/fit-c2:`scripts/analysis/c2_fit.py:1146-1174` |

**The brain's one-step map is already autograd-capable, and that has been exercised.** LB 126's
imitation probe (`supporting/126-c2-run-mode/probes/imitation_d.py.txt:118-147`) worked as follows:

- **The data:** 720,000 (state, teacher action) pairs per round, from C.1 driving rollouts. A shadow
  C.2 advanced its own modulators along them.
- **The training:** `forward_with_hidden_batched` → `loss.backward()` with Adam, in three rounds of
  20 epochs, about 3.3 h.
- **The fit** (`imitation_training.log.txt:22, 44, 66`):
  - R² for speed was 0.28–0.29, and for reversal 0.08–0.29;
  - omega reached 0.14–0.23;
  - curvature reached 0.001–0.07, since C.1's curvature noise draw cannot be learned.
- **Closed-loop**, the result left the lawn (D_fit 8.0; LB 126:97-101).

The logbook's diagnosis is that the memoryless student "learns the teacher's average action given the
senses, not its roaming or dwelling state". That is direct evidence on two points:

- open-loop gradient fitting of this architecture is cheap;
- its fidelity is limited by the absence of neural state, and by distribution shift when the loop is
  closed.

### 4.2 What is non-differentiable in the evaluator, primitives and grading

1. **Bout starts are Bernoulli draws.** `rng.random() < 1 − exp(−Σhazards·dt)` runs per idle animal
   in Python (`bat:733-735`; `prim:417-418`).

2. **The bout kind is a categorical draw** (`bat:736-743`).

3. **Durations are gamma draws and turn angles are rejection-sampled truncated normals**
   (`prim:62-90`).

4. **During a bout the controller's outputs move nothing**, so the gradient is zero for seconds at a
   time (`prim:16-22`; `bat:694-703`).

5. **Hard clamps and lookups:**

   - action clip (`bat:666`);
   - hazard clip in the controllers (`ctl:275-281`);
   - arena clip and circle projection (`bat:557-573, 707-710`);
   - grid-cell lookup by `np.rint` for food contact and eating (`bat:531-555`);
   - lawn depletion (`bat:780-783`).

6. **Discontinuous sensing state:** the STAM ring buffer, and the action-entropy feature computed with
   `np.unique` (`bat:651-656`).

7. **Measurement:** resampled real-tracker noise and run-length losses, drawn per animal
   (`benchmark/measurement.py:1-18`).

8. **Grading:**

   - Viterbi-style HMM state labels;
   - window rules (on, off or unclassified);
   - censoring-aware bout means;
   - exit counts;
   - a thresholded reorientation detector (0.03 mm floor, >135° flips);
   - z-capping at ±10;
   - Bonferroni pass and fail;
   - resampled presence tests;
   - "undefined" statistics.

   None of these is a smooth function of the trajectory.

9. **Horizon:** a small-lawn assay is 2,400 one-second steps, and López-Cruz food removal is 5,400
   (`scripts/analysis/c2_encoder_probe.py:1785`). Backpropagating closed-loop through thousands of
   steps of a stochastic hybrid system is the core difficulty, independent of framework.

### 4.3 Line counts of what a differentiable backend would touch

| Module | Lines | Role |
|---|---|---|
| `brain/arch/connectome_ppo.py` | 5,309 (about 1,600–3,900 is the topology and forward; much of the rest is legacy options and PPO plumbing) | brain |
| `connectome/encoder.py` | 656 | encoder |
| `connectome/neuromodulator_layer.py` | 309 | layer spec |
| `connectome/signs.py`, `loader.py`, `neurons.py`, `neurotransmitters.py`, `neuron_classes.py`, `model.py`, `measured_weights.py`, `rewiring.py` | 486, 598, 408, 236, 73, 178, 150, 375 | **reusable as they are** (data → Python objects) |
| `evaluation/batched.py`, `controllers.py`, `odour_field.py`, `reference.py` | 854, 722, 103, 102 | evaluator |
| `env/primitives.py` | 461 | primitives |
| `env/continuous_2d.py`, `env/lawns.py`, `env/assay.py`, `agent/stam.py`, `utils/step_duration.py` | 961, 292, 108, 472, 454 | used by the evaluator at construction or in the per-step runner |
| `brain/arch/_policy.py`, `learning_rules/ppo.py` | 658, 535 | sampling, PPO |
| `benchmark/*.py` (grading, chemotaxis, measurement, readouts, references, track_import) | 2,392 | grading |
| `scripts/analysis/benchmark_grade.py`, `c2_encoder_probe.py`, feat/fit-c2:`c2_fit.py` | 945, 2,116, 2,122 | fitting and grading drivers |
| Tests: evaluator parity, `test_connectome_encoder.py`, connectome-brain tests | 1,645, 724, about 11,000 in total | parity guarantees a port would have to re-earn |

The whole package is about 96,000 lines of Python. NumPy (≥ 2.2.4), torch (≥ 2.7, an extra) and
`cma` are dependencies; JAX is not (`packages/quantum-nematode/pyproject.toml:37-77`).

### 4.4 Three options, with rough effort

The effort estimates are my judgement from the code's structure, not measured. They assume one
engineer familiar with the codebase.

**(a) A differentiable neural model only, fitted open-loop to recorded stimuli and activity.**

- **Reuse:**
  - the loaders (Emmons or Cook);
  - `per_connection_signs()`;
  - the neuron tables and classes;
  - the neuromodulator `LayerSpec`, including the knockouts and the receptor-placement null;
  - `rewire_degree_preserving` for nulls;
  - `compose_chemical` and `compose_injection` (`enc:302-342`) as the count- and sign-based wiring
    composition.
- **New:**
  - a continuous-time torch rate model with per-neuron (or per-class) time constants, biases and
    gains, and possibly per-edge weights on the mask. The existing leaky integrator is a precedent,
    but has one τ for all neurons (`cppo:2579-2609`);
  - the modulator pools moved into autograd;
  - a calcium observation model;
  - a stimulus-input model per sensory class, and an optogenetic current-injection path, which the
    injection map makes easy;
  - data ingestion;
  - a training loop (BPTT or multiple shooting over the recordings);
  - held-out evaluation on held-out neurons, stimuli or animals.
- **Size:** roughly 1,000–2,500 new lines, plus data ingestion.
- **Effort:** about 2–5 weeks of engineering. **Compute is small**: 302×302 maps over recordings of
  hours at 1–5 Hz.
- **The binding constraint is data.** No recordings are vendored. The Randi atlas is GPL-3 or
  unlicensed (3.2), and the project's rule is CC0 or CC BY. The licences of other whole-brain sets
  (for example Atanas 2023, Kato 2015, Dag 2023) were **not checked** in this audit.
- **Connecting it to the benchmark** still needs a readout from the fitted network to the primitives
  (the existing pooled readout could serve) and the closed-loop evaluator. That loop is
  non-differentiable, and that is where the behaviour test would happen.

**(b) A PyTorch adapter that runs the existing brain and evaluator differentiably.**

- **Brain side, small** (about 100–300 lines):
  - move the concentration update into torch with gradient;
  - make the timescales `Parameter`s;
  - write a differentiable z → tensor map;
  - keep the OU noise as a reparameterised process.
- **Evaluator side, large:**
  - port `batched.py`'s per-step arithmetic (about 850 lines) to torch;
  - replace the discrete primitives with relaxations (Gumbel-softmax or concrete starts and kinds,
    reparameterised gamma durations) or with score-function (REINFORCE-style) gradients;
  - soften the clamps and lookups;
  - replace the grading with differentiable surrogates, such as HMM forward-backward posteriors
    instead of decoded labels, expected bout lengths and soft counts.
- **A surrogate objective is not the registered benchmark**, so a fit would still need re-grading
  with the real one.
- **Effort:** roughly 6–12 weeks.
- **Risk is high:**
  - gradient variance and bias over 2,400–5,400-step stochastic closed loops;
  - surrogate–benchmark mismatch;
  - re-earning the parity and byte-identity checks (LB 104, 118, 122).
- **A cheaper variant (my inference, untested):** use the hazards' exact likelihood. Bout starts are
  competing Poisson hazards, so the log-likelihood of an observed sequence of runs and bouts given the
  senses is differentiable. It could be evaluated **open-loop on real tracks**, with senses
  reconstructed along each real path, instead of on simulated ones. Scheer's tracks carry per-frame
  reversal flags (AGENTS.md; `data/benchmark/PROVENANCE.md`). This is "fit to recordings of
  behaviour" rather than neural activity. LB 126's imitation is its closest precedent, and shows the
  memoryless brain's limit.

**(c) A rewrite, for example in JAX** (brain, environment, primitives and grading surrogates).

- **Size:** the brain core (about 1,500 relevant lines plus the 656-line encoder and the 309-line
  layer), the evaluator and primitives (about 1,500) and grading (about 2,400). That is roughly
  6–8k lines to port, plus tests.
- **Effort:** about 3–6 months.
- **Gains:** jit, vmap and GPU, and one differentiable code path.
- **Losses:**
  - the byte-identity link to about 30 logbooks' committed records;
  - the batched-evaluator parity tests (1,645 lines);
  - a new dependency.
- **The fundamental obstacles of (b) remain:** discrete bouts, thresholded statistics and long
  horizons.

### 4.5 What fitting to neural data would need that does not exist

- **Recordings with a usable licence**, vendored with provenance. None are present; the Randi atlas
  is excluded by licence.
- **A calcium observation model:**
  - GCaMP kinetics (rise and decay convolution);
  - a per-neuron scale, offset and nonlinearity from rate or voltage to ΔF/F;
  - noise;
  - bleaching and motion artefacts;
  - the sampling rate (about 1–5 Hz against the model's 1-s step).
- **Neural time constants and persistent state.** The settling model has none, and resets every
  second. Calcium data have dynamics on seconds to minutes (for example AVA/RIM reversal states and
  roaming/dwelling-related activity), which the current model can only express through modulator
  pools.
- **Unrecorded neurons.** Head-imaging datasets identify a subset; Creamer's table covers 125 of 302
  and reaches no body motor neuron. The fit needs latent-neuron treatment (marginalisation or
  Kalman-style inference, as Creamer's LDS uses) or explicit unconstrained neurons.
- **Neuron identity matching** to the 302 names (NeuroPAL-style IDs). The name tables and alias
  handling exist (`connectome/neurons.py`, `CANONICAL_NAME_ALIASES`).
- **Stimulus inputs:**
  - cell-specific stimulus channels (odour pulses on AWA or AWC, salt on ASE, temperature on AFD,
    optogenetic drive), in place of the single abstract food signal shared by all 11 classes
    (`enc:43-56, 345-370`);
  - the thermal and mechanosensory projections exist only in the legacy, non-encoder path
    (`cppo:2132-2258`).
- **For freely moving data**, a mapping from recorded behaviour (velocity, head curvature) to the
  model's motor side. The current readout pools four ventral-cord classes that head imaging rarely
  records.
- **A unit bridge** if fitted weights are reused in the rate model. LB 072 matched RMS by a constant,
  and noted that the LDS coefficients are "in no unit this rate model shares"
  (`measured_weights.py:7-9`).

______________________________________________________________________

## 5. Not verified, and caveats

- **Pilot 3 of launch 133:** its state comes only from off-repo logs reported by a forked audit
  (`build/c2-fit/pilot.log`). `state.pkl` was not opened.
- **The off-food reversal share of LB 131's final answer** is not in `hinge4.json`. LB 129 reports
  5.5–7% for its answers.
- **LB 116 and 117 numbers** come from AGENTS.md, not from re-reading those logbooks.
- **No C.3 (reward-trained C.2) result** was found. I did not search exhaustively beyond the logbook
  index.
- **How the uniform well's scales "rest on the small lawn's"** (LB 109:190) was not traced in code.
- **The licences of external whole-brain datasets** other than Randi's (Atanas 2023, Kato 2015,
  Dag 2023) were not checked.
- **Effort estimates in 4.4 are judgements**, not measurements. The cheaper likelihood variant in
  4.4(b) is my inference and has not been tried in the repository.
- **The 17-fixed and 104-fitted split of modulator rows** is reconciled from `layer.json` and LB 122.
  `PROVENANCE.md` counts 16 and 105 by couple. Both are consistent once the drive and the
  producerless flp-24 couple are accounted for. I did not load the spec in code to confirm.
