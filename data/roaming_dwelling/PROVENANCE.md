# Roaming and dwelling reference data

## `reference_hmm.json`

The two-state roaming/dwelling hidden Markov model of Scheer & Bargmann 2023:

- **Citation:** Scheer, E. & Bargmann, C. I. (2023). Sensory neurons couple arousal and foraging
  decisions in *Caenorhabditis elegans*. *eLife* 12:RP88657.
- **Deposit:** Dryad doi:10.5061/dryad.47d7wm3jf, mirrored at Zenodo record 8310289.
- **Licence:** CC0-1.0.
- **Source file:** `PD1074_od2_LL_Data_Centroid_RoamingDwellingHMM_081721.pkl`, 623 bytes, MD5
  `84f3c8255ca1e1905cd657a325451eb8` (as published), downloaded 2026-10-10.

**How the parameters were read.** The source is a pickled `ssm.hmm.HMM` (Linderman lab `ssm`). It was
loaded once with a restricted unpickler: the `ssm` classes became empty stand-ins, so no `ssm` code
ran, and only numpy's array reconstruction was allowed. Three arrays were taken from it:

- `init_state_distn.log_pi0`;
- `transitions.log_Ps`;
- `observations.logits`, for two categories.

Each was normalised with a log-softmax, as `ssm` normalises categorical logits, and written here.

**State order.** State 1 is roaming: it emits roaming observations (category 1) 53% of the time,
against dwelling's 1%. The authors' code orders states so roaming is the higher-speed state
(`permuteRDHMMStates`).

**Observations.** Each 10-second bin, in-lawn only, is a roaming observation when
`midbody speed × 450 > midbody angular speed`. This is `decision_slope = 450` in
`LawnLeavingAnalysis/preProcessing.py` of github.com/BargmannLab/Scheer_Bargmann2023 (MIT). This
project recalibrates the slope at its own 5-second step and keeps the model; see
`quantumnematode/validation/roaming_dwelling.py`.

## `ji2021_readings.json`

Fractions of time roaming from Ji et al. 2021, used as **directions**, never matched as absolute
values:

- **Citation:** Ji, N., Madan, G. K., Fabre, G. I., Dayan, A., Baker, C. M., Kramer, T. S.,
  Nwabudike, I. & Flavell, S. W. (2021). A neural circuit for flexible control of persistent
  behavioral states. *eLife* 10:e62889.
- **Deposit:** Dryad doi:10.5061/dryad.3bk3j9kh3.
- **Licence:** CC0-1.0.
- **Source file:** `RD_manuscript_data.mat`, MD5 `8afe85564b77d5249fa55d91237f592e`, SHA-256
  `aed6b71a26f19e9e4de9b9e5cebfa2ecabb33b2a9458c1e07741a889c153c807`, downloaded 2026-10-10 by
  browser, since Dryad refuses scripted downloads. It is not vendored; this file is derived from it.

**Mapping.** The deposit names variables by the preprint's figures: its `Data_F6` is the published
paper's Figure 7, the patch foraging assay. Matched against the published legends:

- `Data_F6.BEF` is Fig. 7B and 7E and Figure 7—figure supplement 1C. Each array holds fractions of
  time roaming on sparse food, with NaN entries. The count of non-NaN values is close to the
  legend's n but not equal to it:

  | arm | non-NaN values | legend's n |
  |---|---|---|
  | wild type | 248 | 288 |
  | uniform-food control | 128 | 194 |
  | *pdfr-1* | 103 | 99 |
  | *tph-1* | 233 | 212 |

  So each value is read as roughly one per animal. Their mean and median are reported. Only
  directions are used from them, so the exact unit does not change any reading.

- `Data_F6.G` is Fig. 7F, wild type on uniform food at two densities:

  - **Fractions:** 200 tightly clustered values per condition, read as resamples, and their mean is
    reported.
  - **Durations:** the legend does not give their unit, so they are used only as ratios between
    the two densities.

**Directions** (verified against the legends):

- wild type roams more on sparse food with a dense patch nearby than on uniform sparse food;
- wild type roams less, and dwells longer, at the higher density;
- *tph-1* roams more than wild type in the patch assay, and *pdfr-1* less.

## `scheer2023_windows.npz` and `calibration.json`

**`scheer2023_windows.npz`** is derived from Scheer & Bargmann 2023's wild-type deposit (CC0):

- **Source file:** `PD1074_od2_Fig1_021523.pkl` (Zenodo record 8310289), 4,871,119,532 bytes, MD5
  `f8556a4684e7a7f30a3dc01242591556` as published, downloaded 2026-10-10. It is not vendored.

- **How it was read.** Its opcodes were scanned first: it references only numpy, pandas,
  `builtins.slice`, `datetime.date` and the `ssm` model classes. It was then loaded with an unpickler
  that admits only those modules and replaces `ssm` with stand-ins.

- **What each animal contributes.** The animal's midbody position (`Midbody_cent_x/y`) is sampled
  every 15 frames (5 s at 3 frames/s) and divided by its own `pixpermm`. Its 10-second windows are
  then measured exactly as simulated tracks are.

- **What each window keeps:**

  - its speed and angular speed;
  - whether it lies in an in-lawn run (`InLawnRunMask`);
  - the authors' label (`RD_states_Matrix_exog`: 1 roaming, 0 dwelling, -1 masked off-lawn).

  The windows line up with the authors' 10-second bins.

- **Size:** 1,586 animals, 240 windows each.

**The model reads correctly.** Fed the authors' own bin measures (`bin_Midbody_absSpeed_inLawn`,
`bin_Midbody_angspeed_inLawn`, slope 450), the vendored model decoded by this project's Viterbi
reproduces their labels on 99.2% of 372,860 bins.

**`calibration.json`** is the output of `scripts/analysis/roaming_dwelling_calibration.py calibrate`:
the slope fitted on half the animals (split seed 2026), and the held-out agreement against the gate
of kappa >= 0.6 that was fixed before calibration.

## `calibration_retry.json`

The one registered retry after `calibration.json` failed its gate. It was registered in the design at
commit f015a7ea before it was computed, and produced by `roaming_dwelling_calibration.py retry`:

- **Model:** a two-state Gaussian-emission HMM on each window's `(log(speed + 0.001 mm/s), angular speed)`.
- **Fitting:** with the authors' labels, on the same calibration half (split seed 2026).
- **Decoding:** Viterbi per on-lawn run.
- **Held-out result:** κ = 0.632 against the gate of 0.6, accuracy 91.1%. On-lawn roaming is 17.4%
  against the authors' 10.6%.

Its `parameters` are the instrument this project uses (`load_calibrated_hmm`).
