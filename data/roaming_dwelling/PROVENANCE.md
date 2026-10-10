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
