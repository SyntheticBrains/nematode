# Locomotor primitive kinematics

`primitives.json` fixes the point worm's locomotor primitives under `continuous.locomotion: primitives`.
These are run speed and curvature bounds, reversal speed and duration, and omega and delta turn angles
and durations, plus the range of the controller's start hazards.

Every value but two is transcribed from published numbers. No dataset is vendored here.

- **Status.** Each entry says where it comes from, under which condition, and how it was obtained:
  - `verified`, read in the text;
  - `figure`, read from a figure by eye (about ±10%);
  - `measured`, computed from deposited tracks. The entry names the script (`script`) and its
    committed output (`record`), and the loader refuses a `measured` entry without both.
- **The published sources are off food.**
  - Broekmans et al. 2016 recorded worms foraging off food inside a copper ring, which its reviewers
    noted is aversive.
  - Zhao et al. 2003 and Pierce-Shimomura et al. 1999 are off-food assays.
- **Delta turns** in Broekmans et al. 2016 are the deepest coils, reorienting dorsally. They are not
  shallow turns.
- **Check.** The roadmap's block B (B.1) checked these values against Scheer & Bargmann 2023's
  on-food movement flags, on cohorts held out of the benchmark (Logbook 105). On the lawn, reversals
  were shorter and slower than every published off-food value.
- **On food** (`on_food_reversal_duration_s`, `on_food_reversal_speed_mm_per_s`, added 2026-10-24).
  - Measured on the OD 2 wild type's held-out recording dates, from bouts starting more than 0.5 mm
    inside the lawn: 2.047 ± 0.899 s (n = 24,890) at 0.069 mm/s (n = 17,632).
  - Script: `scripts/analysis/primitive_kinematics_check.py`. Record:
    `docs/experiments/logbooks/supporting/105-reference-extraction/kinematics.json`.
  - Scheer & Bargmann's deposit is CC0 (Zenodo 8310289).
  - A reversal that starts on a lawn draws these, unless `continuous.reversal_kinematics` is
    `off_food` (Logbook 108).

Sources:

- Zhao, Khare, Feldman & Dent 2003, *J Neurosci* 23:5319, doi:10.1523/JNEUROSCI.23-12-05319.2003.
- Pierce-Shimomura, Morse & Lockery 1999, *J Neurosci* 19:9557, doi:10.1523/JNEUROSCI.19-21-09557.1999.
- Broekmans, Rodgers, Ryu & Stephens 2016, *eLife* 5:e17227, doi:10.7554/eLife.17227.
- Gray, Hill & Bargmann 2005, *PNAS* 102:3184, doi:10.1073/pnas.0409009101.
- Ben Arous, Laffont & Chatenay 2009, *PLoS One* 4:e7584, doi:10.1371/journal.pone.0007584.
- Scheer & Bargmann 2023, *eLife* 12:RP88657, doi:10.7554/eLife.88657.

Transcribed 2026-10-24.
