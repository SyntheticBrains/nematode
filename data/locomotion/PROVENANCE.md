# Locomotor primitive kinematics

`primitives.json` fixes the point worm's locomotor primitives under `continuous.locomotion: primitives`.
These are run speed and curvature bounds, reversal speed and duration, and omega and delta turn angles
and durations, plus the range of the controller's start hazards.

Every value is transcribed from published numbers. No dataset is vendored.

- **Status.** Each entry says where it comes from, under which condition, and whether it was verified
  in the text (`verified`) or read from a figure by eye (`figure`, about ±10%).
- **All sources are off food.**
  - Broekmans et al. 2016 recorded worms foraging off food inside a copper ring, which its reviewers
    noted is aversive.
  - Zhao et al. 2003 and Pierce-Shimomura et al. 1999 are off-food assays.
- **Delta turns** in Broekmans et al. 2016 are the deepest coils, reorienting dorsally. They are not
  shallow turns.
- **Check.** The roadmap's block B (B.1) checks these values against Scheer & Bargmann 2023's on-food
  movement flags, on cohorts held out of the benchmark.

Sources:

- Zhao, Khare, Feldman & Dent 2003, *J Neurosci* 23:5319, doi:10.1523/JNEUROSCI.23-12-05319.2003.
- Pierce-Shimomura, Morse & Lockery 1999, *J Neurosci* 19:9557, doi:10.1523/JNEUROSCI.19-21-09557.1999.
- Broekmans, Rodgers, Ryu & Stephens 2016, *eLife* 5:e17227, doi:10.7554/eLife.17227.
- Gray, Hill & Bargmann 2005, *PNAS* 102:3184, doi:10.1073/pnas.0409009101.
- Ben Arous, Laffont & Chatenay 2009, *PLoS One* 4:e7584, doi:10.1371/journal.pone.0007584.

Transcribed 2026-10-24.
