# Posture data: provenance

Two files, vendored verbatim and pinned by SHA-256, for the body-level validation's posture
instruments. Both are stored in Git LFS, which keeps their bytes exact.

## `EigenWorms.csv` — the eigenworm basis

- **What**: 100 eigenworms over 100 tangent angles, head to tail. **Each column is one mode**, in
  decreasing order of variance; the first four are the ones the instruments read.
- **Source**: the Stephens group's WormPose (Hebert, Ahamed, Costa, O'Shaughnessy & Stephens 2021),
  `extras/EigenWorms.csv` in `iteal/wormpose` at commit `fb1d77eac56538569f639ab58a62470bff10f04c`:
  <https://raw.githubusercontent.com/iteal/wormpose/fb1d77eac56538569f639ab58a62470bff10f04c/extras/EigenWorms.csv>
- **SHA-256**: `bc806b9073972f6c6f6b0c48ecf8b9278bff2433ee2b39906423d58b156d930f`
- **Licence**: BSD-3-Clause, Copyright (c) 2020, Okinawa Institute of Science and Technology. The
  licence text is [LICENSE-wormpose.txt](LICENSE-wormpose.txt), from the same commit.
- **Identity with Stephens et al. 2008 is inferred, not stated by the file.** WormPose projects onto
  eigenworms citing Stephens, Johnson-Kerner, Bialek & Ryu 2008 (*PLoS Comput Biol* 4:e1000028), and
  the group's Broekmans et al. 2016 took its eigenworms from that paper. The real postures below,
  which their tutorial introduces as coming from Stephens et al.'s experiment, are consistent with it:
  the first four modes capture **96.49%** of their pooled sum of squares here. Wormlight, reading the
  same two files, reports 96.46% by its own computation.

## `shapes.csv` — real postures

- **What**: 6,655 postures of real *C. elegans*, each 100 tangent angles head to tail with its mean
  removed, which the tutorial introduces as coming from the experiment of Stephens et al. 2008.
- **Source**: the OIST Physics of Behavior tutorials v1.0 (Korshok, Béraud, Stephens et al.), Zenodo
  doi:[10.5281/zenodo.15099731](https://doi.org/10.5281/zenodo.15099731), `data/shapes.csv` at the
  commit that archive holds, `9ce75a49356dbe718b42b92f85495b2cba4bddd8`:
  <https://raw.githubusercontent.com/oist/Physics-of-Behavior-Tutorials/9ce75a49356dbe718b42b92f85495b2cba4bddd8/data/shapes.csv>
- **SHA-256**: `410abb65af193b86d273bdbba670406c088032b9774e5808b02bb138950fb4b1`
- **Licence**: CC BY 4.0, per the Zenodo record of v1.0
  (<https://creativecommons.org/licenses/by/4.0/>). Attribution: *Physics of Behavior Tutorials*,
  OIST Biological Physics Theory Unit, v1.0, doi:10.5281/zenodo.15099731. Unmodified.
- **Use**: the reference for the eigenworm basis above, and the real distribution of peak curvature
  the body's amplitude is set beside.

Both pins are the ones [Wormlight](https://github.com/chrisjz/wormlight) records in its
`DATA_SOURCES.md`, and the SHA-256s match its; checked when vendored on 2026-10-10.
