# Benchmark reference data

Derived files only: each is produced from its public deposit by
`scripts/analysis/extract_benchmark_references.py`, and read back by
`quantumnematode.benchmark.references`. The deposits are not vendored. Positions are stored as
integers, a stated unit times a fixed factor, with the integer type's minimum marking a missing value.

## `scheer2023_tracks.npz` and `scheer2023_split.json`

- **Citation:** Scheer, E. & Bargmann, C. I. (2023). Sensory neurons couple arousal and foraging
  decisions in *Caenorhabditis elegans*. *eLife* 12:RP88657.

- **Deposit:** Dryad doi:10.5061/dryad.47d7wm3jf, mirrored at Zenodo record 8310289. **Licence:**
  CC0-1.0. The authors' analysis code (github.com/BargmannLab/Scheer_Bargmann2023, MIT) defines each
  field (`LawnLeavingAnalysis/preProcessing.py`).

- **Source files** (downloaded 2026-10-24, scripted from Zenodo; neither is vendored):

  - `GenotypeComparisons_021523.pkl`, 20,597,126,711 bytes, MD5 `41d93e6b65725890908c2acb611d8208` as
    published;
  - `PD1074_od2_Fig1_021523.pkl`, 4,871,119,532 bytes, MD5 `f8556a4684e7a7f30a3dc01242591556` as
    published, read only to match its animals.

- **How it was read:** through the restricted unpickler in `scripts/analysis/_scheer_pickle.py`
  (numpy, pandas, `slice` and `datetime.date` admitted; the `ssm` classes replaced by stand-ins). The
  genotype file was written with pandas 1.x: its numeric index classes are read as `pandas.Index`, as
  pandas' own `read_pickle` reads them.

- **Arms** (deposit keys), 2,847 animals of 40 min at 3 frames/s:

  | Arm | Animals | Dates |
  |---|---|---|
  | `PD1074_od2_Data` (wild type, OD 2) | 1,586 | 65 |
  | `PD1074_od1_Data` | 45 | 1 |
  | `PD1074_od4_Data` | 83 | 3 |
  | `PD1074_od2_uniformlawn_Data` | 100 | 3 |
  | `N2_od2_Data` (wild type, the N2 strain) | 1,380 | 70 |
  | `MT15434_tph1mg280_od2_Data` (tph-1) | 139 | 5 |
  | `CX14295_pdfr1ok3425_od2_Data` (pdfr-1) | 395 | 16 |
  | `CX13078_tax4p678_od2_Data` (tax-4) | 499 | 19 |

  N2 is the batch-matched control of tph-1, an N2-background strain recorded only beside N2: all five
  of tph-1's dates have N2 animals and none has PD1074. pdfr-1 shares 11 of its 16 dates with PD1074
  and 2 with N2; tax-4 shares 17 of 19 with PD1074. The two wild-type strains are separate arms, never
  pooled.

- **Positions** (`x_um`, `y_um`, int16 µm): the midbody (`Midbody_cent_x/y`) at frames 0, 3, 6, …,
  divided by the animal's own `pixpermm` (median 116 px/mm, so a pixel is about 8.6 µm), from its lawn
  centre. −32768 marks a missing frame.

- **The lawn's geometry.** The lawn is re-detected in each background segment of a recording.
  `Center_Point_x/y` has one column per background slot (20), and `Lawn_Boundary_Pts_x/y` one 360-point
  polygon per slot in a block of rows per animal; unused slots hold 0 or NaN. A frame's slot is its
  `bgvidindex` (1-based). Each animal's centre is the median over the slots its frames use (99th
  percentile of the centre's spread across slots: 11–66 µm by arm), and its 64 boundary radii
  (`boundary_um`) the median of those slots' radii about that centre.

  - **Checked against the authors':** the signed midbody distance to this boundary differs from their
    `Centroid_Lawn_Boundary_Dist` by a median of 0.022–0.025 mm by arm (95th percentile ≤ 0.085 mm),
    and its sign agrees with their `MidbodyInLawn` on 99.0–99.8% of samples. Their
    `Lawn_Boundary_Dist` is the head's distance (it agrees with `HeadInLawn` on every frame), so it is
    not the comparison.
  - **The uniform lawn** has no boundary fields, nor `HeadPokesAll` or `MidbodyInLawn` (stored as
    zero). Its positions are from each animal's first valid position (`origin = 1`), and its radii are
    the sentinel.

- **Flags** (`frame_flags`, bit-packed per frame): bit 0 `MovingReverse`, 1 `Omega`, 2 `Pause`, 3
  `HeadPokesAll`, 4 `MidbodyInLawn`, 5 `missingdata`. `MovingReverse` and `Pause` are the authors'
  thresholds on midbody speed. Every `Omega` run touches `missingdata`: the skeleton fails while the
  worm coils.

- **Per-bin fields** (10 s, 240 bins): `in_lawn_run` (`InLawnRunMask`), `out_lawn_run`
  (`OutLawnRunMask`) and `labels` (`RD_states_Matrix_exog`: 1 roaming, 0 dwelling, −1 masked).

- **Cohort fields:** `genotype` (index into `genotypes`), `date` (YYYYMMDD, parsed by the authors from
  the video name), `video` (index into `videos`), `source_row`, and `fig1_row`, the animal's row in the
  Fig. 1 file. All 1,586 Fig. 1 animals match `PD1074_od2_Data` by video and source file, in the same
  order, with equal positions.

- **Reproduces D.1's windows.** Re-deriving the 5-s windows from this store reproduces
  `data/roaming_dwelling/scheer2023_windows.npz` on every Fig. 1 animal: masks, labels and missing
  windows identical, speeds within the rounding's bound (worst 0.25 µm/s against 0.28), the calibrated
  model's likelier emission state on 99.92% and its decoded states on 99.96% of on-food windows
  (`tests/quantumnematode_tests/benchmark/test_scheer_store.py`).

- **The split** (`scheer2023_split.json`): cohorts are recording dates, everything on a date on one
  side. It was drawn from each animal's date and arm alone, before any behavioural statistic was
  computed, by `extract_benchmark_references.py split` (seed 2027, draw k = 16 of at most 10,000).
  The OD 2 wild type has 24 of 65 dates held out (36% of its animals), N2 28 of 70 (39%), tph-1 2 of 5
  (25%), pdfr-1 6 of 16 (35%), tax-4 7 of 19 (38%), OD 4 and the uniform lawn 1 of 3 each. OD 1 was
  recorded on one date and has no held-out cohort. **Held out means held out of the bands, not unseen
  by the instrument:** the roaming/dwelling classifier was calibrated on an animal-level split (seed
  2026\) across every wild-type date, and `scheer2023_leaving.json` used every wild-type animal.

- **Size:** 32.5 MB.

- **Commands:** `extract_benchmark_references.py scheer --genotypes GenotypeComparisons_021523.pkl --fig1 PD1074_od2_Fig1_021523.pkl --out data/benchmark/scheer2023_tracks.npz --record <record.json>`, then `extract_benchmark_references.py split --tracks data/benchmark/scheer2023_tracks.npz --out data/benchmark/scheer2023_split.json`.

## `calhoun2014_tracks.npz`

- **Citation:** Calhoun, A. J., Chalasani, S. H. & Sharpee, T. O. (2014). Maximally informative foraging
  by *Caenorhabditis elegans*. *eLife* 3:e04220.
- **Deposit:** Dryad doi:10.5061/dryad.3j2j9. **Licence:** CC0-1.0.
- **Download:** 2026-10-24, by browser (Dryad's API refuses scripted downloads), as the dataset zip
  `doi_10_5061_dryad_3j2j9__v20151128.zip`, SHA-256
  `67f12c9470aa74b885557ee6f3280b16d64102aabf6b1ad2e42a574e41a0d0fc`. Its inner `Wormdata.xlsx` is
  5,731,077 bytes, SHA-256 `f7974224c7a2f0a2e31f16a1d4a55ba696ee5c3b9db7837202cef7e58e3b411d`.
- **Assay** (the paper's methods): animals taken off food and corralled in a 1-inch square by filter
  paper soaked in 200 mM CuSO₄, recorded for 30 min at 3 frames/s. The authors ignore the first 2 min.
- **Contents:** 47 worms × 5,400 frames, from the sheets `X position`, `Y position`, `Pirouettes`,
  `Omega turns` and `Non-omega turns`.
  - **Positions** are in pixels. Neither the paper nor the file states a pixel scale, so they are
    vendored in pixels (`units = "px"`) and serve only unit-free statistics: turn timing and rates,
    never speeds or distances. They are sampled at 1 s (every third frame) and stored as hundredths of
    a pixel (`x_cpx`, `y_cpx`, int32).
  - **Missing positions:** 27,902 cells are "NaN". Another 27 cells hold malformed text ("272.6.",
    "Na3", …); they are counted at extraction and treated as missing rather than guessed.
  - **Flags** (`frame_flags`, bit-packed per frame at 3 frames/s): pirouette (bit 0), omega (bit 1)
    and non-omega turn (bit 2). Each is a one-frame onset marker. The deposit's `Non-omega turns` sheet
    is all zeros. `Pirouettes` has 5,406 columns against 5,400 elsewhere: aligned at the start, every
    one of the 709 omega onsets coincides with a pirouette flag, and at any other offset none does, so
    its last six columns are dropped. There are 3,966 pirouette onsets.
  - **No cohort fields:** Calhoun validates a reorientation detector but cannot serve as a
    cohort-held-out control.
- **Command:** `extract_benchmark_references.py calhoun --xlsx Wormdata.xlsx --out data/benchmark/calhoun2014_tracks.npz`.

## `roberts2016_tracks.npz`

- **Citation:** Roberts, W. M., Augustine, S. B., Lawton, K. J., Lindsay, T. H., Thiele, T. R.,
  Izquierdo, E. J., Faumont, S., Lindsay, R. A., Britton, M. C., Pokala, N., Bargmann, C. I. &
  Lockery, S. R. (2016). A stochastic neuronal model predicts random search behaviors at multiple
  spatial scales in *C. elegans*. *eLife* 5:e12572.
- **Deposit:** Dryad doi:10.5061/dryad.35qv6. **Licence:** CC0-1.0.
- **Download:** 2026-10-24, by browser, as `doi_10_5061_dryad_35qv6__v20160203.zip`, SHA-256
  `fc7aaec5ceaafe5489d8611469c0bcc1ba4802e11477bd913e80943b58866cea`. Its inner
  `wormtracks archive.zip` is 222,322,502 bytes, SHA-256
  `11b27aa8a0ccae0dd37274765eff7704e8c5da20fe2062b5877bd8cf2f7947be`.
- **Format:** Igor binary waves (versions 2 and 5), read by a minimal reader in the extraction script.
  What each wave means is taken from the authors' program, `wormtracks.ipf` (296,502 bytes), in the
  same archive:
  - `xx`, `yy`: the tracking spot's position in **mm** (the program computes speed as
    `fps * sqrt(dx² + dy²)`, "converted to mm/sec");
  - **29.97 frames/s** (`fps=29.97`);
  - `worm`: the worm number at each frame;
  - `state`: per frame, **2 forward, 1 reverse, 0 paused**. The README calls these "the initial state
    identifications that were made by eye when the data were collected"; the program calls them the
    tracker's forward/reverse classifications;
  - `IBW files/C<n>_Viterbi.ibw`: the authors' four-state decoding, codes **0 X, 1 R, 2 F, 3 Y**
    (`stateX=0, stateR=1, stateF=2, stateY=3`), where X is a pause after forward movement and Y one
    after reversal. The program drops two points at each end of every worm before decoding, and the
    store pads those with −1.
- **Contents:** the five intact wild-type cohorts in the archive's `Wild type/` folder (C15, C21, C57,
  C162 and C186: 122 worms of about 9 min each) and the four dead-worm tracks in `dead worm/`, a
  tracking-noise floor. Positions are sampled at the frame nearest each whole second and stored in µm
  (`x_um`, `y_um`, int32: C103 reaches 47.5 mm, beyond int16). The tracker and Viterbi states are
  kept per frame (`state`, `viterbi`), −1 outside each worm's recording.
- **Not vendored**, named by the program's figure labels: wild-type ablation controls C103, C111,
  C113, C117 and C175; ablations C104 (AVB), C110 (PVC), C112 (AVE), C116 (AVD) and C176 (AVA); the DEP
  mutant cohorts C14 and C19; and the HYP mutant cohorts C16, C20 and C56.
- **Command:** `extract_benchmark_references.py roberts --archive <extracted archive> --out data/benchmark/roberts2016_tracks.npz`.

## `lopezcruz2019_reorientations.json`

- **Citation:** López-Cruz, A., Sordillo, A., Pokala, N., Liu, Q., McGrath, P. T. & Bargmann, C. I.
  (2019). Parallel multimodal circuits control an innate foraging behavior. *Neuron* 102:407–419.
- **Deposit:** Mendeley Data doi:10.17632/prdtw5jgcd.1. **Licence:** CC BY 4.0.
- **Download:** 2026-10-24, scripted from Mendeley's public API: `CompleteForagingAssayData.xls`,
  155,648 bytes, SHA-256 `1ee8cf962883d2d37456cab7eb48d646921ce8644b26c81e76f7267d7aa596f2`.
- **Contents:** every row of the sheet `data_2min_bins` ("Values in the table: Average # of
  reorientations ± S.E.M"), 184 rows. Each keeps its genotype, figure, recording dates and N, and each
  of 22 2-min bins from 0 to 44 min after removal (the sheet marks 0–2 min "not plotted") as a mean and
  SEM parsed from the "m +/- s" text; "N/A" cells are null. Each mutant row is linked to the nearest
  wild-type row above it with the same dates (`control_row`), where one exists.
- **D37's mutants:** pdfr-1 (ok3425) is present (row 91, N = 56, beside wild type N = 57 on the same
  dates). tax-4 and tph-1 are absent.
- **Command:** `uv run --with xlrd python scripts/analysis/extract_benchmark_references.py lopezcruz --xls CompleteForagingAssayData.xls --out data/benchmark/lopezcruz2019_reorientations.json`
  (`xlrd` reads the legacy `.xls`; the project does not depend on it).
