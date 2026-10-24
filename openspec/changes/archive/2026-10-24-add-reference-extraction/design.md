## Context

**What the repository holds.** D.1 ([Logbook 087](../../../docs/experiments/logbooks/087-patchy-lawns.md))
read Scheer & Bargmann 2023's wild-type pickle once, through a restricted unpickler
(`scripts/analysis/roaming_dwelling_calibration.py`, `_SafeUnpickler`). It vendored 10-second window
measures at the 5-s step (`data/roaming_dwelling/scheer2023_windows.npz`, 1,586 animals × 240
windows): speed, angular speed, the in- and out-of-lawn run masks and the authors' labels. Positions,
cohort fields and mutants were not kept.

**What the deposit holds** (Zenodo 8310289, CC0, 27.2 GB; read 2026-10-24 from its README and the
authors' `preProcessing.py`, MIT):

- `GenotypeComparisons_021523.pkl` (20.6 GB) is `(bins, binSize, genotypesData)`. `genotypesData`
  maps keys such as `PD1074_od2_Data`, `MT15434_tph1mg280_od2_Data`, `CX14295_pdfr1ok3425_od2_Data` and
  `CX13078_tax4p678_od2_Data` to dicts of per-animal arrays at 3 frames/s over 40 minutes (7,200
  frames), and of 10-second bins (240).
- `PD1074_od2_Fig1_021523.pkl` (4.9 GB) holds Figure 1's wild type, "otherwise identical".
- Each animal carries:
  - `videoname`, `dates` (parsed from the video name) and `matfilename`;
  - `pixpermm`;
  - `Center_Point_x/y` and `Lawn_Boundary_Pts_x/y`, but only where the recording has a lawn
    boundary. The authors' code skips them otherwise, with defaults it calls "OK for UL data", so the
    uniform-lawn arm is expected to lack them;
  - `Midbody_cent_x/y` in pixels;
  - per-frame `MovingReverse`, `Omega`, `Pause`, `HeadPokesAll`, `MidbodyInLawn` and `missingdata`;
  - per-bin `InLawnRunMask`, `OutLawnRunMask` and `RD_states_Matrix_exog`.

**Other sources**, each opened on 2026-10-24:

| Source | Deposit | Licence | Content |
|---|---|---|---|
| Calhoun, Chalasani & Sharpee 2014 | Dryad 10.5061/dryad.3j2j9: `Wormdata.xlsx`, 5,731,077 bytes | CC0 | 47 worms × 5,400 frames (30 min at 3 frames/s): sheets `X position`, `Y position`, `Pirouettes`, `Omega turns` and `Non-omega turns`, the flags per frame as 0/1. Positions are in what look like pixels (values 150–400); the file states no scale. No cohort fields |
| Roberts et al. 2016 | Dryad 10.5061/dryad.35qv6: `wormtracks archive.zip`, 222,322,502 bytes | CC0 | Igor binary waves. Each cohort folder holds `xx.ibw` and `yy.ibw` (the tracking spot), `worm.ibw` (the worm number at each time point) and `state.ibw` ("the initial state identifications that were made by eye when the data were collected"). `IBW files/` holds Viterbi states for the five wild-type cohorts. `wormtracks.ipf` is the authors' Igor program, as text |
| López-Cruz et al. 2019 | Mendeley 10.17632/prdtw5jgcd.1: `CompleteForagingAssayData.xls`, sheet `data_2min_bins` | CC BY 4.0 | about 190 rows of average reorientations ± SEM in 22 2-min bins (0–44 min, the first "not plotted"), with figure, recording dates and N. Each mutant row follows a wild-type row of the same dates. Of D37's mutants only pdfr-1 is present (row 91, N = 56, beside wild type N = 57) |

**Roberts's cohorts.** The README names wild type (C15, C21, C57, C162, C186), DEP (C14, C19) and HYP
(C16, C20, C56), and ten more in the root folder without a genotype (C103–C176). A `dead worm/` folder
tracks dead animals.

**Downloads.** Dryad's download API answers 401 without a login, so the maintainer downloaded both
Dryad datasets by browser, as whole-dataset zips (`doi_10_5061_dryad_3j2j9__v20151128.zip` and
`doi_10_5061_dryad_35qv6__v20160203.zip`). Zenodo and Mendeley serve scripted downloads.

The maintainer's decisions of 2026-10-24:

- vendor 1-s tracks;
- the D37 set plus the density arms;
- recording dates as cohorts, with one third held out;
- the kinematics check on the held-out cohorts.

## Goals / Non-Goals

**Goals**

- Real tracks that the project's own classifier and B.2's detector can re-grade anywhere, including
  CI, without the 27 GB deposit.
- Cohort fields on every animal, and a split fixed before any behavioural statistic is read.
- Every source vendored with its licence, unit convention, checksums and the code that derived it.
- E.2's owed check of the primitives' kinematics on food.

**Non-goals**

- Readouts, bands and grading rules: B.2 and B.3.
- Track import from arbitrary formats: B.5.
- The second-tier genotypes (npr-1, cat-2), the optogenetics file and the AR-HMMs.
- Roberts's DEP, HYP and unlabelled cohorts, unless `wormtracks.ipf` or the spreadsheet names their
  genotype.

## Decisions

### Decision A: Scheer's genotypes come from one file, checked against the other

All seven arms are read from `GenotypeComparisons_021523.pkl`:

- `PD1074_od2_Data`, `PD1074_od1_Data`, `PD1074_od4_Data` and `PD1074_od2_uniformlawn_Data`;
- `MT15434_tph1mg280_od2_Data`, `CX14295_pdfr1ok3425_od2_Data` and `CX13078_tax4p678_od2_Data`.

*(Amended 2026-10-24, the maintainer's decision, after extraction showed it:)* **`N2_od2_Data` is an
eighth arm.** tph-1 is an N2-background strain recorded only beside the N2 wild type: all five of its
dates have N2 animals and none has PD1074. Without N2 it would have no batch-matched control, which D34
requires. N2 is kept as its own arm, never pooled with PD1074, and the split constrains it like the
other arms.

The Fig. 1 file is read only to compare its wild type with `PD1074_od2_Data`. Animals are matched by
video name, source file and row. The result is recorded either way: the matched count, and whether the
positions are equal. `PD1074_od2_Data` is what is stored, since the mutants' batch-matched controls are
drawn from it.

*Alternative considered:* the other pdfr-1 and tax-4 alleles and the rescues in the file. They are left
out: D37 names one allele each, and the rescues are not phenotypes the controllers encode.

The restricted unpickler moves from the D.1 script into a shared helper,
`scripts/analysis/_scheer_pickle.py`. Both scripts import it.

### Decision B: The store

`data/benchmark/scheer2023_tracks.npz`, compressed, with these arrays for A animals:

| Array | Shape, type | Content |
|---|---|---|
| `x_um`, `y_um` | (A, 2400) int16 | the midbody position at frames 0, 3, 6, …, in µm from the origin below; −32768 where missing |
| `frame_flags` | (A, 7200) uint8 | bits 0–5: reversal, omega, pause, head poke, midbody in lawn, missing |
| `in_lawn_run`, `out_lawn_run` | (A, 240) bool | the authors' per-bin run masks |
| `labels` | (A, 240) int8 | the authors' roaming (1), dwelling (0), or −1 where masked |
| `boundary_um` | (A, 64) int16 | the lawn boundary's radius at 64 equally spaced angles from +x; −32768 where the arm has no boundary |
| `pixpermm` | (A,) float32 | the animal's own scale |
| `genotype`, `video` | (A,) int16 | indices into the string arrays `genotypes` and `videos` |
| `date` | (A,) int32 | YYYYMMDD |
| `source_row` | (A,) int32 | the animal's row in its source dict |
| `origin` | (A,) int8 | 0: the lawn centre; 1: the animal's first valid position |
| `fig1_row` | (A,) int32 | the animal's row in the Fig. 1 file, −1 if it has none (Decision A's match, kept so Decision D's check needs only vendored files) |

**Why micrometres in int16.** Tracking resolves about one pixel, some tens of µm, so 1 µm is far below
the noise. int16 holds ±32.767 mm. If any position exceeds that, the store's positions are int32
instead, and the loader reads either. *(Amended in implementation: one type for the whole store rather
than per-arm wide arrays, which the loader would have had to merge.)*

**Why 1 s for positions and 3 frames/s for flags.** The point worm steps at 1 s (D35), and positions at
1 s give B.2's detector its input. The flags keep the source's own resolution, because a reversal lasts
2–3 s and the kinematics check measures its duration. Bit-packed flags compress to little.

**The origin.** Where the arm has a lawn boundary, the origin is `Center_Point_x/y`. If the centre
varies within an animal, the per-animal median is used, and the spread is recorded. The boundary radii
are interpolated from `Lawn_Boundary_Pts` in polar coordinates about that centre. Where the arm has no
boundary (expected: the uniform lawn), the origin is the animal's first valid position, `origin` is 1,
the radii are the sentinel, and no boundary-distance stratum applies.

### Decision C: The cohort split

A cohort is a recording date: everything recorded on one date is one batch (decided). The split is
drawn from counts alone, never from behaviour:

1. Let D be the dates, and for each arm g its dates D_g and its animal count n_g.
2. **Feasibility first.** For each arm with at least two dates, enumerate the subsets of its dates (or,
   for more than 20 dates, sample 100,000). An arm with no subset holding 20–45% of its animals is
   recorded as *share-infeasible*. For it, the share constraint is replaced by "at least one date on
   each side".
3. For k = 0, 1, …, 9,999, draw with `default_rng(SPLIT_SEED + k)` a held-out flag for each date, with
   probability 1/3.
4. Accept the first draw in which:
   - every arm with at least two dates has at least one date on each side;
   - every arm that is not share-infeasible has 20–45% of its animals held out;
   - the OD 2 wild type has 28–38% held out.
5. An arm with one date is all on the `bands` side, and is recorded as having no held-out cohort.
6. If no draw passes in 10,000, the extraction stops and reports the closest draw. The rule is not
   loosened ad hoc; a new rule would be a registered amendment.

`SPLIT_SEED = 2027`. It is distinct from D.1's animal split (2026), so the two are not confused. The
split is written to `data/benchmark/scheer2023_split.json`: each date's side, k, the constraints, and
any arm that is share-infeasible or single-date. A test re-draws it from the store's counts and requires
the same result.

Batch-matched controls follow their date: a held-out mutant date takes its same-date wild types with it.

**What "held out" means here.** The held-out cohorts are held out of the bands, not unseen by the
instrument. D.1 calibrated the project's classifier on an animal-level split (seed 2026) drawn across
every date, and `scheer2023_leaving.json` used every wild-type animal. The split record says so. B.3
decides whether to re-calibrate on band cohorts only, and B.4 reports whichever applies.

### Decision D: The extraction's positive control

The 1-s store must reproduce what D.1 vendored. The 5-s windows are re-derived from `x_um`/`y_um`
(every fifth sample, through `rd.window_measures`) for the OD 2 wild-type animals matched to the
Fig. 1 file (Decision A), in the Fig. 1 order. At least 95% of the Fig. 1 animals must match. If fewer
do, the check fails and the extraction stops. The re-derived windows must give:

- `on_food`, `off_lawn` and `labels` identical to `scheer2023_windows.npz` on the matched animals;
- `speed` within the bound implied by the 0.5 µm rounding: each step's displacement is off by at most
  √2 µm, so a window's mean speed by at most √2 µm / 5 s;
- the calibrated model's per-window evidence, the state whose Gaussian emission is likelier, identical
  on at least 99.5% of on-food windows. *(Amended in implementation: the instrument is the
  Gaussian-emission model, `calibration_retry.json`. The slope line's observation belongs to the model
  that failed its gate.)*
- the calibrated classifier's decoded states agreeing on at least 99% of on-food windows.

Angular speed has no closed bound: a turn's rounding error grows without limit as a step's
displacement goes to zero. It is therefore checked through the two classifier conditions, which are
what it feeds.

The check runs on the vendored files alone, in the default tier if it takes under 10 s, and in the
slow tier otherwise.

### Decision E: The on-food kinematics check

`scripts/analysis/primitive_kinematics_check.py` reads the held-out OD 2 wild-type cohorts' flags and
positions only. It writes `docs/experiments/logbooks/supporting/105-reference-extraction/kinematics.json`.

**Bouts.** A reversal bout is a run of the reversal flag after gaps of one frame are merged. Runs
shorter than two frames (0.67 s) are dropped. An omega bout is built the same way from the omega flag.
Both the filtered and the unfiltered values are reported; the rule reads the filtered one. A bout is kept whether or not it touches a frame the tracker lost (`missing`). *(Noted in
implementation: the rule as written does not exclude them. The first run of the script did, and that
reading is reported beside, as description. It removed every omega, since the omega flag always
coincides with lost frames while the worm coils.)*

| Quantity | From | Compared with |
|---|---|---|
| reversal bout duration | reversal bouts | Zhao 2003: 2.3 ± 1.6 s (n = 249) and 3.4 ± 1.8 s (n = 135) |
| reversal speed | position displacement over reversal bouts, at 1 s | Zhao 2003: 0.14 mm/s |
| omega duration | omega bouts | Broekmans 2016: 7 ± 3 s, escape omegas (foraging omega duration is not reported) |
| forward speed, 95th percentile | 1-s displacement outside every bout | `v_max_mm_per_s` (0.2) |
| turning rate, 95th percentile | 1-s heading change outside every bout | `omega_max_deg_per_s` (50) |
| pause fraction | the pause flag | reported only: a pause is a run at zero speed |

Each is computed in three strata of the boundary distance: more than 0.5 mm inside, within 0.5 mm of
it, and outside. Intervals are bootstraps over cohorts (dates), not animals.

**The rule, fixed here.** A quantity is *acted on* if its source states a mean with an SD and an n. Its
reference span runs from the lowest source mean − 2 SEM to the highest + 2 SEM, with SEM = SD / √n:

- reversal duration: 2.3 − 2 × 1.6/√249 to 3.4 + 2 × 1.8/√135, about 2.10–3.71 s;
- omega duration: Broekmans's n is read from the figure supplement at implementation. If it cannot be
  read, the quantity is reported only. Its source condition (escape) differs from foraging, and that
  is stated beside it;
- reversal speed: acted on only if Zhao's Fig. 5A gives an SD and n at implementation; otherwise
  reported only.

A **disagreement** is the inside-lawn stratum's 95% cohort interval lying wholly outside the reference
span.

**What a disagreement does.** It is recorded in `kinematics.json` and Logbook 105, and it **comes back
to the maintainer before B.3 registers**. `primitives.json` is not edited by this change. The current
values are off-food, and food removal, Calhoun and Roberts are all off-food assays: replacing them with
on-food values would mis-set those. The amendment's form is the maintainer's to choose. Two forms are
foreseen:

- food-dependent kinematics, keyed on the existing food-contact sense;
- the off-food values kept, with the on-food gap stated as a limit of the on-lawn grading.

Either form is a dated D36 amendment. The other quantities are reported, not acted on.

*Why the held-out cohorts (decided).* The check reads movement flags, never a graded statistic, and the
positive control grades real tracks, not the simulator. Sharing the cohorts therefore contaminates
neither, and the bands keep two thirds of the dates.

### Decision F: Calhoun, Roberts and López-Cruz

- **Calhoun 2014.** The five sheets are read as 47 worms × 5,400 frames. The pixel scale is taken from
  the paper's methods (arena size or camera resolution). If the paper does not state it, positions are
  vendored in pixels, with `units: "px"` in the file. Their use is then restricted to unit-free
  statistics: turn timing and rates, never speeds or distances. Vendored as
  `data/benchmark/calhoun2014_tracks.npz`:

  - positions at 1 s (every third frame), as hundredths of a pixel in int32;
  - the per-frame pirouette, omega and non-omega flags, bit-packed.

  The file has no cohort fields, so Calhoun can validate B.2's detector but cannot be a
  cohort-held-out control.

- **Roberts 2016.** A minimal reader for Igor binary waves (versions 2 and 5, numeric waves only) lives
  in the extraction script; there is no new dependency. The scale (pixels or mm per unit), the frame
  rate and the codes of `state` are taken from `wormtracks.ipf`, and quoted in PROVENANCE.md. Vendored
  as `data/benchmark/roberts2016_tracks.npz`:

  - the five wild-type cohorts' positions at 1 s, in µm;
  - each frame's by-eye `state` and the Viterbi states from `IBW files/`, kept per frame at 29.97
    frames/s. *(Amended in implementation: a reversal lasts a few seconds, so a 1-s state would blur the
    starts B.2's detector is timed against. Per-frame states compress to little.)*
  - the cohort IDs;
  - the dead-worm tracks, separately, as a tracking-noise floor for B.2's false-positive rate.

  If `wormtracks.ipf` names the genotypes of the DEP, HYP or root cohorts, those are listed in
  PROVENANCE.md, but are not vendored by this change.

- **López-Cruz 2019.** The whole `data_2min_bins` sheet is vendored as
  `data/benchmark/lopezcruz2019_reorientations.json`. Each row keeps its genotype, figure, dates, N, and
  each bin's mean and SEM, parsed from the "m ± s" strings, with the sheet's own labels quoted. Each
  mutant row is linked to its same-dates wild-type row. The sheet has pdfr-1, but no tax-4 or tph-1;
  that absence is recorded. B.2 reads the wild-type rows for γ.

Each source's frame rate, units and coordinate origin are stated in `data/benchmark/PROVENANCE.md`,
with the published MD5 or the SHA-256 of the downloaded file. For the Dryad zips that is the zip's
SHA-256 and each inner file's size. Only derived files are vendored, never the originals.

### Decision G: The loader

`quantumnematode/benchmark/references.py` gives:

- `load_scheer_tracks()`, returning a `ScheerTracks` with:
  - positions in mm, NaN where missing;
  - the flags, unpacked on request;
  - the masks, labels, boundary, origin and cohort fields;
- `scheer_split()`, returning each date's side;
- `load_search_tracks(source)`, for `"calhoun2014"` or `"roberts2016"`, carrying each set's units;
- `load_lopezcruz()`.

It reads the vendored files only, and depends on numpy alone.

## Risks / Trade-offs

- **The genotype file is 20.6 GB.** Unpickling it needs about as much memory again. The machine has 137
  GB. The extraction reads the file once, then frees each arm as it is stored.
- **Whether the store fits the size estimate depends on the arms' animal counts.** If it exceeds 40 MB
  compressed, the flags drop to 1 s (any frame in the second set), and the kinematics check reads
  durations at 1-s resolution. That is recorded if it happens.
- **A date can hold few animals of a mutant.** The feasibility step and the fallback in Decision C
  handle that, and record it; nothing is forced.
- **Calhoun's scale may be unstated.** Decision F then restricts Calhoun to unit-free statistics.
- **Roberts's `state` is by eye, at collection.** B.2 treats it as a second annotator beside Calhoun's
  flags, not as ground truth.
