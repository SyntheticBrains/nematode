## Context

**What the repository holds.** D.1 (Logbook 085) read Scheer & Bargmann 2023's wild-type pickle once,
through a restricted unpickler (`scripts/analysis/roaming_dwelling_calibration.py`, `_SafeUnpickler`).
It vendored 10-second window measures at the 5-s step (`data/roaming_dwelling/scheer2023_windows.npz`,
1,586 animals × 240 windows): speed, angular speed, the in- and out-of-lawn run masks and the authors'
labels. Positions, cohort fields and mutants were not kept.

**What the deposit holds** (Zenodo 8310289, CC0, 27.2 GB; read 2026-10-24 from its README and the
authors' `preProcessing.py`, MIT):

- `GenotypeComparisons_021523.pkl` (20.6 GB) is `(bins, binSize, genotypesData)`. `genotypesData`
  maps keys such as `PD1074_od2_Data`, `MT15434_tph1mg280_od2_Data`, `CX14295_pdfr1ok3425_od2_Data` and
  `CX13078_tax4p678_od2_Data` to dicts of per-animal arrays at 3 frames/s over 40 minutes (7,200
  frames), and of 10-second bins (240).
- `PD1074_od2_Fig1_021523.pkl` (4.9 GB) holds Figure 1's wild type, "otherwise identical".
- Each animal carries:
  - `videoname`, `dates` (parsed from the video name) and `matfilename`;
  - `pixpermm`, `Center_Point_x/y` and `Lawn_Boundary_Pts_x/y`;
  - `Midbody_cent_x/y` in pixels;
  - per-frame `MovingReverse`, `Omega`, `Pause`, `HeadPokesAll`, `MidbodyInLawn` and `missingdata`;
  - per-bin `InLawnRunMask`, `OutLawnRunMask` and `RD_states_Matrix_exog`.

**Other sources** (read 2026-10-24 from their repositories):

| Source | Deposit | Licence | Content |
|---|---|---|---|
| Calhoun, Chalasani & Sharpee 2014 | Dryad 10.5061/dryad.3j2j9, `Wormdata.xlsx` (5.7 MB) | CC0 | positions, omega and non-omega turns, 3 frames/s, 30 min |
| Roberts et al. 2016 | Dryad 10.5061/dryad.35qv6, `wormtracks archive.zip` (222 MB) | CC0 | raw search tracks with the authors' analysis program |
| López-Cruz et al. 2019 | Mendeley 10.17632/prdtw5jgcd.1, `CompleteForagingAssayData.xls` (0.16 MB) | CC BY 4.0 | reorientations per 2-min bin after removal, by genotype |

Dryad's download API answers 401 without a login, so both Dryad files are downloaded by browser.
Zenodo and Mendeley serve scripted downloads.

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

## Decisions

### Decision A: Scheer's genotypes come from one file, checked against the other

All seven arms are read from `GenotypeComparisons_021523.pkl`:

- `PD1074_od2_Data`, `PD1074_od1_Data`, `PD1074_od4_Data` and `PD1074_od2_uniformlawn_Data`;
- `MT15434_tph1mg280_od2_Data`, `CX14295_pdfr1ok3425_od2_Data` and `CX13078_tax4p678_od2_Data`.

The Fig. 1 file is read only to confirm that its wild type is the same animals as `PD1074_od2_Data`:
the same video names in the same order, and equal positions. If they differ, both are recorded and
`PD1074_od2_Data` is used, since the mutants' batch-matched controls are drawn from it.

*Alternative considered:* the other pdfr-1 and tax-4 alleles and rescues in the file. They are left
out: D37 names one allele each, and the rescues are not phenotypes the controllers encode.

### Decision B: The store

`data/benchmark/scheer2023_tracks.npz`, compressed, with these arrays for A animals:

| Array | Shape, type | Content |
|---|---|---|
| `x_um`, `y_um` | (A, 2400) int16 | the midbody position at frames 0, 3, 6, …, in µm from the lawn centre; −32768 where missing |
| `frame_flags` | (A, 7200) uint8 | bits 0–5: reversal, omega, pause, head poke, midbody in lawn, missing |
| `in_lawn_run`, `out_lawn_run` | (A, 240) bool | the authors' per-bin run masks |
| `labels` | (A, 240) int8 | the authors' roaming (1), dwelling (0), or −1 where masked |
| `boundary_um` | (A, 64) int16 | the lawn boundary's radius at 64 equally spaced angles from +x |
| `pixpermm` | (A,) float32 | the animal's own scale |
| `genotype`, `video` | (A,) int16 | indices into the string arrays `genotypes` and `videos` |
| `date` | (A,) int32 | YYYYMMDD |
| `source_row` | (A,) int32 | the animal's row in its source dict |

**Why micrometres in int16.** Tracking resolves about one pixel, some tens of µm, so 1 µm is far below
the noise. int16 holds ±32.767 mm, wider than the camera's field. The extraction refuses any coordinate
outside it rather than clip it.

**Why 1 s for positions and 3 frames/s for flags.** The point worm steps at 1 s (D35), and positions at
1 s give B.2's detector its input. The flags keep the source's own resolution, because a reversal
lasts 2–3 s and the kinematics check measures its duration. Bit-packed flags compress to little.

**The lawn centre** is `Center_Point_x/y`. If it varies within an animal, the per-animal median is
used, and the spread is recorded. The boundary radii are interpolated from `Lawn_Boundary_Pts` in
polar coordinates about that centre.

### Decision C: The cohort split

A cohort is a recording date: everything recorded on one date is one batch (decided). The split is
drawn from counts alone, never from behaviour:

1. Let D be the dates, and for each arm g its dates D_g and its animal count n_g.
2. For k = 0, 1, 2, …, draw with `default_rng(SPLIT_SEED + k)` a held-out flag for each date, with
   probability 1/3.
3. Accept the first draw in which:
   - every arm with at least two dates has at least one date on each side;
   - every such arm has between 20% and 45% of its animals held out;
   - the OD 2 wild type has between 28% and 38% held out.
4. An arm with one date is all on the `bands` side, and is recorded as having no held-out cohort.

`SPLIT_SEED = 2027`. It is distinct from D.1's animal split (2026), so the two are not confused. The
split is written to `data/benchmark/scheer2023_split.json` with each date's side, k and the
constraints. A test re-draws it from the store's counts and requires the same result.

Batch-matched controls follow their date: a held-out mutant date takes its same-date wild types with it.

### Decision D: The extraction's positive control

The 1-s store must reproduce what D.1 vendored. Re-deriving the 5-s windows from `x_um`/`y_um` (every
fifth sample, through `rd.window_measures`) on the OD 2 wild type, in `PD1074_od2_Data`'s order, gives:

- `on_food`, `off_lawn` and `labels` identical to `scheer2023_windows.npz`;
- `speed` and `angular` within the quantisation's bound. The bound is computed from the 0.5 µm rounding
  over a 10-s window, not chosen;
- the calibrated classifier's decoded states agreeing on at least 99% of on-food windows.

If the Fig. 1 file's wild type differs from `PD1074_od2_Data` (Decision A), the check runs on the Fig. 1
order instead.

This runs in the default test tier, on the vendored files alone.

### Decision E: The on-food kinematics check

`scripts/analysis/primitive_kinematics_check.py` reads the held-out OD 2 wild-type cohorts' flags and
positions only. It writes `docs/experiments/logbooks/supporting/105-reference-extraction/kinematics.json`.

| Quantity | From | Compared with |
|---|---|---|
| reversal bout duration | runs of the reversal flag | `reversal_duration_s` (gamma, 2.69 ± 1.75 s) |
| reversal speed | position displacement over reversal frames | `reversal_speed_mm_per_s` (0.14) |
| omega duration | runs of the omega flag | `omega_duration_s` (7 ± 3 s) |
| forward speed, 95th percentile | 1-s displacement outside every bout | `v_max_mm_per_s` (0.2) |
| turning rate, 95th percentile | 1-s heading change outside every bout | `omega_max_deg_per_s` (50) |
| pause fraction | the pause flag | reported only: a pause is a run at zero speed |

Each is computed in three strata of the boundary distance: more than 0.5 mm inside, within 0.5 mm of
it, and outside. Intervals are bootstraps over cohorts (dates), not animals.

**The rule, fixed here.** For the three quantities with a published spread (reversal duration, reversal
speed and omega duration), a disagreement is the inside-lawn stratum's 95% interval lying wholly
outside the span of the sources' reported mean ± SD. A disagreement becomes a dated D36 amendment, and
`primitives.json` takes the on-food value with Scheer as its source, before B.1 closes. For the other
quantities the comparison is reported, not acted on: a maximum's percentile is not a published value
with a spread.

*Why the held-out cohorts (decided).* The check reads movement flags, never a graded statistic, and the
positive control grades real tracks, not the simulator. Sharing the cohorts therefore contaminates
neither, and the bands keep two thirds of the dates.

### Decision F: Calhoun, Roberts and López-Cruz

- **Calhoun 2014.** `Wormdata.xlsx` is read with its units checked against the paper's stated arena and
  frame rate. It is vendored as `data/benchmark/calhoun2014_tracks.npz`: positions at 1 s in µm (int16,
  or int32 if the arena exceeds int16) and per-frame omega and non-omega flags.
- **Roberts 2016.** The archive's track files are read by the format its README PDF states. They are
  vendored as `data/benchmark/roberts2016_tracks.npz`, at 1 s with the turn flags the authors' program
  writes. If the archive holds only raw video-derived files without flags, positions alone are
  vendored, and the absence is recorded.
- **López-Cruz 2019.** The sheet's bins for the wild type, pdfr-1, tax-4 and tph-1, where present, are
  vendored as `data/benchmark/lopezcruz2019_reorientations.json`, with each bin's mean, SEM and n, and
  the sheet's own labels quoted.

Each source's frame rate, units and coordinate origin are stated in `data/benchmark/PROVENANCE.md`,
with the published MD5 or the SHA-256 of the downloaded file. Only derived files are vendored, never the
originals.

### Decision G: The loader

`quantumnematode/benchmark/references.py` gives:

- `load_scheer_tracks()`, returning a `ScheerTracks` with positions in mm (NaN where missing), the
  flags unpacked on request, the masks, labels, boundary and cohort fields;
- `scheer_split()`, returning each date's side;
- `load_search_tracks(source)`, for `"calhoun2014"` or `"roberts2016"`;
- `load_lopezcruz()`.

It reads the vendored files only, and depends on numpy alone.

## Risks / Trade-offs

- **The genotype file is 20.6 GB.** Unpickling it needs about as much memory again. The machine has 137
  GB. The extraction reads one arm at a time, then frees it.
- **Whether the store fits the size estimate depends on the arms' animal counts.** If it exceeds 40 MB
  compressed, the flags drop to 1 s (any frame in the second set), and the kinematics check reads
  durations at 1-s resolution. That is recorded if it happens.
- **A date can hold few animals of a mutant.** The split's constraints keep each arm's held-out share
  between 20% and 45%. An arm the rule cannot split is recorded, not forced.
- **The Calhoun and Roberts formats are unread until download.** Decision F says what is vendored if a
  file lacks flags. A file that cannot be read is recorded, and B.2's validation then uses the other.
