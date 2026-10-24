## Why

Phase 10's benchmark (block B, D34) grades simulated worms against real ones, and is valid only if
real tracks pass it. Its positive control holds out **whole cohorts**, and its mutant control holds
out real mutant cohorts (D37). Neither is possible with what the repository holds today:

- `data/roaming_dwelling/scheer2023_windows.npz` keeps Scheer & Bargmann 2023's wild type as 10-second
  window measures at the 5-s step. It has no positions, no cohort IDs and no mutants.
- No search tracks are vendored. Calhoun 2014 and Roberts 2016 are the deposited off-food search
  tracks, with omega flags, that B.2's reorientation detector is validated against.
- López-Cruz 2019's reorientation counts after food removal were read in E.3, but not vendored.

E.2's primitives also carry a debt: their kinematics are published off-food values, and E.2's spec
owes a check against Scheer's on-food movement flags on held-out cohorts, before any fit.

This is tracker item B.1: the reference re-extraction, each source with its licence, unit convention
and provenance.

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **Scheer & Bargmann 2023 re-extracted** from the Zenodo mirror (record 8310289, CC0) through the
  existing restricted unpickler:

  - **genotypes (decided):** wild type (PD1074, OD 2) with tph-1, pdfr-1 and tax-4 (D37's mutants),
    plus wild type at OD 1, at OD 4 and on a uniform lawn, for the food-density target. npr-1 and
    cat-2 stay out (second tier);
  - **a 1-s track store, vendored (decided):** per animal, the midbody position in micrometres
    relative to the lawn centre, as int16; the authors' per-frame flags at 3 frames/s (reversal,
    omega, pause, head poke, midbody in lawn, in- and out-of-lawn runs, missing), bit-packed; their
    10-second roaming/dwelling labels; the lawn boundary as 64 radii; and the animal's genotype,
    recording date, video and source index;
  - about 15–30 MB compressed, in `data/benchmark/`.

- **Cohorts are recording dates, one third held out (decided).** Everything recorded on one date is
  one batch. A seeded, genotype-stratified split of dates is fixed in this change, from counts alone,
  before any behavioural statistic is read. It assigns every date to `bands` or `held_out`, and B.3
  and B.4 cite it.

- **An extraction positive control.** The 5-s windows re-derived from the 1-s store reproduce the
  vendored `scheer2023_windows.npz`: masks and labels identical, measures within the store's
  quantisation, and the calibrated classifier's decoded states agreeing on at least 99% of windows.

- **The on-food kinematics check (decided: on the held-out cohorts).** It measures reversal duration
  and speed, omega duration, forward speed and turning rate, from Scheer's flags on held-out wild-type
  cohorts, split by distance to the lawn boundary. It compares each with `primitives.json`. A value
  outside the sources' spread is recorded as a dated D36 amendment, and `primitives.json` is updated,
  before B closes. The check reads movement flags only, never a graded statistic, and the positive
  control grades real tracks, not the simulator, so sharing the held-out cohorts contaminates neither.

- **Calhoun 2014** (Dryad 10.5061/dryad.3j2j9, CC0) and **Roberts 2016** (Dryad 10.5061/dryad.35qv6,
  CC0): positions and turn flags, converted to millimetres at 1 s and vendored with their provenance.
  Dryad refuses scripted downloads, so the maintainer downloads both by browser; their checksums are
  recorded.

- **López-Cruz 2019** (Mendeley 10.17632/prdtw5jgcd.1, CC BY 4.0): reorientations per 2-min bin after
  food removal for the wild type, pdfr-1, tax-4 and tph-1 where present, with each bin's n, vendored
  as JSON. B.2 sets the readout, time zero and fit start.

- **A loader**, `quantumnematode/benchmark/references.py`: the stores read back as per-animal tracks
  with their cohort fields and split, in millimetres and seconds.

## Capabilities

### New Capabilities

- `benchmark-references`: the vendored real-worm reference sets, their provenance, the cohort split,
  the extraction's positive control and the on-food kinematics check.

## Impact

- **Code:**
  - new `quantumnematode/benchmark/` (`__init__.py`, `references.py`);
  - `scripts/analysis/extract_benchmark_references.py`, one subcommand per source;
  - `scripts/analysis/primitive_kinematics_check.py`.
- **Data:** new `data/benchmark/` with each store and a `PROVENANCE.md`.
- **Default behaviour:** none. No simulation path changes, unless the kinematics check amends
  `primitives.json`, which changes only the `*_primitives` configs, all introduced in Phase 10 and none
  fitted yet.
- **Cost:** about 3–4 engineering days. The 25 GB download takes a few hours, and extraction under an
  hour.
- **Seeds:** none simulated. The split's seed is the record's own (`SPLIT_SEED = 2027`), not a band.
- **Downstream:**
  - B.2 validates its detector on Calhoun's and Roberts's flags;
  - B.3 bands on the `bands` cohorts;
  - B.4 holds out the `held_out` cohorts;
  - B.5 imports the same tracks through its own path.
