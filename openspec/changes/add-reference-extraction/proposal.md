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
  existing restricted unpickler, moved into a shared helper:

  - **genotypes (decided):** wild type (PD1074, OD 2) with tph-1, pdfr-1 and tax-4 (D37's mutants),
    plus wild type at OD 1, at OD 4 and on a uniform lawn, for the food-density target. npr-1 and
    cat-2 stay out (second tier);
  - **a 1-s track store, vendored (decided):** per animal, the midbody position in micrometres as
    int16, relative to the lawn centre (or, on the uniform lawn, which has no boundary, to the
    animal's first position); the authors' per-frame flags at 3 frames/s (reversal, omega, pause, head
    poke, midbody in lawn, missing), bit-packed; their per-bin run masks and 10-second roaming/dwelling
    labels; the lawn boundary as 64 radii; and the animal's genotype, recording date, video and source
    row;
  - about 15–30 MB compressed, in `data/benchmark/`.

- **Cohorts are recording dates, one third held out (decided).** Everything recorded on one date is
  one batch. A seeded, genotype-stratified split of dates is fixed in this change, from counts alone,
  before any behavioural statistic is read. Each arm's constraint is checked for feasibility first,
  and the draw is bounded. It assigns every date to `bands` or `held_out`, and B.3 and B.4 cite it.
  The held-out cohorts are held out of the bands, not unseen by the instrument: D.1 calibrated the
  classifier on animals from every date, and the record says so.

- **An extraction positive control.** The 5-s windows re-derived from the 1-s store reproduce the
  vendored `scheer2023_windows.npz` on the animals matched to it: masks and labels identical, speed
  within the rounding's bound, the classifier's observation identical on 99.5% of windows and its
  decoded states on 99%.

- **The on-food kinematics check (decided: on the held-out cohorts).** It measures reversal duration
  and speed, omega duration, forward speed and turning rate from Scheer's flags on held-out wild-type
  cohorts, split by distance to the lawn boundary. The bout rule and the comparison are fixed here:
  the on-food mean's interval against the span of the sources' means ± 2 SEM (about 2.10–3.71 s for
  reversal duration). **A disagreement is brought to the maintainer before B.3 registers; this change
  does not edit `primitives.json`.** The primitives are off-food values, and the food-removal and
  search assays are off food, so an on-food value cannot simply replace them. The check reads
  movement flags only, never a graded statistic, and the positive control grades real tracks, not the
  simulator, so sharing the held-out cohorts contaminates neither.

- **Calhoun 2014** (Dryad 10.5061/dryad.3j2j9, CC0): 47 worms at 3 frames/s for 30 min, with per-frame
  pirouette, omega and non-omega flags. Positions are in pixels with no scale in the file: µm if the
  paper states the scale, otherwise pixels, marked, for unit-free statistics only. It has no cohorts.

- **Roberts 2016** (Dryad 10.5061/dryad.35qv6, CC0): Igor binary waves, read by a minimal reader with
  no new dependency. The five wild-type cohorts are vendored with positions, worm numbers, by-eye
  states, Viterbi states and cohort IDs, and the dead-worm tracks beside them as a noise floor. The
  scale and state codes come from the authors' `wormtracks.ipf`.

- Dryad refuses scripted downloads, so the maintainer downloaded both by browser, as dataset zips; their
  checksums are recorded.

- **López-Cruz 2019** (Mendeley 10.17632/prdtw5jgcd.1, CC BY 4.0): the whole 2-min-bin sheet, about 190
  rows of mean ± SEM per bin with N, each mutant linked to its same-dates wild type. Of D37's mutants
  only pdfr-1 is present. B.2 sets the readout, time zero and fit start.

- **A loader**, `quantumnematode/benchmark/references.py`: the stores read back as per-animal tracks
  with their cohort fields and split, in their stated units and seconds.

## Capabilities

### New Capabilities

- `benchmark-references`: the vendored real-worm reference sets, their provenance, the cohort split,
  the extraction's positive control and the on-food kinematics check.

## Impact

- **Code:**
  - new `quantumnematode/benchmark/` (`__init__.py`, `references.py`);
  - `scripts/analysis/_scheer_pickle.py`, the restricted unpickler shared with D.1's script;
  - `scripts/analysis/extract_benchmark_references.py`, one subcommand per source;
  - `scripts/analysis/primitive_kinematics_check.py`.
- **Data:** new `data/benchmark/` with each store and a `PROVENANCE.md`.
- **Default behaviour:** none. No simulation path changes; a kinematics disagreement is brought back to
  the maintainer, not applied here.
- **Cost:** about 3–4 engineering days. The 25 GB download takes a few hours, and extraction under an
  hour.
- **Seeds:** none simulated. The split's seed is the record's own (`SPLIT_SEED = 2027`), not a band.
- **Downstream:**
  - B.2 validates its detector on Calhoun's flags and Roberts's by-eye states, with the dead worms as
    its noise floor;
  - B.3 bands on the `bands` cohorts;
  - B.4 holds out the `held_out` cohorts;
  - B.5 imports the same tracks through its own path.
