# Tasks: B.1 — the reference re-extraction

Phase 10, block B, tracker item B.1. The maintainer's decisions of 2026-10-24:

- vendor 1-s tracks;
- the D37 set plus the density arms;
- recording dates as cohorts, with one third held out;
- the kinematics check on the held-out cohorts.

## 1. Downloads

- [ ] 1.1 Scheer's two pickles from Zenodo 8310289, into gitignored `build/scheer2023/`, with the
  published MD5s checked.
- [ ] 1.2 Calhoun's `Wormdata.xlsx` and Roberts's `wormtracks archive.zip`, by the maintainer's
  browser, into `build/dryad/`, with their SHA-256s recorded.
- [ ] 1.3 López-Cruz's `CompleteForagingAssayData.xls` from Mendeley.

## 2. Scheer

- [ ] 2.1 `scripts/analysis/extract_benchmark_references.py scheer`: the restricted unpickler
  (shared with D.1's), one arm at a time, the store of design Decision B, and the range refusal.
- [ ] 2.2 The Fig. 1 wild-type comparison (Decision A), recorded.
- [ ] 2.3 The split (Decision C), drawn and written before any statistic is read.
- [ ] 2.4 Tests: the unpickler's refusal, the range refusal on a synthetic animal, the split re-draw,
  and the windows' re-derivation (Decision D).

## 3. The other sources

- [ ] 3.1 `extract_benchmark_references.py calhoun`, `roberts` and `lopezcruz` (Decision F), with
  their units checked and stated.
- [ ] 3.2 Tests on the vendored files: shapes, units within the paper's arena, flag counts against the
  counts recorded at extraction.

## 4. The loader

- [ ] 4.1 `quantumnematode/benchmark/references.py` (Decision G), with tests.

## 5. The kinematics check

- [ ] 5.1 `scripts/analysis/primitive_kinematics_check.py` (Decision E), reading held-out cohorts only.
- [ ] 5.2 Run it and commit `kinematics.json`. Apply any amendment to `primitives.json`, dated in the
  roadmap's D36 row and the tracker.

## 6. Documentation and tracker

- [ ] 6.1 `data/benchmark/PROVENANCE.md`, Logbook 105, AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [ ] 6.2 Close B.1 in the tracker, with the split's sizes and the check's outcome.

## 7. Close-out

- [ ] 7.1 Full suite and `pre-commit run -a`.
- [ ] 7.2 `openspec validate add-reference-extraction --strict`.
- [ ] 7.3 Archive.
- [ ] 7.4 PR.
