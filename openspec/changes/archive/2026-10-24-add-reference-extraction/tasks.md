# Tasks: B.1 — the reference re-extraction

Phase 10, block B, tracker item B.1. The maintainer's decisions of 2026-10-24:

- vendor 1-s tracks;
- the D37 set plus the density arms;
- recording dates as cohorts, with one third held out;
- the kinematics check on the held-out cohorts.

## 1. Downloads

- [x] 1.1 Scheer's two pickles from Zenodo 8310289, into gitignored `build/scheer2023/`, with the
  published MD5s checked.
- [x] 1.2 Calhoun's and Roberts's Dryad datasets, by the maintainer's browser, into `build/dryad/`
  (2026-10-24), with the zips' SHA-256s and inner sizes recorded.
- [x] 1.3 López-Cruz's `CompleteForagingAssayData.xls` from Mendeley, into `build/mendeley/`.

## 2. Scheer

- [x] 2.1 `scripts/analysis/_scheer_pickle.py`: the restricted unpickler moved out of D.1's script,
  which imports it.
- [x] 2.2 `scripts/analysis/extract_benchmark_references.py scheer`: the store of design Decision B,
  with the boundary-less origin, the int32 fallback and the non-finite refusal.
- [x] 2.3 The Fig. 1 wild-type match (Decision A), recorded, with the 95% floor.
- [x] 2.4 The split (Decision C): feasibility first, at most 10,000 draws, drawn and written before any
  statistic is read, with the instrument's calibration stated.
- [x] 2.5 Tests: the unpickler's refusal, the boundary-less arm on a synthetic animal, the split
  re-draw, a share-infeasible arm and a single-date arm on synthetic counts, and the windows'
  re-derivation (Decision D), timed for its tier.

## 3. The other sources

- [x] 3.1 `extract_benchmark_references.py calhoun`: the scale from the paper's methods, or pixels,
  marked.
- [x] 3.2 `extract_benchmark_references.py roberts`: the minimal Igor-binary-wave reader, with the scale,
  frame rate and state codes from `wormtracks.ipf`; the wild-type cohorts, their Viterbi states and the
  dead worms.
- [x] 3.3 `extract_benchmark_references.py lopezcruz`: the whole sheet, parsed, with the mutant–control
  links and the absent D37 mutants recorded.
- [x] 3.4 Tests on the vendored files: shapes, units, flag and state counts against the counts recorded
  at extraction, and the IBW reader on a synthetic wave.

## 4. The loader

- [x] 4.1 `quantumnematode/benchmark/references.py` (Decision G), with tests.

## 5. The kinematics check

- [x] 5.1 `scripts/analysis/primitive_kinematics_check.py` (Decision E), reading held-out cohorts only,
  with the bout rule, the reference spans and the acted-on set fixed in code before it runs.
- [x] 5.2 Run it and commit `kinematics.json`. Any disagreement is reported to the maintainer before B.3
  registers; `primitives.json` is not edited here.

## 6. Documentation and tracker

- [x] 6.1 `data/benchmark/PROVENANCE.md`, Logbook 105, AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [x] 6.2 Close B.1 in the tracker, with the split's sizes, the check's outcome and any disagreement
  owed to the maintainer.

## 7. Close-out

- [x] 7.1 Full suite and `pre-commit run -a`.
- [x] 7.2 `openspec validate add-reference-extraction --strict`.
- [x] 7.3 Archive.
- [x] 7.4 PR.
