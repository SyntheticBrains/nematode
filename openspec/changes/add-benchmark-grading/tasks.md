# Tasks: B.3 — bands and the grading rule

Phase 10, block B, tracker item B.3. The maintainer's decisions of 2026-10-24:

- D36's form: food-dependent kinematics, in its own change (`amend-food-dependent-kinematics`);
- recalibrate the classifier on the band cohorts only;
- grade the core lawn, search and mutants;
- between-date bands, with a z distance.

Gates and seeds fixed here are registered by this change's commit, before any band is read.

## 1. The classifier

- [ ] 1.1 `scripts/analysis/benchmark_classifier.py`:
  - the OD 2 band dates split two thirds and one third by date (seed 2029);
  - windows from the store under the geometric rule;
  - `GaussianHMM.fit_labelled` on the two thirds, over runs of windows both on the lawn and labelled;
  - κ on the third, against the 0.6 gate;
  - a nonzero exit with no model written on failure.
- [ ] 1.2 Run it. Commit `data/benchmark/classifier.json` with its split, gate reading and provenance,
  or stop at the gate.

## 2. The grading module

- [ ] 2.1 `quantumnematode/benchmark/grading.py`:
  - windows from 1-s positions and on-lawn samples, every 1-s sample in a window's span;
  - the uniform well's rule: every sample on the lawn;
  - the phase-aware reader for `BehaviourStep` series;
  - censoring-aware bout means;
  - the 11 wild-type statistics and the mutant Δs;
  - z with its cap, the pass rule, D and D_fit.
- [ ] 2.2 Unit tests on synthetic windows:
  - a censored bout;
  - an uncomputable statistic failing;
  - a boundary window;
  - a uniform-lawn track with missing radii, read as on the lawn;
  - a thin arm's direction read from μ ± 1.96σ after inflation;
  - D = 0 at the centres;
  - a wrong direction failing inside the band;
  - normal leaving by magnitude alone;
  - the reader refusing a short analysis window.

## 3. The bands

- [ ] 3.1 `scripts/analysis/benchmark_bands.py`:
  - per-animal sufficient statistics from the store's band dates, so each resample is a sum;
  - the cluster bootstrap at n (seed 2030);
  - paired Δ for mutants, and thin-arm inflation;
  - López-Cruz's batches and their split (seed 2031), pdfr-1's pair kept in the bands, written to
    `data/benchmark/lopezcruz2019_split.json`;
  - the bands at n = 200 and n = 64.
- [ ] 3.2 The rule's own checks (seed 2032, 200 half-splits), before the bands are written:
  - the small lawn at N = 200, a population drawn from the held-back half;
  - López-Cruz per held-back batch;
  - each gated at a joint pass rate of 0.90.
- [ ] 3.3 The reported rows:
  - density, and the authors' labels beside ours;
  - Flavell's published timescales;
  - B.2's pooled γ beside the band's centre;
  - the quintile ratio's reference;
  - the expected directions beside the references' intervals;
  - the geometric rule's agreement with D.1's in-lawn runs and with `midbody_in_lawn`.
- [ ] 3.4 Run it. Commit `data/benchmark/bands.json` and the record under
  `docs/experiments/logbooks/supporting/109-benchmark-grading/`.
- [ ] 3.5 A test that the bands script never loads a held-out date or batch.

## 4. The uniform well

- [ ] 4.1 `generate_assay_configs.py`: `uniform_lawn_scheer`, with the lawn's radius the well's and
  `wall_clearance_mm: 0`. Check that `lawn_at` counts the rim as on the lawn, as `_cells_under` does.
  Generate both arms' configs, and smoke them on 5991–5999 in both runners.

## 5. Grading a run

- [ ] 5.1 `scripts/analysis/benchmark_grade.py`: grade a batched evaluation or a set of captures against
  `bands.json` at the run's n. Write each statistic's value, z and pass, the families' passes, D and
  D_fit.
- [ ] 5.2 An end-to-end test on a smoke seed's captures.

## 6. The preflight (after `amend-food-dependent-kinematics` merges)

- [ ] 6.1 ConstantHazards (0.15 mm/s; 0.02, 0.01 and 0.01/s) and PirouetteHazards (the same, β = 3)
  at n = 64 on seeds 9101–9164, on the small lawn, the
  uniform well and López-Cruz's assay, through the batched evaluator. Commit the record, as
  description.

## 7. The novelty check

- [ ] 7.1 Record NeuroSparkSNT (Yazır, Research Square rs-10240803, 2026), PGOB (He, Zenodo 2026) and
  Morrison & Young 2025 against the novelty map's first claim, in the logbook.

## 8. Documentation and tracker

- [ ] 8.1 Logbook 109:
  - the classifier's gate;
  - the limits: tracking noise, and the bout estimator's exponential form;
  - the bands and the rule's check;
  - the reported rows;
  - the preflight;
  - the novelty check.
- [ ] 8.2 The tracker: close B.3, mark B.5's NeuroSparkSNT check done, and add 9101–9164 to the seed
  table.
- [ ] 8.3 AGENTS.md (the grading command), `docs/usage.md`, `data/benchmark/PROVENANCE.md`, and the
  CHANGELOG.

## 9. Close-out

- [ ] 9.1 Full suite and `pre-commit run -a`.
- [ ] 9.2 `openspec validate add-benchmark-grading --strict`.
- [ ] 9.3 Archive, with the new spec's Purpose fixed.
- [ ] 9.4 PR.
