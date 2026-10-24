# Tasks: hardening the benchmark's grading before B.4

The maintainer's decisions of 2026-10-24, after B.3's review:

- direction by a paired t-test across dates;
- narrow the pdfr-1/tax-4 dissociation claim, with Ji 2021 reported beside;
- report pdfr-1's local search, don't grade it;
- keep all 11 wild-type statistics, and let B.4 decide.

## 1. The rule

- [ ] 1.1 `grading.local_search_statistics`: γ undefined at a grid bound (within 1%) or α ≤ 0.
- [ ] 1.2 `grading`: a paired-t direction helper, and `Band` carrying a resolved direction instead of
  deriving one from its scale.
- [ ] 1.3 `benchmark_bands.py`:
  - the family and mutant builders as functions at any n;
  - mutant directions by paired t;
  - pdfr-1's search and Ji 2021 in `reported`;
  - pdfr-1's bar at 2.24.

## 2. Grading at each family's size

- [ ] 2.1 `benchmark_grade.py`: per-family n, with bands from `bands.json` where present and built on
  demand otherwise.

## 3. Rebuild and re-check

- [ ] 3.1 Rebuild `bands.json` and the record. Check that every centre, scale, κ and check is
  unchanged, and list what changed.
- [ ] 3.2 Re-check the real López-Cruz band batches' γ under the new rule.
- [ ] 3.3 Re-run the preflight (9101–9164), and report beside B.3's.

## 4. Tests

- [ ] 4.1 A flat curve's γ is undefined and fails at the cap. A decaying one is unchanged.
- [ ] 4.2 The paired t resolves a direction the band scale did not; a single paired date resolves
  none; a wrong sign fails.
- [ ] 4.3 On-demand bands at 200 equal the vendored ones; per-family n grades.
- [ ] 4.4 pdfr-1's family has two statistics, and no López-Cruz statistic enters it.

## 5. Docs and tracker

- [ ] 5.1 Logbook 109 addendum: every changed value beside its original.
- [ ] 5.2 The roadmap's novelty map: the dissociation claim narrowed.
- [ ] 5.3 The tracker:
  - B.3 amended;
  - B.4's notes (multiplicity across controls, thin controls, per-statistic discrimination as
    description);
  - C.2's note (`time_on_lawn`'s scale);
  - C's mutant table (directions resolved: tph-1 and tax-4 roaming only).
- [ ] 5.4 `data/benchmark/PROVENANCE.md`, the CHANGELOG, and AGENTS.md's grading entry.
- [ ] 5.5 Pin `mdformat-frontmatter` in `.pre-commit-config.yaml`.

## 6. Close-out

- [ ] 6.1 Full suite and `pre-commit run -a`.
- [ ] 6.2 `openspec validate harden-benchmark-grading --strict`.
- [ ] 6.3 Archive.
- [ ] 6.4 PR.
