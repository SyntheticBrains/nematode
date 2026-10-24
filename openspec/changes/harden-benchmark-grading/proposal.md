## Why

B.3 (#485, Logbook 109) shipped the benchmark's rule. Its review, and a check on band data afterwards, found four things that would weaken B.4 and the days-long fits after it. The maintainer chose to fix them in one PR before B.4, and to record the rest in the tracker (2026-10-24).

1. **The direction rule is mis-specified.** A mutant's reference direction is resolved from μ ± 1.96σ, where σ is the band's scale: the spread of a new 200-animal population. Whether the reference's Δ is non-zero is a question about the centre's own uncertainty, which is about half that scale. On band data:

   | Mutant Δ | Centre | z on band scale (current) | Paired t across dates |
   |---|---|---|---|
   | tph-1 roaming | +0.270 | +5.2 | +8.9 (2 df) |
   | pdfr-1 roaming | −0.053 | −1.1 | −2.1 (7 df) |
   | tax-4 roaming | −0.116 | −2.6 | −4.8 (10 df) |
   | tax-4 exits | +0.306 | +0.8 | +1.7 (10 df) |

2. **B.4 cannot grade its real controls.**
   - The bands exist only at n = 64 and 200, and the grader requires one n across families.
   - B.4's positive controls come at their own sizes: 573 held-out wild-type animals, 31 on the uniform well, single López-Cruz batches, and mutants at 35–190.
3. **γ is meaningless without a decay.** A flat curve's fit lands on the grid's ceiling (3/min), as both preflight controllers' did. It fails at the cap, as it should, but a number is reported where none exists.
4. **pdfr-1's local search cannot be positive-controlled.** Its only paired batch is in the bands, so B.4 has no held-out cohort for it. D34 makes the benchmark valid only if its controls pass.

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **Direction by a paired t-test across dates** (decided).
  - A mutant's reference direction is resolved when a two-sided paired t-test of the per-date differences rejects zero at 0.05, with dates − 1 degrees of freedom.
  - The magnitude test keeps the band's scale.
  - On band data, tph-1 and tax-4 roaming stay resolved. pdfr-1 roaming and every leaving difference stay unresolved.
- **Bands at any size, per family.**
  - The band builders become library functions that the grader calls at each family's own n.
  - Each (family, n) and (mutant, n) draws from its own generator, `default_rng([2030, index, n])`, so a band does not depend on what was built before it. The reference build used one generator in sequence, so its scales move at bootstrap-noise level when rebuilt. Each is reported beside its original.
  - `bands.json` keeps its sizes as the reference build.
  - The grader accepts a different n per family.
- **γ is undefined without a decay.** When the fitted γ lies on the search grid's bound, or the fitted α ≤ 0, γ is NaN. It fails and enters the distance at the cap, so the outcome is the same.
- **pdfr-1's local-search Δ is reported, not graded** (decided). pdfr-1's family is Δ roaming and Δ exits, z\* = 2.24.
- **The dissociation claim is narrowed** (decided). The roadmap's novelty map, its biological validation targets and the tracker restate it:
  - tax-4 roams less, with leaving inside its band;
  - pdfr-1 is graded by magnitude.

  Ji 2021's vendored patch-assay direction for pdfr-1 (0.217 roaming against the wild type's 0.365) is reported beside the grade, not graded.
- **All 11 wild-type statistics are kept** (decided). B.4's matched negatives decide whether the family rule and the distance discriminate. Each statistic's discrimination is recorded there as description.
- **The mdformat hook's dependencies are pinned** in `.pre-commit-config.yaml`, so local and CI formatting agree: `mdformat-frontmatter` 2.1.2, `markdown-it-py` 4.2.0 and `mdit-py-plugins` 0.6.1, the versions the hook resolves locally. The list spacing CI enforced comes from `markdown-it-py`'s parser, not the frontmatter plugin.
- **Tracker notes**, for items that are not this PR's:
  - B.4 must fix a rule for multiplicity across its roughly six positive controls. At a 95% family-wise rate each, a chance failure voids the benchmark about a quarter of the time.
  - C.2's weights must reckon with `time_on_lawn`'s scale of 0.006, which will dominate an unweighted fit.
  - B.4's thin controls: the uniform well has one held-out date (31 animals), and tph-1 two held-out paired dates.

## Capabilities

### Modified Capabilities

- `benchmark-grading`: the direction rule, bands at any size, γ without a decay, pdfr-1's local search reported, and Ji 2021 beside.

## Impact

- **Code**:
  - `quantumnematode/benchmark/grading.py` (γ, the direction test, per-family n);
  - `scripts/analysis/benchmark_bands.py` (the builders as functions, direction by paired t, pdfr-1's search reported);
  - `scripts/analysis/benchmark_grade.py` (bands at each family's n).
- **Data**: `data/benchmark/bands.json` rebuilt.
  - Its centres, κ and the rule's checks are unchanged. Its scales move at bootstrap-noise level, from the per-band generators.
  - The direction fields are re-read, and pdfr-1's search moves to the record.
  - The preflight is re-run, and every changed value is reported beside B.3's.
  - The γ rule changes no real band batch: all 17 fit γ between 0.06 and 0.18 per minute with α > 0. So López-Cruz's band is unchanged, verified before the change.
- **Held-out data**: never read.
- **Docs**:
  - Logbook 109 gains an addendum;
  - the roadmap's novelty map;
  - the tracker (B.4 and C.2 notes);
  - `data/benchmark/PROVENANCE.md`.
- **Cost**: about a day. No campaign.
- **Seeds**: none new. The preflight re-runs on 9101–9164.
