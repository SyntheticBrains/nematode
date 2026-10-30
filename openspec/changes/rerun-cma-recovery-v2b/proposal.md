## Why

F.0b-i on B v2 ([Logbook 116](../../../docs/experiments/logbooks/116-cma-recovery-v2.md)) failed its gate: the median gain was 0.690 against 0.75, so C.1b may not run. sep-CMA recovered the two truths whose statistics are stable on v2's scales (gains 0.914 and 1.007) but not the two noisy ones far from the wild type (0.466 and 0.056). The failure belongs to the fit's objective, which had two defects:

- **capped**, it is flat beyond the cap: the first pilot stopped at 0.511;
- **uncapped** (amendment 1), an undefined statistic's flat 10 undercut a far defined one, so the noisy truths' fits escaped into undefined families.

A re-investigation (2026-10-29, on the stored simulations) bears on the fix:

- **undefined statistics are pervasive:** 80% of the 201 prior references have at least one, and γ is undefined in 64%;
- **the box centre** has every statistic defined, so a fit starts in a defined region;
- **noise grows with distance from the wild type** (floor against exit rate, Spearman 0.65).

The maintainer's decisions of 2026-10-29:

- the objective is 10·asinh(z/10), with undefined statistics ranked behind defined ones (made lexicographic at the spec review);
- the registered four truths are reused;
- the cap of 272 is kept;
- a pre-registration probe on discarded seeds fits the two failed truths and one stable one.

## What Changes

- **The fit's ranking objective,** lexicographic in definedness: 1,000 × (the statistics undefined in only one record) + the weighted RMS of 10·asinh(z/10) over the defined ones. Fewer undefined statistics always rank ahead, and near the truth each term is z to within 1.5%. The registered D, the gates and the gains stay capped.
- **The probe, before registration.** Truths 1164 and 6985 are fitted to the guarded plateau (cap 200), and 9105 for 80 generations, on discarded seeds. Each must reach a gain of at least 0.80 (the bar plus a 0.05 margin), or the change returns to the maintainer. The reused observations' file is pinned by its SHA-256 in Part A.
- **The registered re-run:**
  - the four registered truths and their committed observations, floors and priors;
  - 4 × 4 fits on new seeds (revised after the probe; see below), at the cap of 272, with the guarded plateau;
  - the same gate, a median best-restart gain of at least 0.75, and the same verdict map.
- **Launch record 117** and **Logbook 117**.

## Revision after the probe (2026-10-30)

The probe returned to the maintainer. On discarded seeds the ranking score recovered 6985 (gain 0.981; attempt 1 reached 0.056) and 9105 (1.000), and left no statistic undefined in any answer. Truth 1164 reached 0.787, under the 0.80 bar. The maintainer then had a second restart of 1164 fitted on the probe's seeds (`probe-b2`, CMA seed 5998). It gained 0.523 in a different wrong basin, and the gate's selection keeps the first restart, at 0.787.

The diagnosis (`probe_diagnosis.json`): 1164's answers sit 21 and 10 band scales off on the speed medians (first restart) and across the small lawn and kinematics (second). Its own repeats sit at 0.2 on those statistics. So the shortfall is restart-to-restart variance, not the objective, the truth's noise or the box's bounds (every truth has 1–3 parameters within 0.05 of a bound).

The maintainer's decisions of 2026-10-30:

- **1164 is accepted below the probe's bar.** The probe was the go/no-go for spending the registered run, and the registered gate is unchanged.
- **Each truth is fitted with 4 restarts instead of 2.** The gate's selection already follows the final objective (4.06 → 0.787, 5.00 → 0.523), so more restarts reduce the chance that a truth loses every restart. Four is also closer to the 8 restarts C.1b fits with, and the run measures the per-restart success rate C.1b's restart count needs.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `fit-recovery`: the fit ranks on a compressed, uncapped distance, with undefined statistics costed above every defined term.

## Impact

- `scripts/analysis/cma_recovery_v2.py`: the ranking score and the attempt-2 stages (`probe-b`, `diagnose-b`, `probe-b2`, `run-b`, `read-b`), on attempt 1's records and code.
- `docs/experiments/logbooks/supporting/117-cma-recovery-v2b/`; Logbook 117; the tracker.
- **Cost:** the probe took about 4 h and the second restart about 1.5 h. The 16 registered fits take about 26 h at the probe's 54 s per generation and plateaus near 110 generations, and at most about 65 h if every fit reaches the cap.
