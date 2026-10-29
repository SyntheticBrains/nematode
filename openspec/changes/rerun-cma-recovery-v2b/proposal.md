## Why

F.0b-i on B v2 ([Logbook 116](../../../docs/experiments/logbooks/116-cma-recovery-v2.md)) failed its gate: the median gain was 0.690 against 0.75, so C.1b may not run. sep-CMA recovered the two truths whose statistics are stable on v2's scales (gains 0.914 and 1.007) but not the two noisy ones far from the wild type (0.466 and 0.056). The failure belongs to the fit's objective, which had two defects:

- **capped**, it is flat beyond the cap: the first pilot stopped at 0.511;
- **uncapped** (amendment 1), an undefined statistic's flat 10 undercut a far defined one, so the noisy truths' fits escaped into undefined families.

A re-investigation (2026-10-29, on the stored simulations) bears on the fix:

- **undefined statistics are pervasive:** 80% of the 201 prior references have at least one, and γ is undefined in 64%;
- **the box centre** has every statistic defined, so a fit starts in a defined region;
- **noise grows with distance from the wild type** (floor against exit rate, Spearman 0.65).

The maintainer's decisions of 2026-10-29:

- the objective is 10·asinh(z/10), with an undefined statistic costing 100;
- the registered four truths are reused;
- the cap of 272 is kept;
- a pre-registration probe on discarded seeds fits the two failed truths and one stable one.

## What Changes

- **The fit's ranking objective.** Each defined statistic contributes 10·asinh(z/10): linear near the truth, slowly growing far from it. A statistic undefined in only one record contributes 100, more than any plausible defined term, since reaching 100 takes about 110,000 band scales. The registered D, the gates and the gains stay capped.
- **The probe, before registration.** Truths 1164 and 6985 are fitted to the guarded plateau (cap 200), and 9105 for 80 generations, on discarded seeds. Each must reach a gain of at least 0.75, or the change returns to the maintainer.
- **The registered re-run:**
  - the four registered truths and their committed observations, floors and priors;
  - 4 × 2 fits on new seeds, at the cap of 272, with the guarded plateau;
  - the same gate, a median best-restart gain of at least 0.75, and the same verdict map.
- **Launch record 117** and **Logbook 117**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `fit-recovery`: the fit ranks on a compressed, uncapped distance, with undefined statistics costed above every defined term.

## Impact

- `scripts/analysis/cma_recovery_v2.py`: a `compressed` objective and the attempt-2 stages (`probe-b`, `run-b`, `read-b`), on attempt 1's records and code.
- `docs/experiments/logbooks/supporting/117-cma-recovery-v2b/`; Logbook 117; the tracker.
- **Cost:** the probe takes about 4–6 h, and the registered fits about 15 h (attempt 1's ran 15.3 h).
