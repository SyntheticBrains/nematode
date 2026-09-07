# Panel 3: replicate the Hebbian wiring contrast on fresh seeds

## Why

Panel 2 (Logbook 041) left one question open that is cheap to close. The reward-free Hebbian
wiring contrast — wild-type Hebbian over degree-preserving rewired-null Hebbian — held its size on
fresh seeds (+16.2 on panel 1's seeds 1–8, +11.9 on seeds 9–16, +14.1 pooled with the 80%
interval clear of zero) and failed only its test: both arms are bimodal (alignment finds a
competent fixed point or a dead one, set by the seed), the paired deltas spread 36.5 points, and a
rank test at n = 16 reaches q = 0.28. That is a power problem with a known remedy, not an
ambiguity about the effect. With the observed spread, a true effect of +12 to +14 gives roughly
75–85% power at n = 48 for the registered one-sided paired test; and the competent fraction (9/16
against 5/16 in panel 2) is the statistic suited to a bimodal outcome and can be registered beside
it.

It costs about an hour: two arms on seeds 17–64 at 1000 episodes is 96 runs, and the frozen
floors for those seeds already exist from panel 2's prior sweep, so learning gains come free. It
has to be now: the imitation warm start (S.2) changes the initial policy on every seed, after
which the reward-free question cannot be revisited without re-running floors. And it would be the
project's first registered wiring-specific result — Logbook 034 found the wiring inert under
gradient learning; a confirmed `specific_wiring` under an unmodulated local rule is a structure
claim, not the Phase 7 headline, but it is citable and it tells S.2's design whether the two
wirings' Hebbian fixed points differ before a warm start is layered on.

Ratified with Chris 2026-09-07 as the item before S.2.

## What Changes

- **A registration** for a replication on disjoint seeds: the two degree-scaled Hebbian arms on
  seeds 17–64 (48 fresh, paired), 1000 episodes, the single registered extension to 1500; a
  two-test BH-FDR family — R1 the paired one-sided Wilcoxon on the plateau tails (primary, decides
  the verdict) and R2 a paired exact test on competent-fraction discordance (registered secondary,
  annotates); the verdict map in the rewired-null control's vocabulary; seeds 1–16 reported
  pooled and labelled descriptive, never confirmatory; no pilot; no new arm, config or mechanism.
- **A harness** (`scripts/analysis/l4_panel3.py`) reusing panel 2's registry and readers,
  enforcing the replication seed range, reading the frozen floors for seeds 17–64 from panel 2's
  sweep for the learning gains, and reading panel 2's committed per-seed table for the pooled
  descriptive.
- The launch record before the run, the run, records under `supporting/042-l4-panel3/`, tests,
  docs.

Out of scope: any other arm (count-init, plastic, MLP); any rule change; the logbook (the next
tracker item).

## Capabilities

**Modified**: `l4-plasticity-panel` — gains the panel-3 replication protocol as a further
requirement.

## Impact

- New: the analysis script and its tests, the supporting directory. Edited: `CHANGELOG.md`, the
  tracker. No package code, no configs.
