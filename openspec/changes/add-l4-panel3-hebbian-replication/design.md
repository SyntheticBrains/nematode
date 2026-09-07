# Design: panel 3 — replicating the Hebbian wiring contrast

## Context

Panel 2 measured wild-type Hebbian minus rewired-null Hebbian at +14.1 over seeds 1–16 (interval
clear of zero, q = 0.28), with both arms bimodal and the paired deltas spreading 36.5 points.
Seeds 1–16 have been seen and cannot be confirmatory again. Panel 2's prior sweep ran the four
frozen arms on seeds 1–64, so the degree-scaled frozen floors for seeds 17–64 exist and are
reproducible from their campaign logs.

## Goals / Non-Goals

**Goals**

- Test the Hebbian wiring contrast at a sample size that can carry it, on seeds never used for
  it, with a statistic suited to a bimodal outcome registered beside the rank test.
- Change nothing about the arms: same configs, recipe, budget and extension rule as panel 2.

**Non-Goals**

- Any new arm, initialisation law, rule variant or config.
- Any claim about reward-modulated learning; this is a structure claim under a reward-free rule.

## Decisions

### D1. Arms, seeds, budget

`wt_hebbian` and `rn_hebbian` (panel 2's degree-scaled Hebbian stems, unchanged) on seeds
**17–64**, paired by run seed (`rewire_seed` derived from it, as in every earlier panel), 1000
episodes. A seed the committed plateau detector marks non-converged at 1000 receives the single
registered extension, a fresh run at 1500 replacing the shorter log. The frozen floors for seeds
17–64 are panel 2's sweep runs (`wt_frozen`, `rn_frozen`, 600 episodes), read from that campaign
for the learning gains; nothing is re-run. Cost: 96 runs, about seventy minutes on 16 workers
(panel 2's 64 Hebbian runs took 47 minutes). No pilot: every value is panel 1's pin.

### D2. Metric and statistics

The committed plateau-tail full-clear success and panel 2's reader. Two registered tests, one
BH-FDR family at α = 0.05:

| id | test | direction | statistic |
|---|---|---|---|
| **R1** | `wt_hebbian` vs `rn_hebbian`, seeds 17–64 | wild-type > rewired | the committed paired one-sided Wilcoxon, 80% bootstrap CI |
| **R2** | competent-fraction discordance, seeds 17–64 | wild-type > rewired | exact binomial on the discordant pairs: with `b` seeds where only the wild-type is competent (plateau tail ≥ 20%) and `c` where only the rewired is, `P(X ≥ b)` for `X ~ Bin(b + c, ½)` |

R1 passes at q < 0.05 with a positive mean delta; R2 at q < 0.05 with `b > c`. R2's threshold is
panel 2's registered competent threshold, 20%, unchanged.

**Power** (stated so the result can be read against it): with the observed spread of 36.5 and a
true effect of +12 to +14, R1 at n = 48 has roughly 75–85% power one-sided at α = 0.05. With
panel 2's competent fractions (0.56 against 0.31) and its discordance rate, R2 at n = 48 has
comparable power. Neither is high; a null here is "not confirmed at n = 48", and the pooled
descriptive over 64 seeds is reported beside it.

### D3. Verdict

From R1 alone, by the map used in Logbook 034 and panel 2: `insufficient_seeds` (fewer than two
common seeds), `specific_wiring` (R1 passes), `rewired_beats_wild_type` (R1's interval entirely
below zero), `degree_statistics` (interval spans zero), `inconclusive` (otherwise). R2 annotates:
`competent_fraction_confirms` (R2 passes) is recorded with the verdict, and the report names the
case where R2 passes and R1 does not — the bimodal case the rank test cannot carry — without
promoting it. A `specific_wiring` verdict requires R1; that is the registered claim.

### D4. Descriptive

Seeds 1–16 (panel 2's per-seed values, read from the committed `per-seed.csv`) pooled with 17–64
for a 64-seed mean delta, interval, sign count and competent fractions, labelled descriptive; the
learning gain of each Hebbian arm over its own frozen floor on seeds 17–64 and pooled; the
distribution of plateau tails per arm; the per-seed sign count for R1.

### D5. Harness and records

`scripts/analysis/l4_panel3.py`: imports panel 2's registry, reader, `paired`, `restrict` and
`distribution`; `REPLICATION_SEEDS = 17–64`; confirmatory grouping rejects a Hebbian log outside
17–64; `--campaign-dir` for the Hebbian runs, `--sweep-dir` for panel 2's frozen sweep,
`--panel2-csv` for the descriptive pooling; R1/R2, the verdict, the annotations, the pooled
descriptive, per-seed CSV and curves; a launch record before the run; everything promoted to
`supporting/042-l4-panel3/`. Tests on synthetic values: seed-range enforcement, R2's exact
p-value on a known `(b, c)`, R1/R2 directions, every verdict row, the R2-annotation case, the
pooled descriptive's seed set, the CSV shape.

## Risks / Trade-offs

- **Power is moderate, not high.** Stated in D2; a null is reported as unconfirmed at n = 48, and
  the design is not extended further on the same hypothesis without a new registration.
- **R2's threshold was chosen before panel 2 and the discordance framing after it.** The
  threshold is unchanged; the test is new and is registered here on seeds that have not been
  used, which is what makes it confirmatory.
- **The frozen floors come from a different campaign.** They are the same configs at the same
  seeds and reproducible from their logs; the harness reads them by seed and reports where they
  came from.

## Open Questions

None.
