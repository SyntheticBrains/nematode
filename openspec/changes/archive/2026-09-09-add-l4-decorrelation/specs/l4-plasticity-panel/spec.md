## ADDED Requirements

### Requirement: The decorrelation test asks whether a decorrelating term recovers what grounding cost

The sign-grounding test found that grounding the substrate's synapse signs made reward-free
Hebbian learning substantially worse, and predicted that a rule with an anti-Hebbian or
decorrelating term would recover the loss. That prediction SHALL be tested by re-running the
sign-grounding test's own Hebbian protocol under each decorrelating variant: the wild-type and
rewired-null grounded Hebbian arms, paired seeds 1–16, a 1000-episode budget, the committed
plateau-tail full-clear metric, and the single registered extension of a fresh run at 1.5×
replacing a shorter log the plateau detector marks non-converged. The comparators SHALL be the
sign-grounding test's committed per-seed values on the same seeds 1–16, read from its published
table and paired seed for seed, and no grounded Hebbian arm SHALL be re-run.

The confirmatory family SHALL be exactly four one-sided paired tests corrected together under
BH-FDR: **D1** the wild-type anti-Hebbian arm over the committed wild-type grounded Hebbian
values, **D2** the wild-type Oja arm over the same, **D3** the wild-type over the rewired null
under the anti-Hebbian variant, and **D4** the same under the Oja variant.

The verdict SHALL be assigned in order as `insufficient_seeds`; then **`no_recovery`** when
neither D1 nor D2 confirms; then `recovery_specific` (D1 only), `recovery_general` (D2 only) or
`recovery_both`. D3 and D4 SHALL annotate the verdict and never change it. A launch record SHALL
be committed before any run.

Two annotations SHALL be computed and reported without changing the verdict: `full_recovery`,
true when the 80% bootstrap interval of a recovered arm's mean plateau tail over seeds 1–16
includes or exceeds the committed random-sign mean for the same wiring,
which separates a term that helps from one that restores what grounding cost; and the mean
decorrelation share from the rule's telemetry, so that a recovery whose share is near zero is
recorded as attributable to something other than the term.

#### Scenario: The prediction can fail

- **GIVEN** neither D1 nor D2 confirming
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `no_recovery`
- **AND** the record SHALL state that the missing inhibitory brake was not what limited the rule

#### Scenario: Helping and restoring are distinguished

- **WHEN** a variant's arm confirms its recovery test
- **THEN** the record SHALL report whether the 80% bootstrap interval of its mean plateau tail
  includes or exceeds the committed random-sign mean for the same wiring

#### Scenario: A recovery is attributed to the term or not

- **WHEN** a variant's arm confirms its recovery test
- **THEN** the record SHALL report that arm's mean decorrelation share

#### Scenario: The wiring contrast annotates and never decides

- **WHEN** D3 or D4 confirms or fails
- **THEN** the verdict SHALL be unchanged by it

#### Scenario: The panel stays gated

- **WHEN** a variant recovers
- **THEN** the 2×2 panel SHALL still require a passing clone assay before it is re-run
