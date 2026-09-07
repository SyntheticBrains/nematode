## ADDED Requirements

### Requirement: Panel 3 replicates the Hebbian wiring contrast on fresh seeds

The third L4 panel SHALL run the two degree-scaled Hebbian arms of panel 2, unchanged, on paired
seeds 17–64 at 1000 episodes, with no pilot and no other arm. The frozen floors for those seeds
SHALL be panel 2's prior-sweep runs, read by seed and not re-run. The per-seed ranked metric SHALL
be the committed plateau-tail full-clear success. The confirmatory family SHALL be exactly two
one-sided paired tests corrected together at α = 0.05: (R1) wild-type Hebbian over rewired-null
Hebbian by the committed paired Wilcoxon, the primary; (R2) wild-type over rewired-null in
competent-fraction discordance, an exact binomial on the discordant pairs at the 20% competent
threshold. The verdict SHALL be assigned from R1 alone as `insufficient_seeds`, `specific_wiring`,
`rewired_beats_wild_type`, `degree_statistics` or `inconclusive`, in that order and by the same
rules as panel 2; R2 SHALL annotate the verdict and never change it. Seeds 1–16 SHALL enter only
a pooled descriptive summary read from panel 2's committed per-seed table, never a confirmatory
test. A launch record SHALL be committed before the campaign runs, and a seed without a detected
plateau at 1000 episodes SHALL receive exactly one extension, a fresh run at 1500 replacing the
shorter log.

#### Scenario: Only the replication seeds are confirmatory

- **WHEN** the panel-3 harness groups a campaign
- **THEN** a Hebbian-arm log with a seed outside 17–64 SHALL be rejected
- **AND** panel 2's seeds 1–16 SHALL appear only in the pooled descriptive summary

#### Scenario: The family has two members and the verdict follows R1

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly R1 and R2 SHALL be corrected together
- **AND** the verdict SHALL follow R1 by panel 2's map
- **AND** R2's outcome SHALL be recorded as an annotation, including the case where R2 passes and
  R1 does not

#### Scenario: The discordance test is exact

- **GIVEN** `b` seeds where only the wild-type arm is competent and `c` where only the rewired arm is
- **WHEN** R2 is computed
- **THEN** its p-value SHALL be the exact binomial `P(X ≥ b)` for `X ~ Bin(b + c, ½)`, and R2 SHALL
  pass only when its corrected q is below α and `b > c`

#### Scenario: The floors come from panel 2's sweep

- **WHEN** the harness computes learning gains for seeds 17–64
- **THEN** each Hebbian arm's floor SHALL be panel 2's frozen run at the same seed, and the record
  SHALL name the campaign they were read from

#### Scenario: The launch precedes the run and the extension is bounded

- **WHEN** the campaign is launched
- **THEN** a launch record naming the commit, command, seeds and budget SHALL already be committed
- **AND** a seed still without a plateau at 1000 SHALL be re-run once at 1500 and its log SHALL
  replace the shorter one
