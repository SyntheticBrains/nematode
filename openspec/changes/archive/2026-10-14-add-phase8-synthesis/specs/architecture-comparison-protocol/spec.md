## ADDED Requirements

### Requirement: A panel sized from a small pilot sizes from an upper bound on the pilot's spread

Where a panel's seed count is computed from a pilot of fewer than 16 seeds, the launch record SHALL
size it from an upper confidence bound on the pilot's per-seed spread, the one-sided 80% chi-square
bound on the standard deviation, or SHALL state why the point estimate is used instead. The logbook
SHALL report the achieved spread beside the pilot's, as a ratio.

#### Scenario: The upper bound sizes the panel

- **GIVEN** a pilot of four seeds with a per-seed standard deviation `s`
- **WHEN** the panel is sized
- **THEN** the seed count SHALL be computed from `s × sqrt((n − 1) / χ²(0.20; n − 1))`, not from `s`
- **AND** the launch record SHALL give both figures

#### Scenario: A point estimate is used with its reason

- **GIVEN** a reason to size from the point estimate, such as a seed cap the bound would exceed
- **WHEN** the panel is registered
- **THEN** the launch record SHALL state the reason and the sensitivity the bound would have given

#### Scenario: The achieved spread is compared with the pilot's

- **GIVEN** a panel that has read out
- **WHEN** its logbook is written
- **THEN** it SHALL report the achieved spread over the pilot's as a ratio, and the verdict SHALL
  stand as registered whatever the ratio

### Requirement: A positive control that fails its fixed rule is retried only as a new registration

Where a positive control fails the rule fixed for it, its verdict SHALL stand as recorded. Any retry
SHALL be a new registration: its changed settings fixed and stated with their reason before it runs,
its seeds disjoint from every earlier attempt's, its pass rule the original rule, and its failure
stopping the item without a further retry.

#### Scenario: The failed verdict stands

- **GIVEN** a positive control that read `fallback` or `fails` under its fixed rule
- **WHEN** a retry is decided
- **THEN** the first verdict SHALL be committed and cited unchanged, and the retry SHALL NOT re-read it

#### Scenario: The retry is fixed before it runs

- **GIVEN** a retry with a changed budget, target or other setting
- **WHEN** it is registered
- **THEN** the change and its reason SHALL be recorded before any retry run, the seeds SHALL be
  disjoint from every earlier attempt's, and the pass rule SHALL be unchanged

#### Scenario: A failed retry stops the item

- **GIVEN** a retry that fails the same rule
- **WHEN** it reads out
- **THEN** the item SHALL close with both verdicts and no further retry
