## ADDED Requirements

### Requirement: A swept level is shown to reach the learner it is set on

Where a campaign varies a pinned setting in order to report how a result depends on it, the record SHALL establish that each level of that setting is read by the learner the arm runs, and SHALL NOT treat an accepted configuration value as evidence that it took effect. A setting that is declared, validated and then never read produces an arm that is indistinguishable from a swept one in its configuration, in its logs and in its score, and a sweep is the one design whose entire output is a claim about settings.

#### Scenario: The level is asserted to reach the learner

- **GIVEN** a sweep arm that differs from its parent in one setting
- **WHEN** the arm is registered
- **THEN** a test SHALL establish that the setting changes what the configured learner computes or updates, at the level the arm sets
- **AND** where the setting is declared on a shared configuration but read only by some learners, the record SHALL name which learners read it and SHALL confine the sweep to those

#### Scenario: A setting accepted by configuration is not evidence of a manipulation

- **GIVEN** a setting that passes its declared bounds and any load-time validation
- **WHEN** the arm runs under a learner that does not consume it
- **THEN** the arm SHALL NOT be reported as a level of that setting
- **AND** a surface SHALL NOT include a level whose only evidence of taking effect is that the configuration accepted it

#### Scenario: A key silently dropped is named rather than inferred

- **GIVEN** a configuration mechanism that discards unrecognised keys with a warning rather than an error
- **WHEN** a sweep templates one setting across more than one architecture
- **THEN** the record SHALL state, per architecture, whether the setting is declared on that architecture
- **AND** an architecture on which it is not declared SHALL be excluded from the sweep rather than run and reported as swept
