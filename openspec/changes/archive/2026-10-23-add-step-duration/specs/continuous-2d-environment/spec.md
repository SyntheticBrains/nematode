## MODIFIED Requirements

### Requirement: The step's duration in worm time

The environment SHALL record the duration of a step in worm-seconds. It SHALL be the configured
`step_seconds` when set, and otherwise `max_step_mm` divided by the crawl speed. The crawl speed and the
undulation period SHALL be named constants beside it. When `step_seconds` is set, the derived
`max_step_mm` divided by the crawl speed SHALL equal it.

#### Scenario: Block V's step is five worm-seconds

- **WHEN** the constant is evaluated at `max_step_mm: 1.0` with `step_seconds` unset
- **THEN** it SHALL be 5.0 worm-seconds, about three undulation periods

#### Scenario: A configured step is the environment's step

- **WHEN** a lawn config with `max_step_mm: 1.0` sets `step_seconds: 1.0`
- **THEN** the environment's step SHALL be 1.0 worm-second, and its full-speed step 0.2 mm
