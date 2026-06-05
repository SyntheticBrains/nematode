## ADDED Requirements

### Requirement: Continuous-2D environment configuration

The configuration system SHALL accept an environment-type discriminator and continuous-2D environment fields — world bounds (physical units), worm body length scale, per-step maximum displacement, food capture radius, and klinotaxis head-sweep amplitude — validated with documented defaults, while existing grid configurations remain valid unchanged.

#### Scenario: Continuous-2D config parsed

- **WHEN** a YAML config specifies the continuous-2D environment type with continuous fields
- **THEN** the config loads into the typed configuration model with the continuous fields populated (or defaulted) and validated

#### Scenario: Existing grid configs unchanged

- **WHEN** an existing grid scenario YAML (no environment-type field) is loaded
- **THEN** it loads and behaves exactly as before this change

### Requirement: Continuous-action configuration

The configuration system SHALL accept continuous-action policy settings (action mode, action bounds for speed and turn, and Gaussian log-std parameterisation/clamp) for PPO-family brains, defaulting to discrete on the grid substrate.

#### Scenario: Continuous-action settings parsed

- **WHEN** a config selects the continuous-action mode with action bounds
- **THEN** the brain config loads with the continuous-action settings and the bounds are applied to the tanh-squashed Gaussian head

#### Scenario: Discrete default preserved

- **WHEN** a config does not specify continuous-action settings
- **THEN** the brain defaults to discrete action behaviour, unchanged from before this change
