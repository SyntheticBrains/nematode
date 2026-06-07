## ADDED Requirements

### Requirement: Environment-type selection

The system SHALL select the environment implementation from configuration via an environment-type discriminator, dispatching to either the discrete grid environment (default) or the continuous-2D environment through a single factory, with no caller depending on a hard-coded environment class.

#### Scenario: Grid selected by default

- **WHEN** a configuration omits the environment type or sets it to the grid type
- **THEN** the factory constructs the discrete grid environment, preserving existing behaviour

#### Scenario: Continuous-2D selected explicitly

- **WHEN** a configuration sets the environment type to continuous-2D
- **THEN** the factory constructs the continuous-2D environment, and all environment construction sites route through the factory

#### Scenario: Callers are environment-agnostic

- **WHEN** the simulation, agent, and screenshot/export paths obtain an environment
- **THEN** they do so through the factory and operate against the base environment interface, not a concrete subclass alias
