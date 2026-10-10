## ADDED Requirements

### Requirement: Internal-State Sensory Module

The system SHALL provide an `internal_state` sensory module of width 1 that carries the agent's satiety
as a fraction of its maximum, read from `BrainParams.satiety`.

#### Scenario: Satiety reaches the brain

- **GIVEN** a brain configured with `internal_state` among its sensory modules
- **WHEN** the agent's satiety is half its maximum
- **THEN** the module's feature SHALL be 0.5
