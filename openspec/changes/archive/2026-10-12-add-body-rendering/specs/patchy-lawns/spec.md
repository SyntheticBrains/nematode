## MODIFIED Requirements

### Requirement: The lawn model's scope

A configuration with `food_model: lawns` SHALL be refused if it runs more than one agent or off the
continuous substrate, and a lawn environment SHALL refuse the grid `pixel` renderer, which draws no
lawns. The continuous `pixel_continuous` renderer SHALL be accepted and SHALL draw the lawns.
Point-food keys (`foods_on_grid`, `target_foods_to_collect`) SHALL be ignored under lawns rather
than validated.

#### Scenario: Multi-agent lawns are refused

- **WHEN** a lawn configuration declares two agents
- **THEN** loading it SHALL fail with a message naming the lawn model

#### Scenario: The continuous renderer runs lawns

- **WHEN** a lawn environment is created with the `pixel_continuous` theme
- **THEN** it SHALL place its lawns, and the grid `pixel` theme SHALL still be refused
