## ADDED Requirements

### Requirement: The body-drive action space

Continuous brains SHALL offer `action_space: body_drive` beside the default `speed_turn`: 25 numbers in
`[-1, 1]`, dorsal and ventral drive for each of 12 segments and one direction channel. A simulation
configuration SHALL be refused when a brain's `body_drive` and the environment's kinematic body
disagree. Brains that do not implement it SHALL refuse it.

#### Scenario: The action is 25 bounded numbers

- **WHEN** a `body_drive` brain acts
- **THEN** its action SHALL have 25 entries, each in `[-1, 1]`

#### Scenario: A brain and an environment that disagree are refused

- **WHEN** a configuration pairs `body_drive` with a point worm, or a kinematic body with `speed_turn`
- **THEN** loading SHALL raise, naming both settings
