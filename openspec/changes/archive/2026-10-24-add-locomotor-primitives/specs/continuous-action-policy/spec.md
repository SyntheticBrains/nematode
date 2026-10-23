## ADDED Requirements

### Requirement: The primitives action space

Continuous brains SHALL offer `action_space: primitives`: five numbers in `[-1, 1]`, the run speed,
the run curvature and three start hazards. A simulation configuration SHALL be refused when a brain's
`primitives` and the environment's `locomotion: primitives` disagree, or when `primitives` is paired
with `signed_speed`, a state-dependent standard deviation, or a brain other than `connectomeppo` or
`mlpppo`.

#### Scenario: The action is five bounded numbers

- **WHEN** a `primitives` brain acts
- **THEN** its action SHALL have five entries, each in `[-1, 1]`

#### Scenario: A brain and an environment that disagree are refused

- **WHEN** a configuration pairs `primitives` with free locomotion, or primitive locomotion with
  `speed_turn`
- **THEN** loading SHALL raise, naming both settings
