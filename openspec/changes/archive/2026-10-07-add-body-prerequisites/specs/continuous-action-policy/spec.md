## ADDED Requirements

### Requirement: Signed speed bounds that agree with the environment

Continuous brains SHALL take their action bounds from one shared helper. Under `signed_speed: true`
the speed bound SHALL be `[-1, 1]`; otherwise it SHALL be `[0, 1]` as before, with turn `[-1, 1]` either
way. A simulation configuration whose continuous brain's `signed_speed` differs from the environment's
`allow_reversal` SHALL be refused at load.

#### Scenario: Every continuous brain reads the shared bounds

- **WHEN** any continuous PPO brain is built with `signed_speed: true`
- **THEN** its action bounds SHALL be `[-1, -1]` to `[1, 1]`, and with it false they SHALL be unchanged

#### Scenario: A brain and an environment that disagree are refused

- **WHEN** a configuration sets `signed_speed` on the brain and not `allow_reversal` on the environment,
  or the reverse
- **THEN** loading SHALL raise, naming both settings

#### Scenario: Signed speed needs continuous actions

- **WHEN** a discrete-action brain sets `signed_speed: true`
- **THEN** validation SHALL raise
