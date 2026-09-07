## ADDED Requirements

### Requirement: Per-seed weight paths

A configured `weights_path` MAY contain the placeholder `{seed}`. The simulation entry point
SHALL substitute the run seed for it before loading, SHALL reject a path that still contains a
brace after substitution, and SHALL leave a path without the placeholder unchanged.

#### Scenario: The placeholder resolves to the run seed

- **GIVEN** a config whose `weights_path` is `clones/arm_seed{seed}.pt`
- **WHEN** a run starts at seed 5
- **THEN** the entry point SHALL load `clones/arm_seed5.pt`

#### Scenario: An unresolved placeholder is rejected

- **WHEN** the substituted path still contains `{` or `}`
- **THEN** the entry point SHALL exit with an error before building the agent

#### Scenario: A plain path is untouched

- **WHEN** `weights_path` holds no placeholder
- **THEN** it SHALL be used as written
