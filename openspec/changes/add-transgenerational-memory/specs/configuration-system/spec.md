# Configuration System — Transgenerational Memory Delta

## ADDED Requirements

### Requirement: Transgenerational Memory Configuration Schema

The `quantumnematode/utils/config_loader.py` module SHALL define a `TransgenerationalConfig` Pydantic model and a `LawnScheduleEntry` Pydantic model that together describe the transgenerational-memory evolution arm.

`TransgenerationalConfig` SHALL contain:

- `enabled: bool` — ablation switch for paired-arm comparison. When `false`, the substrate is structurally absent: the loop sets `tei_prior_source = None` in every worker tuple, `fitness.evaluate` omits the kwarg, and `brain.tei_prior` is never set (LSTMPPO's `__init__` default of `None` is preserved).
- `decay_factor: float` constrained to `0.0 ≤ x ≤ 1.0`, defaulting to `0.6` (matches the planned F0=1.0 / F1=0.6 / F2=0.36 / F3=0.216 cascade).
- `extraction_seed: int` constrained to `≥ 0`, defaulting to a fixed sentinel (e.g. `424242`). Controls determinism of the F0 telemetry-pass.
- `lawn_schedule: list[LawnScheduleEntry]` — per-generation overrides for pathogen-lawn enablement and training-episode count.

`LawnScheduleEntry` SHALL contain:

- `generation: int` constrained to `≥ 0`. Identifies the generation this entry applies to.
- `pathogen_lawns_enabled: bool`. When `true`, the env-build step for that generation SHALL include at least one `STATIONARY` predator entity (the pathogen lawn). When `false`, the env-build step SHALL exclude `STATIONARY` predators for that generation.
- `ppo_train_episodes: int` constrained to `≥ 0`. Overrides `EvolutionConfig.learn_episodes_per_eval` for that generation only.

The `transgenerational` block SHALL be optional on the YAML's `evolution:` block. When absent, the loop SHALL behave byte-equivalently to its current behaviour (no per-gen schedule consulted).

#### Scenario: Default config has no transgenerational block

- **GIVEN** any evolution YAML without a `transgenerational` field under `evolution:`
- **WHEN** the YAML is loaded via `load_simulation_config`
- **THEN** `EvolutionConfig.transgenerational` SHALL be `None`
- **AND** the loop SHALL construct the appropriate `InheritanceStrategy` from `evolution.inheritance` (defaulting to `NoInheritance`)
- **AND** no per-gen schedule SHALL be consulted

#### Scenario: TransgenerationalConfig validates decay_factor range

- **GIVEN** a YAML with `evolution.transgenerational.decay_factor: -0.1` (or `1.5`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state the valid range `0.0 ≤ decay_factor ≤ 1.0`

#### Scenario: Lawn schedule entries are validated against generation range

- **GIVEN** a YAML with `evolution.generations: 4` and `transgenerational.lawn_schedule` whose entries reference a generation outside `[0, 4)` (e.g. `generation: 5` or `generation: -1`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that schedule entries MUST reference generations in `[0, evolution.generations)`

### Requirement: Inheritance ↔ Transgenerational Pairing Validator

The `EvolutionConfig` SHALL include a model validator enforcing the paired-arm contract: `transgenerational.enabled = true` SHALL require `evolution.inheritance = "transgenerational"`, and `transgenerational.enabled = false` SHALL require `evolution.inheritance = "none"`.

The validator's purpose SHALL be to guarantee that the M6.6 paired-cohort ablation is a one-bit difference: the TEI-on and TEI-off arms differ ONLY in whether the substrate is transmitted, with no implicit weight-flow or trait-flow pathway carrying signal between arms.

#### Scenario: Enabled TEI requires transgenerational inheritance

- **GIVEN** a YAML with `evolution.transgenerational.enabled: true` AND `evolution.inheritance: lamarckian` (or `baldwin` or `none`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that `transgenerational.enabled=true` requires `inheritance: transgenerational` and SHALL explain that the paired-arm ablation contract requires the substrate to be the only difference between arms
- **AND** the message SHALL list the offending values

#### Scenario: Disabled TEI requires NoInheritance

- **GIVEN** a YAML with `evolution.transgenerational.enabled: false` AND `evolution.inheritance: transgenerational` (or `lamarckian` or `baldwin`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that `transgenerational.enabled=false` requires `inheritance: none` (the paired control arm)
- **AND** the message SHALL list the offending values

#### Scenario: Valid pairings are accepted

- **GIVEN** a YAML with `transgenerational.enabled: true` AND `inheritance: transgenerational`, OR a YAML with `transgenerational.enabled: false` AND `inheritance: none`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL succeed
- **AND** the resolved `EvolutionConfig` SHALL contain a `transgenerational` field with the requested values
