# transformer-brain Specification

## Purpose

TBD - created by archiving change add-continuous-2d-and-action-heads. Update Purpose after archive.

## Requirements

### Requirement: Transformer/attention PPO architecture

The system SHALL provide a Transformer/attention-based PPO brain registered through the plugin interface, conforming to the `Brain` Protocol and selectable by name like any other architecture. It serves as the Gate-2 plugin-parity verification vehicle and an optional comparison row.

#### Scenario: Registered and instantiable via the registry

- **WHEN** the transformer brain name is requested through `instantiate_brain` with a matching config
- **THEN** the registry returns a constructed transformer brain through the same code path as every other architecture, with no per-architecture dispatcher branch

#### Scenario: Trains on klinotaxis without collapse

- **WHEN** the transformer brain is trained on a klinotaxis smoke configuration
- **THEN** training runs without NaN/Inf and mean episode return over the last quarter exceeds that over the first quarter (a learning signal, not policy collapse)

### Requirement: Transformer addition obeys the plugin-parity budget

Adding the transformer architecture SHALL touch no more than 6 files and SHALL introduce no per-architecture branches in the simulation or training loops, demonstrating L1 plugin parity (Gate 2 G2.b/G2.c).

#### Scenario: File-count budget met

- **WHEN** the transformer addition is measured via `git diff --name-only` for its addition commit(s)
- **THEN** the number of files touched to register and wire the architecture is ≤ 6 (new module, enum member, package import/export, config union, optional infra-kwargs branch, test)

#### Scenario: No per-architecture loop branches

- **WHEN** the post-addition simulation and training loops are reviewed
- **THEN** they contain no branch keyed on the transformer (or any) concrete architecture type
