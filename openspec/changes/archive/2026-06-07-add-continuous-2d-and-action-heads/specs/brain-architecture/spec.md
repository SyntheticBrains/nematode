## ADDED Requirements

### Requirement: Continuous-action heads on MUST PPO brains

The MUST PPO-family brains (MLP-PPO, LSTM-PPO, CfC-PPO, connectome-PPO) SHALL support a continuous-action mode via the shared action-policy module, producing a continuous action on the continuous-2D substrate and retaining their discrete behaviour on the grid substrate.

#### Scenario: Continuous mode on continuous substrate

- **WHEN** a MUST PPO brain runs in the continuous-2D environment
- **THEN** it emits a continuous `(speed, turn)` action via the shared policy module and updates via PPO using the continuous log-probability and entropy

#### Scenario: Discrete mode retained on grid

- **WHEN** a MUST PPO brain runs in the existing grid environment
- **THEN** it emits a discrete action exactly as before this change (subject to the migration-regression bar)

### Requirement: Discrete-path migration regression bar

Migrating the existing discrete action/PPO logic of the MUST brains onto the shared action-policy module SHALL preserve their pre-refactor behaviour, demonstrated on at least one discrete smoke configuration per brain. MLP-PPO and LSTM-PPO SHALL be byte-equivalent; CfC-PPO and connectome-PPO MAY declare an up-front seeded-RNG tolerance.

#### Scenario: Byte-equivalence for MLP-PPO and LSTM-PPO

- **WHEN** MLP-PPO or LSTM-PPO is run pre- and post-migration on a fixed-seed discrete smoke config
- **THEN** the training curves are byte-identical

#### Scenario: Declared tolerance for CfC and connectome

- **WHEN** CfC-PPO or connectome-PPO is run pre- and post-migration on a fixed-seed discrete smoke config
- **THEN** any divergence is within the tolerance declared before the migration, and the tolerance is documented in the T5 logbook

### Requirement: Plugin-parity verification (Gate 2)

Adding a new architecture family through the L1 plugin interface SHALL touch ≤ 6 files and introduce no per-architecture branches in the simulation or training loops, verified during T5 by adding the transformer architecture and recording engineer-hours (documented, not load-bearing).

#### Scenario: Parity budget enforced by test

- **WHEN** the plugin-parity test evaluates the transformer addition
- **THEN** it asserts the files-touched count is ≤ 6 and that no per-architecture branch exists in the simulation/training loops

#### Scenario: Entrypoint default is registry-driven

- **WHEN** a simulation is started without a config file for a registered architecture
- **THEN** the default config is resolved through the registry rather than a per-architecture `match`/`if` branch in the entrypoint
