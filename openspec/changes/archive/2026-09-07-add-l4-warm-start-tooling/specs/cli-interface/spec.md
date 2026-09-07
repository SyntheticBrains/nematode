## ADDED Requirements

### Requirement: Rollout recording flag

The simulation entry point SHALL accept `--record-rollouts PATH`. When given, a rollout recorder
SHALL be attached to the agent and every step of every episode SHALL be appended to `PATH` as
one JSON line. When absent, no recorder SHALL be attached and the run SHALL be bit-identical to a
run without this requirement.

#### Scenario: Recording writes one line per step

- **WHEN** a headless run is started with `--record-rollouts PATH`
- **THEN** `PATH` SHALL hold one line per environment step, each carrying the episode, the step,
  the `BrainParams` the brain read, the sampled action, the action mean and its probability

#### Scenario: No flag, no recorder

- **WHEN** a run is started without the flag
- **THEN** no file SHALL be written and the runners SHALL make no recorder call
