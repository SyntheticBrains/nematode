# experiment-tracking Specification

## MODIFIED Requirements

### Requirement: Experiment Storage and Retrieval

The system SHALL store experiment metadata as JSON files in a self-contained per-experiment folder, with the originating config preserved alongside, and provide query capabilities for filtering, searching, and comparing experiments.

The `Experiment Storage` scenario is respecified to the folder layout the code implements. The previous scenario specified a flat `experiments/{experiment_id}.json`; `storage.py` writes `experiments/<experiment_id>/<experiment_id>.json` and discovers experiments by scanning subdirectories for a same-named JSON. This drift was surfaced by migrating `benchmark-management`'s more-accurate `Experiment Folder Structure` scenario into this capability under the same change, and is corrected here rather than landed as a second contradictory scenario.

#### Scenario: Experiment Storage

- **GIVEN** experiment metadata has been captured
- **WHEN** the metadata is saved
- **THEN** a JSON file SHALL be written to `experiments/{experiment_id}/{experiment_id}.json`
- **AND** the config file used SHALL be copied into the same folder
- **AND** the folder SHALL be self-contained for reproducibility
- **AND** the JSON SHALL be formatted with indentation for readability
- **AND** the file SHALL be written atomically to prevent corruption
- **AND** the experiments directory SHALL be created if it doesn't exist

#### Scenario: Ad-hoc Experiment Storage

- **WHEN** experiments are referenced in logbooks or other documentation
- **THEN** experiment folders MAY be stored in `artifacts/experiments/`
- **AND** this storage SHALL follow the same self-contained folder layout

## ADDED Requirements

### Requirement: Reproducibility Through Seeding

The system SHALL ensure all experiments are reproducible by automatically generating and tracking random seeds.

*Migrated verbatim from `benchmark-management` under the `remove-nematodebench` change. It was never benchmark-submission behaviour, and it is the only live-spec coverage of single-agent experiment seeding — `multi-agent` § Reproducible Seeding covers only the multi-agent case. The paired-seed statistics in `architecture-comparison-protocol` rest on this requirement.*

#### Scenario: Automatic Seed Generation

- **WHEN** an experiment is started without a seed parameter
- **THEN** the system SHALL generate a cryptographically random seed using `secrets.randbelow(2**32)`
- **AND** SHALL store the generated seed for the experiment
- **AND** SHALL use this seed consistently for all random number generation
- **AND** the seed SHALL be included in the experiment output

#### Scenario: Environment Reproducibility

- **WHEN** an environment is initialized with a seed
- **THEN** all random operations (food spawning, predator movement, initial positions) SHALL be deterministic
- **AND** running the same seed twice SHALL produce identical episode results

#### Scenario: Brain Reproducibility

- **WHEN** a brain is initialized with a seed
- **THEN** weight initialization SHALL be deterministic
- **AND** action selection (for stochastic policies) SHALL be reproducible
- **AND** PyTorch and NumPy random states SHALL be seeded consistently

#### Scenario: Per-Run Seed Tracking

- **WHEN** multiple runs are executed in an experiment
- **THEN** each run SHALL have its own seed
- **AND** per-run seeds SHALL be recorded in the experiment JSON
- **AND** any individual run SHALL be reproducible using its recorded seed

### Requirement: Convergence-Derived Metrics

The system SHALL compute learning-speed and stability metrics from convergence analysis of experiment results.

*Migrated from `benchmark-management` § Enhanced Metrics for Benchmarks under the `remove-nematodebench` change, renamed to drop the benchmark framing, and reduced to its two live scenarios. The third scenario (`Statistical Aggregation`) was scoped to NematodeBench submission roll-up and is removed with `AggregateMetrics`. The implementing module moves to `experiment/convergence.py`; the ranked-metric plateau-detection semantics it feeds remain specified by `architecture-comparison-protocol`.*

#### Scenario: Learning Speed Calculation

- **WHEN** convergence analysis is performed on experiment results
- **THEN** the system SHALL calculate episodes to reach 80% rolling success rate
- **AND** SHALL compute learning_speed = 1.0 - (episodes_to_80 / max_episodes)
- **AND** learning_speed SHALL be in range [0, 1] where 1 = instant learning

#### Scenario: Stability Metric Calculation

- **WHEN** metrics are aggregated across multiple sessions
- **THEN** the system SHALL compute stability from coefficient of variation
- **AND** stability = 1.0 - (std / mean) for success rates, clamped to [0, 1]
- **AND** higher stability indicates more consistent results
