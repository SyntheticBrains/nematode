# experiment-tracking Specification (Delta)

## Purpose

Extends experiment tracking to support the plasticity evaluation protocol's multi-phase sessions, where a single evaluation run spans multiple environment objectives with metric segmentation per phase.

## MODIFIED Requirements

### Requirement: Experiment Metadata Capture

The system SHALL automatically capture comprehensive metadata for simulation runs including configuration, git state, system information, and results when experiment tracking is enabled. For plasticity evaluation runs, metadata SHALL additionally capture per-phase breakdowns and cross-phase metrics.

#### Scenario: Basic Metadata Capture

- **GIVEN** a user runs a simulation with `--track-experiment` flag
- **WHEN** the simulation completes
- **THEN** an experiment metadata JSON file SHALL be created in `experiments/{timestamp}.json`
- **AND** the file SHALL contain experiment ID, timestamp, config file path, and config hash
- **AND** the file SHALL contain git commit hash, branch name, and dirty state
- **AND** the file SHALL contain environment parameters (type, grid size, food count, satiety settings)
- **AND** the file SHALL contain brain parameters (type, architecture-specific settings)
- **AND** the file SHALL contain aggregated results (success rate, avg steps, avg reward, foraging metrics)
- **AND** the file SHALL contain system information (Python version, Qiskit version, device type)

#### Scenario: Plasticity evaluation metadata capture

- **GIVEN** a plasticity evaluation run completes
- **WHEN** experiment tracking captures metadata
- **THEN** the metadata SHALL contain a `plasticity` section with:
  - Protocol parameters (training episodes per phase, eval episodes, seeds)
  - Per-phase results (phase name, environment config hash, training metrics, eval metrics)
  - Aggregate plasticity metrics (backward forgetting, forward transfer, plasticity retention)
- **AND** the experiment type SHALL be tagged as `"plasticity_evaluation"`

#### Scenario: Metadata Capture Without Git Repository

- **GIVEN** the simulation is run outside a git repository
- **WHEN** experiment tracking captures metadata
- **THEN** git-related fields SHALL be set to null
- **AND** the experiment SHALL still be saved successfully
- **AND** a warning SHALL be logged about missing git context

#### Scenario: Config File Hash Generation

- **GIVEN** an experiment uses a configuration file
- **WHEN** metadata is captured
- **THEN** a SHA256 hash of the configuration file content SHALL be computed
- **AND** the hash SHALL be stored in the metadata
- **AND** the hash SHALL enable detection of config changes even with same filename
