# experiment-tracking Specification

## Purpose
This specification defines the experiment metadata capture, storage, and query system. It enables reproducible research by automatically tracking configuration, git state, system information, and results for simulation runs.

## Requirements
### Requirement: Experiment Metadata Capture
The system SHALL automatically capture comprehensive metadata for simulation runs including configuration, git state, system information, and results when experiment tracking is enabled.

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

### Requirement: Experiment Storage and Retrieval
The system SHALL store experiment metadata as JSON files and provide query capabilities for filtering, searching, and comparing experiments.

#### Scenario: Experiment Storage
- **GIVEN** experiment metadata has been captured
- **WHEN** the metadata is saved
- **THEN** a JSON file SHALL be written to `experiments/{experiment_id}.json`
- **AND** the JSON SHALL be formatted with indentation for readability
- **AND** the file SHALL be written atomically to prevent corruption
- **AND** the experiments directory SHALL be created if it doesn't exist

#### Scenario: Query Experiments by Environment Type
- **GIVEN** multiple experiments are stored with different environment types
- **WHEN** a user queries for experiments with `environment_type="dynamic"`
- **THEN** only experiments using dynamic foraging environments SHALL be returned
- **AND** experiments SHALL be sorted by timestamp (most recent first)

#### Scenario: Query Experiments by Brain Type
- **GIVEN** multiple experiments are stored with different brain architectures
- **WHEN** a user queries for experiments with `brain_type="modular"`
- **THEN** only experiments using modular quantum brain SHALL be returned

#### Scenario: Query Experiments by Success Rate Threshold
- **GIVEN** multiple experiments with varying success rates
- **WHEN** a user queries for experiments with `min_success_rate=0.8`
- **THEN** only experiments with success rate >= 80% SHALL be returned

#### Scenario: Compare Two Experiments
- **GIVEN** two experiment IDs
- **WHEN** a comparison is requested
- **THEN** a structured diff SHALL be generated showing:
  - Configuration differences (environment, brain, hyperparameters)
  - Results comparison (success rate, avg steps, foraging metrics)
  - System differences (versions, device types)
- **AND** the comparison SHALL highlight which experiment performed better on each metric

### Requirement: Automatic Tracking Integration
The system SHALL integrate experiment tracking seamlessly with the existing simulation workflow without disrupting normal operation.

#### Scenario: Opt-in Experiment Tracking
- **GIVEN** a user runs a simulation without `--track-experiment` flag
- **WHEN** the simulation completes
- **THEN** no experiment metadata SHALL be saved
- **AND** simulation SHALL run exactly as before (backward compatible)

#### Scenario: Tracking with Per-Run Data
- **GIVEN** a user runs a simulation with both `--track-experiment` and `--track-per-run` flags
- **WHEN** the simulation completes
- **THEN** experiment metadata SHALL reference the exports directory path
- **AND** the metadata SHALL include a link to detailed per-run data
- **AND** both tracking mechanisms SHALL work together without interference

#### Scenario: Tracking Overhead
- **GIVEN** a simulation with experiment tracking enabled
- **WHEN** metadata capture and storage occurs
- **THEN** the overhead SHALL be less than 100ms
- **AND** the overhead SHALL not impact simulation performance
- **AND** file I/O SHALL be performed after simulation completes

### Requirement: Git Context Capture
The system SHALL capture git repository state to enable reproducibility and verification of experimental results.

#### Scenario: Clean Git State
- **GIVEN** a git repository with no uncommitted changes
- **WHEN** experiment metadata is captured
- **THEN** git_commit SHALL contain the current commit hash
- **AND** git_branch SHALL contain the current branch name
- **AND** git_dirty SHALL be false

#### Scenario: Dirty Git State
- **GIVEN** a git repository with uncommitted changes
- **WHEN** experiment metadata is captured
- **THEN** git_commit SHALL contain the current commit hash
- **AND** git_branch SHALL contain the current branch name
- **AND** git_dirty SHALL be true
- **AND** a warning SHALL be logged recommending committing changes for reproducibility

#### Scenario: Detached HEAD State
- **GIVEN** a git repository in detached HEAD state
- **WHEN** experiment metadata is captured
- **THEN** git_commit SHALL contain the current commit hash
- **AND** git_branch SHALL be null or "HEAD"
- **AND** a warning SHALL be logged about detached HEAD state

### Requirement: System Information Capture
The system SHALL capture relevant system and dependency version information to support reproducibility across different environments.

#### Scenario: Python and Package Versions
- **GIVEN** an experiment is being tracked
- **WHEN** system metadata is captured
- **THEN** Python version SHALL be recorded (e.g., "3.12.0")
- **AND** Qiskit version SHALL be recorded
- **AND** PyTorch version SHALL be recorded if torch is installed
- **AND** version information SHALL be obtained from installed packages

#### Scenario: Device Type Detection
- **GIVEN** an experiment is being tracked
- **WHEN** system metadata is captured
- **THEN** device_type SHALL be "cpu" for CPU-only simulations
- **AND** device_type SHALL be "gpu" for GPU-accelerated simulations
- **AND** device_type SHALL be "qpu" for quantum hardware execution
- **AND** qpu_backend SHALL be recorded for QPU executions (e.g., "ibm_brisbane")

### Requirement: Experiment Query CLI
The system SHALL provide a command-line interface for querying, comparing, and analyzing stored experiments.

#### Scenario: List All Experiments
- **GIVEN** multiple experiments are stored
- **WHEN** a user runs `scripts/experiment_query.py list`
- **THEN** a table SHALL be displayed with experiment ID, config, success rate, and timestamp
- **AND** experiments SHALL be sorted by timestamp (most recent first)
- **AND** the table SHALL be formatted for terminal display

#### Scenario: Filter Experiments
- **GIVEN** multiple experiments are stored
- **WHEN** a user runs `scripts/experiment_query.py list --env-type dynamic --brain-type modular`
- **THEN** only matching experiments SHALL be displayed
- **AND** the filter criteria SHALL be shown in the output

#### Scenario: View Experiment Details
- **GIVEN** an experiment ID
- **WHEN** a user runs `scripts/experiment_query.py show {experiment_id}`
- **THEN** full experiment metadata SHALL be displayed in a readable format
- **AND** configuration parameters SHALL be grouped by category
- **AND** results SHALL be highlighted with formatting

#### Scenario: Compare Experiments
- **GIVEN** two experiment IDs
- **WHEN** a user runs `scripts/experiment_query.py compare {id1} {id2}`
- **THEN** a side-by-side comparison SHALL be displayed
- **AND** differences SHALL be highlighted
- **AND** performance improvements/regressions SHALL be clearly indicated
