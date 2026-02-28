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
- **AND** experiment metadata JSON file I/O SHALL be performed after simulation completes
- **NOTE** This constraint applies to experiment metadata capture only; incremental CSV data exports (paths, detailed tracking) are written per-episode by design (see Per-Episode Data Lifecycle)

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

### Requirement: Predator Experiment Metadata Capture

The experiment tracking system SHALL capture comprehensive predator configuration and performance data to enable analysis of multi-objective learning dynamics.

#### Scenario: Predator Configuration in Experiment Metadata

- **GIVEN** an experiment run with `predators.enabled: true`
- **WHEN** experiment metadata is recorded
- **THEN** the following predator configuration SHALL be captured:
  - `predators_enabled` (boolean)
  - `num_predators` (integer)
  - `predator_speed` (float)
  - `predator_movement_pattern` (string)
  - `predator_detection_radius` (integer)
  - `predator_kill_radius` (integer)
  - `predator_gradient_decay_constant` (float)
  - `predator_gradient_strength` (float)
- **AND** this SHALL be stored in experiment JSON metadata
- **NOTE**: The proximity penalty is stored in the reward configuration, not environment metadata

#### Scenario: Predator Metrics in Experiment Results

- **GIVEN** an experiment with predators enabled
- **WHEN** per-episode results are recorded
- **THEN** each episode SHALL include predator metrics:
  - `predator_encounters` (integer)
  - `successful_evasions` (integer)
  - `predator_deaths` (boolean or integer)
  - `foods_collected_before_death` (integer, if applicable)
- **AND** these SHALL be included in the episode performance data

#### Scenario: Experiment Metadata When Predators Disabled

- **GIVEN** an experiment with `predators.enabled: false` or predators not configured
- **WHEN** experiment metadata is recorded
- **THEN** `predators_enabled` SHALL be false
- **AND** other predator configuration fields MAY be omitted or null
- **AND** predator metrics SHALL be omitted or null in episode results
- **AND** this SHALL maintain backward compatibility with pre-predator experiments

### Requirement: Predator-Specific Experiment Queries

The experiment tracking system SHALL support querying and filtering experiments by predator configuration and performance metrics.

#### Scenario: Query Experiments with Predators Enabled

- **GIVEN** a set of historical experiments, some with predators enabled
- **WHEN** querying for predator-enabled experiments
- **THEN** the system SHALL filter by `predators_enabled: true`
- **AND** SHALL return only experiments where predators were active
- **AND** SHALL include all predator configuration in results

#### Scenario: Query by Predator Count

- **GIVEN** experiments with varying predator counts (0, 2, 3, 5)
- **WHEN** querying for experiments with specific num_predators
- **THEN** the system SHALL filter experiments by exact num_predators value
- **AND** SHALL enable analysis of difficulty scaling with predator numbers

#### Scenario: Query by Predator Death Rate

- **GIVEN** completed experiments with predator performance data
- **WHEN** querying for experiments with high predator death rate (e.g., >50%)
- **THEN** the system SHALL calculate death_by_predator percentage across episodes
- **AND** SHALL return experiments meeting the threshold
- **AND** SHALL enable identification of challenging predator configurations

#### Scenario: Compare Predator vs Non-Predator Experiments

- **GIVEN** experiments with same brain and environment but varying predator_enabled
- **WHEN** comparing experiments
- **THEN** the system SHALL support side-by-side comparison
- **AND** SHALL show impact of predators on:
  - Success rate
  - Average episode length
  - Foods collected
  - Total reward
- **AND** SHALL highlight performance degradation or adaptation patterns

### Requirement: Predator Learning Curve Visualization

The experiment tracking system SHALL support visualization of predator-specific learning curves showing adaptation over training.

#### Scenario: Predator Encounter Rate Over Time

- **GIVEN** an experiment with predators enabled and training over 1000 episodes
- **WHEN** learning curve visualization is generated
- **THEN** a plot SHALL show `predator_encounters` per episode over time
- **AND** SHALL enable analysis of whether agent learns to avoid predators
- **AND** decreasing encounters SHALL indicate improved avoidance behavior

#### Scenario: Evasion Success Rate Over Time

- **GIVEN** an experiment tracking evasions across episodes
- **WHEN** learning curve visualization is generated
- **THEN** a plot SHALL show `successful_evasions / predator_encounters` ratio over time
- **AND** increasing ratio SHALL indicate improved escape skills
- **AND** SHALL be smoothed over rolling window (e.g., 100 episodes)

#### Scenario: Survival Rate vs Foraging Trade-off

- **GIVEN** an experiment with both predator and foraging metrics
- **WHEN** trade-off visualization is generated
- **THEN** a plot SHALL show:
  - X-axis: foods collected per episode
  - Y-axis: survival rate (episodes without predator death)
- **AND** SHALL show Pareto frontier of optimal trade-offs
- **AND** SHALL indicate whether agent learns balanced strategy or specializes

#### Scenario: Predator Death Rate Over Training

- **GIVEN** an experiment with episode-level predator death tracking
- **WHEN** learning curve visualization is generated
- **THEN** a plot SHALL show percentage of episodes ending in predator death over time
- **AND** decreasing rate SHALL indicate successful learning
- **AND** SHALL be compared against starvation rate to show termination cause shifts

### Requirement: Predator Experiment Comparison and Benchmarking

The experiment tracking system SHALL enable systematic comparison of experiments with different predator configurations to identify optimal difficulty curves.

#### Scenario: Compare Different Predator Counts

- **GIVEN** experiments with num_predators ranging from 1 to 5
- **WHEN** comparison is performed
- **THEN** the system SHALL show how success rate changes with num_predators
- **AND** SHALL identify the count at which learning becomes infeasible
- **AND** SHALL support curriculum design (start with fewer predators, increase gradually)

#### Scenario: Compare Detection Radius Impact

- **GIVEN** experiments with detection_radius values of 5, 8, and 12
- **WHEN** comparison is performed
- **THEN** the system SHALL show how detection radius affects:
  - Encounter frequency
  - Evasion success rate
  - Overall survival rate
- **AND** SHALL identify optimal radius for learning signal (not too easy, not impossible)

#### Scenario: Compare Proximity Penalty Effectiveness

- **GIVEN** experiments with proximity_penalty of 0.0, -0.05, -0.1, and -0.2
- **WHEN** comparison is performed
- **THEN** the system SHALL analyze whether proximity penalty improves:
  - Proactive avoidance (lower encounter rate)
  - Faster learning (fewer episodes to convergence)
  - Overall performance (higher success rate)
- **AND** SHALL identify whether penalty is necessary or redundant with gradient signal

### Requirement: Experiment Export with Predator Data

The experiment tracking system SHALL include predator metrics in CSV exports and data dumps for external analysis.

#### Scenario: CSV Export with Predator Columns

- **GIVEN** an experiment with predators enabled
- **WHEN** CSV export is generated
- **THEN** the CSV SHALL include columns:
  - `predator_encounters`
  - `successful_evasions`
  - `predator_deaths` (boolean)
  - `foods_collected` (existing, but crucial for predator context)
  - `termination_reason` (showing "predator" when applicable)
- **AND** columns SHALL be empty/null when predators disabled (backward compatibility)

#### Scenario: JSON Export with Full Predator Configuration

- **GIVEN** an experiment with predators enabled
- **WHEN** JSON export is generated
- **THEN** the JSON SHALL include full `predator_config` object with all parameters
- **AND** SHALL include per-episode `predator_metrics` object
- **AND** SHALL be machine-readable for automated analysis pipelines

#### Scenario: Export Filtering by Predator Status

- **GIVEN** a dataset containing both predator and non-predator experiments
- **WHEN** export is filtered to predator-only experiments
- **THEN** the export SHALL include only experiments with `predators_enabled: true`
- **AND** SHALL include all predator-specific columns
- **AND** SHALL enable focused analysis on survival-foraging scenarios

### Requirement: Per-Episode Data Lifecycle

SimulationResult per-step data (path, food_history, satiety_history, health_history, temperature_history) SHALL be flushed after each episode to reduce memory, with scalar snapshots preserved for post-loop consumers.

#### Scenario: Snapshot Extraction Before Flush

- **GIVEN** a completed episode with satiety_history, health_history, and path data
- **WHEN** the episode data is processed after the main step loop
- **THEN** SimulationResult SHALL have `path_length` set to `len(path)`
- **AND** `max_satiety` set to `max(satiety_history)` if satiety_history exists
- **AND** `final_health` set to `health_history[-1]` if health_history exists
- **AND** `max_health` set to `max(health_history)` if health_history exists
- **AND** per-step fields (path, food_history, satiety_history, health_history, temperature_history) SHALL be cleared

#### Scenario: Incremental Path CSV Export

- **GIVEN** a simulation session with N episodes
- **WHEN** each episode completes
- **THEN** path data for that episode SHALL be written to `paths.csv` incrementally
- **AND** the final CSV SHALL be identical to the batch-written version

#### Scenario: Incremental Detailed Brain Tracking Export

- **GIVEN** a simulation session with brain tracking enabled
- **WHEN** each episode completes
- **THEN** step-by-step brain history data SHALL be written to `detailed/*.csv` incrementally
- **AND** the full BrainHistoryData SHALL be replaced with a BrainDataSnapshot (last value per attribute)

#### Scenario: Per-Episode Chemotaxis Metrics

- **GIVEN** a simulation with `--track-experiment` enabled and food_history present
- **WHEN** each episode completes
- **THEN** ChemotaxisMetrics SHALL be computed from that episode's path and food_history
- **AND** the pre-computed metrics SHALL be passed to `aggregate_results_metadata` at session end
- **AND** results SHALL be identical to batch computation
- **AND** the post-convergence validation logic (biological comparison, validation level) SHALL execute regardless of whether metrics were pre-computed or computed from results

### Requirement: Post-Loop Consumer Snapshot Fallback

All post-loop consumers that previously accessed per-step data SHALL use snapshot fields when per-step data has been flushed (is None or empty).

#### Scenario: Summary With Flushed Data

- **GIVEN** all_results with flushed health_history (None) but populated final_health
- **WHEN** `summary()` is called
- **THEN** it SHALL use `result.final_health` instead of `result.health_history[-1]`

#### Scenario: Plot Results With Flushed Data

- **GIVEN** all_results with flushed satiety_history and health_history
- **WHEN** `plot_results()` generates satiety and health plots
- **THEN** it SHALL use `result.max_satiety`, `result.final_health`, and `result.max_health` snapshots
- **AND** plots SHALL be visually identical to pre-change output

#### Scenario: Single-Run Progression Plots With Flushed Data

- **GIVEN** all_results with flushed satiety_history and health_history
- **WHEN** `plot_results()` generates single-run progression plots for predator environments
- **THEN** it SHALL use the last run's per-step histories preserved from the main loop
- **AND** the satiety and health progression plots SHALL be generated identically to pre-change output

#### Scenario: Session Tracking Plots With BrainDataSnapshot

- **GIVEN** tracking_data.brain_data containing BrainDataSnapshot instances
- **WHEN** `plot_tracking_data_by_session()` is called
- **THEN** it SHALL read last values from `snapshot.last_values[key]`
- **AND** plots SHALL be identical to those generated from full BrainHistoryData

#### Scenario: Interrupt Handler With Incremental Files

- **GIVEN** incremental CSV writers are open during the simulation loop
- **WHEN** a KeyboardInterrupt occurs
- **THEN** incremental CSV file handles SHALL be closed cleanly
- **AND** partial results export SHALL work correctly using snapshot fields
