# experiment-tracking Delta

## ADDED Requirements

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
- **AND** other predator configuration fields SHALL be null
- **AND** predator metrics SHALL be null in episode results
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
- **AND** SHALL be calculated as: (predator_deaths / total_episodes) × 100
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
- **THEN** a plot SHALL show the evasion success rate over time where:
  - Rate is calculated as `successful_evasions / predator_encounters`
  - When `predator_encounters == 0`, the rate SHALL be treated as "N/A" (or null)
  - N/A values SHALL be omitted from numeric plotting
  - N/A values SHALL be excluded from rolling-window smoothing computations
  - N/A values SHALL NOT affect adjacent smoothed values in the rolling window
  - Visualizations SHALL render gaps or explicit "N/A" markers for these points
- **AND** increasing ratio SHALL indicate improved escape skills
- **AND** SHALL be smoothed over rolling window (e.g., 100 episodes) with N/A points excluded from window computation

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
