# validation-system Specification Delta

## ADDED Requirements

### Requirement: Chemotaxis Index Calculation
The system SHALL calculate chemotaxis index (CI) from agent trajectories using the standard formula from C. elegans literature.

#### Scenario: Standard CI Calculation
- **WHEN** chemotaxis index is calculated from an episode trajectory with agent positions and food positions
- **THEN** the system SHALL compute CI = (N_attractant - N_control) / N_total
- **AND** N_attractant SHALL be steps within attractant_zone_radius of any food (default 5.0)
- **AND** N_control SHALL be steps outside the attractant zone
- **AND** N_total SHALL be total episode steps
- **AND** CI SHALL be in range [-1, 1] where 1 = perfect attraction, -1 = perfect avoidance

#### Scenario: Multi-Food Environment CI
- **WHEN** determining if a position is in the attractant zone in an environment with multiple food sources
- **THEN** the system SHALL check distance to ALL food sources
- **AND** SHALL count as attractant if within radius of ANY food
- **AND** SHALL handle dynamic food spawning correctly

#### Scenario: Empty Episode Handling
- **WHEN** chemotaxis index is calculated for an episode with zero steps (immediate termination)
- **THEN** the system SHALL return CI = 0.0 (neutral)
- **AND** SHALL NOT divide by zero
- **AND** SHALL flag the result as potentially unreliable

### Requirement: Additional Chemotaxis Metrics
The system SHALL compute supplementary metrics that provide deeper insight into agent navigation behavior.

#### Scenario: Time in Attractant Zone
- **WHEN** chemotaxis metrics are calculated from an episode trajectory
- **THEN** the system SHALL compute time_in_attractant = N_attractant / N_total
- **AND** SHALL be in range [0, 1]
- **AND** SHALL indicate fraction of episode spent near food

#### Scenario: Approach Frequency
- **WHEN** chemotaxis metrics are calculated from an episode trajectory with gradient information
- **THEN** the system SHALL compute how often agent moves toward the gradient
- **AND** approach_frequency = (steps moving toward food) / N_total
- **AND** SHALL use gradient direction to determine "toward food"

#### Scenario: Path Efficiency
- **WHEN** chemotaxis metrics are calculated from an episode trajectory from start to first food collection
- **THEN** the system SHALL compute path_efficiency = optimal_distance / actual_distance
- **AND** optimal_distance SHALL be Euclidean distance from start to nearest food
- **AND** actual_distance SHALL be sum of step distances
- **AND** SHALL be in range (0, 1] where 1 = optimal path

### Requirement: Literature Dataset Integration
The system SHALL load and manage published C. elegans chemotaxis data from peer-reviewed literature.

#### Scenario: Dataset Loading
- **WHEN** the validation system is initialized with a JSON file at `data/chemotaxis/literature_ci_values.json`
- **THEN** the system SHALL load published CI values
- **AND** SHALL parse citation information
- **AND** SHALL parse experimental conditions
- **AND** SHALL parse CI ranges (min, typical, max)

#### Scenario: Dataset Structure
- **WHEN** data is accessed from the literature dataset
- **THEN** each entry SHALL include citation (author, year, journal)
- **AND** SHALL include attractant type (e.g., diacetyl, bacteria)
- **AND** SHALL include wild-type CI value
- **AND** SHALL include CI range for biological variability
- **AND** SHALL include experimental conditions

#### Scenario: Multiple Sources
- **WHEN** validation is performed with multiple literature sources having different CI values
- **THEN** the system SHALL support selecting which source to compare against
- **AND** SHALL default to the most relevant source (food chemotaxis)
- **AND** SHALL document source selection in validation output

### Requirement: Biological Validation Benchmark
The system SHALL compare agent behavior to biological data and produce validation results.

#### Scenario: Validation Against Literature
- **WHEN** validation is requested with calculated agent chemotaxis metrics
- **THEN** the system SHALL compare agent CI to biological CI range
- **AND** SHALL return whether agent falls within biological range
- **AND** SHALL return the difference between agent CI and biological CI
- **AND** SHALL indicate validation level (minimum, target, excellent)

#### Scenario: Validation Thresholds
- **WHEN** interpreting validation results
- **THEN** minimum threshold SHALL be CI >= 0.4
- **AND** target threshold SHALL be CI >= 0.6
- **AND** excellent threshold SHALL be CI >= 0.75
- **AND** thresholds SHALL be configurable via dataset

#### Scenario: Validation Result Output
- **WHEN** validation results are reported
- **THEN** the system SHALL output agent_ci value
- **AND** SHALL output biological_ci_range (min, max)
- **AND** SHALL output matches_biology boolean
- **AND** SHALL output validation_level (minimum/target/excellent/none)
- **AND** SHALL output the literature source used

### Requirement: Experiment Tracking Integration
The system SHALL integrate chemotaxis validation with existing experiment tracking.

#### Scenario: Metadata Extension
- **WHEN** experiment metadata is saved for an experiment with chemotaxis calculation enabled
- **THEN** the metadata SHALL include chemotaxis_index field
- **AND** SHALL include additional metrics (time_in_attractant, approach_frequency, path_efficiency)
- **AND** SHALL include validation_result if validation was performed

#### Scenario: CLI Integration
- **WHEN** a simulation completes with `--validate-chemotaxis` flag
- **THEN** the system SHALL calculate chemotaxis metrics
- **AND** SHALL compare against literature data
- **AND** SHALL display validation results in terminal output
- **AND** SHALL include validation in experiment metadata

#### Scenario: Backward Compatibility
- **WHEN** experiment metadata is saved for an experiment run without chemotaxis validation
- **THEN** chemotaxis fields SHALL be null or absent
- **AND** existing experiment query tools SHALL continue to work
- **AND** no errors SHALL occur for missing chemotaxis data
