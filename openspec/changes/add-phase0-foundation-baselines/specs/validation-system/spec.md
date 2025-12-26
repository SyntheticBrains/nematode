# validation-system Specification Delta

## ADDED Requirements

### Requirement: Chemotaxis Index Calculation
The system SHALL calculate chemotaxis index (CI) from agent trajectories using the standard formula from C. elegans literature.

#### Scenario: Standard CI Calculation
**Given** an episode trajectory with agent positions and food positions
**When** chemotaxis index is calculated
**Then** the system SHALL compute CI = (N_attractant - N_control) / N_total
**And** N_attractant SHALL be steps within attractant_zone_radius of any food (default 5.0)
**And** N_control SHALL be steps outside the attractant zone
**And** N_total SHALL be total episode steps
**And** CI SHALL be in range [-1, 1] where 1 = perfect attraction, -1 = perfect avoidance

#### Scenario: Multi-Food Environment CI
**Given** an environment with multiple food sources
**When** determining if a position is in the attractant zone
**Then** the system SHALL check distance to ALL food sources
**And** SHALL count as attractant if within radius of ANY food
**And** SHALL handle dynamic food spawning correctly

#### Scenario: Empty Episode Handling
**Given** an episode with zero steps (immediate termination)
**When** chemotaxis index is calculated
**Then** the system SHALL return CI = 0.0 (neutral)
**And** SHALL NOT divide by zero
**And** SHALL flag the result as potentially unreliable

### Requirement: Additional Chemotaxis Metrics
The system SHALL compute supplementary metrics that provide deeper insight into agent navigation behavior.

#### Scenario: Time in Attractant Zone
**Given** an episode trajectory
**When** chemotaxis metrics are calculated
**Then** the system SHALL compute time_in_attractant = N_attractant / N_total
**And** SHALL be in range [0, 1]
**And** SHALL indicate fraction of episode spent near food

#### Scenario: Approach Frequency
**Given** an episode trajectory with gradient information
**When** chemotaxis metrics are calculated
**Then** the system SHALL compute how often agent moves toward the gradient
**And** approach_frequency = (steps moving toward food) / N_total
**And** SHALL use gradient direction to determine "toward food"

#### Scenario: Path Efficiency
**Given** an episode trajectory from start to first food collection
**When** chemotaxis metrics are calculated
**Then** the system SHALL compute path_efficiency = optimal_distance / actual_distance
**And** optimal_distance SHALL be Euclidean distance from start to nearest food
**And** actual_distance SHALL be sum of step distances
**And** SHALL be in range (0, 1] where 1 = optimal path

### Requirement: Literature Dataset Integration
The system SHALL load and manage published C. elegans chemotaxis data from peer-reviewed literature.

#### Scenario: Dataset Loading
**Given** a JSON file at `data/chemotaxis/literature_ci_values.json`
**When** the validation system is initialized
**Then** the system SHALL load published CI values
**And** SHALL parse citation information
**And** SHALL parse experimental conditions
**And** SHALL parse CI ranges (min, typical, max)

#### Scenario: Dataset Structure
**Given** the literature dataset
**When** data is accessed
**Then** each entry SHALL include citation (author, year, journal)
**And** SHALL include attractant type (e.g., diacetyl, bacteria)
**And** SHALL include wild-type CI value
**And** SHALL include CI range for biological variability
**And** SHALL include experimental conditions

#### Scenario: Multiple Sources
**Given** multiple literature sources with different CI values
**When** validation is performed
**Then** the system SHALL support selecting which source to compare against
**And** SHALL default to the most relevant source (food chemotaxis)
**And** SHALL document source selection in validation output

### Requirement: Biological Validation Benchmark
The system SHALL compare agent behavior to biological data and produce validation results.

#### Scenario: Validation Against Literature
**Given** calculated agent chemotaxis metrics
**When** validation is requested
**Then** the system SHALL compare agent CI to biological CI range
**And** SHALL return whether agent falls within biological range
**And** SHALL return the difference between agent CI and biological CI
**And** SHALL indicate validation level (minimum, target, excellent)

#### Scenario: Validation Thresholds
**Given** the validation system
**When** interpreting results
**Then** minimum threshold SHALL be CI >= 0.4
**And** target threshold SHALL be CI >= 0.6
**And** excellent threshold SHALL be CI >= 0.75
**And** thresholds SHALL be configurable via dataset

#### Scenario: Validation Result Output
**Given** a completed validation
**When** results are reported
**Then** the system SHALL output agent_ci value
**And** SHALL output biological_ci_range (min, max)
**And** SHALL output matches_biology boolean
**And** SHALL output validation_level (minimum/target/excellent/none)
**And** SHALL output the literature source used

### Requirement: Experiment Tracking Integration
The system SHALL integrate chemotaxis validation with existing experiment tracking.

#### Scenario: Metadata Extension
**Given** an experiment with chemotaxis calculation enabled
**When** experiment metadata is saved
**Then** the metadata SHALL include chemotaxis_index field
**And** SHALL include additional metrics (time_in_attractant, approach_frequency, path_efficiency)
**And** SHALL include validation_result if validation was performed

#### Scenario: CLI Integration
**Given** a user running simulation with `--validate-chemotaxis` flag
**When** the simulation completes
**Then** the system SHALL calculate chemotaxis metrics
**And** SHALL compare against literature data
**And** SHALL display validation results in terminal output
**And** SHALL include validation in experiment metadata

#### Scenario: Backward Compatibility
**Given** an experiment run without chemotaxis validation
**When** experiment metadata is saved
**Then** chemotaxis fields SHALL be null or absent
**And** existing experiment query tools SHALL continue to work
**And** no errors SHALL occur for missing chemotaxis data
