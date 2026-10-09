## MODIFIED Requirements

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
- **AND** the post-convergence chemotaxis summary (the index and its validation level) SHALL be computed regardless of whether metrics were pre-computed or computed from results

## ADDED Requirements

### Requirement: No literature verdict on the simulated chemotaxis index

The experiment tracker SHALL record the simulated chemotaxis index and its validation level, a banding
of the index at 0.4, 0.6 and 0.75. It SHALL NOT record a literature CI range, a typical literature CI,
a literature citation or a `matches_biology` verdict, because the simulated index is a time-in-zone
fraction and every published chemotaxis index it could be set against is an endpoint count of worms.
Those fields SHALL remain readable, and SHALL be `None` on new records.

#### Scenario: A tracked run records no literature verdict

- **WHEN** a tracked run with food history completes
- **THEN** its record SHALL carry the post-convergence chemotaxis index and validation level
- **AND** its literature range, typical value, citation and `matches_biology` SHALL be `None`

#### Scenario: Older records still load

- **WHEN** an experiment record written before this change, with those fields set, is loaded
- **THEN** it SHALL load with its recorded values

### Requirement: A verified chemotaxis reference set

The chemotaxis reference file SHALL list only values verified in the cited paper, each with its correct
citation, what the paper assayed, the index it reports, and whether the value was stated in the text
or read from a figure.

#### Scenario: Every entry is traceable

- **WHEN** the reference file is loaded
- **THEN** every entry SHALL carry a citation, an assay description and a value source of `text` or
  `figure`
