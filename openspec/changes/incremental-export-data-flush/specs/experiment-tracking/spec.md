## CHANGED Requirements

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
