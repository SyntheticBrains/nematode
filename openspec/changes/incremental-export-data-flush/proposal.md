# Change: Incremental Export + Per-Step Data Flush

## Why

Running 2000-episode sessions accumulates ~4-5 GB in `all_results` (per-step paths, satiety/health/temperature histories, food history) and ~1 GB in `tracking_data.brain_data` (deep-copied `BrainHistoryData` per episode). This data is held in RAM until session end, preventing parallel execution — 4 sessions consume ~48 GB, exceeding a 32 GB MacBook's RAM.

An audit of all 16 post-loop consumers reveals that only 3 need per-step data, and all 3 can work incrementally:

1. `_export_path_data` — writes `(run, step, x, y)` CSV rows from `result.path`
2. `_export_detailed_tracking_data` — writes step-by-step brain history to CSV
3. Chemotaxis validation — computes metrics from `result.path` + `result.food_history`

The remaining 13 consumers use only scalar fields (steps, reward, success, foods_collected, etc.) or last-values from per-step data (`health_history[-1]`, `max(satiety_history)`).

## What Changes

### 1. Snapshot Fields on SimulationResult

Add scalar snapshot fields to `SimulationResult` that capture values from per-step histories before they are flushed:

- `path_length: int | None` — preserves `len(path)` after path is flushed
- `max_satiety: float | None` — `max(satiety_history)`, used by satiety plot y-axis
- `final_health: float | None` — `health_history[-1]`, used by health plot and summary
- `max_health: float | None` — `max(health_history)`, used by health plot y-axis

### 2. BrainDataSnapshot Model

Replace the full `deepcopy(agent.brain.history_data)` (~0.5 MB per episode, 13 unbounded lists) with a `BrainDataSnapshot` that stores only the last value of each history attribute (~500 bytes per episode). All session-level plots and exports that use `tracking_data.brain_data` only access the last value per run.

### 3. Incremental Path CSV Writer

Write path data to CSV incrementally after each episode completes, then flush `result.path` from memory. Creates the CSV file before the main loop and appends rows per episode. The post-loop `_export_path_data` is skipped since data was already written.

### 4. Incremental Detailed Tracking Writer

Write step-by-step brain history data to CSV incrementally after each episode, before replacing the full `BrainHistoryData` with a `BrainDataSnapshot`. Uses an `IncrementalDetailedTrackingWriter` class that lazily opens one CSV per brain history key. The post-loop `_export_detailed_tracking_data` is skipped.

### 5. Per-Episode Chemotaxis Metrics

When `--track-experiment` is enabled, compute `ChemotaxisMetrics` for each episode immediately (requires `result.path` + `result.food_history`), storing only the small metrics struct. Pass pre-computed metrics to `aggregate_results_metadata()` at session end instead of recomputing from already-flushed data.

### 6. Per-Episode Data Flush

After incremental writes and snapshot extraction, flush heavy fields from each episode's `SimulationResult`:

- `result.path = []`
- `result.food_history = None`
- `result.satiety_history = None`
- `result.health_history = None`
- `result.temperature_history = None`

And store lightweight `EpisodeTrackingData` (only `foods_collected` + `distance_efficiencies`).

### 7. Post-Loop Consumer Updates

Update all consumers that accessed per-step data to use snapshot fields with fallback:

- `_export_main_results`: `len(result.path)` → `result.path_length`
- `summary()`: `result.health_history[-1]` → `result.final_health`
- `plot_results()`: `max(satiety_history)` → `result.max_satiety`, `health_history[-1]` → `result.final_health`, `max(health_history)` → `result.max_health`
- `plot_tracking_data_by_session()`: handle `BrainDataSnapshot` type
- `_export_session_tracking_data()`: handle `BrainDataSnapshot` type
- `export_tracking_data_to_csv()`: key discovery from snapshot
- `manage_simulation_halt()`: close incremental file handles on interrupt

## Impact

- **Memory savings**: ~4-5 GB per session (from ~12 GB to ~2-3 GB)
- **Enables**: 8-12 parallel sessions on 32 GB RAM (vs. current limit of ~3-4)
- **CSV output**: Identical to pre-change (same format, same data)
- **Risk**: Medium — touches 7 files but each change is straightforward with clear fallback patterns

## Files

| File | Changes |
|------|---------|
| `quantumnematode/report/dtypes.py` | Snapshot fields, `BrainDataSnapshot`, `TrackingData` type |
| `quantumnematode/report/csv_export.py` | Incremental writers, snapshot handling, skip params |
| `quantumnematode/report/summary.py` | Snapshot fallbacks |
| `quantumnematode/report/plots.py` | `BrainDataSnapshot` handling |
| `quantumnematode/experiment/tracker.py` | Pre-computed chemotaxis parameter |
| `scripts/run_simulation.py` | Loop restructuring, snapshots, flush |
| `quantumnematode/utils/interrupt_handler.py` | Incremental file cleanup |
