# Tasks: Incremental Export + Per-Step Data Flush

## Overview

Reduce per-session memory from ~12 GB to ~2-3 GB by writing heavy per-step data to CSV incrementally and flushing it from memory after each episode. Scalar snapshots are preserved for post-loop consumers (plots, summary, exports).

______________________________________________________________________

## 1. Data Models

### 1.1 Add Snapshot Fields to SimulationResult

- [ ] Add `path_length: int | None = None` to `SimulationResult`
- [ ] Add `max_satiety: float | None = None` to `SimulationResult`
- [ ] Add `final_health: float | None = None` to `SimulationResult`
- [ ] Add `max_health: float | None = None` to `SimulationResult`

**File:** `packages/quantum-nematode/quantumnematode/report/dtypes.py`

### 1.2 Add BrainDataSnapshot Model

- [ ] Create `BrainDataSnapshot(BaseModel)` with `last_values: dict[str, Any]`
- [ ] Update `TrackingData.brain_data` type to `dict[TrackingRunIndex, BrainHistoryData | BrainDataSnapshot]`

**File:** `packages/quantum-nematode/quantumnematode/report/dtypes.py`

______________________________________________________________________

## 2. Incremental CSV Writers

### 2.1 Incremental Path Writer

- [ ] Add `create_path_csv_writer(filepath) -> tuple[IO, csv.DictWriter]` — opens file, writes header
- [ ] Add `write_path_data_row(writer, result)` — writes one run's path rows
- [ ] Add `skip_path_data` parameter to `export_simulation_results_to_csv()` to skip `_export_path_data`

**File:** `packages/quantum-nematode/quantumnematode/report/csv_export.py`

### 2.2 Incremental Detailed Tracking Writer

- [ ] Create `IncrementalDetailedTrackingWriter` class:
  - `__init__(data_dir, file_prefix)` — stores config, defers file creation
  - `write_run(run, brain_history)` — lazily opens CSV files on first call (determines structure from data), writes step-by-step rows
  - `close()` — closes all file handles
  - Handles dict, `ActionData`, and scalar data types (reuse logic from `_export_detailed_tracking_data`)
- [ ] Add `skip_detailed` parameter to `export_tracking_data_to_csv()` to skip `_export_detailed_tracking_data`

**File:** `packages/quantum-nematode/quantumnematode/report/csv_export.py`

______________________________________________________________________

## 3. Main Loop Restructuring

### 3.1 Initialize Incremental Writers Before Loop

- [ ] Create `data_dir` before the loop
- [ ] Open path CSV writer via `create_path_csv_writer`
- [ ] Create `IncrementalDetailedTrackingWriter` instance
- [ ] Initialize `chemotaxis_metrics_per_run: list[tuple[int, ChemotaxisMetrics]] = []`

**File:** `scripts/run_simulation.py`

### 3.2 Per-Episode Processing (after result append, before environment reset)

Execute in this order (critical — some operations need full data):

- [ ] Write incremental path CSV (needs `result.path`)
- [ ] Write incremental detailed brain tracking CSV (needs `agent.brain.history_data`)
- [ ] Compute chemotaxis metrics if `--track-experiment` and `result.food_history` present
- [ ] Keep existing `deepcopy` + `track_per_run` block (needs full data, executes before flush)
- [ ] Extract scalar snapshots into result fields (`path_length`, `max_satiety`, `final_health`, `max_health`)
- [ ] Replace `tracking_data.brain_data[run_num]` with `BrainDataSnapshot`
- [ ] Flush heavy fields from result (`path=[]`, `food_history=None`, etc.)
- [ ] Store lightweight `EpisodeTrackingData` (only `foods_collected` + `distance_efficiencies`)

**File:** `scripts/run_simulation.py`

### 3.3 Close Writers After Loop

- [ ] Close path CSV file handle in `finally` block
- [ ] Close `IncrementalDetailedTrackingWriter` in `finally` block

**File:** `scripts/run_simulation.py`

### 3.4 Pass Skip Flags to Post-Loop Exports

- [ ] Pass `skip_path_data=True` to `export_simulation_results_to_csv()`
- [ ] Pass `skip_detailed=True` to `export_tracking_data_to_csv()`

**File:** `scripts/run_simulation.py`

______________________________________________________________________

## 4. Post-Loop Consumer Updates

### 4.1 `_export_main_results`

- [ ] Replace `len(result.path)` with `result.path_length if result.path_length is not None else len(result.path)`

**File:** `packages/quantum-nematode/quantumnematode/report/csv_export.py` (line 80)

### 4.2 `summary()`

- [ ] Replace `result.health_history[-1]` with fallback to `result.final_health`
- [ ] Replace `result.path` debug log with `result.path_length`

**File:** `packages/quantum-nematode/quantumnematode/report/summary.py` (lines 59-60, 151)

### 4.3 `plot_results()`

- [ ] Line 1072: `max(satiety_history)` → fallback to `result.max_satiety`
- [ ] Line 1085: `r.health_history[-1]` → fallback to `r.final_health`
- [ ] Line 1090: `max(health_history)` → fallback to `result.max_health`

**File:** `scripts/run_simulation.py` (lines 1067-1091)

### 4.4 `plot_tracking_data_by_session()`

- [ ] Handle `BrainDataSnapshot`: use `run_data.last_values.get(key)` instead of `getattr(run_data, key)[-1]`
- [ ] Handle key discovery: use `last_values.keys()` for snapshot type

**File:** `packages/quantum-nematode/quantumnematode/report/plots.py` (lines 203-310)

### 4.5 `_export_session_tracking_data()`

- [ ] Handle `BrainDataSnapshot` same pattern as 4.4

**File:** `packages/quantum-nematode/quantumnematode/report/csv_export.py` (lines 378-416)

### 4.6 `export_tracking_data_to_csv()` Key Discovery

- [ ] Handle `BrainDataSnapshot` in key discovery (line 361)

**File:** `packages/quantum-nematode/quantumnematode/report/csv_export.py` (line 361)

______________________________________________________________________

## 5. Chemotaxis Pre-Computation

### 5.1 Update `aggregate_results_metadata`

- [ ] Add `precomputed_chemotaxis: list[tuple[int, ChemotaxisMetrics]] | None = None` parameter
- [ ] When provided, use pre-computed metrics instead of recomputing from `result.path`/`result.food_history`
- [ ] Maintain identical chemotaxis validation logic (convergence filtering, biological comparison)

**File:** `packages/quantum-nematode/quantumnematode/experiment/tracker.py` (lines 370-474)

### 5.2 Pass Pre-Computed Metrics From Main Loop

- [ ] Pass `chemotaxis_metrics_per_run` to `capture_experiment_metadata` call

**File:** `scripts/run_simulation.py` (lines 787-834)

______________________________________________________________________

## 6. Interrupt Handler

### 6.1 Clean Up Incremental Files on Interrupt

- [ ] Pass incremental file handles (or a cleanup callback) to `manage_simulation_halt`
- [ ] Close file handles before generating partial results

**File:** `packages/quantum-nematode/quantumnematode/utils/interrupt_handler.py`
**File:** `scripts/run_simulation.py` (lines 660-673)

______________________________________________________________________

## 7. Verification

- [ ] `uv run pre-commit run -a` passes
- [ ] `uv run pytest -m "not smoke and not nightly"` passes (all unit/integration tests)
- [ ] `uv run pytest -m smoke -v` passes (smoke tests run actual simulations)
- [ ] Manual: run 10-episode simulation, diff CSV outputs against pre-change baseline
- [ ] Manual: run with `--track-experiment`, confirm chemotaxis metrics match
- [ ] Manual: run with `--track-per-run`, confirm per-run plots/exports work
- [ ] Manual: `Ctrl-C` mid-simulation, verify partial results export correctly
