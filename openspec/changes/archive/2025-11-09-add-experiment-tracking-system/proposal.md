# Experiment Tracking and Benchmarking System Proposal

## Why

The current simulation framework saves comprehensive data exports (CSVs, plots) but lacks persistent experiment tracking, cross-session comparison capabilities, and a curated benchmark leaderboard. Users cannot easily:

- Compare experiments across different sessions or configurations
- Reproduce past results without manually tracking configuration and environment details
- Share and verify benchmark results with proper metadata (commit hash, contributor, system info)
- Discover best-performing configurations for different tasks

To enable reproducible research, facilitate collaboration, and showcase project capabilities, we need a lightweight experiment tracking system with optional benchmark curation.

## What Changes

- **Add experiment metadata tracking** with automatic capture of configuration, git state, system info, and results
- **Implement two-tier tracking system**: lightweight auto-tracking for all runs + explicit benchmark mode for curated results
- **Create JSON-based experiment storage** in `experiments/` directory with queryable metadata
- **Add benchmark submission workflow** via CLI flag `--save-benchmark` with semi-automatic metadata capture
- **Create benchmark leaderboard** with summary table in README.md and detailed results in BENCHMARKS.md
- **Organize benchmarks hierarchically** by environment type (Static/Dynamic) → brain architecture → metrics
- **Add CLI query tools** for viewing, comparing, and analyzing stored experiments
- **Implement reproducibility helpers** to re-run experiments from stored metadata
- **Add validation** to ensure benchmark submissions include required metadata and pass quality checks
- **Maintain backward compatibility** - experiment tracking is opt-in, existing workflows unchanged

## Impact

### Affected Specs

- **NEW**: `experiment-tracking` - Creating new specification for experiment metadata capture and storage
- **NEW**: `benchmark-management` - Creating new specification for benchmark curation and leaderboard
- **MODIFIED**: `cli-interface` - Adding experiment tracking and benchmark CLI flags and commands

### Affected Code

- **Core Implementation**:

  - [`quantumnematode/experiment/`](../../../packages/quantum-nematode/quantumnematode/experiment/) - New module for experiment tracking (NEW)
    - `metadata.py` - Experiment metadata models (ExperimentMetadata, BenchmarkMetadata)
    - `storage.py` - JSON-based experiment storage and retrieval
    - `tracker.py` - Experiment tracking integration with simulation runs
  - [`quantumnematode/benchmark/`](../../../packages/quantum-nematode/quantumnematode/benchmark/) - New module for benchmark management (NEW)
    - `submission.py` - Benchmark submission workflow and validation
    - `leaderboard.py` - Leaderboard generation and formatting
    - `query.py` - Benchmark querying and comparison tools

- **CLI Integration**:

  - [`scripts/run_simulation.py`](../../../scripts/run_simulation.py) - Add `--track-experiment` and `--save-benchmark` flags
  - [`scripts/experiment_query.py`](../../../scripts/experiment_query.py) - New CLI tool for querying experiments (NEW)
  - [`scripts/benchmark_submit.py`](../../../scripts/benchmark_submit.py) - New CLI tool for benchmark management (NEW)

- **Documentation**:

  - [`README.md`](../../../README.md) - Add "Top Benchmarks" section with summary table
  - [`BENCHMARKS.md`](../../../BENCHMARKS.md) - Detailed benchmark results and reproduction instructions (NEW)
  - [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) - Add benchmark submission guidelines

- **Storage Structure**:

  - `experiments/` - Auto-tracked experiment metadata (JSON files) (NEW)
  - `benchmarks/` - Curated benchmark submissions (JSON files) (NEW)
  - `.gitignore` - Update to handle experiment storage (experiments/ tracked, exports/ ignored)

### Migration Path

- No migration needed - new functionality is additive and opt-in
- Existing simulation runs continue to work unchanged
- Users explicitly opt into experiment tracking with `--track-experiment` flag
- Benchmark submission requires explicit `--save-benchmark` flag

### Breaking Changes

None - all new functionality is opt-in and backward compatible.

### Performance Considerations

- Experiment metadata capture adds ~10-50ms overhead per simulation run (negligible)
- JSON file writes are fast (\<5ms) and non-blocking
- Query operations read from disk but with reasonable index sizes (\<1000 experiments)
- No impact on simulation performance itself

### Testing Strategy

- Unit tests for metadata capture, serialization, and storage
- Integration tests for experiment tracking workflow
- Validation tests for benchmark submission requirements
- CLI tests for query and submission tools
- Documentation tests to ensure examples work correctly
