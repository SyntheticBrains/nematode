# Implementation Tasks

## 1. Core Data Models and Storage
- [x] 1.1 Create `quantumnematode/experiment/` module directory
- [x] 1.2 Implement `ExperimentMetadata` Pydantic model with all required fields
- [x] 1.3 Implement `EnvironmentMetadata`, `BrainMetadata`, `ResultsMetadata`, `SystemMetadata` models
- [x] 1.4 Implement `BenchmarkMetadata` model for benchmark-specific fields
- [x] 1.5 Add JSON serialization/deserialization methods
- [x] 1.6 Implement config file hash generation (SHA256)
- [x] 1.7 Create `experiments/` directory structure with `.gitignore` entry
- [x] 1.8 Create `benchmarks/` directory structure with category subdirectories
- [x] 1.9 Implement atomic file writing for experiment storage
- [x] 1.10 Add validation for metadata completeness

## 2. Git Context Capture
- [x] 2.1 Implement git repository detection
- [x] 2.2 Implement current commit hash extraction
- [x] 2.3 Implement current branch name extraction
- [x] 2.4 Implement git dirty state detection (uncommitted changes)
- [x] 2.5 Handle edge cases (no git repo, detached HEAD, etc.)
- [x] 2.6 Add warning logging for suboptimal git states

## 3. System Information Capture
- [x] 3.1 Implement Python version detection
- [x] 3.2 Implement Qiskit version detection
- [x] 3.3 Implement PyTorch version detection (if installed)
- [x] 3.4 Implement device type detection (CPU/GPU/QPU)
- [x] 3.5 Implement QPU backend name extraction
- [x] 3.6 Add system metadata to experiment data model

## 4. Experiment Metadata Capture
- [x] 4.1 Create `tracker.py` module for experiment tracking logic
- [x] 4.2 Implement environment metadata extraction from config
- [x] 4.3 Implement brain metadata extraction from config
- [x] 4.4 Implement results metadata aggregation from SimulationResult list
- [x] 4.5 Implement foraging-specific metrics extraction
- [x] 4.6 Implement termination reason breakdown calculation
- [x] 4.7 Add metadata capture to run_simulation.py workflow
- [x] 4.8 Add reference to exports directory in metadata
- [x] 4.9 Test metadata capture with all brain types
- [x] 4.10 Test metadata capture with all environment types

## 5. Experiment Storage and Retrieval
- [x] 5.1 Create `storage.py` module for file I/O operations
- [x] 5.2 Implement `save_experiment()` function
- [x] 5.3 Implement `load_experiment()` function
- [x] 5.4 Implement `list_experiments()` function with filtering
- [x] 5.5 Implement experiment ID generation (timestamp-based)
- [x] 5.6 Add error handling for file I/O failures
- [x] 5.7 Implement experiment query by environment type
- [x] 5.8 Implement experiment query by brain type
- [x] 5.9 Implement experiment query by success rate threshold
- [x] 5.10 Implement experiment query by date range

## 6. Experiment Comparison
- [x] 6.1 Implement `compare_experiments()` function
- [x] 6.2 Generate configuration diff structure
- [x] 6.3 Generate results comparison structure
- [x] 6.4 Calculate performance deltas and improvements
- [x] 6.5 Format comparison output for terminal display
- [x] 6.6 Add statistical significance indicators

## 7. Benchmark Categorization
- [x] 7.1 Create `quantumnematode/benchmark/` module directory
- [x] 7.2 Implement category detection logic
- [x] 7.3 Map environment types to categories (static, dynamic_small, dynamic_medium, dynamic_large)
- [x] 7.4 Map brain types to quantum/classical classification
- [x] 7.5 Generate category directories automatically
- [x] 7.6 Test categorization with all combinations

## 8. Benchmark Validation
- [x] 8.1 Create `validation.py` module for benchmark validation
- [x] 8.2 Implement minimum runs validation (20+ runs)
- [x] 8.3 Implement git clean state validation
- [x] 8.4 Implement config file existence validation
- [x] 8.5 Implement success rate threshold warnings
- [x] 8.6 Implement statistical consistency checks
- [x] 8.7 Implement configuration schema validation
- [x] 8.8 Add validation error messages and user prompts
- [x] 8.9 Allow validation bypass with user confirmation

## 9. Benchmark Submission Workflow
- [x] 9.1 Create `submission.py` module for benchmark submission
- [x] 9.2 Implement interactive contributor name prompt
- [x] 9.3 Implement interactive GitHub username prompt
- [x] 9.4 Implement interactive optimization notes prompt
- [x] 9.5 Implement benchmark file creation with all metadata
- [x] 9.6 Integrate validation into submission workflow
- [x] 9.7 Display submission instructions (PR creation)
- [x] 9.8 Add benchmark submission to run_simulation.py via `--save-benchmark` flag
- [x] 9.9 Support non-interactive mode with CLI arguments
- [x] 9.10 Test submission workflow end-to-end

## 10. Leaderboard Generation
- [x] 10.1 Create `leaderboard.py` module for leaderboard generation
- [x] 10.2 Implement benchmark ranking logic (composite score)
- [x] 10.3 Implement README.md summary table generation (top 3 per category)
- [x] 10.4 Implement BENCHMARKS.md full table generation
- [x] 10.5 Generate reproduction instructions per benchmark
- [x] 10.6 Format GitHub links for config files and commits
- [x] 10.7 Handle empty categories gracefully
- [x] 10.8 Group quantum vs classical brains separately
- [x] 10.9 Add generation timestamp to tables
- [x] 10.10 Implement leaderboard regeneration command

## 11. Experiment Query CLI
- [x] 11.1 Create `scripts/experiment_query.py` CLI tool
- [x] 11.2 Implement `list` command with filtering options
- [x] 11.3 Implement `show` command for detailed view
- [x] 11.4 Implement `compare` command for side-by-side comparison
- [x] 11.5 Add table formatting with aligned columns
- [x] 11.6 Add color coding for terminal output (optional)
- [x] 11.7 Add `--json` flag for machine-readable output
- [x] 11.8 Implement argument parsing and help text
- [x] 11.9 Add error handling for missing experiments
- [x] 11.10 Test CLI with various filter combinations

## 12. Benchmark Management CLI
- [x] 12.1 Create `scripts/benchmark_submit.py` CLI tool
- [x] 12.2 Implement `submit` command for promoting experiments to benchmarks
- [x] 12.3 Implement `leaderboard` command for viewing rankings
- [x] 12.4 Implement `regenerate` command for updating documentation
- [x] 12.5 Implement `verify` command for maintainer verification (future)
- [x] 12.6 Add argument parsing and help text
- [x] 12.7 Add interactive prompts for metadata
- [x] 12.8 Display git diff after regeneration
- [x] 12.9 Add error handling and validation
- [x] 12.10 Test CLI with real benchmark submissions

## 13. Integration with run_simulation.py
- [x] 13.1 Add `--track-experiment` CLI flag
- [x] 13.2 Add `--save-benchmark` CLI flag
- [x] 13.3 Add `--benchmark-notes` CLI flag
- [x] 13.4 Integrate experiment tracking into simulation completion
- [x] 13.5 Integrate benchmark submission into simulation completion
- [x] 13.6 Display experiment ID and file path in output
- [x] 13.7 Display benchmark submission instructions
- [x] 13.8 Ensure tracking doesn't impact simulation performance
- [x] 13.9 Add help text for new flags
- [x] 13.10 Test integration with all brain and environment types

## 14. Documentation Updates
- [x] 14.1 Create initial `BENCHMARKS.md` with structure and submission guidelines
- [x] 14.2 Add "🏆 Top Benchmarks" section to README.md
- [x] 14.3 Add benchmark submission guidelines to CONTRIBUTING.md
- [x] 14.4 Document experiment tracking CLI in CONTRIBUTING.md
- [x] 14.5 Add example commands for experiment tracking
- [x] 14.6 Add example commands for benchmark submission
- [x] 14.7 Document metadata structure and fields
- [x] 14.8 Add troubleshooting section for common issues
- [x] 14.9 Create example benchmark submission PR template
- [x] 14.10 Update README with link to experiment tracking features

## 15. Testing
- [x] 15.1 Unit tests for ExperimentMetadata serialization
- [x] 15.2 Unit tests for BenchmarkMetadata serialization
- [x] 15.3 Unit tests for git context capture
- [x] 15.4 Unit tests for system information capture
- [x] 15.5 Unit tests for experiment storage and retrieval
- [x] 15.6 Unit tests for experiment query filtering
- [x] 15.7 Unit tests for experiment comparison logic
- [x] 15.8 Unit tests for benchmark categorization
- [x] 15.9 Unit tests for benchmark validation rules
- [x] 15.10 Unit tests for leaderboard generation
- [x] 15.11 Integration test for full experiment tracking workflow
- [x] 15.12 Integration test for full benchmark submission workflow
- [x] 15.13 CLI tests for experiment_query.py commands
- [x] 15.14 CLI tests for benchmark_submit.py commands
- [x] 15.15 Test with all brain types (Modular, MLP, QModular, QMLP, Spiking)
- [x] 15.16 Test with all environment types (Static, Dynamic Small/Medium/Large)
- [x] 15.17 Test edge cases (no git repo, missing config, etc.)
- [x] 15.18 Test backward compatibility (experiments disabled)

## 16. Validation and Quality Assurance
- [x] 16.1 Run full test suite and ensure all tests pass
- [x] 16.2 Test experiment tracking with real simulations
- [x] 16.3 Test benchmark submission with real experiments
- [x] 16.4 Verify leaderboard generation produces correct output
- [x] 16.5 Verify metadata completeness for all test cases
- [x] 16.6 Check file permissions and directory creation
- [x] 16.7 Validate JSON structure with external tools
- [x] 16.8 Test query performance with 100+ experiments
- [x] 16.9 Verify backward compatibility with existing workflows
- [x] 16.10 Code review for consistency and best practices
- [x] 16.11 Documentation review for clarity and completeness
- [x] 16.12 Run pyright and ruff validation
