# Implementation Tasks

## 1. Core Data Models and Storage
- [ ] 1.1 Create `quantumnematode/experiment/` module directory
- [ ] 1.2 Implement `ExperimentMetadata` Pydantic model with all required fields
- [ ] 1.3 Implement `EnvironmentMetadata`, `BrainMetadata`, `ResultsMetadata`, `SystemMetadata` models
- [ ] 1.4 Implement `BenchmarkMetadata` model for benchmark-specific fields
- [ ] 1.5 Add JSON serialization/deserialization methods
- [ ] 1.6 Implement config file hash generation (SHA256)
- [ ] 1.7 Create `experiments/` directory structure with `.gitignore` entry
- [ ] 1.8 Create `benchmarks/` directory structure with category subdirectories
- [ ] 1.9 Implement atomic file writing for experiment storage
- [ ] 1.10 Add validation for metadata completeness

## 2. Git Context Capture
- [ ] 2.1 Implement git repository detection
- [ ] 2.2 Implement current commit hash extraction
- [ ] 2.3 Implement current branch name extraction
- [ ] 2.4 Implement git dirty state detection (uncommitted changes)
- [ ] 2.5 Handle edge cases (no git repo, detached HEAD, etc.)
- [ ] 2.6 Add warning logging for suboptimal git states

## 3. System Information Capture
- [ ] 3.1 Implement Python version detection
- [ ] 3.2 Implement Qiskit version detection
- [ ] 3.3 Implement PyTorch version detection (if installed)
- [ ] 3.4 Implement device type detection (CPU/GPU/QPU)
- [ ] 3.5 Implement QPU backend name extraction
- [ ] 3.6 Add system metadata to experiment data model

## 4. Experiment Metadata Capture
- [ ] 4.1 Create `tracker.py` module for experiment tracking logic
- [ ] 4.2 Implement environment metadata extraction from config
- [ ] 4.3 Implement brain metadata extraction from config
- [ ] 4.4 Implement results metadata aggregation from SimulationResult list
- [ ] 4.5 Implement foraging-specific metrics extraction
- [ ] 4.6 Implement termination reason breakdown calculation
- [ ] 4.7 Add metadata capture to run_simulation.py workflow
- [ ] 4.8 Add reference to exports directory in metadata
- [ ] 4.9 Test metadata capture with all brain types
- [ ] 4.10 Test metadata capture with all environment types

## 5. Experiment Storage and Retrieval
- [ ] 5.1 Create `storage.py` module for file I/O operations
- [ ] 5.2 Implement `save_experiment()` function
- [ ] 5.3 Implement `load_experiment()` function
- [ ] 5.4 Implement `list_experiments()` function with filtering
- [ ] 5.5 Implement experiment ID generation (timestamp-based)
- [ ] 5.6 Add error handling for file I/O failures
- [ ] 5.7 Implement experiment query by environment type
- [ ] 5.8 Implement experiment query by brain type
- [ ] 5.9 Implement experiment query by success rate threshold
- [ ] 5.10 Implement experiment query by date range

## 6. Experiment Comparison
- [ ] 6.1 Implement `compare_experiments()` function
- [ ] 6.2 Generate configuration diff structure
- [ ] 6.3 Generate results comparison structure
- [ ] 6.4 Calculate performance deltas and improvements
- [ ] 6.5 Format comparison output for terminal display
- [ ] 6.6 Add statistical significance indicators

## 7. Benchmark Categorization
- [ ] 7.1 Create `quantumnematode/benchmark/` module directory
- [ ] 7.2 Implement category detection logic
- [ ] 7.3 Map environment types to categories (static, dynamic_small, dynamic_medium, dynamic_large)
- [ ] 7.4 Map brain types to quantum/classical classification
- [ ] 7.5 Generate category directories automatically
- [ ] 7.6 Test categorization with all combinations

## 8. Benchmark Validation
- [ ] 8.1 Create `validation.py` module for benchmark validation
- [ ] 8.2 Implement minimum runs validation (20+ runs)
- [ ] 8.3 Implement git clean state validation
- [ ] 8.4 Implement config file existence validation
- [ ] 8.5 Implement success rate threshold warnings
- [ ] 8.6 Implement statistical consistency checks
- [ ] 8.7 Implement configuration schema validation
- [ ] 8.8 Add validation error messages and user prompts
- [ ] 8.9 Allow validation bypass with user confirmation

## 9. Benchmark Submission Workflow
- [ ] 9.1 Create `submission.py` module for benchmark submission
- [ ] 9.2 Implement interactive contributor name prompt
- [ ] 9.3 Implement interactive GitHub username prompt
- [ ] 9.4 Implement interactive optimization notes prompt
- [ ] 9.5 Implement benchmark file creation with all metadata
- [ ] 9.6 Integrate validation into submission workflow
- [ ] 9.7 Display submission instructions (PR creation)
- [ ] 9.8 Add benchmark submission to run_simulation.py via `--save-benchmark` flag
- [ ] 9.9 Support non-interactive mode with CLI arguments
- [ ] 9.10 Test submission workflow end-to-end

## 10. Leaderboard Generation
- [ ] 10.1 Create `leaderboard.py` module for leaderboard generation
- [ ] 10.2 Implement benchmark ranking logic (composite score)
- [ ] 10.3 Implement README.md summary table generation (top 3 per category)
- [ ] 10.4 Implement BENCHMARKS.md full table generation
- [ ] 10.5 Generate reproduction instructions per benchmark
- [ ] 10.6 Format GitHub links for config files and commits
- [ ] 10.7 Handle empty categories gracefully
- [ ] 10.8 Group quantum vs classical brains separately
- [ ] 10.9 Add generation timestamp to tables
- [ ] 10.10 Implement leaderboard regeneration command

## 11. Experiment Query CLI
- [ ] 11.1 Create `scripts/experiment_query.py` CLI tool
- [ ] 11.2 Implement `list` command with filtering options
- [ ] 11.3 Implement `show` command for detailed view
- [ ] 11.4 Implement `compare` command for side-by-side comparison
- [ ] 11.5 Add table formatting with aligned columns
- [ ] 11.6 Add color coding for terminal output (optional)
- [ ] 11.7 Add `--json` flag for machine-readable output
- [ ] 11.8 Implement argument parsing and help text
- [ ] 11.9 Add error handling for missing experiments
- [ ] 11.10 Test CLI with various filter combinations

## 12. Benchmark Management CLI
- [ ] 12.1 Create `scripts/benchmark_submit.py` CLI tool
- [ ] 12.2 Implement `submit` command for promoting experiments to benchmarks
- [ ] 12.3 Implement `leaderboard` command for viewing rankings
- [ ] 12.4 Implement `regenerate` command for updating documentation
- [ ] 12.5 Implement `verify` command for maintainer verification (future)
- [ ] 12.6 Add argument parsing and help text
- [ ] 12.7 Add interactive prompts for metadata
- [ ] 12.8 Display git diff after regeneration
- [ ] 12.9 Add error handling and validation
- [ ] 12.10 Test CLI with real benchmark submissions

## 13. Integration with run_simulation.py
- [ ] 13.1 Add `--track-experiment` CLI flag
- [ ] 13.2 Add `--save-benchmark` CLI flag
- [ ] 13.3 Add `--benchmark-notes` CLI flag
- [ ] 13.4 Integrate experiment tracking into simulation completion
- [ ] 13.5 Integrate benchmark submission into simulation completion
- [ ] 13.6 Display experiment ID and file path in output
- [ ] 13.7 Display benchmark submission instructions
- [ ] 13.8 Ensure tracking doesn't impact simulation performance
- [ ] 13.9 Add help text for new flags
- [ ] 13.10 Test integration with all brain and environment types

## 14. Documentation Updates
- [ ] 14.1 Create initial `BENCHMARKS.md` with structure and submission guidelines
- [ ] 14.2 Add "🏆 Top Benchmarks" section to README.md
- [ ] 14.3 Add benchmark submission guidelines to CONTRIBUTING.md
- [ ] 14.4 Document experiment tracking CLI in CONTRIBUTING.md
- [ ] 14.5 Add example commands for experiment tracking
- [ ] 14.6 Add example commands for benchmark submission
- [ ] 14.7 Document metadata structure and fields
- [ ] 14.8 Add troubleshooting section for common issues
- [ ] 14.9 Create example benchmark submission PR template
- [ ] 14.10 Update README with link to experiment tracking features

## 15. Testing
- [ ] 15.1 Unit tests for ExperimentMetadata serialization
- [ ] 15.2 Unit tests for BenchmarkMetadata serialization
- [ ] 15.3 Unit tests for git context capture
- [ ] 15.4 Unit tests for system information capture
- [ ] 15.5 Unit tests for experiment storage and retrieval
- [ ] 15.6 Unit tests for experiment query filtering
- [ ] 15.7 Unit tests for experiment comparison logic
- [ ] 15.8 Unit tests for benchmark categorization
- [ ] 15.9 Unit tests for benchmark validation rules
- [ ] 15.10 Unit tests for leaderboard generation
- [ ] 15.11 Integration test for full experiment tracking workflow
- [ ] 15.12 Integration test for full benchmark submission workflow
- [ ] 15.13 CLI tests for experiment_query.py commands
- [ ] 15.14 CLI tests for benchmark_submit.py commands
- [ ] 15.15 Test with all brain types (Modular, MLP, QModular, QMLP, Spiking)
- [ ] 15.16 Test with all environment types (Static, Dynamic Small/Medium/Large)
- [ ] 15.17 Test edge cases (no git repo, missing config, etc.)
- [ ] 15.18 Test backward compatibility (experiments disabled)

## 16. Validation and Quality Assurance
- [ ] 16.1 Run full test suite and ensure all tests pass
- [ ] 16.2 Test experiment tracking with real simulations
- [ ] 16.3 Test benchmark submission with real experiments
- [ ] 16.4 Verify leaderboard generation produces correct output
- [ ] 16.5 Verify metadata completeness for all test cases
- [ ] 16.6 Check file permissions and directory creation
- [ ] 16.7 Validate JSON structure with external tools
- [ ] 16.8 Test query performance with 100+ experiments
- [ ] 16.9 Verify backward compatibility with existing workflows
- [ ] 16.10 Code review for consistency and best practices
- [ ] 16.11 Documentation review for clarity and completeness
- [ ] 16.12 Run pyright and ruff validation
