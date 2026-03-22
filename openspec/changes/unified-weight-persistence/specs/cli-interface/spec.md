## ADDED Requirements

### Requirement: Weight Persistence CLI Flags

The `run_simulation.py` script SHALL accept CLI arguments for loading and saving brain weights.

#### Scenario: Load Weights Flag

- **WHEN** a user runs `scripts/run_simulation.py --load-weights path/to/weights.pt`
- **THEN** the CLI SHALL accept the argument as a valid file path string
- **AND** after brain construction, the system SHALL call `load_weights(brain, Path(args.load_weights))`
- **AND** training SHALL continue from the loaded weight state

#### Scenario: Save Weights Flag

- **WHEN** a user runs `scripts/run_simulation.py --save-weights path/to/output.pt`
- **THEN** the CLI SHALL accept the argument as a valid file path string
- **AND** after training completes, the system SHALL call `save_weights(brain, Path(args.save_weights))`

#### Scenario: Both Flags Together

- **WHEN** a user runs `scripts/run_simulation.py --load-weights stage1.pt --save-weights stage2.pt`
- **THEN** the system SHALL load weights before training starts
- **AND** SHALL save weights after training completes
- **AND** the loaded and saved paths MAY be different

#### Scenario: Load Weights CLI Overrides Config

- **WHEN** both `--load-weights path/cli.pt` and `config.weights_path: path/config.pt` are specified
- **THEN** the CLI `--load-weights` path SHALL take precedence
- **AND** `config.weights_path` SHALL be ignored

#### Scenario: Help Text for Weight Flags

- **WHEN** a user runs `scripts/run_simulation.py --help`
- **THEN** `--load-weights` SHALL be documented as "Path to saved weights to load before training"
- **AND** `--save-weights` SHALL be documented as "Path to save weights after training completes"

#### Scenario: Invalid Load Path

- **WHEN** `--load-weights nonexistent.pt` is specified and the file does not exist
- **THEN** the system SHALL raise a `FileNotFoundError` with the path
- **AND** SHALL exit before starting the training loop

#### Scenario: Non-Implementing Brain With CLI Flags

- **WHEN** `--load-weights` or `--save-weights` is specified but the brain does not implement `WeightPersistence`
- **THEN** the system SHALL raise a `TypeError` with a message naming the brain class
- **AND** SHALL exit before starting the training loop

### Requirement: Auto-Save Final Weights

The training loop in `run_simulation.py` SHALL auto-save final weights to the session export directory after training completes.

#### Scenario: Auto-Save on Normal Completion

- **WHEN** the training loop completes all episodes
- **AND** the brain implements `WeightPersistence`
- **THEN** the system SHALL save brain weights to `exports/{session_id}/weights/final.pt`
- **AND** SHALL log the save path for user reference
- **AND** this SHALL happen regardless of whether `--save-weights` is specified

#### Scenario: Auto-Save on KeyboardInterrupt

- **WHEN** the training loop is interrupted by KeyboardInterrupt
- **AND** the brain implements `WeightPersistence`
- **THEN** the system SHALL save brain weights to `exports/{session_id}/weights/final.pt` with the current training state
- **AND** SHALL log the save path for user reference

#### Scenario: Auto-Save Skipped for Non-Implementing Brain

- **WHEN** the training loop completes but the brain does not implement `WeightPersistence`
- **THEN** the system SHALL skip auto-save silently (debug log only)

#### Scenario: Auto-Save Directory Creation

- **WHEN** the `exports/{session_id}/weights/` directory does not exist
- **THEN** the system SHALL create it before saving

#### Scenario: Auto-Save Plus Explicit Save

- **WHEN** `--save-weights custom/path.pt` is specified
- **THEN** the system SHALL save to BOTH:
  - `exports/{session_id}/weights/final.pt` (auto-save)
  - `custom/path.pt` (explicit save)
