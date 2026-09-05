# cli-interface Specification

## Purpose

This specification defines the command-line interfaces for running simulations and managing experiments. It ensures consistent CLI argument parsing, help text, error handling, and output formatting across all CLI tools.

## Requirements

### Requirement: Brain Type Argument Extension

The CLI argument parser SHALL accept "spiking" as a valid brain type option.

#### Scenario: Spiking Brain Selection

- **GIVEN** a user wants to run simulation with spiking neural network
- **WHEN** they execute `python scripts/run_simulation.py --brain spiking`
- **THEN** the CLI SHALL parse "spiking" as a valid brain type
- **AND** SHALL pass the selection to the brain factory
- **AND** SHALL not raise validation errors

#### Scenario: Brain Type Help Text

- **GIVEN** a user requests help for brain type options
- **WHEN** they execute `python scripts/run_simulation.py --help`
- **THEN** the help text SHALL list "spiking" among valid brain types
- **AND** SHALL provide brief description of spiking neural network approach

### Requirement: Configuration Compatibility

The CLI SHALL support loading spiking brain configurations through existing configuration mechanisms.

#### Scenario: Spiking Configuration Loading

- **GIVEN** a YAML configuration file with spiking brain parameters
- **WHEN** loaded via `--config spikingreinforce_foraging_medium.yml`
- **THEN** the CLI SHALL parse the configuration
- **AND** SHALL validate spiking-specific parameters
- **AND** SHALL initialize the spiking brain with specified parameters

#### Scenario: Brain Type Override

- **GIVEN** a configuration file specifies a different brain type
- **WHEN** user provides `--brain spiking` CLI argument
- **THEN** the CLI argument SHALL override the configuration file
- **AND** SHALL use spiking brain regardless of config file brain type

### Requirement: Error Handling

The CLI SHALL provide clear error messages for spiking brain configuration issues.

#### Scenario: Invalid Spiking Parameters

- **GIVEN** a configuration with invalid spiking brain parameters
- **WHEN** the CLI attempts to initialize the brain
- **THEN** SHALL provide specific error message about invalid parameters
- **AND** SHALL suggest valid parameter ranges
- **AND** SHALL exit gracefully with appropriate error code

#### Scenario: Missing Dependencies

- **GIVEN** spiking brain is selected but required dependencies are missing
- **WHEN** the CLI attempts initialization
- **THEN** SHALL provide clear error about missing dependencies
- **AND** SHALL suggest installation commands

### Requirement: Experiment Tracking CLI Flags

The run_simulation.py script SHALL accept a command-line flag for experiment tracking.

#### Scenario: Track Experiment Flag

- **GIVEN** a user wants to track experiment metadata
- **WHEN** the user runs `scripts/run_simulation.py --track-experiment ...`
- **THEN** experiment metadata SHALL be captured and saved after simulation completes
- **AND** the experiment ID SHALL be displayed in the output
- **AND** the path to the saved metadata file SHALL be shown

#### Scenario: Help Text for Tracking Flags

- **GIVEN** a user runs `scripts/run_simulation.py --help`
- **WHEN** help text is displayed
- **THEN** the `--track-experiment` flag SHALL be documented as "Save experiment metadata for reproducibility and comparison"

### Requirement: Experiment Query CLI

The system SHALL provide a dedicated CLI tool for querying and analyzing stored experiments.

#### Scenario: Experiment List Command

- **GIVEN** experiments are stored
- **WHEN** a user runs `scripts/experiment_query.py list`
- **THEN** a formatted table of experiments SHALL be displayed
- **AND** the table SHALL show: ID, config, env type, brain type, success rate, date
- **AND** experiments SHALL be sorted by date (most recent first)

#### Scenario: Experiment Filter Options

- **GIVEN** a user runs `scripts/experiment_query.py list --help`
- **WHEN** help text is displayed
- **THEN** filter options SHALL include: --env-type, --brain-type, --min-success-rate, --since, --limit
- **AND** each filter option SHALL have clear documentation

#### Scenario: Experiment Show Command

- **GIVEN** an experiment ID
- **WHEN** a user runs `scripts/experiment_query.py show {experiment_id}`
- **THEN** detailed metadata SHALL be displayed in a readable format
- **AND** configuration, results, and system info SHALL be organized in sections
- **AND** the path to detailed exports SHALL be shown if available

#### Scenario: Experiment Compare Command

- **GIVEN** two experiment IDs
- **WHEN** a user runs `scripts/experiment_query.py compare {id1} {id2}`
- **THEN** a side-by-side comparison SHALL be displayed
- **AND** differences SHALL be highlighted with color coding (if terminal supports it)
- **AND** performance delta SHALL be shown for key metrics

### Requirement: CLI Output Formatting

The experiment CLI tools SHALL provide well-formatted, readable output for terminal display.

#### Scenario: Table Formatting

- **GIVEN** query results to display
- **WHEN** output is generated
- **THEN** tables SHALL be formatted with aligned columns
- **AND** headers SHALL be clearly distinguished
- **AND** terminal width SHALL be respected (no wrapping)

#### Scenario: Color Coding

- **GIVEN** a terminal that supports ANSI colors
- **WHEN** output is displayed
- **THEN** success indicators SHALL be green
- **AND** warnings SHALL be yellow
- **AND** errors SHALL be red
- **AND** important values SHALL be bold

#### Scenario: JSON Output Option

- **GIVEN** a user wants machine-readable output
- **WHEN** the user adds `--json` flag to any command
- **THEN** output SHALL be formatted as JSON
- **AND** all data SHALL be included (not truncated for display)
- **AND** JSON SHALL be pretty-printed for readability

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

#### Scenario: Non-Implementing Brain With Weight Persistence Request

- **WHEN** `--load-weights`, `--save-weights`, or `config.weights_path` is specified but the brain does not implement `WeightPersistence`
- **THEN** the system SHALL raise a `TypeError` with a message naming the brain class and the source of the request (CLI flags or config field)
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

### Requirement: Accelerator selection and availability validation

The device options offered by the CLI SHALL correspond to backends the running build can actually provide, and a request for an unavailable accelerator SHALL fail at startup with an actionable message rather than an unhandled error raised later during brain construction.

`DeviceType` SHALL expose a `mps` member mapping to PyTorch's `"mps"` device string, so Apple GPUs are selectable and therefore measurable. `gpu` SHALL continue to map to `"cuda"` and SHALL NOT be silently redirected to another accelerator: a redirect would deliver a materially different performance profile under a name the user chose deliberately.

When a selected accelerator is unavailable in the running build, the CLI SHALL raise a clear error before brain construction that names the requested device, states that it is unavailable, and names the platform-appropriate alternative.

The availability check SHALL apply to brains that place tensors on the PyTorch device. It SHALL NOT be applied to brains in the `quantum` family, for which `gpu` selects the Qiskit simulator's own GPU device: that backend has separate requirements, so PyTorch's view of CUDA is not evidence about it, and rejecting on that basis would refuse a working configuration.

#### Scenario: Requesting CUDA on a build without CUDA

- **GIVEN** a PyTorch build without CUDA support
- **WHEN** the CLI is invoked with the `gpu` device and a brain that places tensors on the PyTorch device
- **THEN** it SHALL exit with a clear error naming the unavailable device and the platform-appropriate alternative
- **AND** SHALL NOT surface a raw torch assertion from inside brain construction

#### Scenario: Quantum brains are not judged by PyTorch's accelerator availability

- **GIVEN** a PyTorch build without CUDA support
- **WHEN** the CLI is invoked with the `gpu` device and a brain in the `quantum` family
- **THEN** the selection SHALL be accepted, leaving availability to the Qiskit backend that owns it

#### Scenario: Requesting MPS where it is available

- **GIVEN** a build where the MPS backend is available
- **WHEN** the CLI is invoked with the `mps` device and a brain that runs on the PyTorch backend
- **THEN** brain tensors SHALL be placed on the MPS device
- **AND** the run SHALL proceed without error

#### Scenario: Requesting MPS where it is unavailable

- **GIVEN** a build where the MPS backend is unavailable
- **WHEN** the CLI is invoked with the `mps` device
- **THEN** it SHALL exit with a clear error naming the unavailable device
- **AND** SHALL NOT surface a raw torch error from inside brain construction

#### Scenario: CPU remains the default

- **WHEN** no device is specified
- **THEN** the CLI SHALL select CPU
- **AND** the selection SHALL require no accelerator availability check

### Requirement: Device selection is validated against the brain family

`DeviceType` is shared by two unrelated backends: the PyTorch device for classical/spiking brains, and the Qiskit Aer / IBM Runtime backend selector for quantum brains, which reach it as an upper-cased string. A device meaningful to one backend is not necessarily meaningful to the other, and the quantum backend accepts unknown device strings **without raising**, so an unchecked selection is recorded in experiment metadata as though it were a legitimate backend.

The CLI SHALL therefore reject, at startup and before brain construction, any device that the selected brain's family cannot use. Specifically, a PyTorch-only accelerator SHALL NOT be accepted for a brain registered in the `quantum` family, whose device value is consumed as a simulator selector rather than a tensor placement. The error SHALL name the requested device, the brain, and the devices that brain does accept.

The brain family SHALL be read from the existing plugin-registry `families` metadata rather than from a hand-maintained list, so a newly registered architecture inherits the correct validation without a per-architecture branch. A brain tagged with both `quantum` and another family SHALL be treated as quantum for this check, because its device value still reaches the simulator.

This requirement adds validation only for accelerators introduced by this change; it SHALL NOT alter the pre-existing tolerance whereby a non-quantum brain selecting `qpu` falls back to CPU tensor placement.

#### Scenario: PyTorch-only accelerator rejected for a quantum brain

- **GIVEN** a config selecting a brain registered in the `quantum` family
- **WHEN** the CLI is invoked with the `mps` device
- **THEN** it SHALL exit with a clear error naming the device, the brain, and the accepted devices
- **AND** SHALL NOT construct a simulator backend from the rejected device string

#### Scenario: Hybrid quantum brains are treated as quantum

- **GIVEN** a brain whose registry entry tags it both `quantum` and `classical`
- **WHEN** the CLI is invoked with a PyTorch-only accelerator
- **THEN** the selection SHALL be rejected on the same grounds as a purely quantum brain

#### Scenario: Quantum brains keep their own devices

- **WHEN** a quantum brain is run with `cpu`, `gpu`, or `qpu`
- **THEN** the selection SHALL be accepted
- **AND** the simulator backend string SHALL be unchanged from before this change

#### Scenario: Validation derives from registry metadata

- **WHEN** a new architecture is registered with a `quantum` family tag
- **THEN** it SHALL be covered by this validation without any change to the validation code
