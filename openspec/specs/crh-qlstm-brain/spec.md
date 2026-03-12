## ADDED Requirements

### Requirement: CRH-QLSTM brain type registration

The system SHALL register `crhqlstm` as a valid brain type in `BrainType` enum, `CLASSICAL_BRAIN_TYPES` set, `BRAIN_CONFIG_MAP`, and `__all__` exports. The brain SHALL be selectable via `brain.name: crhqlstm` in YAML config files.

#### Scenario: Brain type recognized from config

- **WHEN** a YAML config specifies `brain.name: crhqlstm`
- **THEN** the config loader SHALL instantiate `CRHQLSTMBrainConfig` and the simulation SHALL create a `CRHQLSTMBrain` instance

#### Scenario: Brain classified as classical

- **WHEN** the brain type is `CRHQLSTM`
- **THEN** it SHALL be included in `CLASSICAL_BRAIN_TYPES` (the reservoir is classical even though QLIF gates may be quantum)

### Requirement: CRH-QLSTM config

`CRHQLSTMBrainConfig` SHALL accept the same LSTM readout, QLIF gate, PPO, LR schedule, critic, and sensory module parameters as `QRHQLSTMBrainConfig`, with CRH-specific reservoir parameters replacing QRH reservoir parameters:

- **Reservoir**: `num_reservoir_neurons` (default 10), `reservoir_depth` (default 3), `spectral_radius` (default 0.9), `input_connectivity` (default "sparse"), `input_scale` (default 1.0), `feature_channels` (default ["raw", "cos_sin", "pairwise"]), `input_encoding` (default "linear")

#### Scenario: Config with defaults

- **WHEN** a config specifies only `brain.name: crhqlstm` with no config overrides
- **THEN** all fields SHALL use their default values and the brain SHALL initialize successfully

### Requirement: CRH-QLSTM reservoir feature extraction

The brain SHALL instantiate a CRH classical reservoir (Echo State Network) internally and use it exclusively as a feature extractor. The reservoir SHALL produce features based on configured feature channels (raw activations, cos/sin transforms, pairwise products). The reservoir SHALL NOT be trained.

#### Scenario: Feature extraction pipeline

- **WHEN** `run_brain()` is called with sensory input
- **THEN** the brain SHALL preprocess sensory input, pass features through the CRH classical reservoir, and produce a feature vector whose dimension depends on the configured feature channels

### Requirement: CRH-QLSTM shared behavior with QRH-QLSTM

The CRH-QLSTM brain SHALL share all non-reservoir behavior with QRH-QLSTM via a common `ReservoirLSTMBase` class. This includes: QLIF-LSTM temporal readout, actor-critic architecture ([features, h_t] input), recurrent PPO with truncated BPTT, LR scheduling, feature normalization, and brain lifecycle.

#### Scenario: Identical readout behavior

- **WHEN** QRH-QLSTM and CRH-QLSTM receive the same reservoir feature vector
- **THEN** the LSTM readout, actor, critic, and PPO training SHALL behave identically

#### Scenario: Classical ablation comparison

- **WHEN** QRH-QLSTM and CRH-QLSTM are evaluated on the same environment with matching hyperparameters
- **THEN** performance differences SHALL be attributable to the reservoir type (quantum vs classical), not the readout implementation
