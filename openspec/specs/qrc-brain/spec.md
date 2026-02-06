# qrc-brain Specification

## Purpose

Define the requirements for the Quantum Reservoir Computing (QRC) brain architecture. QRC uses a fixed random quantum reservoir circuit to generate rich feature representations, with only a classical readout network trained via gradient descent. This architecture inherently avoids barren plateau issues that affect parameterized quantum circuits.

## ADDED Requirements

### Requirement: QRC Brain Architecture

The system SHALL support a Quantum Reservoir Computing brain architecture that uses a fixed quantum reservoir with a trainable classical readout layer.

#### Scenario: QRC Brain Instantiation

- **WHEN** a QRCBrain is instantiated with default configuration
- **THEN** the system SHALL create a quantum reservoir with configurable qubits (default 8, recommended 4 for efficiency) and 3 entangling layers
- **AND** SHALL create a trainable MLP readout network with configurable hidden units (default 32)
- **AND** SHALL initialize the reservoir with a deterministic seed for reproducibility
- **AND** SHALL use orthogonal weight initialization for better gradient flow

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qrc --config config.yml`
- **THEN** the system SHALL initialize a QRCBrain instance
- **AND** the simulation SHALL proceed using the quantum reservoir for feature extraction

### Requirement: Fixed Quantum Reservoir Circuit

The QRCBrain SHALL implement a fixed quantum reservoir that generates diverse feature representations without trainable quantum parameters.

#### Scenario: Reservoir Circuit Construction

- **WHEN** a QRCBrain constructs its reservoir circuit
- **THEN** the system SHALL apply Hadamard gates to all qubits for initial superposition
- **AND** SHALL use data re-uploading pattern: interleave input encoding with reservoir layers
- **AND** SHALL apply random RX, RY, RZ rotations to each qubit per layer (angles in [0, π/2], determined by seed)
- **AND** SHALL apply CZ entangling gates in a circular topology between adjacent qubits
- **AND** SHALL repeat the [input encoding → rotation → entangling] pattern for the configured number of layers

#### Scenario: Reservoir Reproducibility

- **WHEN** two QRCBrain instances are created with the same `reservoir_seed`
- **THEN** both instances SHALL produce identical reservoir circuits
- **AND** SHALL generate identical outputs for the same input

#### Scenario: Reservoir Immutability

- **WHEN** the QRCBrain is trained over multiple episodes
- **THEN** the quantum reservoir circuit parameters SHALL remain unchanged
- **AND** only the classical readout network parameters SHALL be updated

### Requirement: Sensory Input Encoding

The QRCBrain SHALL encode sensory inputs from BrainParams into the quantum reservoir using rotation gates.

#### Scenario: Input Feature Encoding

- **WHEN** the QRCBrain receives a BrainParams input
- **THEN** the system SHALL compute gradient strength normalized to [0, 1]
- **AND** SHALL compute relative angle normalized to [-1, 1]
- **AND** SHALL encode features using data re-uploading: input is encoded before EACH reservoir layer
- **AND** SHALL encode even-indexed features (e.g., gradient_strength) as RY rotations on ALL qubits
- **AND** SHALL encode odd-indexed features (e.g., relative_angle) as RZ rotations on ALL qubits

#### Scenario: Input Scaling

- **WHEN** encoding a feature value `v`
- **THEN** the system SHALL apply rotation angle `θ = v * π`
- **AND** SHALL apply the rotation to ALL reservoir qubits (dense encoding)

### Requirement: Reservoir State Extraction

The QRCBrain SHALL extract the quantum reservoir state as a probability distribution for the classical readout.

#### Scenario: Measurement-Based State Extraction

- **WHEN** the reservoir circuit is executed with encoded input
- **THEN** the system SHALL measure all qubits in the computational basis
- **AND** SHALL compute the probability distribution over all 2^n bitstrings
- **AND** SHALL return a numpy array of shape (2^num_qubits,)

#### Scenario: Shot-Based Probability Estimation

- **WHEN** extracting reservoir state with `shots` measurement repetitions
- **THEN** the system SHALL execute the circuit `shots` times (default 1024)
- **AND** SHALL estimate probabilities as `count / total_shots` for each bitstring

### Requirement: Classical Readout Network

The QRCBrain SHALL implement a trainable classical readout network that maps reservoir states to action logits.

#### Scenario: MLP Readout Architecture

- **WHEN** the readout_type is "mlp"
- **THEN** the system SHALL create a two-layer neural network
- **AND** SHALL have input dimension equal to 2^num_qubits
- **AND** SHALL have hidden dimension equal to `readout_hidden` (default 32)
- **AND** SHALL have output dimension equal to `num_actions` as provided to `QRCBrain.__init__`
- **AND** SHALL use ReLU activation between layers

#### Scenario: Linear Readout Architecture

- **WHEN** the readout_type is "linear"
- **THEN** the system SHALL create a single linear layer
- **AND** SHALL have input dimension equal to 2^num_qubits
- **AND** SHALL have output dimension equal to `num_actions` as provided to `QRCBrain.__init__`

### Requirement: REINFORCE Learning for Readout

The QRCBrain SHALL train the classical readout using REINFORCE policy gradients, matching the MLPReinforceBrain algorithm.

#### Scenario: Episode-Level Learning

- **WHEN** an episode completes with a sequence of (state, action, reward) tuples
- **THEN** the system SHALL compute discounted returns: `G_t = r_t + γ·G_{t+1}`
- **AND** SHALL normalize returns for variance reduction
- **AND** SHALL compute policy loss: `L = -Σ log_prob(a_t) · (G_t - baseline)`
- **AND** SHALL compute entropy bonus: `H = -Σ π(a) log π(a)` for exploration
- **AND** SHALL compute total loss: `L_total = L_policy - entropy_coef * H`
- **AND** SHALL backpropagate through the readout network only
- **AND** SHALL apply gradient clipping (max_norm=1.0)
- **AND** SHALL update readout parameters using Adam optimizer

#### Scenario: Action Selection

- **WHEN** the QRCBrain selects an action given a reservoir state
- **THEN** the system SHALL pass the reservoir state through the readout network
- **AND** SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution
- **AND** SHALL store the log probability for policy gradient computation

### Requirement: ClassicalBrain Protocol Compliance

The QRCBrain SHALL implement the ClassicalBrain protocol for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods

- **WHEN** QRCBrain is used in a simulation
- **THEN** it SHALL implement `run_brain(params, reward, input_data, top_only, top_randomize)`
- **AND** SHALL implement `learn(params, reward, episode_done)`
- **AND** SHALL implement `update_memory(reward)`
- **AND** SHALL implement `prepare_episode()` and `post_process_episode(episode_success)`
- **AND** SHALL implement `copy()` returning an independent clone

#### Scenario: Data Structure Compatibility

- **WHEN** QRCBrain returns brain data
- **THEN** it SHALL return ActionData with action, probability, and metadata
- **AND** SHALL populate BrainData fields compatible with existing tracking

### Requirement: QRC Configuration Schema

The configuration system SHALL support QRC-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Parameters

- **WHEN** parsing a QRCBrain configuration
- **THEN** the system SHALL accept `num_reservoir_qubits` (int, default 8)
- **AND** SHALL accept `reservoir_depth` (int, default 3)
- **AND** SHALL accept `reservoir_seed` (int, default 42)
- **AND** SHALL accept `readout_hidden` (int, default 32)
- **AND** SHALL accept `readout_type` (literal "mlp" | "linear", default "mlp")
- **AND** SHALL accept `shots` (int, default 1024)
- **AND** SHALL accept `entropy_coef` (float, default 0.01) for exploration regularization
- **AND** SHALL accept `baseline_alpha` (float, default 0.05) for baseline smoothing
- **AND** SHALL accept standard learning parameters (gamma, learning_rate)

#### Scenario: Configuration Validation

- **WHEN** validating QRCBrain configuration
- **THEN** the system SHALL require `num_reservoir_qubits` >= 2
- **AND** SHALL require `reservoir_depth` >= 1
- **AND** SHALL require `readout_hidden` >= 1
- **AND** SHALL require `shots` >= 100
