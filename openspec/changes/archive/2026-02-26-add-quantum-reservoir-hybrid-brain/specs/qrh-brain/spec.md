# qrh-brain Specification

## Purpose

Define the requirements for the Quantum Reservoir Hybrid (QRH) brain architecture. QRH uses a fixed structured quantum reservoir with C. elegans-inspired topology to generate biologically-grounded feature representations, with a classical actor-critic readout trained via PPO. The structured topology and X/Y/Z+ZZ feature extraction address the failure of QRC's random reservoir approach.

## ADDED Requirements

### Requirement: QRH Brain Architecture

The system SHALL support a Quantum Reservoir Hybrid brain architecture that uses a fixed structured quantum reservoir with a PPO-trained classical actor-critic readout.

#### Scenario: QRH Brain Instantiation

- **WHEN** a QRHBrain is instantiated with default configuration
- **THEN** the system SHALL create a structured quantum reservoir with 8 qubits and 3 entanglement layers
- **AND** SHALL create a classical actor MLP (64 hidden units, 2 layers) for action logits
- **AND** SHALL create a classical critic MLP (64 hidden units, 2 layers) for value estimation
- **AND** SHALL initialize the reservoir with a deterministic seed for reproducibility
- **AND** SHALL use orthogonal weight initialization for the readout networks
- **AND** SHALL compute a reservoir feature dimension of 3N + N(N-1)/2 (52 for 8 qubits, 75 for 10 qubits)

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qrh --config config.yml`
- **THEN** the system SHALL initialize a QRHBrain instance
- **AND** the simulation SHALL proceed using the structured quantum reservoir for feature extraction
- **AND** SHALL train the readout via PPO

### Requirement: Structured Quantum Reservoir Circuit

The QRHBrain SHALL implement a fixed quantum reservoir with C. elegans-inspired topology that generates discriminative feature representations without trainable quantum parameters.

#### Scenario: Structured Reservoir Construction

- **WHEN** a QRHBrain constructs its reservoir circuit with `use_random_topology=False` (default)
- **THEN** the system SHALL map the C. elegans sensory-interneuron subnetwork onto qubits: ASEL/ASER (qubits 0-1), AIYL/AIYR (qubits 2-3), AIAL/AIAR (qubits 4-5), AVAL/AVAR (qubits 6-7)
- **AND** SHALL apply Hadamard gates to all qubits for initial superposition
- **AND** SHALL use data re-uploading: encode input features as RY/RZ rotations on sensory qubits before each reservoir layer
- **AND** SHALL apply CZ gates for gap junction connections (bilateral neuron pairs + feedforward)
- **AND** SHALL apply controlled rotations (CRY/CRZ) for chemical synapse connections with angles derived from connectome synaptic weights (Cook et al. 2019)
- **AND** SHALL repeat the [input encoding → CZ entanglement → fixed rotations] pattern for the configured number of layers

#### Scenario: Random Reservoir Construction (MI Comparison)

- **WHEN** a QRHBrain constructs its reservoir circuit with `use_random_topology=True`
- **THEN** the system SHALL generate random CZ pairs with the same density as the structured topology
- **AND** SHALL generate random CRY/CRZ rotation angles using the reservoir seed
- **AND** SHALL maintain the same data re-uploading and Hadamard initialization pattern

#### Scenario: Reservoir Reproducibility

- **WHEN** two QRHBrain instances are created with the same `reservoir_seed` and `use_random_topology` setting
- **THEN** both instances SHALL produce identical reservoir circuits
- **AND** SHALL generate identical feature vectors for the same input

#### Scenario: Reservoir Immutability

- **WHEN** the QRHBrain is trained over multiple episodes
- **THEN** the quantum reservoir circuit parameters SHALL remain unchanged
- **AND** only the classical readout network parameters SHALL be updated

### Requirement: X/Y/Z-Expectation and ZZ-Correlation Feature Extraction

The QRHBrain SHALL extract per-qubit X/Y/Z-expectations and pairwise ZZ-correlations from the quantum reservoir state, providing richer features than raw probability distributions.

#### Scenario: Feature Extraction from Statevector

- **WHEN** the reservoir circuit is executed and the statevector |ψ⟩ is obtained
- **THEN** the system SHALL compute per-qubit X/Y/Z-expectations from probability amplitudes (full Bloch sphere per qubit)
- **AND** SHALL compute pairwise ZZ-correlations: ⟨Z_i Z_j⟩ = Σ_k (-1)^(bit(k,i)+bit(k,j)) |ψ_k|² for each pair (i,j) where i < j
- **AND** SHALL concatenate X-expectations, Y-expectations, Z-expectations, and ZZ-correlations into a single feature vector
- **AND** SHALL return a numpy array of shape (3N + N(N-1)/2,) where N is the number of qubits
- **AND** SHALL apply LayerNorm to normalize heterogeneous feature scales before the readout MLPs

#### Scenario: Feature Value Range

- **WHEN** feature extraction produces X/Y/Z-expectations and ZZ-correlations
- **THEN** all expectation values SHALL be in the range [-1, 1]
- **AND** all ZZ-correlation values SHALL be in the range [-1, 1]

#### Scenario: Feature Sensitivity

- **WHEN** different sensory inputs are encoded into the reservoir
- **THEN** the extracted features SHALL produce statistically different feature vectors
- **AND** different inputs SHALL NOT produce identical feature vectors (non-degeneracy)

NOTE: The hypothesis that structured topology produces higher mutual information than random topology is validated by the MI decision gate script (`scripts/qrh_mi_analysis.py`), not by unit tests.

### Requirement: Statevector Simulation

The QRHBrain SHALL use statevector simulation for exact feature computation.

#### Scenario: Statevector Execution

- **WHEN** the reservoir circuit is executed for feature extraction
- **THEN** the system SHALL use Qiskit Statevector simulation (not shot-based measurement)
- **AND** SHALL compute exact probability amplitudes without shot noise
- **AND** SHALL NOT add measurement gates to the circuit

### Requirement: PPO Training for Readout

The QRHBrain SHALL train the classical actor-critic readout using Proximal Policy Optimization with Generalized Advantage Estimation.

#### Scenario: Rollout Buffer Collection

- **WHEN** the QRHBrain interacts with the environment
- **THEN** the system SHALL store (state, action, log_prob, value, reward, done) tuples in a rollout buffer
- **AND** SHALL trigger a PPO update when the buffer reaches the configured size (default 512 steps)
- **AND** SHALL also trigger a PPO update at episode boundaries if the buffer contains sufficient data

#### Scenario: PPO Update

- **WHEN** the rollout buffer is full or the episode ends with sufficient data
- **THEN** the system SHALL compute GAE advantages with γ=0.99 and λ=0.95
- **AND** SHALL normalize advantages to zero mean and unit variance
- **AND** SHALL iterate over the buffer for the configured number of PPO epochs (default 4)
- **AND** SHALL split each epoch into minibatches (default 4)
- **AND** SHALL compute the clipped surrogate policy loss with clip_ε=0.2
- **AND** SHALL compute value function loss (MSE) weighted by `value_loss_coef` (default 0.5)
- **AND** SHALL compute entropy bonus for exploration weighted by `entropy_coeff`
- **AND** SHALL update actor, critic, and feature normalization via a single combined Adam optimizer
- **AND** SHALL apply gradient clipping (max_norm=0.5)

#### Scenario: Action Selection

- **WHEN** the QRHBrain selects an action given reservoir features
- **THEN** the system SHALL pass features through the actor MLP to get action logits
- **AND** SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution
- **AND** SHALL compute the value estimate from the critic MLP
- **AND** SHALL store the log probability and value for the PPO buffer

### Requirement: ClassicalBrain Protocol Compliance

The QRHBrain SHALL implement the ClassicalBrain protocol for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods

- **WHEN** QRHBrain is used in a simulation
- **THEN** it SHALL implement `run_brain(params, reward, input_data, top_only, top_randomize)`
- **AND** SHALL implement `learn(params, reward, episode_done)`
- **AND** SHALL implement `update_memory(reward)`
- **AND** SHALL implement `prepare_episode()` and `post_process_episode(episode_success)`
- **AND** SHALL implement `copy()` returning an independent clone

#### Scenario: Data Structure Compatibility

- **WHEN** QRHBrain returns brain data
- **THEN** it SHALL return ActionData with action, probability, and metadata
- **AND** SHALL populate BrainData fields compatible with existing tracking

#### Scenario: Brain Copy Independence

- **WHEN** `qrh_brain.copy()` is called
- **THEN** the system SHALL return an independent copy of the QRHBrain
- **AND** the copy SHALL produce identical reservoir circuits for the same input (same seed, same topology)
- **AND** the copy SHALL have independent readout network weights
- **AND** modifications to the copy's readout SHALL NOT affect the original

### Requirement: QRH Configuration Schema

The configuration system SHALL support QRH-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Parameters

- **WHEN** parsing a QRHBrain configuration
- **THEN** the system SHALL accept `num_reservoir_qubits` (int, default 8)
- **AND** SHALL accept `reservoir_depth` (int, default 3)
- **AND** SHALL accept `reservoir_seed` (int, default 42)
- **AND** SHALL accept `shots` (int, default 1024)
- **AND** SHALL accept `readout_hidden_dim` (int, default 64)
- **AND** SHALL accept `readout_num_layers` (int, default 2)
- **AND** SHALL accept `actor_lr` (float, default 0.0003)
- **AND** SHALL accept `critic_lr` (float, default 0.0003)
- **AND** SHALL accept `gamma` (float, default 0.99)
- **AND** SHALL accept `gae_lambda` (float, default 0.95)
- **AND** SHALL accept `ppo_clip_epsilon` (float, default 0.2)
- **AND** SHALL accept `ppo_epochs` (int, default 4)
- **AND** SHALL accept `ppo_minibatches` (int, default 4)
- **AND** SHALL accept `ppo_buffer_size` (int, default 512)
- **AND** SHALL accept `entropy_coeff` (float, default 0.01)
- **AND** SHALL accept `max_grad_norm` (float, default 0.5)
- **AND** SHALL accept `value_loss_coef` (float, default 0.5)
- **AND** SHALL accept `use_random_topology` (bool, default False)
- **AND** SHALL accept `num_sensory_qubits` (int | None, default None) — auto-computed from `sensory_modules` if not set
- **AND** SHALL accept `lr_warmup_episodes` (int, default 0) — number of episodes for LR warmup phase
- **AND** SHALL accept `lr_warmup_start` (float | None, default None) — starting LR during warmup
- **AND** SHALL accept `lr_decay_episodes` (int, default 0) — number of episodes for LR decay phase
- **AND** SHALL accept `lr_decay_end` (float | None, default None) — ending LR after decay
- **AND** SHALL accept `sensory_modules` (list of ModuleName or None, default None)

#### Scenario: Configuration Validation

- **WHEN** validating QRHBrain configuration
- **THEN** the system SHALL require `num_reservoir_qubits` >= 2
- **AND** SHALL require `reservoir_depth` >= 1
- **AND** SHALL require `shots` >= 100
- **AND** SHALL require `readout_hidden_dim` >= 1

### Requirement: Sensory Module Integration

The QRHBrain SHALL support both legacy and unified sensory module input modes.

#### Scenario: Legacy Input Mode

- **WHEN** `sensory_modules` is None (default)
- **THEN** the system SHALL use 2 features: gradient_strength [0, 1] and relative_angle [-1, 1]

#### Scenario: Unified Sensory Input Mode

- **WHEN** `sensory_modules` is configured with a list of ModuleName values
- **THEN** the system SHALL use `extract_classical_features()` from `brain/modules.py`
- **AND** SHALL compute feature dimension as the sum of each module's classical_dim
- **AND** SHALL encode all features into the reservoir via data re-uploading
