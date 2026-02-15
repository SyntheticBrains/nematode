# qsnn-reinforce-brain Specification

## Purpose

Define the requirements for the Quantum Spiking Neural Network (QSNN) brain architecture. QSNN implements Quantum Leaky Integrate-and-Fire (QLIF) neurons with trainable parameters and hybrid quantum-classical learning. This architecture addresses QRC's representation problem (trainable quantum parameters) while avoiding QVarCircuitBrain's barren plateau issue (classical surrogate gradients instead of quantum backpropagation).

The QSNN Reinforce variant (`QSNNReinforceBrain`) uses REINFORCE policy gradient with surrogate gradient backward pass. Shared QLIF neuron infrastructure is extracted into `_qlif_layers.py` for reuse by other QSNN variants.

## ADDED Requirements

### Requirement: QSNN Brain Architecture

The system SHALL support a Quantum Spiking Neural Network brain architecture that uses QLIF neurons organized in layers with trainable synaptic weights.

#### Scenario: QSNN Brain Instantiation

- **WHEN** a QSNNReinforceBrain is instantiated with default configuration
- **THEN** the system SHALL create a network with configurable sensory neurons (default 8), hidden neurons (default 16), and motor neurons (default 4 matching action space)
- **AND** SHALL initialize trainable synaptic weight matrices between layers
- **AND** SHALL initialize the network with a deterministic seed for reproducibility

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qsnnreinforce --config config.yml`
- **THEN** the system SHALL initialize a QSNNReinforceBrain instance
- **AND** the simulation SHALL proceed using quantum spiking dynamics for action selection

### Requirement: Quantum Leaky Integrate-and-Fire Neuron

The QSNNReinforceBrain SHALL implement QLIF neurons using a minimal 2-gate quantum circuit per neuron based on Brand & Petruccione (2024).

#### Scenario: QLIF Circuit Structure

- **WHEN** a QLIF neuron processes input
- **THEN** the system SHALL apply the circuit: `|0> -> RY(theta_membrane + tanh(weighted_input / sqrt(fan_in)) * pi) -> RX(theta_leak) -> Measure`
- **AND** SHALL use `theta_membrane` as the trainable membrane potential parameter
- **AND** SHALL compute `weighted_input` as the sum of `w_ij * spike_j` for all presynaptic neurons
- **AND** SHALL apply fan-in-aware scaling (`/ sqrt(fan_in)`) to prevent gradient death in wider networks
- **AND** SHALL compute `theta_leak` as `(1 - membrane_tau) * pi` where `membrane_tau` is the leak time constant

#### Scenario: Firing Probability

- **WHEN** a QLIF neuron is measured
- **THEN** the system SHALL interpret measurement outcome `|1>` as a spike (fired)
- **AND** SHALL interpret measurement outcome `|0>` as no spike
- **AND** SHALL compute firing probability from shot statistics

#### Scenario: Refractory Period

- **WHEN** a QLIF neuron fires (outputs spike)
- **THEN** the system SHALL suppress neuron activity for `refractory_period` timesteps (default 0)
- **AND** SHALL track refractory state per neuron

### Requirement: Network Topology

The QSNNReinforceBrain SHALL organize QLIF neurons into sensory, hidden, and motor layers with trainable connections.

#### Scenario: Layer Structure

- **WHEN** a QSNNReinforceBrain is constructed
- **THEN** the system SHALL create a sensory layer with neurons mapped to sensory modalities (food chemotaxis, nociception, thermotaxis, etc.)
- **AND** SHALL create a hidden interneuron layer for processing
- **AND** SHALL create a motor layer with neurons corresponding to actions (forward, backward, left, right)

#### Scenario: Synaptic Connections

- **WHEN** the network processes a timestep
- **THEN** the system SHALL compute sensory-to-hidden connections using weight matrix `W_sh`
- **AND** SHALL compute hidden-to-motor connections using weight matrix `W_hm`
- **AND** SHALL allow all weights to be updated during learning
- **AND** SHALL clamp weights to [-weight_clip, weight_clip] after updates (default weight_clip=3.0)

### Requirement: Sensory Spike Encoding

The QSNNReinforceBrain SHALL convert continuous sensory inputs from BrainParams into spike probabilities.

#### Scenario: Input Feature Encoding

- **WHEN** the QSNNReinforceBrain receives a BrainParams input
- **THEN** the system SHALL compute gradient magnitude from `gradient_x` and `gradient_y`
- **AND** SHALL convert continuous values to spike probabilities using sigmoid function
- **AND** SHALL scale inputs appropriately (e.g., `sigmoid(gradient_magnitude * 5.0)`)

#### Scenario: Multi-Sensory Support

- **WHEN** the QSNNReinforceBrain is configured with `sensory_modules`
- **THEN** the system SHALL use `extract_classical_features()` for unified feature extraction
- **AND** SHALL map each sensory module to corresponding sensory neurons

### Requirement: Timestep Dynamics

The QSNNReinforceBrain SHALL implement biologically-inspired spiking dynamics per simulation timestep.

#### Scenario: Forward Pass

- **WHEN** the QSNNReinforceBrain processes a single timestep
- **THEN** the system SHALL encode sensory inputs as spike probabilities
- **AND** SHALL propagate spikes through sensory-to-hidden-to-motor layers
- **AND** SHALL execute QLIF circuits for all neurons
- **AND** SHALL measure motor neurons to obtain firing probabilities
- **AND** SHALL convert motor firing probabilities to action logits

#### Scenario: Multi-Timestep Integration

- **WHEN** the QSNNReinforceBrain processes a decision step
- **THEN** the system SHALL average spike probabilities across `num_integration_steps` timesteps (default 10)
- **AND** SHALL reduce quantum shot noise variance proportionally

#### Scenario: Action Selection

- **WHEN** motor neuron firing probabilities are computed
- **THEN** the system SHALL apply logit scaling: `(spike_prob - 0.5) * logit_scale`
- **AND** SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution
- **AND** SHALL store action and log probability for learning

### Requirement: Surrogate Gradient Learning

The QSNNReinforceBrain SHALL implement REINFORCE policy gradient with surrogate gradient backward pass as the primary learning mode.

#### Scenario: Surrogate Gradient Backward Pass

- **WHEN** gradients are backpropagated through the spike function
- **THEN** the system SHALL use `QLIFSurrogateSpike` autograd function
- **AND** SHALL compute surrogate gradient as sigmoid centered at pi/2 (RY gate transition point)
- **AND** SHALL enable gradient flow from motor outputs through hidden layer to sensory weights

#### Scenario: REINFORCE Policy Gradient Update

- **WHEN** an episode completes or the update window is reached
- **THEN** the system SHALL compute discounted returns and normalize advantages
- **AND** SHALL compute policy loss with entropy bonus
- **AND** SHALL clip gradients and update weights via Adam optimizer

#### Scenario: Multi-Epoch REINFORCE

- **WHEN** `num_reinforce_epochs` > 1
- **THEN** epoch 0 SHALL run quantum circuits and cache spike probabilities
- **AND** subsequent epochs SHALL reuse cached spike probs but recompute RY angles from updated weights
- **AND** SHALL provide additional gradient passes without proportional quantum circuit cost

#### Scenario: Adaptive Entropy Regulation

- **WHEN** entropy drops below 0.5 nats
- **THEN** the system SHALL scale entropy_coef up to 20x to prevent policy collapse
- **WHEN** entropy exceeds 95% of maximum
- **THEN** the system SHALL suppress entropy bonus to let policy gradient sharpen

### Requirement: Hebbian Learning (Legacy Mode)

The QSNNReinforceBrain SHALL support reward-modulated Hebbian learning as a legacy alternative.

#### Scenario: Local Learning Mode

- **WHEN** `use_local_learning` is True
- **THEN** the system SHALL use 3-factor local learning rule (eligibility trace * reward)
- **WHEN** `use_local_learning` is False (default)
- **THEN** the system SHALL use REINFORCE with surrogate gradients

### Requirement: ClassicalBrain Protocol Compliance

The QSNNReinforceBrain SHALL implement the ClassicalBrain protocol for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods

- **WHEN** QSNNReinforceBrain is used in a simulation
- **THEN** it SHALL implement `run_brain(params, reward, input_data, top_only, top_randomize)`
- **AND** SHALL implement `learn(params, reward, episode_done)`
- **AND** SHALL implement `update_memory(reward)`
- **AND** SHALL implement `prepare_episode()` and `post_process_episode(episode_success)`
- **AND** SHALL implement `copy()` returning an independent clone

#### Scenario: Data Structure Compatibility

- **WHEN** QSNNReinforceBrain returns brain data
- **THEN** it SHALL return ActionData with action, probability, and metadata
- **AND** SHALL populate BrainData fields compatible with existing tracking

### Requirement: QSNN Configuration Schema

The configuration system SHALL support QSNN-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Parameters

- **WHEN** parsing a QSNNReinforceBrain configuration
- **THEN** the system SHALL accept `num_sensory_neurons` (int, default 8)
- **AND** SHALL accept `num_hidden_neurons` (int, default 16)
- **AND** SHALL accept `num_motor_neurons` (int, default 4)
- **AND** SHALL accept `membrane_tau` (float, default 0.9) for leak time constant
- **AND** SHALL accept `threshold` (float, default 0.5) for firing threshold
- **AND** SHALL accept `refractory_period` (int, default 0) for post-spike suppression
- **AND** SHALL accept `use_local_learning` (bool, default False) for learning rule selection
- **AND** SHALL accept `shots` (int, default 1024) for quantum measurement
- **AND** SHALL accept `num_integration_steps` (int, default 10) for multi-timestep integration
- **AND** SHALL accept `num_reinforce_epochs` (int, default 1) for multi-epoch REINFORCE
- **AND** SHALL accept `logit_scale` (float, default 5.0) for spike-to-logit conversion
- **AND** SHALL accept `weight_clip` (float, default 3.0) for weight clamping bounds
- **AND** SHALL accept `theta_motor_max_norm` (float, default 2.0) for theta motor L2 norm clamping
- **AND** SHALL accept `advantage_clip` (float, default 2.0) for advantage clipping
- **AND** SHALL accept standard learning parameters (gamma, learning_rate, entropy_coef)
- **AND** SHALL accept schedule parameters (exploration_decay_episodes, lr_decay_episodes, lr_min_factor)

#### Scenario: Configuration Validation

- **WHEN** validating QSNNReinforceBrain configuration
- **THEN** the system SHALL require `num_sensory_neurons` >= 1
- **AND** SHALL require `num_hidden_neurons` >= 1
- **AND** SHALL require `num_motor_neurons` >= 2
- **AND** SHALL require `membrane_tau` in range (0, 1\]
- **AND** SHALL require `threshold` in range (0, 1)
- **AND** SHALL require `shots` >= 100

### Requirement: Shared QLIF Module

The system SHALL provide a shared QLIF neuron module (`_qlif_layers.py`) for reuse across QSNN brain variants.

#### Scenario: Module Contents

- **WHEN** a QSNN brain variant imports from `_qlif_layers`
- **THEN** it SHALL have access to `build_qlif_circuit()` for circuit construction
- **AND** SHALL have access to `QLIFSurrogateSpike` for differentiable spike function
- **AND** SHALL have access to `encode_sensory_spikes()` for input encoding
- **AND** SHALL have access to `execute_qlif_layer()` for non-differentiable forward pass
- **AND** SHALL have access to `execute_qlif_layer_differentiable()` for gradient-enabled forward pass
- **AND** SHALL have access to `execute_qlif_layer_differentiable_cached()` for multi-epoch caching

### Requirement: Reproducibility

The QSNNReinforceBrain SHALL support deterministic execution for reproducibility.

#### Scenario: Seeded Initialization

- **WHEN** two QSNNReinforceBrain instances are created with the same seed
- **THEN** both instances SHALL produce identical initial weight matrices
- **AND** SHALL produce identical circuit structures

#### Scenario: Deterministic Execution

- **WHEN** the same inputs are provided with the same seed
- **THEN** the system SHALL produce identical spike patterns (given same shot count and seed)

### Requirement: Backward Compatibility

The system SHALL maintain backward compatibility with the original `qsnn` name.

#### Scenario: Config Loading

- **WHEN** a configuration file specifies brain type as `"qsnn"`
- **THEN** the system SHALL resolve this to `QSNNReinforceBrainConfig`

#### Scenario: BrainType Enum

- **WHEN** code references `BrainType.QSNN`
- **THEN** the system SHALL treat this as equivalent to `BrainType.QSNN_REINFORCE`
