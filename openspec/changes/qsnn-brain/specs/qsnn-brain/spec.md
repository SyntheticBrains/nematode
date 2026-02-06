# qsnn-brain Specification

## Purpose

Define the requirements for the Quantum Spiking Neural Network (QSNN) brain architecture. QSNN implements Quantum Leaky Integrate-and-Fire (QLIF) neurons with trainable parameters and local learning rules. This architecture addresses QRC's representation problem (trainable quantum parameters) while avoiding QVarCircuitBrain's barren plateau issue (local learning rules instead of global backpropagation).

## ADDED Requirements

### Requirement: QSNN Brain Architecture

The system SHALL support a Quantum Spiking Neural Network brain architecture that uses QLIF neurons organized in layers with trainable synaptic weights.

#### Scenario: QSNN Brain Instantiation

- **WHEN** a QSNNBrain is instantiated with default configuration
- **THEN** the system SHALL create a network with configurable sensory neurons (default 6), hidden neurons (default 4), and motor neurons (default 4-5 matching action space)
- **AND** SHALL initialize trainable synaptic weight matrices between layers
- **AND** SHALL initialize the network with a deterministic seed for reproducibility

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qsnn --config config.yml`
- **THEN** the system SHALL initialize a QSNNBrain instance
- **AND** the simulation SHALL proceed using quantum spiking dynamics for action selection

### Requirement: Quantum Leaky Integrate-and-Fire Neuron

The QSNNBrain SHALL implement QLIF neurons using a minimal 2-gate quantum circuit per neuron based on Brand & Petruccione (2024).

#### Scenario: QLIF Circuit Structure

- **WHEN** a QLIF neuron processes input
- **THEN** the system SHALL apply the circuit: `|0⟩ → RY(θ_membrane + weighted_input) → RX(θ_leak) → Measure`
- **AND** SHALL use `θ_membrane` as the trainable membrane potential parameter
- **AND** SHALL compute `weighted_input` as the sum of `w_ij × spike_j` for all presynaptic neurons
- **AND** SHALL compute `θ_leak` as `(1 - membrane_tau) × π` where `membrane_tau` is the leak time constant

#### Scenario: Firing Probability

- **WHEN** a QLIF neuron is measured
- **THEN** the system SHALL interpret measurement outcome `|1⟩` as a spike (fired)
- **AND** SHALL interpret measurement outcome `|0⟩` as no spike
- **AND** SHALL compute firing probability from shot statistics

#### Scenario: Refractory Period

- **WHEN** a QLIF neuron fires (outputs spike)
- **THEN** the system SHALL suppress neuron activity for `refractory_period` timesteps (default 2)
- **AND** SHALL track refractory state per neuron

### Requirement: Network Topology

The QSNNBrain SHALL organize QLIF neurons into sensory, hidden, and motor layers with trainable connections.

#### Scenario: Layer Structure

- **WHEN** a QSNNBrain is constructed
- **THEN** the system SHALL create a sensory layer with neurons mapped to sensory modalities (food chemotaxis, nociception, thermotaxis, etc.)
- **AND** SHALL create a hidden interneuron layer for processing
- **AND** SHALL create a motor layer with neurons corresponding to actions (forward, backward, left, right, optionally stay)

#### Scenario: Synaptic Connections

- **WHEN** the network processes a timestep
- **THEN** the system SHALL compute sensory→hidden connections using weight matrix `W_sh`
- **AND** SHALL compute hidden→motor connections using weight matrix `W_hm`
- **AND** SHALL allow all weights to be updated during learning

### Requirement: Sensory Spike Encoding

The QSNNBrain SHALL convert continuous sensory inputs from BrainParams into spike probabilities.

#### Scenario: Input Feature Encoding

- **WHEN** the QSNNBrain receives a BrainParams input
- **THEN** the system SHALL compute gradient magnitude from `gradient_x` and `gradient_y`
- **AND** SHALL convert continuous values to spike probabilities using sigmoid function
- **AND** SHALL scale inputs appropriately (e.g., `sigmoid(gradient_magnitude × 5.0)`)

#### Scenario: Multi-Sensory Support

- **WHEN** the QSNNBrain is configured with `sensory_modules`
- **THEN** the system SHALL use `extract_classical_features()` for unified feature extraction
- **AND** SHALL map each sensory module to corresponding sensory neurons

### Requirement: Timestep Dynamics

The QSNNBrain SHALL implement biologically-inspired spiking dynamics per simulation timestep.

#### Scenario: Forward Pass

- **WHEN** the QSNNBrain processes a single timestep
- **THEN** the system SHALL encode sensory inputs as spike probabilities
- **AND** SHALL propagate spikes through sensory→hidden→motor layers
- **AND** SHALL execute QLIF circuits for all neurons
- **AND** SHALL measure motor neurons to obtain firing probabilities
- **AND** SHALL convert motor firing probabilities to action logits

#### Scenario: Action Selection

- **WHEN** motor neuron firing probabilities are computed
- **THEN** the system SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution
- **AND** SHALL store action and log probability for learning

### Requirement: Local Learning Rule

The QSNNBrain SHALL implement reward-modulated Hebbian learning (3-factor rule) to avoid barren plateaus.

#### Scenario: Eligibility Trace Computation

- **WHEN** a synaptic weight update is computed
- **THEN** the system SHALL compute eligibility trace as `pre_spike × post_spike`
- **AND** SHALL maintain eligibility traces per synapse

#### Scenario: Weight Update

- **WHEN** reward is received at episode end
- **THEN** the system SHALL compute weight update as `Δw = lr × eligibility × reward`
- **AND** SHALL apply updates to all synaptic weight matrices
- **AND** SHALL optionally apply weight clipping for stability

#### Scenario: Local vs Global Learning Mode

- **WHEN** `use_local_learning` is True (default)
- **THEN** the system SHALL use the 3-factor local learning rule
- **WHEN** `use_local_learning` is False
- **THEN** the system SHALL fall back to REINFORCE with surrogate gradients for comparison

### Requirement: ClassicalBrain Protocol Compliance

The QSNNBrain SHALL implement the ClassicalBrain protocol for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods

- **WHEN** QSNNBrain is used in a simulation
- **THEN** it SHALL implement `run_brain(params, reward, input_data, top_only, top_randomize)`
- **AND** SHALL implement `learn(params, reward, episode_done)`
- **AND** SHALL implement `update_memory(reward)`
- **AND** SHALL implement `prepare_episode()` and `post_process_episode(episode_success)`
- **AND** SHALL implement `copy()` returning an independent clone

#### Scenario: Data Structure Compatibility

- **WHEN** QSNNBrain returns brain data
- **THEN** it SHALL return ActionData with action, probability, and metadata
- **AND** SHALL populate BrainData fields compatible with existing tracking

### Requirement: QSNN Configuration Schema

The configuration system SHALL support QSNN-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Parameters

- **WHEN** parsing a QSNNBrain configuration
- **THEN** the system SHALL accept `num_sensory_neurons` (int, default 6)
- **AND** SHALL accept `num_hidden_neurons` (int, default 4)
- **AND** SHALL accept `num_motor_neurons` (int, default 4)
- **AND** SHALL accept `membrane_tau` (float, default 0.9) for leak time constant
- **AND** SHALL accept `threshold` (float, default 0.5) for firing threshold
- **AND** SHALL accept `refractory_period` (int, default 2) for post-spike suppression
- **AND** SHALL accept `use_local_learning` (bool, default True) for learning rule selection
- **AND** SHALL accept `shots` (int, default 1024) for quantum measurement
- **AND** SHALL accept standard learning parameters (gamma, learning_rate)

#### Scenario: Configuration Validation

- **WHEN** validating QSNNBrain configuration
- **THEN** the system SHALL require `num_sensory_neurons` >= 1
- **AND** SHALL require `num_hidden_neurons` >= 1
- **AND** SHALL require `num_motor_neurons` >= 2
- **AND** SHALL require `membrane_tau` in range (0, 1\]
- **AND** SHALL require `threshold` in range (0, 1)
- **AND** SHALL require `shots` >= 100

### Requirement: Reproducibility

The QSNNBrain SHALL support deterministic execution for reproducibility.

#### Scenario: Seeded Initialization

- **WHEN** two QSNNBrain instances are created with the same seed
- **THEN** both instances SHALL produce identical initial weight matrices
- **AND** SHALL produce identical circuit structures

#### Scenario: Deterministic Execution

- **WHEN** the same inputs are provided with the same seed
- **THEN** the system SHALL produce identical spike patterns (given same shot count and seed)
