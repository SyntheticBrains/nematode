# brain-architecture Specification

## Purpose

Define the requirements for brain architectures that control agent decision-making in simulation environments. This specification covers multiple brain types (MLP, Spiking, Quantum Modular, QMLP) with their learning algorithms, configuration schemas, and integration with the simulation infrastructure.

## Requirements

### Requirement: Spiking Neural Network Architecture Support

The system SHALL support a spiking neural network brain architecture using surrogate gradient descent for learning, enabling biologically-plausible neural dynamics with effective gradient-based optimization.

#### Scenario: CLI Brain Selection

**Given** a user wants to run a simulation with a spiking neural network brain
**When** they execute `python scripts/run_simulation.py --brain spiking --config config.yml`
**Then** the system SHALL initialize a SpikingBrain instance using surrogate gradient descent
**And** the simulation SHALL proceed using LIF neural dynamics for decision-making
**And** SHALL learn via policy gradient optimization (REINFORCE)

#### Scenario: Configuration Loading

**Given** a configuration file specifies brain type as "spiking"
**When** the configuration is loaded
**Then** the system SHALL validate spiking-specific parameters
**And** SHALL require policy gradient parameters (gamma, baseline_alpha, learning_rate)
**And** SHALL require network architecture parameters (num_timesteps, num_hidden_layers, hidden_size)
**And** SHALL initialize the spiking brain with surrogate gradient learning enabled

### Requirement: Biological Neural Dynamics

The SpikingBrain SHALL implement biologically plausible LIF neuron dynamics while maintaining differentiability for gradient-based learning through surrogate gradients.

#### Scenario: LIF Neuron Forward Pass

**Given** a LIF neuron layer receives input current
**When** the membrane potential is updated using leaky integration
**Then** the neuron SHALL generate spikes when potential exceeds threshold
**And** SHALL reset membrane potential for spiking neurons
**And** SHALL maintain membrane state across timesteps within an episode

#### Scenario: Surrogate Gradient Backward Pass

**Given** a spike has occurred during the forward pass
**When** gradients are backpropagated through the spike function
**Then** the system SHALL use a smooth surrogate gradient approximation
**And** SHALL enable gradient flow to upstream layers
**And** SHALL use sigmoid-based surrogate: `∂spike/∂v ≈ α·σ(α(v - v_th))·(1 - σ(α(v - v_th)))`

#### Scenario: Temporal Dynamics Simulation

**Given** a state input to the spiking network
**When** simulating for `num_timesteps` steps
**Then** each LIF layer SHALL update membrane potentials at each timestep
**And** SHALL accumulate spikes over the simulation period
**And** SHALL convert total spike counts to action logits

### Requirement: Input/Output Encoding

The SpikingBrain SHALL encode continuous environmental states as constant currents and decode spike patterns to action probabilities, using relative angles between agent orientation and goal direction.

#### Scenario: State Preprocessing with Relative Angles

**Given** environmental state with gradient strength and gradient direction
**When** preprocessing the state
**Then** the system SHALL compute gradient strength normalized to [0, 1]
**And** SHALL compute relative angle: `(gradient_direction - agent_facing_angle + π) mod 2π - π`
**And** SHALL normalize relative angle to [-1, 1]: `rel_angle / π`
**And** SHALL produce feature vector: `[grad_strength, rel_angle_normalized]`
**And** SHALL match MLPBrain preprocessing exactly

#### Scenario: Current-Based Input Encoding

**Given** preprocessed state features [grad_strength, rel_angle]
**When** encoding to neural input
**Then** the system SHALL optionally apply population coding (multiple neurons per feature with Gaussian tuning)
**And** SHALL pass features through a linear layer to produce input currents
**And** SHALL apply the same input current at each simulation timestep (constant input)
**And** SHALL NOT use stochastic Poisson spike generation
**And** SHALL enable deterministic forward passes for reduced variance

#### Scenario: Population Coding (Optional)

**Given** population_coding is enabled in configuration
**When** encoding input features
**Then** the system SHALL expand each feature to neurons_per_feature neurons (default 8)
**And** SHALL use Gaussian tuning curves with population_sigma width (default 0.25)
**And** SHALL improve input discrimination for gradient-based inputs

#### Scenario: Spike-Based Action Selection

**Given** output layer spike counts accumulated over simulation period
**When** selecting an action
**Then** the system SHALL sum spikes over all timesteps for each output neuron
**And** SHALL pass total spike counts through a linear layer to produce action logits
**And** SHALL apply softmax to obtain action probabilities
**And** SHALL sample action from categorical distribution
**And** SHALL store log probability for policy gradient learning

### Requirement: Protocol Compatibility

The SpikingBrain SHALL implement the ClassicalBrain protocol while using policy gradient learning instead of STDP.

#### Scenario: Brain Interface Compliance

**Given** the existing brain architecture framework
**When** SpikingBrain is instantiated
**Then** it SHALL implement all required ClassicalBrain methods
**And** SHALL return compatible data structures (ActionData, BrainData)
**And** SHALL integrate with existing simulation infrastructure
**And** SHALL use episode buffers for state/action/reward sequences
**And** SHALL support intra-episode updates via update_frequency parameter (0 = end of episode only)

#### Scenario: Configuration Schema for Policy Gradients

**Given** a YAML configuration for spiking brain
**When** parsing configuration parameters
**Then** the system SHALL accept policy gradient parameters: gamma, baseline_alpha, entropy_beta, entropy_beta_final, entropy_decay_episodes
**And** SHALL accept network architecture parameters: num_timesteps, num_hidden_layers, hidden_size
**And** SHALL accept LIF neuron parameters: tau_m, v_threshold, v_reset
**And** SHALL accept learning parameters: learning_rate, lr_decay_rate, surrogate_alpha, update_frequency
**And** SHALL accept input encoding parameters: population_coding, neurons_per_feature, population_sigma
**And** SHALL accept initialization parameters: weight_init (orthogonal, kaiming, default)
**And** SHALL reject STDP-specific parameters with meaningful error messages

### Requirement: Brain Factory Extension

The brain factory method SHALL support spiking neural network instantiation.

#### Scenario: Brain Type Resolution

**Given** a configuration specifies brain type as "spiking"
**When** the brain factory creates a brain instance
**Then** it SHALL return a SpikingBrain object
**And** SHALL pass through all spiking-specific configuration parameters

### Requirement: CLI Argument Extension

The command-line interface SHALL accept "spiking" as a valid brain type option.

#### Scenario: Argument Validation

**Given** a user specifies `--brain spiking`
**When** command-line arguments are parsed
**Then** the system SHALL recognize "spiking" as a valid brain type
**And** SHALL pass the selection to the brain factory

### Requirement: Evolutionary Parameter Optimization

The system SHALL support evolutionary optimization of brain parameters as an alternative to gradient-based learning.

#### Scenario: CMA-ES Optimization

- **GIVEN** a quantum brain with N trainable parameters
- **WHEN** the user runs evolution with CMA-ES algorithm
- **THEN** the system SHALL create a population of candidate parameter sets
- **AND** SHALL evaluate fitness by running multiple episodes per candidate
- **AND** SHALL update the search distribution based on fitness rankings
- **AND** SHALL return the best-performing parameters after convergence

#### Scenario: Genetic Algorithm Optimization

- **GIVEN** a quantum brain with N trainable parameters
- **WHEN** the user runs evolution with genetic algorithm
- **THEN** the system SHALL maintain a population of parameter sets
- **AND** SHALL select elite performers for reproduction
- **AND** SHALL apply crossover and mutation to generate offspring
- **AND** SHALL return the best-performing parameters after generations complete

#### Scenario: Parallel Fitness Evaluation

- **GIVEN** a population of candidate parameter sets
- **WHEN** fitness evaluation is requested with parallel workers > 1
- **THEN** the system SHALL evaluate candidates concurrently using multiprocessing
- **AND** SHALL aggregate episode results into per-candidate fitness scores

### Requirement: Fitness Function Interface

The system SHALL provide a configurable fitness function for evolutionary optimization.

#### Scenario: Success Rate Fitness

- **GIVEN** a candidate parameter set
- **WHEN** fitness is evaluated with episodes_per_evaluation = N
- **THEN** the system SHALL run N episodes with those parameters
- **AND** SHALL compute fitness as negative success rate (for minimization)
- **AND** SHALL reset the environment between episodes

#### Scenario: Fitness Aggregation

- **GIVEN** multiple episode results for a candidate
- **WHEN** aggregating to a single fitness value
- **THEN** the system SHALL compute mean success rate across episodes
- **AND** MAY optionally penalize high variance

### Requirement: Brain Parameter Interface

Brain implementations SHALL expose a uniform interface for parameter manipulation required by evolutionary optimization.

#### Scenario: Parameter Export

- **GIVEN** a brain instance with trainable parameters
- **WHEN** `brain.get_parameter_array()` is called
- **THEN** the system SHALL return a flat numpy array of all trainable parameters
- **AND** SHALL maintain consistent ordering across calls

#### Scenario: Parameter Import

- **GIVEN** a brain instance and a parameter array
- **WHEN** `brain.set_parameter_array(params)` is called
- **THEN** the system SHALL update all trainable parameters from the array
- **AND** SHALL validate array length matches expected parameter count

#### Scenario: Brain Copying

- **GIVEN** a brain instance
- **WHEN** `brain.copy()` is called
- **THEN** the system SHALL return an independent copy of the brain
- **AND** modifications to the copy SHALL NOT affect the original

### Requirement: Evolution Configuration

The configuration system SHALL support evolutionary optimization parameters.

#### Scenario: CMA-ES Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** algorithm is set to "cmaes"
- **THEN** the system SHALL accept population_size, generations, sigma0 parameters
- **AND** SHALL validate parameter ranges

#### Scenario: GA Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** algorithm is set to "ga"
- **THEN** the system SHALL accept elite_fraction, mutation_rate, crossover_rate parameters
- **AND** SHALL validate parameter ranges

#### Scenario: Parallel Workers Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** parallel_workers is specified
- **THEN** the system SHALL use that many processes for fitness evaluation
- **AND** SHALL default to 1 (sequential) if not specified

### Requirement: Evolution Script Interface

The system SHALL provide a command-line script for running evolutionary optimization.

#### Scenario: Basic Evolution Run

- **GIVEN** a user wants to optimize brain parameters
- **WHEN** they execute `python scripts/run_evolution.py --config evolution.yml`
- **THEN** the system SHALL load the configuration
- **AND** SHALL run evolutionary optimization
- **AND** SHALL log generation progress (best fitness, mean, std)
- **AND** SHALL save best parameters on completion

#### Scenario: Evolution Checkpoint Resume

- **GIVEN** an interrupted evolution run with checkpoint file
- **WHEN** the user runs with `--resume checkpoint.pkl`
- **THEN** the system SHALL load the optimizer state from checkpoint
- **AND** SHALL continue evolution from the saved generation

### Requirement: Policy Gradient Learning (REINFORCE)

The SpikingBrain SHALL implement policy gradient learning with discounted returns, matching the proven algorithm used by MLPBrain.

#### Scenario: Episode-Level Learning

**Given** an episode has completed with a sequence of (state, action, reward) tuples
**When** the learning phase begins
**Then** the system SHALL compute discounted returns backward through the episode: `G_t = r_t + γ·G_{t+1}`
**And** SHALL normalize returns for variance reduction: `G' = (G - mean(G)) / std(G)`
**And** SHALL compute advantages using baseline: `A_t = G'_t - baseline`
**And** SHALL update baseline with running average: `baseline ← α·episode_return + (1-α)·baseline`

#### Scenario: Policy Gradient Update

**Given** computed advantages for each timestep
**When** updating network parameters
**Then** the system SHALL compute policy loss: `L = -Σ log_prob(a_t) · A_t`
**And** SHALL backpropagate gradients through the spiking network
**And** SHALL clip individual gradient values to [-1, 1]
**And** SHALL clip gradient norm to max_norm=1.0
**And** SHALL update parameters using Adam optimizer
**And** SHALL clear episode buffers after update

#### Scenario: Gradient Clipping

**Given** policy gradients have been computed
**When** gradients exceed the maximum norm threshold
**Then** the system SHALL clip individual values to [-1, 1] first
**And** SHALL scale gradients to max_norm=1.0
**And** SHALL prevent gradient explosion through long spike sequences

### Requirement: Surrogate Gradient Configuration

The configuration system SHALL support surrogate gradient parameters for controlling the differentiable spike approximation.

#### Scenario: Surrogate Alpha Parameter

**Given** a spiking brain configuration
**When** the surrogate_alpha parameter is specified
**Then** the system SHALL use this value to control gradient smoothness
**And** SHALL default to 1.0 if not specified (10.0 causes gradient explosion)
**And** SHALL validate that surrogate_alpha > 0

#### Scenario: Network Architecture Parameters

**Given** a spiking brain configuration
**When** num_timesteps and num_hidden_layers are specified
**Then** the system SHALL simulate the network for num_timesteps steps per decision (default 100)
**And** SHALL create num_hidden_layers LIF layers in the network (default 2)
**And** SHALL use hidden_size neurons per layer (default 256)
**And** SHALL validate num_timesteps >= 10 and num_hidden_layers >= 1

### Requirement: Gradient Flow Testing

The system SHALL provide testing utilities to verify gradient flow through spiking layers.

#### Scenario: Finite Difference Gradient Check

**Given** a LIFLayer with random weights
**When** computing gradients via backpropagation and finite differences
**Then** the gradients SHALL match within tolerance (relative error < 1e-4)
**And** SHALL verify gradient flow through all network parameters
**And** SHALL detect gradient vanishing or explosion issues

### Requirement: Learning Convergence Validation

The system SHALL validate that the spiking brain learns successfully on standard tasks.

#### Scenario: Foraging Task Learning

**Given** a spiking brain with surrogate gradients on foraging environments
**When** trained for 100-200 episodes
**Then** the average reward SHALL show a positive trend
**And** the success rate SHALL reach 100% on static and dynamic foraging (matching MLP)
**And** the success rate SHALL reach >60% on predator environments (vs 92% MLP)
**And** the learning curve SHALL be comparable to MLPBrain within 2x episode count

#### Scenario: Loss Decrease Over Episodes

**Given** a spiking brain training on any environment
**When** monitoring policy loss across episodes
**Then** the policy loss SHALL generally decrease over time
**And** SHALL not diverge (loss increasing consistently for >20 episodes)
**And** SHALL indicate successful gradient-based learning
