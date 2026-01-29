# brain-architecture Specification

## Purpose

Define the requirements for brain architectures that control agent decision-making in simulation environments. This specification covers multiple brain types (MLP, Spiking, Quantum Modular, QMLP) with their learning algorithms, configuration schemas, and integration with the simulation infrastructure.

## Requirements

### Requirement: Spiking Neural Network Architecture Support

The system SHALL support a spiking neural network brain architecture using surrogate gradient descent for learning, enabling biologically-plausible neural dynamics with effective gradient-based optimization.

#### Scenario: CLI Brain Selection

- **GIVEN** a user wants to run a simulation with a spiking neural network brain
- **WHEN** they execute `python scripts/run_simulation.py --brain spiking --config config.yml`
- **THEN** the system SHALL initialize a SpikingBrain instance using surrogate gradient descent
- **AND** the simulation SHALL proceed using LIF neural dynamics for decision-making
- **AND** SHALL learn via policy gradient optimization (REINFORCE)

#### Scenario: Configuration Loading

- **GIVEN** a configuration file specifies brain type as "spiking"
- **WHEN** the configuration is loaded
- **THEN** the system SHALL validate spiking-specific parameters
- **AND** SHALL require policy gradient parameters (gamma, baseline_alpha, learning_rate)
- **AND** SHALL require network architecture parameters (num_timesteps, num_hidden_layers, hidden_size)
- **AND** SHALL initialize the spiking brain with surrogate gradient learning enabled

### Requirement: Biological Neural Dynamics

The SpikingBrain SHALL implement biologically plausible LIF neuron dynamics while maintaining differentiability for gradient-based learning through surrogate gradients.

#### Scenario: LIF Neuron Forward Pass

- **GIVEN** a LIF neuron layer receives input current
- **WHEN** the membrane potential is updated using leaky integration
- **THEN** the neuron SHALL generate spikes when potential exceeds threshold
- **AND** SHALL reset membrane potential for spiking neurons
- **AND** SHALL maintain membrane state across timesteps within an episode

#### Scenario: Surrogate Gradient Backward Pass

- **GIVEN** a spike has occurred during the forward pass
- **WHEN** gradients are backpropagated through the spike function
- **THEN** the system SHALL use a smooth surrogate gradient approximation
- **AND** SHALL enable gradient flow to upstream layers
- **AND** SHALL use sigmoid-based surrogate: `∂spike/∂v ≈ α·σ(α(v - v_th))·(1 - σ(α(v - v_th)))`

#### Scenario: Temporal Dynamics Simulation

- **GIVEN** a state input to the spiking network
- **WHEN** simulating for `num_timesteps` steps
- **THEN** each LIF layer SHALL update membrane potentials at each timestep
- **AND** SHALL accumulate spikes over the simulation period
- **AND** SHALL convert total spike counts to action logits

### Requirement: Input/Output Encoding

The SpikingBrain SHALL encode continuous environmental states as constant currents and decode spike patterns to action probabilities, using relative angles between agent orientation and goal direction.

#### Scenario: State Preprocessing with Relative Angles

- **GIVEN** environmental state with gradient strength and gradient direction
- **WHEN** preprocessing the state
- **THEN** the system SHALL compute gradient strength normalized to [0, 1]
- **AND** SHALL compute relative angle: `(gradient_direction - agent_facing_angle + π) mod 2π - π`
- **AND** SHALL normalize relative angle to \[-1, 1\]: `rel_angle / π`
- **AND** SHALL produce feature vector: `[grad_strength, rel_angle_normalized]`
- **AND** SHALL match MLPBrain preprocessing exactly

#### Scenario: Current-Based Input Encoding

- **GIVEN** preprocessed state features [grad_strength, rel_angle]
- **WHEN** encoding to neural input
- **THEN** the system SHALL optionally apply population coding (multiple neurons per feature with Gaussian tuning)
- **AND** SHALL pass features through a linear layer to produce input currents
- **AND** SHALL apply the same input current at each simulation timestep (constant input)
- **AND** SHALL NOT use stochastic Poisson spike generation
- **AND** SHALL enable deterministic forward passes for reduced variance

#### Scenario: Population Coding (Optional)

- **GIVEN** population_coding is enabled in configuration
- **WHEN** encoding input features
- **THEN** the system SHALL expand each feature to neurons_per_feature neurons (default 8)
- **AND** SHALL use Gaussian tuning curves with population_sigma width (default 0.25)
- **AND** SHALL improve input discrimination for gradient-based inputs

#### Scenario: Spike-Based Action Selection

- **GIVEN** output layer spike counts accumulated over simulation period
- **WHEN** selecting an action
- **THEN** the system SHALL sum spikes over all timesteps for each output neuron
- **AND** SHALL pass total spike counts through a linear layer to produce action logits
- **AND** SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample action from categorical distribution
- **AND** SHALL store log probability for policy gradient learning

### Requirement: Protocol Compatibility

The SpikingBrain SHALL implement the ClassicalBrain protocol while using policy gradient learning instead of STDP.

#### Scenario: Brain Interface Compliance

- **GIVEN** the existing brain architecture framework
- **WHEN** SpikingBrain is instantiated
- **THEN** it SHALL implement all required ClassicalBrain methods
- **AND** SHALL return compatible data structures (ActionData, BrainData)
- **AND** SHALL integrate with existing simulation infrastructure
- **AND** SHALL use episode buffers for state/action/reward sequences
- **AND** SHALL support intra-episode updates via update_frequency parameter (0 = end of episode only)

#### Scenario: Configuration Schema for Policy Gradients

- **GIVEN** a YAML configuration for spiking brain
- **WHEN** parsing configuration parameters
- **THEN** the system SHALL accept policy gradient parameters: gamma, baseline_alpha, entropy_beta, entropy_beta_final, entropy_decay_episodes
- **AND** SHALL accept network architecture parameters: num_timesteps, num_hidden_layers, hidden_size
- **AND** SHALL accept LIF neuron parameters: tau_m, v_threshold, v_reset
- **AND** SHALL accept learning parameters: learning_rate, lr_decay_rate, surrogate_alpha, update_frequency
- **AND** SHALL accept input encoding parameters: population_coding, neurons_per_feature, population_sigma
- **AND** SHALL accept initialization parameters: weight_init (orthogonal, kaiming, default)
- **AND** SHALL reject STDP-specific parameters with meaningful error messages

### Requirement: Brain Factory Extension

The brain factory method SHALL support spiking neural network instantiation.

#### Scenario: Brain Type Resolution

- **GIVEN** a configuration specifies brain type as "spiking"
- **WHEN** the brain factory creates a brain instance
- **THEN** it SHALL return a SpikingBrain object
- **AND** SHALL pass through all spiking-specific configuration parameters

### Requirement: CLI Argument Extension

The command-line interface SHALL accept "spiking" as a valid brain type option.

#### Scenario: Argument Validation

- **GIVEN** a user specifies `--brain spiking`
- **WHEN** command-line arguments are parsed
- **THEN** the system SHALL recognize "spiking" as a valid brain type
- **AND** SHALL pass the selection to the brain factory

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

- **GIVEN** an episode has completed with a sequence of (state, action, reward) tuples
- **WHEN** the learning phase begins
- **THEN** the system SHALL compute discounted returns backward through the episode: `G_t = r_t + γ·G_{t+1}`
- **AND** SHALL normalize returns for variance reduction: `G' = (G - mean(G)) / std(G)`
- **AND** SHALL compute advantages using baseline: `A_t = G'_t - baseline`
- **AND** SHALL update baseline with running average: `baseline ← α·episode_return + (1-α)·baseline`

#### Scenario: Policy Gradient Update

- **GIVEN** computed advantages for each timestep
- **WHEN** updating network parameters
- **THEN** the system SHALL compute policy loss: `L = -Σ log_prob(a_t) · A_t`
- **AND** SHALL backpropagate gradients through the spiking network
- **AND** SHALL clip individual gradient values to [-1, 1]
- **AND** SHALL clip gradient norm to max_norm=1.0
- **AND** SHALL update parameters using Adam optimizer
- **AND** SHALL clear episode buffers after update

#### Scenario: Gradient Clipping

- **GIVEN** policy gradients have been computed
- **WHEN** gradients exceed the maximum norm threshold
- **THEN** the system SHALL clip individual values to [-1, 1] first
- **AND** SHALL scale gradients to max_norm=1.0
- **AND** SHALL prevent gradient explosion through long spike sequences

### Requirement: Surrogate Gradient Configuration

The configuration system SHALL support surrogate gradient parameters for controlling the differentiable spike approximation.

#### Scenario: Surrogate Alpha Parameter

- **GIVEN** a spiking brain configuration
- **WHEN** the surrogate_alpha parameter is specified
- **THEN** the system SHALL use this value to control gradient smoothness
- **AND** SHALL default to 1.0 if not specified (10.0 causes gradient explosion)
- **AND** SHALL validate that surrogate_alpha > 0

#### Scenario: Network Architecture Parameters

- **GIVEN** a spiking brain configuration
- **WHEN** num_timesteps and num_hidden_layers are specified
- **THEN** the system SHALL simulate the network for num_timesteps steps per decision (default 100)
- **AND** SHALL create num_hidden_layers LIF layers in the network (default 2)
- **AND** SHALL use hidden_size neurons per layer (default 256)
- **AND** SHALL validate num_timesteps >= 10 and num_hidden_layers >= 1

### Requirement: Gradient Flow Testing

The system SHALL provide testing utilities to verify gradient flow through spiking layers.

#### Scenario: Finite Difference Gradient Check

- **GIVEN** a LIFLayer with random weights
- **WHEN** computing gradients via backpropagation and finite differences
- **THEN** the gradients SHALL match within tolerance (relative error < 1e-4)
- **AND** SHALL verify gradient flow through all network parameters
- **AND** SHALL detect gradient vanishing or explosion issues

### Requirement: Learning Convergence Validation

The system SHALL validate that the spiking brain learns successfully on standard tasks.

#### Scenario: Foraging Task Learning

- **GIVEN** a spiking brain with surrogate gradients on foraging environments
- **WHEN** trained for 100-200 episodes
- **THEN** the average reward SHALL show a positive trend
- **AND** the success rate SHALL reach 100% on dynamic foraging (matching MLP)
- **AND** the success rate SHALL reach >60% on predator environments (vs 92% MLP)
- **AND** the learning curve SHALL be comparable to MLPBrain within 2x episode count

#### Scenario: Loss Decrease Over Episodes

- **GIVEN** a spiking brain training on any environment
- **WHEN** monitoring policy loss across episodes
- **THEN** the policy loss SHALL generally decrease over time
- **AND** SHALL not diverge (loss increasing consistently for >20 episodes)
- **AND** SHALL indicate successful gradient-based learning
