# brain-architecture Spec Delta

## MODIFIED Requirements

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

### Requirement: Biological Neural Dynamics with Gradient Flow
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
**And** SHALL clip gradients to max norm 0.5
**And** SHALL update parameters using Adam optimizer
**And** SHALL clear episode buffers after update

#### Scenario: Gradient Clipping
**Given** policy gradients have been computed
**When** gradients exceed the maximum norm threshold
**Then** the system SHALL scale gradients to max_norm=0.5
**And** SHALL prevent gradient explosion through long spike sequences

### Requirement: Input/Output Encoding with Relative Angles
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
**Then** the system SHALL pass features through a linear layer to produce input currents
**And** SHALL apply the same input current at each simulation timestep (constant input)
**And** SHALL NOT use stochastic Poisson spike generation
**And** SHALL enable deterministic forward passes for reduced variance

#### Scenario: Spike-Based Action Selection
**Given** output layer spike counts accumulated over simulation period
**When** selecting an action
**Then** the system SHALL sum spikes over all timesteps for each output neuron
**And** SHALL pass total spike counts through a linear layer to produce action logits
**And** SHALL apply softmax to obtain action probabilities
**And** SHALL sample action from categorical distribution
**And** SHALL store log probability for policy gradient learning

### Requirement: Surrogate Gradient Configuration
The configuration system SHALL support surrogate gradient parameters for controlling the differentiable spike approximation.

#### Scenario: Surrogate Alpha Parameter
**Given** a spiking brain configuration
**When** the surrogate_alpha parameter is specified
**Then** the system SHALL use this value to control gradient smoothness
**And** SHALL default to 10.0 if not specified
**And** SHALL validate that surrogate_alpha > 0

#### Scenario: Network Architecture Parameters
**Given** a spiking brain configuration
**When** num_timesteps and num_hidden_layers are specified
**Then** the system SHALL simulate the network for num_timesteps steps per decision
**And** SHALL create num_hidden_layers LIF layers in the network
**And** SHALL validate num_timesteps >= 10 and num_hidden_layers >= 1

### Requirement: Protocol Compatibility with Policy Gradient Semantics
The SpikingBrain SHALL implement the ClassicalBrain protocol while using policy gradient learning instead of STDP.

#### Scenario: Brain Interface Compliance
**Given** the existing brain architecture framework
**When** SpikingBrain is instantiated
**Then** it SHALL implement all required ClassicalBrain methods
**And** SHALL return compatible data structures (ActionData, BrainData)
**And** SHALL integrate with existing simulation infrastructure
**And** SHALL use episode buffers for state/action/reward sequences
**And** SHALL update weights only when episode_done=True

#### Scenario: Configuration Schema for Policy Gradients
**Given** a YAML configuration for spiking brain
**When** parsing configuration parameters
**Then** the system SHALL accept policy gradient parameters: gamma, baseline_alpha, entropy_beta
**And** SHALL accept network architecture parameters: num_timesteps, num_hidden_layers
**And** SHALL accept LIF neuron parameters: tau_m, v_threshold, v_reset
**And** SHALL accept learning parameters: learning_rate, surrogate_alpha
**And** SHALL reject STDP-specific parameters with meaningful error messages

## REMOVED Requirements

### Requirement: Spike-Timing Dependent Plasticity Learning
**Reason**: STDP has fundamental limitations for reinforcement learning tasks - local learning rules cannot implement global credit assignment needed for sparse reward navigation. Experimental results showed zero learning over 400 episodes.

**Migration**: Replace STDP with policy gradient learning (REINFORCE). Users must update configuration files to remove `tau_plus`, `tau_minus`, `a_plus`, `a_minus`, `reward_scaling` parameters and add policy gradient parameters (`gamma`, `baseline_alpha`, `learning_rate`).

### Requirement: Rate Coding Input with Poisson Encoding
**Reason**: Poisson spike encoding adds stochastic noise and reduces learning signal quality. Deterministic current-based encoding provides clearer gradient signals while maintaining biological plausibility.

**Migration**: Remove `max_rate`, `min_rate`, `simulation_duration`, `time_step` parameters from configuration. Replace with `num_timesteps` parameter that controls simulation steps per decision.

## ADDED Requirements

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
**Given** a spiking brain with surrogate gradients on foraging_small environment
**When** trained for 100 episodes
**Then** the average reward SHALL show a positive trend
**And** the success rate SHALL reach >50% by episode 100
**And** the learning curve SHALL be comparable to MLPBrain within 2x episode count

#### Scenario: Loss Decrease Over Episodes
**Given** a spiking brain training on any environment
**When** monitoring policy loss across episodes
**Then** the policy loss SHALL generally decrease over time
**And** SHALL not diverge (loss increasing consistently for >20 episodes)
**And** SHALL indicate successful gradient-based learning
