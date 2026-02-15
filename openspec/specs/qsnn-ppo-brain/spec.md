# qsnn-ppo-brain Specification

## Purpose

Define the requirements for the QSNN-PPO brain architecture. QSNN-PPO pairs a Quantum Spiking Neural Network actor (QLIF circuits with surrogate gradients) with a classical MLP critic, trained via Proximal Policy Optimization. This addresses the three diagnosed root causes of standalone QSNN Reinforce's predator failure: no critic, high REINFORCE variance, and insufficient gradient passes per data collection.

The QSNN actor reuses shared QLIF neuron infrastructure from `_qlif_layers.py`.

## ADDED Requirements

### Requirement: QSNN-PPO Brain Architecture

The system SHALL support a QSNN-PPO brain architecture that combines a QSNN actor with a classical MLP critic, trained via PPO.

#### Scenario: QSNN-PPO Brain Instantiation

- **WHEN** a QSNNPPOBrain is instantiated with default configuration
- **THEN** the system SHALL create a QSNN actor with configurable sensory neurons (default 8), hidden neurons (default 16), and motor neurons (default 4)
- **AND** SHALL create a classical MLP critic accepting raw features and hidden spike rates
- **AND** SHALL initialize separate optimizers for actor and critic
- **AND** SHALL initialize the network with a deterministic seed for reproducibility

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qsnnppo --config config.yml`
- **THEN** the system SHALL initialize a QSNNPPOBrain instance
- **AND** the simulation SHALL proceed using QSNN actor for action selection and MLP critic for value estimation

### Requirement: QSNN Actor

The QSNNPPOBrain actor SHALL use QLIF neurons from the shared `_qlif_layers.py` module for quantum forward pass with surrogate gradient backward pass.

#### Scenario: Actor Forward Pass

- **WHEN** the actor processes sensory input
- **THEN** the system SHALL encode features into sensory spikes via `encode_sensory_spikes()`
- **AND** SHALL propagate spikes through sensory-to-hidden-to-motor QLIF layers
- **AND** SHALL average spike probabilities across `num_integration_steps` timesteps (default 10)
- **AND** SHALL apply logit scaling and softmax to produce action probabilities

#### Scenario: Actor Weight Initialization

- **WHEN** the actor is initialized
- **THEN** the system SHALL create weight matrices W_sh and W_hm with Gaussian scale 0.15
- **AND** SHALL initialize theta_hidden to pi/4 and theta_motor to 0
- **AND** SHALL enable gradients on all actor parameters for surrogate gradient training

#### Scenario: Actor Weight Stability

- **WHEN** actor weights are updated
- **THEN** the system SHALL clamp weights to [-weight_clip, weight_clip] (default weight_clip=3.0)
- **AND** SHALL clamp theta_motor L2 norm to theta_motor_max_norm (default 2.0)

### Requirement: Classical MLP Critic

The QSNNPPOBrain SHALL include a classical MLP critic that estimates state value from raw sensory features and hidden layer spike rates.

#### Scenario: Critic Input Construction

- **WHEN** the critic evaluates a state
- **THEN** the system SHALL concatenate raw sensory features (e.g., 8-dim) with hidden spike rates (e.g., 16-dim, detached from actor graph)
- **AND** SHALL produce a scalar value estimate V(s)

#### Scenario: Critic Architecture

- **WHEN** the critic is initialized
- **THEN** the system SHALL create an MLP with configurable hidden layers (default 2 layers, 64 units each)
- **AND** SHALL use ReLU activations between hidden layers
- **AND** SHALL use orthogonal weight initialization for stable early training

#### Scenario: Gradient Isolation

- **WHEN** the critic computes value loss gradients
- **THEN** hidden spike rates in the critic input SHALL be detached from the actor's autograd graph
- **AND** critic gradients SHALL NOT flow through quantum circuits

### Requirement: Rollout Buffer

The QSNNPPOBrain SHALL use a rollout buffer that stores quantum spike caches alongside standard PPO fields.

#### Scenario: Buffer Storage

- **WHEN** a step is added to the rollout buffer
- **THEN** the system SHALL store features, action, log_prob, value, reward, and done flag
- **AND** SHALL store hidden_spike_rates for critic input reconstruction
- **AND** SHALL store spike_caches for quantum output caching during multi-epoch updates

#### Scenario: Buffer Capacity

- **WHEN** the buffer reaches rollout_buffer_size (default 512) steps
- **THEN** the system SHALL trigger a PPO update
- **AND** SHALL clear the buffer after the update completes

#### Scenario: GAE Advantage Computation

- **WHEN** a PPO update is triggered
- **THEN** the system SHALL compute Generalized Advantage Estimation with configurable gamma (default 0.99) and gae_lambda (default 0.95)
- **AND** SHALL bootstrap the last value from the critic if the episode is not done
- **AND** SHALL normalize advantages to zero mean and unit variance within each minibatch

#### Scenario: Minibatch Generation

- **WHEN** minibatches are requested for PPO update
- **THEN** the system SHALL randomly permute buffer indices using the seeded RNG
- **AND** SHALL split into num_minibatches (default 4) equal-sized batches
- **AND** SHALL yield dictionaries with indices, actions, old_log_probs, returns, and advantages

### Requirement: PPO Training Loop

The QSNNPPOBrain SHALL implement PPO with quantum output caching for multi-epoch updates.

#### Scenario: Multi-Epoch Update with Quantum Caching

- **WHEN** a PPO update is performed with num_epochs > 1
- **THEN** epoch 0 SHALL run quantum circuits and cache spike probabilities for all buffer steps
- **AND** epochs 1+ SHALL reuse cached spike probabilities via `execute_qlif_layer_differentiable_cached()`
- **AND** all epochs SHALL recompute RY angles from updated weights

#### Scenario: Single-Epoch Update

- **WHEN** a PPO update is performed with num_epochs = 1
- **THEN** the system SHALL run quantum circuits with caching for potential future use
- **AND** SHALL NOT require a separate pre-caching pass

#### Scenario: Per-Step Forward Passes

- **WHEN** processing a minibatch during PPO update
- **THEN** the system SHALL iterate over individual buffer steps (not batched)
- **AND** SHALL reset refractory state before each step's forward pass
- **AND** SHALL compute log_prob, entropy, and value for each step independently

#### Scenario: PPO Clipped Surrogate Loss

- **WHEN** computing the actor loss
- **THEN** the system SHALL compute the probability ratio: exp(new_log_prob - old_log_prob)
- **AND** SHALL compute clipped surrogate: min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
- **AND** SHALL negate the result for gradient descent (maximizing the objective)
- **AND** SHALL subtract entropy bonus weighted by entropy_coef

#### Scenario: Critic Loss

- **WHEN** computing the critic loss
- **THEN** the system SHALL use Huber loss (smooth_l1_loss) between predicted values and computed returns
- **AND** SHALL weight the loss by value_loss_coef (default 0.5)

#### Scenario: Gradient Clipping

- **WHEN** gradients are computed for actor or critic
- **THEN** the system SHALL clip gradient norms to max_grad_norm (default 0.5)

### Requirement: Separate Optimizers

The QSNNPPOBrain SHALL use independent optimizers for actor and critic parameters.

#### Scenario: Actor Optimizer

- **WHEN** the actor optimizer is configured
- **THEN** the system SHALL use Adam with learning rate actor_lr (default 0.01)
- **AND** SHALL apply L2 weight decay of actor_weight_decay (default 0.001)
- **AND** SHALL optimize raw tensor parameters: W_sh, W_hm, theta_hidden, theta_motor

#### Scenario: Critic Optimizer

- **WHEN** the critic optimizer is configured
- **THEN** the system SHALL use Adam with learning rate critic_lr (default 0.001)
- **AND** SHALL optimize the critic nn.Module parameters

#### Scenario: Learning Rate Scheduling

- **WHEN** lr_decay_episodes is configured
- **THEN** the system SHALL apply cosine annealing to the actor learning rate
- **AND** SHALL decay from actor_lr to actor_lr * lr_min_factor over lr_decay_episodes

### Requirement: Episode Lifecycle

The QSNNPPOBrain SHALL support the standard episode lifecycle for simulation integration.

#### Scenario: Episode Preparation

- **WHEN** prepare_episode() is called
- **THEN** the system SHALL reset refractory state for all neurons
- **AND** SHALL reset pending features and hidden spikes

#### Scenario: Episode End Update

- **WHEN** learn() is called with episode_done=True and the buffer has enough data
- **THEN** the system SHALL trigger a PPO update with the current buffer contents
- **AND** SHALL clear the buffer after the update

#### Scenario: Post-Episode Processing

- **WHEN** post_process_episode() is called
- **THEN** the system SHALL increment the episode counter
- **AND** SHALL step the learning rate scheduler if configured

### Requirement: QSNN-PPO Configuration Schema

The configuration system SHALL support QSNN-PPO-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Parameters

- **WHEN** parsing a QSNNPPOBrain configuration
- **THEN** the system SHALL accept QLIF network parameters: num_sensory_neurons (default 8), num_hidden_neurons (default 16), num_motor_neurons (default 4), membrane_tau (default 0.9), threshold (default 0.5), refractory_period (default 0), shots (default 1024), num_integration_steps (default 10), logit_scale (default 5.0), weight_clip (default 3.0), theta_motor_max_norm (default 2.0)
- **AND** SHALL accept PPO parameters: gamma (default 0.99), gae_lambda (default 0.95), clip_epsilon (default 0.2), entropy_coef (default 0.01), value_loss_coef (default 0.5), num_epochs (default 4), num_minibatches (default 4), rollout_buffer_size (default 512), max_grad_norm (default 0.5)
- **AND** SHALL accept optimizer parameters: actor_lr (default 0.01), critic_lr (default 0.001), actor_weight_decay (default 0.001)
- **AND** SHALL accept critic architecture parameters: critic_hidden_dim (default 64), critic_num_layers (default 2)
- **AND** SHALL accept sensory_modules (list or None for legacy mode)
- **AND** SHALL accept schedule parameters: lr_decay_episodes (optional), lr_min_factor (default 0.1)

#### Scenario: Configuration Validation

- **WHEN** validating QSNNPPOBrain configuration
- **THEN** the system SHALL require num_sensory_neurons >= 1
- **AND** SHALL require num_hidden_neurons >= 1
- **AND** SHALL require num_motor_neurons >= 2
- **AND** SHALL require membrane_tau in range (0, 1\]
- **AND** SHALL require threshold in range (0, 1)
- **AND** SHALL require shots >= 100

### Requirement: ClassicalBrain Protocol Compliance

The QSNNPPOBrain SHALL implement the ClassicalBrain protocol for integration with the simulation infrastructure.

#### Scenario: Brain Interface Methods

- **WHEN** QSNNPPOBrain is used in a simulation
- **THEN** it SHALL implement `run_brain(params, reward, input_data, top_only, top_randomize)`
- **AND** SHALL implement `learn(params, reward, episode_done)`
- **AND** SHALL implement `update_memory(reward)`
- **AND** SHALL implement `prepare_episode()` and `post_process_episode(episode_success)`
- **AND** SHALL implement `copy()` returning an independent clone

#### Scenario: Data Structure Compatibility

- **WHEN** QSNNPPOBrain returns brain data
- **THEN** it SHALL return ActionData with action, probability, and metadata
- **AND** SHALL populate BrainData fields compatible with existing tracking

### Requirement: Multi-Sensory Support

The QSNNPPOBrain SHALL support both legacy and modular sensory feature extraction.

#### Scenario: Legacy Mode

- **WHEN** sensory_modules is None
- **THEN** the system SHALL use 2 features: gradient_strength and relative_angle
- **AND** SHALL set input_dim to 2

#### Scenario: Unified Sensory Mode

- **WHEN** sensory_modules is configured with a list of module names
- **THEN** the system SHALL use `extract_classical_features()` for feature extraction
- **AND** SHALL set input_dim to the total feature dimension from all modules

### Requirement: Reproducibility

The QSNNPPOBrain SHALL support deterministic execution for reproducibility.

#### Scenario: Seeded Initialization

- **WHEN** two QSNNPPOBrain instances are created with the same seed
- **THEN** both instances SHALL produce identical actor weights, critic weights, and optimizer state

#### Scenario: Deterministic Execution

- **WHEN** the same inputs are provided with the same seed
- **THEN** the system SHALL produce identical actions and value estimates
