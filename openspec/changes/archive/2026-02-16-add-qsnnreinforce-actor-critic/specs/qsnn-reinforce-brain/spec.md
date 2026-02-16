## ADDED Requirements

### Requirement: Actor-Critic Value Estimation

The QSNNReinforceBrain SHALL support an optional classical MLP critic for state value estimation, enabling actor-critic (A2C) variance reduction when `use_critic` is True.

#### Scenario: Critic Instantiation

- **WHEN** QSNNReinforceBrain is instantiated with `use_critic: true`
- **THEN** the system SHALL create a CriticMLP with input dimension equal to `num_sensory_neurons + num_hidden_neurons`
- **AND** SHALL initialize critic weights with orthogonal initialization (gain=sqrt(2)) and zero biases
- **AND** SHALL create a separate Adam optimizer for the critic with learning rate `critic_lr`

#### Scenario: Critic Disabled

- **WHEN** QSNNReinforceBrain is instantiated with `use_critic: false` (default)
- **THEN** the system SHALL NOT create a CriticMLP or critic optimizer
- **AND** SHALL use vanilla REINFORCE with return normalization for advantage computation

#### Scenario: Critic Incompatible with Local Learning

- **WHEN** QSNNReinforceBrain is instantiated with `use_critic: true` and `use_local_learning: true`
- **THEN** the system SHALL log a warning that critic has no effect with local learning
- **AND** SHALL NOT create a CriticMLP

### Requirement: Enriched Critic Input

The CriticMLP SHALL receive both raw sensory features and hidden layer spike rates as input for state value estimation.

#### Scenario: Critic Input Construction

- **WHEN** the critic evaluates a state
- **THEN** the system SHALL concatenate raw sensory features (num_sensory_neurons dimensions) with hidden spike rates (num_hidden_neurons dimensions)
- **AND** SHALL detach hidden spike rates from the autograd graph (no gradient flow from critic through quantum circuits)
- **AND** SHALL produce a critic input vector of dimension `num_sensory_neurons + num_hidden_neurons`

#### Scenario: Hidden Spike Rate Storage

- **WHEN** `run_brain()` executes a forward pass with `use_critic: true`
- **THEN** the system SHALL store the hidden layer spike rates (averaged across integration steps) in the episode buffer alongside sensory features
- **AND** SHALL store spike rates as detached numpy arrays

### Requirement: GAE Advantage Estimation

The QSNNReinforceBrain SHALL compute advantages using Generalized Advantage Estimation (GAE) when the critic is enabled.

#### Scenario: GAE Computation

- **WHEN** a REINFORCE update is triggered with `use_critic: true`
- **THEN** the system SHALL compute TD residuals: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
- **AND** SHALL compute GAE advantages: `A_t = delta_t + gamma * gae_lambda * A_{t+1}`
- **AND** SHALL compute target returns as `returns_t = A_t + V(s_t)`
- **AND** SHALL normalize advantages to zero mean and unit variance

#### Scenario: Bootstrap Value at Window Boundaries

- **WHEN** an intra-episode update window boundary is reached
- **THEN** the system SHALL compute bootstrap value `V(s)` from the critic for the state after the last step
- **AND** SHALL use bootstrap value as the terminal value in GAE computation
- **WHEN** an episode terminates
- **THEN** the system SHALL use bootstrap value of 0.0

#### Scenario: Advantage Clipping

- **WHEN** GAE advantages are computed
- **THEN** the system SHALL clip advantages to `[-advantage_clip, advantage_clip]` (default 2.0)
- **AND** SHALL prevent extreme advantage values from destabilizing policy updates

### Requirement: Critic Training

The CriticMLP SHALL be trained via regression against GAE target returns.

#### Scenario: Critic Update

- **WHEN** a REINFORCE update completes (after all actor epochs)
- **THEN** the system SHALL update the critic once using Huber loss (smooth L1) against target returns
- **AND** SHALL clip critic gradients to the same max norm as actor gradients
- **AND** SHALL update critic weights via its separate Adam optimizer

#### Scenario: Critic Update Ordering

- **WHEN** multi-epoch REINFORCE is configured (`num_reinforce_epochs` > 1)
- **THEN** the system SHALL run all actor epochs first
- **AND** SHALL update the critic once after the final actor epoch
- **AND** SHALL NOT update the critic during intermediate actor epochs

### Requirement: Critic Diagnostics Logging

The QSNNReinforceBrain SHALL log critic performance diagnostics when `use_critic` is True.

#### Scenario: Per-Update Critic Metrics

- **WHEN** a critic update is performed
- **THEN** the system SHALL log `value_loss` (Huber loss value)
- **AND** SHALL log `critic_pred_mean` (mean predicted value across the batch)
- **AND** SHALL log `critic_pred_std` (standard deviation of predictions)
- **AND** SHALL log `target_return_mean` (mean of GAE target returns)
- **AND** SHALL log `explained_variance` computed as `1 - Var(returns - predicted) / Var(returns)`

#### Scenario: Explained Variance Interpretation

- **WHEN** explained_variance is near 1.0
- **THEN** the critic is accurately predicting state values (good baseline)
- **WHEN** explained_variance is near 0.0
- **THEN** the critic predictions are no better than predicting the mean (uninformative baseline)

## MODIFIED Requirements

### Requirement: Surrogate Gradient Learning

The QSNNReinforceBrain SHALL implement REINFORCE policy gradient with surrogate gradient backward pass as the primary learning mode, with optional actor-critic variance reduction.

#### Scenario: Surrogate Gradient Backward Pass

- **WHEN** gradients are backpropagated through the spike function
- **THEN** the system SHALL use `QLIFSurrogateSpike` autograd function
- **AND** SHALL compute surrogate gradient as sigmoid centered at pi/2 (RY gate transition point)
- **AND** SHALL enable gradient flow from motor outputs through hidden layer to sensory weights

#### Scenario: REINFORCE Policy Gradient Update

- **WHEN** an episode completes or the update window is reached
- **THEN** the system SHALL compute advantages (GAE if critic enabled, else normalized discounted returns)
- **AND** SHALL compute policy loss with entropy bonus
- **AND** SHALL clip gradients and update weights via Adam optimizer
- **AND** SHALL update the critic (if enabled) after all actor epochs complete

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

### Requirement: QSNN Configuration Schema

The configuration system SHALL support QSNN-specific parameters via Pydantic BaseModel, including actor-critic parameters.

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
- **AND** SHALL accept `use_critic` (bool, default False) for actor-critic mode
- **AND** SHALL accept `critic_hidden_dim` (int, default 32) for critic MLP hidden layer dimension
- **AND** SHALL accept `critic_num_layers` (int, default 2) for number of critic hidden layers
- **AND** SHALL accept `critic_lr` (float, default 0.001) for critic optimizer learning rate
- **AND** SHALL accept `gae_lambda` (float, default 0.95) for GAE lambda parameter
- **AND** SHALL accept `value_loss_coef` (float, default 0.5) for critic loss coefficient

#### Scenario: Configuration Validation

- **WHEN** validating QSNNReinforceBrain configuration
- **THEN** the system SHALL require `num_sensory_neurons` >= 1
- **AND** SHALL require `num_hidden_neurons` >= 1
- **AND** SHALL require `num_motor_neurons` >= 2
- **AND** SHALL require `membrane_tau` in range (0, 1\]
- **AND** SHALL require `threshold` in range (0, 1)
- **AND** SHALL require `shots` >= 100
- **AND** SHALL require `critic_hidden_dim` >= 1 when `use_critic` is True
- **AND** SHALL require `critic_num_layers` >= 1 when `use_critic` is True
- **AND** SHALL require `critic_lr` > 0 when `use_critic` is True
- **AND** SHALL require `gae_lambda` in range [0, 1]
