# brain-architecture Specification Delta

## ADDED Requirements

### Requirement: PPO Brain Architecture Support

The system SHALL support a Proximal Policy Optimization (PPO) brain architecture as a SOTA classical baseline, using actor-critic networks with clipped surrogate objective.

#### Scenario: CLI Brain Selection

- **WHEN** a user executes `python scripts/run_simulation.py --brain ppo --config config.yml`
- **THEN** the system SHALL initialize a PPOBrain instance
- **AND** the simulation SHALL use actor-critic networks for decision-making
- **AND** SHALL learn via clipped policy gradient optimization

#### Scenario: Configuration Loading

- **WHEN** a configuration file specifying brain type as "ppo" is loaded
- **THEN** the system SHALL validate PPO-specific parameters
- **AND** SHALL require actor-critic parameters (actor_hidden_dim, critic_hidden_dim)
- **AND** SHALL require PPO parameters (clip_epsilon, gae_lambda, num_epochs)
- **AND** SHALL require learning parameters (learning_rate, gamma, entropy_coef)
- **AND** SHALL initialize the PPO brain with both actor and critic networks

### Requirement: Actor-Critic Architecture

The PPOBrain SHALL implement separate actor (policy) and critic (value) networks with shared input preprocessing.

#### Scenario: Actor Network Forward Pass

- **WHEN** the actor network processes a preprocessed state input [gradient_strength, relative_angle]
- **THEN** the system SHALL produce logits for 4 actions (forward, left, right, stay)
- **AND** SHALL apply softmax to obtain action probabilities
- **AND** SHALL sample actions from categorical distribution
- **AND** SHALL store log probabilities for policy updates

#### Scenario: Critic Network Forward Pass

- **WHEN** the critic network processes a preprocessed state input
- **THEN** the system SHALL produce a single scalar value estimate
- **AND** SHALL use the value for advantage computation
- **AND** SHALL use a separate network from the actor (not shared layers)

#### Scenario: State Preprocessing

- **WHEN** preprocessing environmental state with gradient strength and direction
- **THEN** the system SHALL match MLPBrain preprocessing exactly
- **AND** SHALL compute relative angle: `(gradient_direction - agent_facing_angle + pi) mod 2pi - pi`
- **AND** SHALL normalize relative angle to \[-1, 1\]: `rel_angle / pi`
- **AND** SHALL produce feature vector: `[grad_strength, rel_angle_normalized]`

### Requirement: Rollout Buffer and Experience Collection

The PPOBrain SHALL collect experiences in a rollout buffer before performing batch updates.

#### Scenario: Experience Collection

- **WHEN** each step is taken during agent-environment interaction
- **THEN** the system SHALL store (state, action, log_prob, value, reward) in the buffer
- **AND** SHALL continue collecting until buffer_size steps are reached
- **AND** SHALL trigger learning update when buffer is full

#### Scenario: Buffer Reset

- **WHEN** a learning update has completed and the buffer is processed
- **THEN** the system SHALL clear all stored experiences
- **AND** SHALL begin collecting new experiences
- **AND** SHALL maintain episode boundaries for advantage computation

### Requirement: Generalized Advantage Estimation (GAE)

The PPOBrain SHALL compute advantages using GAE for variance reduction.

#### Scenario: GAE Computation

- **WHEN** advantages are computed from a full rollout buffer with rewards and values
- **THEN** the system SHALL compute TD residuals: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
- **AND** SHALL compute GAE: `A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}`
- **AND** SHALL use gae_lambda parameter (default 0.95)
- **AND** SHALL normalize advantages: `A' = (A - mean(A)) / std(A)`

#### Scenario: Return Computation

- **WHEN** returns are needed for value loss from computed advantages
- **THEN** the system SHALL compute returns as: `R_t = A_t + V(s_t)`
- **AND** SHALL use returns as targets for critic updates

### Requirement: Clipped Surrogate Objective

The PPOBrain SHALL use clipped surrogate objective to prevent large policy updates.

#### Scenario: Policy Loss Computation

- **WHEN** computing policy loss from old and new action log probabilities
- **THEN** the system SHALL compute ratio: `r = exp(log_prob_new - log_prob_old)`
- **AND** SHALL compute unclipped objective: `L_unclip = r * A`
- **AND** SHALL compute clipped objective: `L_clip = clip(r, 1-eps, 1+eps) * A`
- **AND** SHALL use minimum: `L = min(L_unclip, L_clip)`
- **AND** SHALL use clip_epsilon parameter (default 0.2)

#### Scenario: Combined Loss

- **WHEN** computing total loss from policy loss, value loss, and entropy
- **THEN** the system SHALL combine: `L_total = L_policy + c1 * L_value - c2 * entropy`
- **AND** SHALL use value_loss_coef (default 0.5) for c1
- **AND** SHALL use entropy_coef (default 0.01) for c2
- **AND** SHALL maximize entropy (subtract from loss)

### Requirement: PPO Training Loop

The PPOBrain SHALL perform multiple epochs of updates on collected experience.

#### Scenario: Multi-Epoch Updates

- **WHEN** learning update is triggered from a full rollout buffer
- **THEN** the system SHALL iterate for num_epochs (default 4)
- **AND** SHALL shuffle and split data into num_minibatches (default 4)
- **AND** SHALL update actor and critic on each minibatch
- **AND** SHALL apply gradient clipping (max_norm default 0.5)

#### Scenario: Learning Rate

- **WHEN** optimizer is created for PPO training
- **THEN** the system SHALL use Adam optimizer
- **AND** SHALL use learning_rate parameter (default 0.0003)
- **AND** MAY optionally support learning rate scheduling

### Requirement: PPO Brain Factory Extension

The brain factory SHALL support PPO instantiation.

#### Scenario: Brain Type Resolution

- **WHEN** the brain factory creates a brain instance from a configuration specifying brain type as "ppo"
- **THEN** it SHALL return a PPOBrain object
- **AND** SHALL pass through all PPO-specific configuration parameters

#### Scenario: Benchmark Categorization

- **WHEN** benchmark category is determined for a benchmark submission using PPO brain
- **THEN** the category SHALL be classified as "classical" (not quantum)
- **AND** SHALL follow same categorization as MLPBrain
