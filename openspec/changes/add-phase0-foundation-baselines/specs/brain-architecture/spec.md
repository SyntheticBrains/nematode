# brain-architecture Specification Delta

## ADDED Requirements

### Requirement: PPO Brain Architecture Support
The system SHALL support a Proximal Policy Optimization (PPO) brain architecture as a SOTA classical baseline, using actor-critic networks with clipped surrogate objective.

#### Scenario: CLI Brain Selection
**Given** a user wants to run a simulation with PPO brain
**When** they execute `python scripts/run_simulation.py --brain ppo --config config.yml`
**Then** the system SHALL initialize a PPOBrain instance
**And** the simulation SHALL use actor-critic networks for decision-making
**And** SHALL learn via clipped policy gradient optimization

#### Scenario: Configuration Loading
**Given** a configuration file specifies brain type as "ppo"
**When** the configuration is loaded
**Then** the system SHALL validate PPO-specific parameters
**And** SHALL require actor-critic parameters (actor_hidden_dim, critic_hidden_dim)
**And** SHALL require PPO parameters (clip_epsilon, gae_lambda, num_epochs)
**And** SHALL require learning parameters (learning_rate, gamma, entropy_coef)
**And** SHALL initialize the PPO brain with both actor and critic networks

### Requirement: Actor-Critic Architecture
The PPOBrain SHALL implement separate actor (policy) and critic (value) networks with shared input preprocessing.

#### Scenario: Actor Network Forward Pass
**Given** a preprocessed state input [gradient_strength, relative_angle]
**When** the actor network processes the input
**Then** the system SHALL produce logits for 4 actions (forward, left, right, stay)
**And** SHALL apply softmax to obtain action probabilities
**And** SHALL sample actions from categorical distribution
**And** SHALL store log probabilities for policy updates

#### Scenario: Critic Network Forward Pass
**Given** a preprocessed state input
**When** the critic network processes the input
**Then** the system SHALL produce a single scalar value estimate
**And** SHALL use the value for advantage computation
**And** SHALL use a separate network from the actor (not shared layers)

#### Scenario: State Preprocessing
**Given** environmental state with gradient strength and direction
**When** preprocessing the state
**Then** the system SHALL match MLPBrain preprocessing exactly
**And** SHALL compute relative angle: `(gradient_direction - agent_facing_angle + pi) mod 2pi - pi`
**And** SHALL normalize relative angle to [-1, 1]: `rel_angle / pi`
**And** SHALL produce feature vector: `[grad_strength, rel_angle_normalized]`

### Requirement: Rollout Buffer and Experience Collection
The PPOBrain SHALL collect experiences in a rollout buffer before performing batch updates.

#### Scenario: Experience Collection
**Given** an agent interacting with the environment
**When** each step is taken
**Then** the system SHALL store (state, action, log_prob, value, reward) in the buffer
**And** SHALL continue collecting until buffer_size steps are reached
**And** SHALL trigger learning update when buffer is full

#### Scenario: Buffer Reset
**Given** a learning update has completed
**When** the buffer is processed
**Then** the system SHALL clear all stored experiences
**And** SHALL begin collecting new experiences
**And** SHALL maintain episode boundaries for advantage computation

### Requirement: Generalized Advantage Estimation (GAE)
The PPOBrain SHALL compute advantages using GAE for variance reduction.

#### Scenario: GAE Computation
**Given** a full rollout buffer with rewards and values
**When** advantages are computed
**Then** the system SHALL compute TD residuals: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
**And** SHALL compute GAE: `A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}`
**And** SHALL use gae_lambda parameter (default 0.95)
**And** SHALL normalize advantages: `A' = (A - mean(A)) / std(A)`

#### Scenario: Return Computation
**Given** computed advantages
**When** returns are needed for value loss
**Then** the system SHALL compute returns as: `R_t = A_t + V(s_t)`
**And** SHALL use returns as targets for critic updates

### Requirement: Clipped Surrogate Objective
The PPOBrain SHALL use clipped surrogate objective to prevent large policy updates.

#### Scenario: Policy Loss Computation
**Given** old and new action log probabilities
**When** computing policy loss
**Then** the system SHALL compute ratio: `r = exp(log_prob_new - log_prob_old)`
**And** SHALL compute unclipped objective: `L_unclip = r * A`
**And** SHALL compute clipped objective: `L_clip = clip(r, 1-eps, 1+eps) * A`
**And** SHALL use minimum: `L = min(L_unclip, L_clip)`
**And** SHALL use clip_epsilon parameter (default 0.2)

#### Scenario: Combined Loss
**Given** policy loss, value loss, and entropy
**When** computing total loss
**Then** the system SHALL combine: `L_total = L_policy + c1 * L_value - c2 * entropy`
**And** SHALL use value_loss_coef (default 0.5) for c1
**And** SHALL use entropy_coef (default 0.01) for c2
**And** SHALL maximize entropy (subtract from loss)

### Requirement: PPO Training Loop
The PPOBrain SHALL perform multiple epochs of updates on collected experience.

#### Scenario: Multi-Epoch Updates
**Given** a full rollout buffer
**When** learning update is triggered
**Then** the system SHALL iterate for num_epochs (default 4)
**And** SHALL shuffle and split data into num_minibatches (default 4)
**And** SHALL update actor and critic on each minibatch
**And** SHALL apply gradient clipping (max_norm default 0.5)

#### Scenario: Learning Rate
**Given** PPO training configuration
**When** optimizer is created
**Then** the system SHALL use Adam optimizer
**And** SHALL use learning_rate parameter (default 0.0003)
**And** MAY optionally support learning rate scheduling

### Requirement: PPO Brain Factory Extension
The brain factory SHALL support PPO instantiation.

#### Scenario: Brain Type Resolution
**Given** a configuration specifies brain type as "ppo"
**When** the brain factory creates a brain instance
**Then** it SHALL return a PPOBrain object
**And** SHALL pass through all PPO-specific configuration parameters

#### Scenario: Benchmark Categorization
**Given** a benchmark submission using PPO brain
**When** benchmark category is determined
**Then** the category SHALL be classified as "classical" (not quantum)
**And** SHALL follow same categorization as MLPBrain
