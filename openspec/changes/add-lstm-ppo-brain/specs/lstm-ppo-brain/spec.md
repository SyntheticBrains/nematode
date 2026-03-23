## ADDED Requirements

### Requirement: LSTM PPO Brain Architecture

The system SHALL provide an LSTM/GRU-augmented PPO brain that processes sensory features through a recurrent layer before actor/critic MLP heads.

#### Scenario: Brain Construction with LSTM

- **WHEN** a brain config specifies `name: lstmppo` with `rnn_type: lstm` and `lstm_hidden_dim: 64`
- **THEN** the brain SHALL construct: LayerNorm(input_dim) → LSTM(input_dim, 64) → Actor MLP(64 → hidden → num_actions) and Critic MLP(64 → hidden → 1)
- **AND** the LSTM, LayerNorm, and actor MLP SHALL share one optimizer
- **AND** the critic MLP SHALL have its own separate optimizer

#### Scenario: Brain Construction with GRU

- **WHEN** a brain config specifies `rnn_type: gru`
- **THEN** the brain SHALL use `nn.GRU` instead of `nn.LSTM`
- **AND** the GRU cell state SHALL be `None` (GRU has no cell state)
- **AND** all other behavior SHALL be identical to LSTM mode

#### Scenario: Sensory Module Feature Extraction

- **WHEN** `run_brain()` is called with BrainParams
- **THEN** the brain SHALL extract features using `extract_classical_features(params, sensory_modules)`
- **AND** the features SHALL be passed through LayerNorm before the LSTM
- **AND** the brain SHALL support all existing sensory modules including temporal, derivative, and STAM

### Requirement: Recurrent Hidden State Management

The brain SHALL maintain LSTM hidden state within episodes and reset at episode boundaries.

#### Scenario: Hidden State Persistence Within Episode

- **WHEN** `run_brain()` is called on consecutive steps within an episode
- **THEN** the LSTM hidden state SHALL persist from one step to the next
- **AND** the brain SHALL use the updated hidden state for the next step's inference

#### Scenario: Hidden State Reset at Episode Boundary

- **WHEN** `prepare_episode()` is called at the start of a new episode
- **THEN** the LSTM hidden state (h and c) SHALL be reset to zeros
- **AND** no state from the previous episode SHALL influence the new episode

#### Scenario: Detached Critic Hidden State

- **WHEN** the LSTM produces hidden state `h_t` during `run_brain()`
- **THEN** the actor head SHALL receive `h_t` with gradients
- **AND** the critic head SHALL receive `h_t.detach()` (no gradient flow to LSTM from critic)

### Requirement: Chunk-Based Truncated BPTT Training

The brain SHALL train using chunk-based truncated backpropagation through time with the PPO algorithm.

#### Scenario: Rollout Buffer with Hidden States

- **WHEN** experience is added to the rollout buffer via `learn()`
- **THEN** the buffer SHALL store the pre-step LSTM hidden state (h, c) alongside the standard PPO data (features, action, log_prob, value, reward, done)

#### Scenario: Sequential Chunk Processing

- **WHEN** the PPO update is triggered (buffer full or episode done)
- **THEN** the buffer SHALL be divided into sequential chunks of `bptt_chunk_length` steps
- **AND** within each chunk, the LSTM SHALL be re-run sequentially from the stored initial hidden state
- **AND** episode boundaries within chunks SHALL reset the hidden state to zeros
- **AND** chunk ORDER SHALL be shuffled across epochs, but step order within chunks SHALL be preserved

#### Scenario: PPO Clipped Objective

- **WHEN** the PPO update computes losses within each chunk
- **THEN** the policy loss SHALL use the clipped surrogate objective
- **AND** the value loss SHALL use MSE or Huber loss
- **AND** an entropy bonus SHALL be subtracted from the loss
- **AND** gradients SHALL be clipped by `max_grad_norm`

### Requirement: Entropy Decay

The brain SHALL support decaying the entropy coefficient over episodes for exploration scheduling.

#### Scenario: Linear Entropy Decay

- **WHEN** `entropy_coef`, `entropy_coef_end`, and `entropy_decay_episodes` are configured
- **THEN** the entropy coefficient SHALL linearly decrease from `entropy_coef` to `entropy_coef_end` over `entropy_decay_episodes` episodes
- **AND** after `entropy_decay_episodes`, the coefficient SHALL remain at `entropy_coef_end`

### Requirement: Learning Rate Scheduling

The brain SHALL support warmup and decay learning rate scheduling for both optimizers.

#### Scenario: LR Warmup and Decay

- **WHEN** `lr_warmup_episodes` and `lr_decay_episodes` are configured
- **THEN** both actor and critic learning rates SHALL follow the warmup → stable → decay schedule
- **AND** the critic LR SHALL scale proportionally to the actor LR

### Requirement: Weight Persistence

The brain SHALL support saving and loading trained weights for curriculum learning.

#### Scenario: Save Weights

- **WHEN** `get_weight_components()` is called
- **THEN** it SHALL return components for: lstm, layer_norm, policy (actor MLP), value (critic MLP), actor_optimizer, critic_optimizer, training_state

#### Scenario: Load Weights

- **WHEN** `load_weight_components()` is called with saved components
- **THEN** the brain SHALL restore all network weights, optimizer states, and training state
- **AND** the rollout buffer SHALL be reset to prevent stale experience

### Requirement: Brain Protocol Compliance

The brain SHALL implement all methods required by the ClassicalBrain protocol.

#### Scenario: Protocol Method Implementation

- **WHEN** LSTMPPOBrain is instantiated
- **THEN** it SHALL implement `run_brain()`, `learn()`, `prepare_episode()`, `post_process_episode()`, `update_memory()` (no-op), `copy()` (raise NotImplementedError)
- **AND** it SHALL implement the `action_set` property with getter and setter
- **AND** `build_brain()` and `update_parameters()` SHALL raise NotImplementedError or be no-ops (not applicable to classical brains)
