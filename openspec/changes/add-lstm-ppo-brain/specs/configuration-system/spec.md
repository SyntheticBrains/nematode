## ADDED Requirements

### Requirement: LSTM PPO Brain Configuration Schema

The configuration system SHALL support the `lstmppo` brain type with LSTM/GRU-specific parameters.

#### Scenario: Brain Type Registration

- **WHEN** a YAML configuration specifies `brain.name: lstmppo`
- **THEN** the system SHALL accept the configuration and create an LSTMPPOBrain instance
- **AND** `lstmppo` SHALL be registered in BRAIN_CONFIG_MAP and BrainType enum
- **AND** it SHALL be classified as a CLASSICAL_BRAIN_TYPE

#### Scenario: LSTM PPO Configuration Parameters

- **WHEN** a YAML configuration includes `lstmppo` brain config
- **THEN** the system SHALL accept:
  - `rnn_type` (string, "lstm" or "gru", default "lstm")
  - `lstm_hidden_dim` (integer, default 64, must be >= 2)
  - `bptt_chunk_length` (integer, default 16, must be >= 4)
  - `actor_hidden_dim` (integer, default 64)
  - `critic_hidden_dim` (integer, default 128)
  - `actor_num_layers` (integer, default 2)
  - `critic_num_layers` (integer, default 2)
  - `actor_lr` (float, default 0.0005)
  - `critic_lr` (float, default 0.0005)
  - `gamma`, `gae_lambda`, `clip_epsilon`, `value_loss_coef` (standard PPO params)
  - `num_epochs` (integer, default 6)
  - `rollout_buffer_size` (integer, default 1024, must be >= bptt_chunk_length)
  - `max_grad_norm` (float, default 0.5)
  - `entropy_coef`, `entropy_coef_end`, `entropy_decay_episodes` (entropy decay)
  - `lr_warmup_episodes`, `lr_warmup_start`, `lr_decay_episodes`, `lr_decay_end` (LR scheduling)
  - `sensory_modules` (list of ModuleName, required for feature extraction)

#### Scenario: Example Configurations

- **WHEN** example configs are provided for `lstmppo`
- **THEN** `lstmppo_foraging_small_derivative.yml` SHALL configure derivative foraging with LSTM PPO
- **AND** `lstmppo_pursuit_predators_small_derivative.yml` SHALL configure derivative pursuit predators with LSTM PPO
- **AND** both SHALL use sensory modules compatible with temporal sensing
