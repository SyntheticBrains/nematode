## ADDED Requirements

### Requirement: QRH-QLSTM brain type registration

The system SHALL register `qrhqlstm` as a valid brain type in `BrainType` enum, `QUANTUM_BRAIN_TYPES` set, `BRAIN_CONFIG_MAP`, and `__all__` exports. The brain SHALL be selectable via `brain.name: qrhqlstm` in YAML config files.

#### Scenario: Brain type recognized from config

- **WHEN** a YAML config specifies `brain.name: qrhqlstm`
- **THEN** the config loader SHALL instantiate `QRHQLSTMBrainConfig` and the simulation SHALL create a `QRHQLSTMBrain` instance

#### Scenario: Brain classified as quantum

- **WHEN** the brain type is `QRHQLSTM`
- **THEN** it SHALL be included in `QUANTUM_BRAIN_TYPES`

### Requirement: QRH-QLSTM config

`QRHQLSTMBrainConfig` SHALL accept the following parameter groups via Pydantic fields:

- **Reservoir**: `num_reservoir_qubits` (default 8), `reservoir_depth` (default 3), `reservoir_seed` (default 42), `use_random_topology` (default False), `num_sensory_qubits` (optional)
- **LSTM readout**: `lstm_hidden_dim` (default 64), `bptt_chunk_length` (default 32)
- **QLIF gates**: `shots` (default 1024), `membrane_tau` (default 0.9), `refractory_period` (default 0), `use_quantum_gates` (default True)
- **PPO**: `actor_lr`, `critic_lr`, `gamma`, `gae_lambda`, `clip_epsilon`, `entropy_coef`, `entropy_coef_end`, `entropy_decay_episodes`, `value_loss_coef`, `num_epochs`, `rollout_buffer_size`, `max_grad_norm`
- **LR schedule**: `lr_warmup_episodes` (optional), `lr_warmup_start` (optional), `lr_decay_episodes` (optional), `lr_decay_end` (optional)
- **Critic**: `critic_hidden_dim` (default 128), `critic_num_layers` (default 2)
- **Sensory**: `sensory_modules` (list of module names)

#### Scenario: Config with defaults

- **WHEN** a config specifies only `brain.name: qrhqlstm` with no config overrides
- **THEN** all fields SHALL use their default values and the brain SHALL initialize successfully

#### Scenario: Config with quantum gates disabled

- **WHEN** config specifies `use_quantum_gates: false`
- **THEN** the QLIF-LSTM cell SHALL use classical sigmoid for forget and input gates instead of quantum measurements

### Requirement: QRH-QLSTM reservoir feature extraction

The brain SHALL instantiate a QRH quantum reservoir internally and use it exclusively as a feature extractor. The reservoir SHALL produce X/Y/Z single-qubit expectations and ZZ pairwise correlations (3N + N(N-1)/2 features for N qubits). The reservoir SHALL NOT be trained — its parameters are fixed.

#### Scenario: Feature extraction pipeline

- **WHEN** `run_brain()` is called with sensory input
- **THEN** the brain SHALL preprocess sensory input via configured sensory modules, pass features through the QRH quantum reservoir, and produce a reservoir feature vector of dimension 3N + N(N-1)/2

#### Scenario: Reservoir isolation

- **WHEN** PPO training updates are performed
- **THEN** gradients SHALL NOT flow into the quantum reservoir parameters — only the LSTM cell, actor head, and critic MLP SHALL be updated

### Requirement: QRH-QLSTM temporal readout

The brain SHALL use a `QLIFLSTMCell` (imported from `qliflstm.py`) as the temporal readout. The LSTM cell SHALL receive reservoir features as input and maintain hidden state (h_t, c_t) across timesteps within an episode.

#### Scenario: LSTM processes reservoir features

- **WHEN** reservoir features are extracted at timestep t
- **THEN** the QLIF-LSTM cell SHALL process `(reservoir_features_t, h_{t-1}, c_{t-1})` and produce `(h_t, c_t)`

#### Scenario: Hidden state persistence within episode

- **WHEN** multiple timesteps occur within a single episode
- **THEN** the LSTM hidden state (h_t, c_t) SHALL persist across timesteps, enabling temporal memory

#### Scenario: Hidden state reset between episodes

- **WHEN** `prepare_episode()` is called
- **THEN** the LSTM hidden state SHALL be reset to zeros

### Requirement: QRH-QLSTM actor-critic architecture

The actor head SHALL receive `[reservoir_features, h_t]` as input and output action logits over 4 actions. The critic MLP SHALL receive `[reservoir_features, h_t.detach()]` as input and output a scalar value estimate.

#### Scenario: Actor input composition

- **WHEN** the actor computes action logits
- **THEN** the input SHALL be the concatenation of current reservoir features and LSTM hidden state h_t (with gradients flowing through h_t to the LSTM)

#### Scenario: Critic input composition

- **WHEN** the critic estimates state value
- **THEN** the input SHALL be the concatenation of current reservoir features and detached h_t (no gradients flowing through h_t to the LSTM from the critic)

### Requirement: QRH-QLSTM recurrent PPO training

The brain SHALL use truncated BPTT for PPO training. The rollout buffer SHALL store reservoir features, actions, log probs, values, rewards, dones, and LSTM hidden states at chunk boundaries.

#### Scenario: Chunk-based BPTT

- **WHEN** a PPO update is triggered (buffer full or episode done)
- **THEN** the buffer SHALL be split into sequential chunks of `bptt_chunk_length` steps, each chunk SHALL be re-run through the LSTM from its stored initial hidden state, and PPO clipped surrogate loss SHALL be computed per chunk

#### Scenario: Episode boundary handling

- **WHEN** an episode boundary occurs within a BPTT chunk
- **THEN** the LSTM hidden state SHALL be reset to zeros at that boundary during the re-run

#### Scenario: Gradient clipping

- **WHEN** PPO gradients are computed
- **THEN** gradient norms SHALL be clipped to `max_grad_norm`

### Requirement: QRH-QLSTM LR scheduling

The brain SHALL support optional LR warmup and decay, following the same schedule as QLIF-LSTM: linear warmup from `lr_warmup_start` to `actor_lr` over `lr_warmup_episodes`, then linear decay from `actor_lr` to `lr_decay_end` over `lr_decay_episodes`. Critic LR SHALL scale proportionally.

#### Scenario: Warmup + decay schedule

- **WHEN** both `lr_warmup_episodes` and `lr_decay_episodes` are configured
- **THEN** LR SHALL warmup linearly for the specified episodes, then decay linearly, with critic LR maintaining the same ratio to actor LR

#### Scenario: No scheduling

- **WHEN** neither `lr_warmup_episodes` nor `lr_decay_episodes` is configured
- **THEN** LR SHALL remain constant at `actor_lr` / `critic_lr`

### Requirement: QRH-QLSTM feature normalization

The brain SHALL apply `LayerNorm` to reservoir features before feeding them to the LSTM cell.

#### Scenario: Normalized input

- **WHEN** reservoir features are extracted
- **THEN** they SHALL be normalized via LayerNorm before being passed to the QLIF-LSTM cell

### Requirement: QRH-QLSTM brain lifecycle

The brain SHALL implement the `ClassicalBrain` protocol: `run_brain()`, `learn()`, `prepare_episode()`, `post_process_episode()`, `copy()`, and `update_memory()`.

#### Scenario: Full episode lifecycle

- **WHEN** a simulation episode runs
- **THEN** `prepare_episode()` SHALL reset LSTM state, `run_brain()` SHALL produce actions each step, `learn()` SHALL store transitions and trigger PPO updates, and `post_process_episode()` SHALL update episode count and LR schedule

#### Scenario: Brain copy

- **WHEN** `copy()` is called
- **THEN** a deep copy SHALL be returned with fresh LSTM hidden states and an independent reservoir instance
