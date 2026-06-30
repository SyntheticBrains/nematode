## ADDED Requirements

### Requirement: Minimal-RNN PPO brain arms

The system SHALL provide two registered classical PPO brain arms, `mingruppo` and `minlstmppo`, that process sensory features through a parallel-form minimal-RNN recurrent core before actor/critic heads, reusing the recurrent-PPO training pipeline shared with the LSTM-PPO brain. Each arm SHALL be selectable by name and SHALL have its own configuration class and `BrainType`.

#### Scenario: minGRU arm construction

- **WHEN** a brain config specifies `name: mingruppo` with `lstm_hidden_dim: 64`
- **THEN** the brain SHALL construct: LayerNorm(input_dim) → minGRU recurrent core(input_dim, 64) → Actor MLP(64 → hidden → action output) and Critic MLP(64 → hidden → 1)
- **AND** the recurrent core, LayerNorm, and actor MLP SHALL share one optimizer and the critic MLP SHALL have its own separate optimizer (as in the LSTM-PPO pipeline)

#### Scenario: minLSTM arm construction

- **WHEN** a brain config specifies `name: minlstmppo`
- **THEN** the brain SHALL use the minLSTM recurrent core (two normalized input-only gates) instead of the minGRU core
- **AND** all other construction and training behaviour SHALL be identical to the minGRU arm

#### Scenario: Sensory module feature extraction

- **WHEN** `run_brain()` is called with BrainParams
- **THEN** the brain SHALL extract features using the configured `sensory_modules` and pass them through LayerNorm before the recurrent core
- **AND** `sensory_modules` SHALL be required (validation SHALL reject None)

### Requirement: Parallel-form minimal-RNN recurrent core

The recurrent core SHALL compute its gates from the current input only (no dependence on the previous hidden state), maintaining a single recurrent state. The state update SHALL be a convex combination of the previous state and an input-derived candidate, so the state remains bounded across arbitrary episode lengths without a squashing nonlinearity.

#### Scenario: minGRU recurrence

- **WHEN** the minGRU core advances one step on input `x` with previous state `h_prev`
- **THEN** it SHALL compute an update gate `z = sigmoid(W_z x)` and candidate `h_tilde = W_h x`, both functions of `x` only
- **AND** it SHALL return `h = (1 - z) * h_prev + z * h_tilde`

#### Scenario: minLSTM recurrence

- **WHEN** the minLSTM core advances one step on input `x` with previous state `h_prev`
- **THEN** it SHALL compute input-only gates `f = sigmoid(W_f x)` and `i = sigmoid(W_i x)`, normalize them to `f' = f/(f+i)` and `i' = i/(f+i)`, and compute candidate `h_tilde = W_h x`
- **AND** it SHALL return `h = f' * h_prev + i' * h_tilde`

#### Scenario: No hidden-to-hidden recurrent matrix

- **WHEN** the recurrent core is constructed and its weights are initialized
- **THEN** the core SHALL contain no hidden-to-hidden (`weight_hh`) parameter, and weight initialization SHALL initialize only the input projections (the saturating-recurrent-matrix orthogonal-init pass SHALL NOT be applied)

### Requirement: Recurrent hidden-state management

The brain SHALL maintain the single recurrent state within an episode and reset it at episode boundaries, with no separate cell state for either variant.

#### Scenario: Hidden state persists within an episode

- **WHEN** `run_brain()` is called on consecutive steps within an episode
- **THEN** the recurrent state SHALL persist from one step to the next and be used for the next step's inference

#### Scenario: Hidden state resets at episode boundary

- **WHEN** `prepare_episode()` is called at the start of a new episode
- **THEN** the recurrent state SHALL be reset to zeros and the cell state SHALL be `None` (single-state core)
- **AND** no state from the previous episode SHALL influence the new episode

### Requirement: Reuse of the recurrent-PPO training pipeline

The arms SHALL train with the same chunk-based truncated BPTT PPO pipeline as the LSTM-PPO brain — a hidden-state rollout buffer, sequential chunk replay with episode-boundary resets, the clipped PPO surrogate with an entropy bonus, separate actor/critic optimizers, gradient clipping, and entropy/learning-rate scheduling — without re-implementing that pipeline.

#### Scenario: Chunk-based BPTT update

- **WHEN** a PPO update is triggered (buffer full or episode done with sufficient data)
- **THEN** the buffer SHALL be divided into sequential chunks, the recurrent core SHALL be re-run sequentially from each chunk's stored initial state, episode boundaries within a chunk SHALL reset the state, and the clipped PPO objective SHALL be optimized with a value loss and entropy bonus

#### Scenario: Weight persistence round-trip

- **WHEN** `get_weight_components()` then `load_weight_components()` are called
- **THEN** the recurrent core, LayerNorm, actor, critic, optimizer, and training-state components SHALL be saved and restored, and the rollout buffer SHALL be reset on load

### Requirement: Discrete and continuous action heads

Each arm SHALL support both the discrete categorical head and the continuous tanh-squashed Gaussian (speed, turn) head, selected by the configuration's action mode, reusing the shared policy helpers.

#### Scenario: Continuous action selection

- **WHEN** the config sets the continuous action mode and `run_brain()` is called
- **THEN** the brain SHALL sample a normalized (speed, turn) action from the tanh-squashed Gaussian head and store the pre-squash sample for the PPO update

#### Scenario: Discrete action selection

- **WHEN** the config uses the default discrete action mode and `run_brain()` is called
- **THEN** the brain SHALL sample a discrete action from the categorical head over the action set

### Requirement: Registry and configuration round-trip

Each arm SHALL self-register via the brain plugin registry such that its registered name, `BrainType` value, and configuration class are mutually consistent and loadable from YAML through the standard configuration pipeline.

#### Scenario: Name, BrainType, and config are consistent

- **WHEN** the brain registry is built at import time
- **THEN** `mingruppo` and `minlstmppo` SHALL each resolve to their `BrainType` and configuration class, the registered name SHALL equal the `BrainType` value, and `BRAIN_CONFIG_MAP` SHALL contain both names

#### Scenario: Scenario YAML loads via the registry

- **WHEN** a scenario YAML names `mingruppo` or `minlstmppo` with valid fields
- **THEN** the configuration pipeline SHALL load it to the arm's configuration class and build the brain without error

#### Scenario: Plain-RNN cell-selection fields are rejected

- **WHEN** a `mingruppo` or `minlstmppo` configuration sets `rnn_type` to a non-default value or enables `recurrent_layernorm`
- **THEN** configuration validation SHALL reject it with an error, because those fields select the plain-RNN cell and are not honoured by the minimal-RNN arms (which always use the single-state minimal core)

### Requirement: Memory-axis evaluation

The arms SHALL be evaluable as memory-axis candidates against the existing recurrent arms using the committed evaluation harnesses, on both a memory-demanding cell and the reactive cell, without a new statistics layer.

#### Scenario: Separation on the memory cell

- **WHEN** the arms are run on the bit-memory delayed-match-to-cue control and analyzed with the committed separation harness
- **THEN** the harness SHALL report each arm's cue-match success rate and its paired-seed comparison against the memoryless MLP, so the arms can be confirmed (or not) as genuine memory arms

#### Scenario: Stability A/B on the reactive cell

- **WHEN** the arms are run on the reactive continuous cell paired-seed against the LSTM-PPO arm
- **THEN** the comparison SHALL use the committed paired-seed statistics layer (one-sided Wilcoxon, bootstrap CI, BH-FDR) to report the stability/return difference versus the LSTM-PPO baseline
