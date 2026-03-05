# qlif-lstm-brain Specification

## Purpose

Define the requirements for the QLIF-LSTM brain architecture. QLIF-LSTM is a quantum-enhanced LSTM where forget and input gates use QLIF quantum neuron measurements (via surrogate gradients) instead of classical sigmoid activations, trained via recurrent PPO with chunk-based truncated BPTT. This is the first temporal architecture in the codebase, introducing within-episode memory.

The QLIF-LSTM cell reuses shared QLIF neuron infrastructure from `_qlif_layers.py`.

## ADDED Requirements

### Requirement: QLIF-LSTM Brain Architecture

The system SHALL support a QLIF-LSTM brain architecture that combines a custom LSTM cell (with quantum QLIF gates) with a classical MLP critic, trained via recurrent PPO.

#### Scenario: QLIF-LSTM Brain Instantiation

- **WHEN** a QLIFLSTMBrain is instantiated with default configuration
- **THEN** the system SHALL create a QLIFLSTMCell with configurable hidden dimension (default 32)
- **AND** SHALL create a classical MLP critic accepting raw sensory features and detached LSTM hidden state
- **AND** SHALL create an actor head (Linear layer) mapping LSTM hidden state to action logits
- **AND** SHALL initialize LSTM hidden state (h_t, c_t) as zero tensors
- **AND** SHALL initialize the network with a deterministic seed for reproducibility

#### Scenario: CLI Brain Selection

- **WHEN** user executes `python scripts/run_simulation.py --brain qliflstm --config config.yml`
- **THEN** the system SHALL initialize a QLIFLSTMBrain instance
- **AND** the simulation SHALL proceed using the QLIF-LSTM cell for temporal processing and MLP critic for value estimation

### Requirement: QLIF-LSTM Cell

The QLIFLSTMCell SHALL implement a custom LSTM cell where forget and input gates use QLIF quantum neuron measurements instead of classical sigmoid activations.

#### Scenario: Cell Forward Pass

- **WHEN** the cell processes input x_t with previous states h\_{t-1} and c\_{t-1}
- **THEN** the system SHALL concatenate x_t and h\_{t-1} into combined input z
- **AND** SHALL compute four linear projections: W_f·z, W_i·z, W_c·z, W_o·z
- **AND** SHALL compute forget gate f_t using QLIF quantum measurement on W_f·z
- **AND** SHALL compute input gate i_t using QLIF quantum measurement on W_i·z
- **AND** SHALL compute cell candidate c_hat_t = tanh(W_c·z)
- **AND** SHALL compute output gate o_t = sigmoid(W_o·z)
- **AND** SHALL compute cell state c_t = f_t * c\_{t-1} + i_t * c_hat_t
- **AND** SHALL compute hidden state h_t = o_t * tanh(c_t)
- **AND** SHALL return (h_t, c_t)

#### Scenario: QLIF Gate Activation

- **WHEN** computing a QLIF gate activation for a single neuron
- **THEN** the system SHALL build a QLIF circuit via `build_qlif_circuit()` with the neuron's linear output as input
- **AND** SHALL execute the circuit on the Qiskit Aer simulator with configurable shots (default 1024)
- **AND** SHALL use `QLIFSurrogateSpike.apply()` to create a differentiable output from the quantum measurement
- **AND** the output SHALL be P(|1⟩) from the quantum measurement, bounded in [0, 1]

#### Scenario: Classical Ablation Mode

- **WHEN** `use_quantum_gates` is False in the configuration
- **THEN** the system SHALL use `torch.sigmoid()` on the linear projection outputs for forget and input gates instead of QLIF circuits
- **AND** all other cell operations SHALL remain unchanged
- **AND** the brain SHALL be fully trainable without any quantum circuit execution

#### Scenario: Cell Output Shapes

- **WHEN** the cell processes input of dimension `input_dim` with hidden dimension `hidden_dim`
- **THEN** h_t SHALL have shape `(hidden_dim,)`
- **AND** c_t SHALL have shape `(hidden_dim,)`

### Requirement: Classical MLP Critic

The QLIFLSTMBrain SHALL include a classical MLP critic that estimates state value from raw sensory features and LSTM hidden state.

#### Scenario: Critic Input Construction

- **WHEN** the critic evaluates a state
- **THEN** the system SHALL concatenate raw sensory features with the LSTM hidden state h_t (detached from the actor's autograd graph)
- **AND** SHALL produce a scalar value estimate V(s)

#### Scenario: Critic Architecture

- **WHEN** the critic is initialized
- **THEN** the system SHALL create an MLP with configurable hidden layers (default 2 layers, 64 units each)
- **AND** SHALL use ReLU activations between hidden layers
- **AND** SHALL use orthogonal weight initialization for stable early training

#### Scenario: Gradient Isolation

- **WHEN** the critic computes value loss gradients
- **THEN** the LSTM hidden state in the critic input SHALL be detached from the actor's autograd graph
- **AND** critic gradients SHALL NOT flow through quantum circuits or the LSTM cell

### Requirement: Recurrent Rollout Buffer

The QLIFLSTMBrain SHALL use a rollout buffer that stores LSTM hidden states at chunk boundaries for truncated BPTT during PPO updates.

#### Scenario: Buffer Storage

- **WHEN** a step is added to the rollout buffer
- **THEN** the system SHALL store features, action, log_prob, value, reward, and done flag
- **AND** SHALL store the LSTM hidden state (h_t, c_t) at the time of collection

#### Scenario: Buffer Capacity

- **WHEN** the buffer reaches rollout_buffer_size (default 256) steps
- **THEN** the system SHALL trigger a PPO update
- **AND** SHALL clear the buffer after the update completes

#### Scenario: Episode-End Flush

- **WHEN** learn() is called with episode_done=True and the buffer has data
- **THEN** the system SHALL trigger a PPO update with the current buffer contents
- **AND** SHALL clear the buffer after the update

#### Scenario: GAE Advantage Computation

- **WHEN** a PPO update is triggered
- **THEN** the system SHALL compute Generalized Advantage Estimation with configurable gamma (default 0.99) and gae_lambda (default 0.95)
- **AND** SHALL bootstrap the last value from the critic if the episode is not done
- **AND** SHALL normalize advantages to zero mean and unit variance

#### Scenario: Sequential Chunk Generation

- **WHEN** chunks are requested for PPO update
- **THEN** the system SHALL split the buffer into sequential chunks of bptt_chunk_length (default 16) steps
- **AND** each chunk SHALL carry its initial LSTM hidden state (h_0, c_0) from collection time
- **AND** chunks SHALL be shuffled across minibatches for PPO but processed sequentially within each chunk
- **AND** episode boundaries within a chunk SHALL reset the LSTM hidden state to zeros

### Requirement: Recurrent PPO Training

The QLIFLSTMBrain SHALL implement PPO with chunk-based truncated BPTT for recurrent training.

#### Scenario: PPO Update with Truncated BPTT

- **WHEN** a PPO update is performed
- **THEN** the system SHALL iterate over num_epochs (default 2) epochs
- **AND** for each epoch SHALL generate sequential chunks from the buffer
- **AND** for each chunk SHALL re-run the QLIFLSTMCell forward pass from the chunk's stored initial (h_0, c_0)
- **AND** SHALL compute new log_probs, entropy, and values for each step in the chunk

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

- **WHEN** gradients are computed
- **THEN** the system SHALL clip gradient norms to max_grad_norm (default 0.5) for all parameters

#### Scenario: Entropy Decay

- **WHEN** entropy_decay_episodes is configured
- **THEN** the system SHALL linearly decay entropy_coef from entropy_coef to entropy_coef_end over entropy_decay_episodes
- **AND** SHALL clamp at entropy_coef_end after the decay period

### Requirement: Episode Lifecycle

The QLIFLSTMBrain SHALL support the standard episode lifecycle with LSTM state management.

#### Scenario: Episode Preparation

- **WHEN** prepare_episode() is called
- **THEN** the system SHALL reset LSTM hidden state h_t to zeros
- **AND** SHALL reset LSTM cell state c_t to zeros
- **AND** SHALL clear any pending buffer transition state

#### Scenario: Post-Episode Processing

- **WHEN** post_process_episode() is called
- **THEN** the system SHALL increment the episode counter

#### Scenario: Brain Copy

- **WHEN** copy() is called
- **THEN** the system SHALL return a deep copy with fresh zero-initialized hidden states
- **AND** the copy SHALL be independent of the original (no shared state)

### Requirement: QLIF-LSTM Configuration Schema

The configuration system SHALL support QLIF-LSTM-specific parameters via Pydantic BaseModel.

#### Scenario: Configuration Defaults

- **WHEN** a QLIFLSTMBrainConfig is created with no arguments
- **THEN** the system SHALL use these defaults:
  - lstm_hidden_dim: 32
  - shots: 1024
  - membrane_tau: 0.9
  - refractory_period: 0
  - gamma: 0.99
  - gae_lambda: 0.95
  - clip_epsilon: 0.2
  - entropy_coef: 0.05
  - entropy_coef_end: 0.005
  - entropy_decay_episodes: 200
  - value_loss_coef: 0.5
  - num_epochs: 2
  - rollout_buffer_size: 256
  - max_grad_norm: 0.5
  - actor_lr: 0.003
  - critic_lr: 0.001
  - critic_hidden_dim: 64
  - critic_num_layers: 2
  - bptt_chunk_length: 16
  - use_quantum_gates: True

#### Scenario: Sensory Module Configuration

- **WHEN** a QLIFLSTMBrainConfig specifies sensory_modules
- **THEN** the system SHALL configure the brain to use the specified sensory modules for feature extraction
- **AND** the default SHALL be FOOD_CHEMOTAXIS and NOCICEPTION (4 features total: 2 per module)

#### Scenario: YAML Config Loading

- **WHEN** a YAML config specifies `brain.name: qliflstm`
- **THEN** the config loader SHALL parse brain config using QLIFLSTMBrainConfig
- **AND** SHALL support all QLIF-LSTM-specific fields
