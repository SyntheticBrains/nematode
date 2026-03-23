## Context

The project has 18 brain architectures. The existing recurrent PPO pattern lives in `_reservoir_lstm_base.py` which uses a quantum-inspired QLIFLSTMCell with a reservoir front-end. The new brain extracts the proven training approach (chunk-based BPTT, separate optimizers, entropy decay) but replaces the quantum components with standard PyTorch LSTM/GRU — creating a clean, fast classical architecture.

Key existing code to build on:

- `_reservoir_lstm_base.py` — chunk-based BPTT training loop, separate optimizers, entropy decay, LR scheduling
- `qliflstm.py:335-487` — `QLIFLSTMRolloutBuffer` with `get_sequential_chunks()` pattern
- `mlpppo.py` — sensory module integration, weight persistence protocol, MLP network building
- `_ppo_buffer.py:70-94` — GAE advantage computation

## Goals / Non-Goals

**Goals:**

- New SOTA classical brain with proper temporal memory for temporal sensing tasks
- Clean implementation with no quantum or reservoir dependencies
- Full compatibility with existing sensory modules, STAM, and weight persistence
- Chunk-based truncated BPTT for memory-efficient recurrent training
- GRU option for ablation

**Non-Goals:**

- Replacing or modifying MLPPPOBrain — it remains the default classical brain
- Attention mechanisms or transformer architectures (future work)
- Multi-layer LSTM (single layer is sufficient for our observation dimensionality)
- Modifying existing reservoir-LSTM architectures

## Decisions

### 1. Shared LSTM with detached critic

**Decision**: Single LSTM processes features. Actor head receives LSTM output `h_t` directly. Critic head receives `h_t.detach()` to prevent critic gradients from affecting LSTM training.

**Rationale**: Proven in `_reservoir_lstm_base.py` (line 483). The actor optimizer trains LSTM + LayerNorm + actor MLP jointly via policy gradient. The critic trains independently on detached features. This prevents the value function loss from distorting the LSTM's learned temporal representations.

### 2. Separate actor/critic optimizers

**Decision**: Two Adam optimizers. Actor optimizer covers LSTM + LayerNorm + actor MLP parameters. Critic optimizer covers only critic MLP parameters.

**Rationale**: Following `_reservoir_lstm_base.py` (lines 359-369). Allows independent LR tuning for policy vs value learning, and ensures critic loss only affects critic weights.

### 3. Chunk-based truncated BPTT

**Decision**: During PPO updates, process the rollout in sequential chunks of `bptt_chunk_length` steps. Within each chunk, re-run the LSTM from stored initial hidden state, resetting at episode boundaries. Chunk ORDER is shuffled across epochs, but step order within chunks is preserved.

**Rationale**: This is the standard approach for recurrent PPO (used by OpenAI, CleanRL, etc.). Full-sequence BPTT (as in the MLPPPOBrain bolt-on) is memory-intensive for long episodes (1000 steps). Shuffled minibatches (standard PPO) break temporal coherence. Chunked BPTT is the proven middle ground.

### 4. LayerNorm on input features

**Decision**: Apply `nn.LayerNorm(input_dim)` to sensory features before the LSTM.

**Rationale**: Stabilizes LSTM input range across different sensory modules with varying scales. Used in `_reservoir_lstm_base.py` (line 331). The LayerNorm is trained jointly with the actor via the actor optimizer.

### 5. LSTMPPORolloutBuffer as new class

**Decision**: Create a new `LSTMPPORolloutBuffer` rather than reusing `QLIFLSTMRolloutBuffer`.

**Rationale**: The QLIF buffer stores hidden states as `(hidden_dim,)` tensors matching the QLIFLSTMCell API. Standard `nn.LSTM` uses `(num_layers, batch, hidden_dim)` tensors. A clean new buffer avoids shape conversion complexity and keeps the code self-contained. The GAE computation and sequential chunk generation patterns are duplicated (simple, ~50 lines total).

### 6. GRU support via config flag

**Decision**: `rnn_type: Literal["lstm", "gru"] = "lstm"`. When "gru", use `nn.GRU` instead of `nn.LSTM`. The buffer stores `c_state=None` for GRU.

**Rationale**: Minimal implementation cost (same API). Enables architecture ablation (LSTM vs GRU) without separate brain classes. GRU has fewer parameters and may train faster for simpler tasks.

## Risks / Trade-offs

**[Risk] Slower per-step inference than MLP PPO** → LSTM forward pass adds ~0.1ms per step. At 1000 steps/episode × 4000 episodes, adds ~7 min total. Acceptable.

**[Risk] Chunk-based BPTT has truncated gradients** → Gradients don't flow across chunk boundaries. With `bptt_chunk_length=16`, the LSTM can learn patterns up to ~16 steps. For temporal sensing (movement-sense-compare cycles of ~3-5 steps), this is sufficient.

**[Trade-off] Separate optimizers add complexity** → But they're proven in this codebase and enable better training dynamics. The alternative (single optimizer) risks critic loss distorting LSTM representations.

**[Trade-off] New buffer class vs reusing existing** → Code duplication of ~50 lines (GAE + chunk generation). But avoids shape compatibility issues and keeps the code self-contained.

## Notes

**No `num_minibatches` parameter**: Unlike MLPPPOBrain which uses `num_minibatches` to divide the rollout into shuffled minibatches, LSTMPPOBrain uses `bptt_chunk_length` to divide into sequential chunks. These serve the same role (controlling gradient batch size) but with different semantics. The `num_minibatches` parameter is intentionally absent from LSTMPPOBrainConfig.

**Supersedes `use_lstm` on MLPPPOBrain**: The `MLPPPOBrain.use_lstm` flag provides a simpler but inferior recurrent PPO (full-sequence processing, no chunk BPTT, no separate optimizers). `LSTMPPOBrain` is the recommended recurrent architecture. The old flag remains for backward compatibility but should not be used for new experiments.
