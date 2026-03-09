## Context

The codebase has 14 brain architectures, all stateless per-step: no architecture maintains temporal context across simulation steps within an episode. The QSNN-PPO brain (`qsnnppo.py`) provides the closest precedent — it pairs QLIF quantum circuits with PPO training using surrogate gradients and quantum caching. The shared `_qlif_layers.py` module provides all quantum circuit infrastructure: `QLIFSurrogateSpike` (autograd function with sigmoid surrogate centered at π/2), `build_qlif_circuit()` (RY+RX 2-gate circuit), and `encode_sensory_spikes()`.

The QLIF-LSTM brain introduces temporal memory by embedding QLIF quantum neurons into the LSTM gating mechanism. This is the first recurrent architecture in the codebase, requiring a new rollout buffer design for chunk-based truncated BPTT.

## Goals / Non-Goals

**Goals:**

- Implement a custom LSTM cell with QLIF quantum gates (forget, input) and classical gates (candidate, output)
- Implement recurrent PPO training with chunk-based truncated BPTT
- Provide classical ablation via `use_quantum_gates` config flag for controlled comparison
- Reuse maximum existing infrastructure from `_qlif_layers.py` and PPO patterns from `qsnnppo.py`
- Create configs for all 4 evaluation stages (foraging, pursuit predators, thermotaxis large)

**Non-Goals:**

- Multi-layer LSTM (config supports it, but initial implementation uses 1 layer)
- Quantum caching for multi-epoch LSTM (unlike QSNN-PPO, LSTM must re-run sequentially per chunk — caching individual gate outputs would add complexity without clear benefit given sequential dependency)
- QRH-QLSTM variant (Stage 4d — deferred to a separate change after H.4 evaluation)
- Changes to environment mechanics (within-episode temporal memory is sufficient)
- QPU execution (simulator only for initial evaluation)

## Decisions

### 1. Custom LSTM Cell vs. nn.LSTM wrapper

**Decision:** Custom `QLIFLSTMCell` as `nn.Module`.

**Rationale:** `nn.LSTM` fuses all four gates internally with sigmoid activations — there's no hook to replace individual gate activations. A custom cell gives explicit access to each gate's linear projection output, allowing QLIF circuits to replace sigmoid on forget and input gates while keeping tanh (candidate) and sigmoid (output gate) classical.

**Alternative considered:** Wrapping nn.LSTMCell and intercepting gate outputs. Rejected because LSTMCell also fuses the gate computation internally.

### 2. Which gates get QLIF quantum activations

**Decision:** Forget gate (f_t) and input gate (i_t) use QLIF. Cell candidate (c_hat_t) uses tanh. Output gate (o_t) uses sigmoid.

**Rationale:** Forget and input gates control information flow — they're the "memory management" gates that determine what to remember and what to add. This is where quantum dynamics could provide richer gating via superposition-based probability distributions. The cell candidate produces unbounded content (tanh is natural), and the output gate modulates final output (sigmoid is natural). This matches the QSNN-QLSTM literature (arXiv:2505.01735).

### 3. Per-neuron QLIF execution (not batched)

**Decision:** Each QLIF gate neuron runs one quantum circuit independently, matching the existing `execute_qlif_layer_differentiable()` pattern.

**Rationale:** Quantum circuits are inherently not batchable — each circuit measures one qubit. The existing QLIF infrastructure already handles per-neuron execution with surrogate gradients. With `lstm_hidden_dim=32`, this means 64 circuits per step (32 forget + 32 input), which is feasible on the Aer simulator.

**Risk:** 64 circuits/step is ~6x more than QSNN-PPO's hidden layer (16 neurons × ~1 integration step). Mitigation: the circuit is minimal (2 gates: RY + RX), and classical ablation mode (`use_quantum_gates=False`) eliminates this cost entirely for rapid iteration.

### 4. Chunk-based truncated BPTT for recurrent PPO

**Decision:** Split rollout buffer into fixed-length chunks (default 16 steps). Store LSTM hidden state (h_t, c_t) at each chunk boundary during collection. During PPO update, re-run LSTM forward pass within each chunk from stored initial state. Chunks are shuffled across minibatches.

**Rationale:** Standard PPO shuffles individual transitions, which destroys temporal ordering needed for recurrence. Full-trajectory BPTT is memory-intensive and doesn't allow shuffling. Chunk-based BPTT is the standard approach for recurrent PPO (used in CleanRL, SB3) — it balances gradient quality with computational cost and allows some degree of shuffling (across chunks, not within).

**Alternative considered:** Storing all LSTM outputs in the buffer and not re-running during PPO. Rejected because weights change across epochs, making cached outputs stale.

### 5. No quantum caching (unlike QSNN-PPO)

**Decision:** Do not implement quantum output caching for multi-epoch PPO.

**Rationale:** QSNN-PPO caches spike probabilities because its forward pass is feedforward — cached outputs at step t are independent of step t-1. LSTM is sequential — step t depends on h\_{t-1}, which depends on updated weights. Caching gate outputs would require caching at every step within every chunk, and the cached values would be invalidated when weights change between epochs. The complexity is not justified for the initial implementation.

### 6. Sensory encoding

**Decision:** Reuse `encode_sensory_spikes()` from `_qlif_layers.py` and `extract_classical_features()` from `modules.py`, matching QSNN-PPO's pattern.

**Rationale:** Proven encoding pipeline. Default sensory modules: FOOD_CHEMOTAXIS (2 features: strength, angle) + NOCICEPTION (2 features: strength, angle) = 4 input features.

### 7. Critic architecture

**Decision:** Classical MLP critic receiving raw sensory features concatenated with detached h_t (LSTM hidden state), matching QSNN-PPO's pattern of detaching hidden representations from the actor graph.

**Rationale:** Keeps critic gradients from flowing through quantum circuits. The LSTM hidden state provides a rich temporal summary for value estimation.

## Risks / Trade-offs

- **[Computational cost]** 64 quantum circuits per step (32 forget + 32 input gate neurons at default hidden_dim=32). → Mitigation: Classical ablation mode for fast iteration; reduce `lstm_hidden_dim` to 16 for faster experiments; minimal 2-gate circuits.

- **[Gradient quality]** Surrogate gradients through LSTM gates add another approximation layer on top of QLIF surrogate. → Mitigation: The surrogate gradient is well-tested in QSNN architectures; truncated BPTT limits gradient chain length to chunk_length steps.

- **[Truncated BPTT information loss]** Chunks of 16 steps may lose longer-range dependencies. → Mitigation: Configurable `bptt_chunk_length`; 16 is conservative — can increase for longer episodes if needed.

- **[First recurrent architecture]** No existing recurrent PPO patterns in codebase to validate against. → Mitigation: Start with classical ablation mode to verify PPO training loop correctness before enabling quantum gates.
