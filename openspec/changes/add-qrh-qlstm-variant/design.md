## Context

QRH and CRH use `ReservoirHybridBase` — an abstract base that handles sensory preprocessing, PPO training (minibatch-based), and episode lifecycle. Subclasses implement `_get_reservoir_features()` to transform sensory input through their reservoir (quantum or classical) and `_compute_feature_dim()` to declare output size. The base class builds MLP actor/critic readouts consuming the reservoir features.

QLIF-LSTM (`qliflstm.py`) is a standalone brain with its own PPO loop using truncated BPTT for recurrent training. Its core component `QLIFLSTMCell` is a reusable `nn.Module` that accepts `(x_t, h_prev, c_prev)` and returns `(h_new, c_new)`.

The QRH-QLSTM variant replaces QRH's MLP readout with an QLIF-LSTM temporal readout. The reservoir remains frozen (no gradients); only the LSTM cell, actor head, and critic MLP are trained.

### Key dimensions

| Component | QRH (8q) | QRH (10q) | CRH (10n) |
|-----------|----------|-----------|-----------|
| Reservoir output dim | 52 | 75 | 75 |
| Current MLP readout | 64-wide, 2-layer | 128-wide, 2-layer | 64-wide, 2-layer |

## Goals / Non-Goals

**Goals:**

- Compose QRH quantum reservoir with QLIF-LSTM temporal readout in a single brain
- Compose CRH classical reservoir with QLIF-LSTM temporal readout (ablation companion)
- Reuse `QLIFLSTMCell` from `qliflstm.py` unchanged
- Reuse reservoir code from QRH/CRH unchanged (import, don't copy)
- Support `use_quantum_gates` toggle for QLIF gate ablation (quantum vs classical sigmoid)
- Support LR warmup + decay scheduling (reuse from QLIF-LSTM)
- Provide evaluation configs for stationary predators (primary test) and pursuit predators (regression test)

**Non-Goals:**

- Modifying the reservoir itself (topology, qubits, encoding)
- Modifying the existing QRH, CRH, or QLIF-LSTM brains
- Multi-layer LSTM (single layer is sufficient for initial evaluation)
- Subclassing `ReservoirHybridBase` — the base class's PPO is minibatch-based and assumes a stateless readout, which conflicts with recurrent BPTT. We compose rather than inherit.

## Decisions

### 1. Composition over inheritance

**Decision**: Create a new `ReservoirLSTMBase` class that composes reservoir feature extraction with QLIF-LSTM readout, rather than subclassing `ReservoirHybridBase`.

**Rationale**: `ReservoirHybridBase.__init__` builds MLP actor/critic networks and a minibatch PPO buffer. Overriding all of this would require bypassing most of the base class. Composition is cleaner — we import and call `_get_reservoir_features()` / `_encode_and_run()` + `_extract_features()` patterns from QRH/CRH directly.

**Alternative considered**: Subclass `ReservoirHybridBase` and override `run_brain()`, `learn()`, `_perform_ppo_update()`. Rejected because the base class's `__init__` constructs MLP networks and optimizer that would be unused — wasteful and confusing.

**Implementation**: `ReservoirLSTMBase` is a new abstract base in `qrh_qlstm.py` that:

- Delegates reservoir computation to a reservoir instance (QRH or CRH brain used as a feature extractor only)
- Owns the QLIF-LSTM cell, actor head, critic MLP, and recurrent PPO buffer
- Implements `run_brain()`, `learn()`, `prepare_episode()` following the QLIF-LSTM pattern

### 2. Reservoir as feature extractor (delegation pattern)

**Decision**: Instantiate the full QRH/CRH brain internally but only use its `_get_reservoir_features()` method. Do not call its `run_brain()` or `learn()`.

**Rationale**: The reservoir setup code (topology generation, index precomputation, statevector simulation) is tightly coupled to the QRH/CRH brain classes. Extracting it into a separate module would require significant refactoring. Using the brain as a feature extractor is simpler and guaranteed to produce identical reservoir features.

**Alternative considered**: Extract reservoir logic into a standalone `QuantumReservoir` class. Cleaner but requires refactoring QRH/CRH internals — higher risk, higher effort, no functional benefit for this change.

### 3. LSTM input: reservoir features (not raw sensory)

**Decision**: The QLIF-LSTM cell receives reservoir features (52 or 75 dims) as input, not raw sensory features (7-9 dims).

**Rationale**: The whole hypothesis is that the reservoir's rich feature representation, combined with temporal memory, produces better spatial navigation than either alone. Feeding raw sensory features would bypass the reservoir entirely.

**LSTM hidden dim**: Default to 64 (matching QRH's readout_hidden_dim). The reservoir outputs 52-75 features, so the LSTM combined input `[x_t, h_prev]` is 116-139 dims — manageable for 4 linear projections.

### 4. Actor receives [reservoir_features, h_t]

**Decision**: Actor head input is `[reservoir_features, h_t]` (not just `h_t`), matching the actor fix applied to QLIF-LSTM in Round 10.

**Rationale**: Proven beneficial — gives the actor direct access to current spatial information (reservoir features) alongside temporal context (LSTM hidden state). The critic also receives `[reservoir_features, h_t.detach()]` (same pattern as QLIF-LSTM).

### 5. Recurrent PPO with truncated BPTT

**Decision**: Use the same truncated BPTT approach from QLIF-LSTM (chunk-based, `bptt_chunk_length=32`), not the base class's minibatch PPO.

**Rationale**: The LSTM readout is recurrent — minibatch PPO (random shuffling of transitions) would break temporal dependencies. Truncated BPTT processes sequential chunks, re-running the LSTM forward from stored hidden states at chunk boundaries.

**Reuse**: Copy the `QLIFLSTMRolloutBuffer` pattern and PPO update loop from `qliflstm.py`. The buffer stores reservoir features (instead of raw sensory features) alongside actions, log probs, values, rewards, dones, and chunk-boundary hidden states.

### 6. Shared base for QRH-QLSTM and CRH-QLSTM

**Decision**: Create `ReservoirLSTMBase` abstract class with two concrete subclasses: `QRHQLSTMBrain` and `CRHQLSTMBrain`.

**Rationale**: The two variants differ only in which reservoir they use. All LSTM readout, PPO training, and lifecycle code is identical. A shared base avoids duplication.

**Structure**:

```text
ReservoirLSTMBase (abstract)
├── _create_reservoir() → abstract, returns reservoir brain instance
├── _get_reservoir_features(sensory) → calls reservoir._get_reservoir_features()
├── QLIFLSTMCell (LSTM readout)
├── actor_head, critic MLP
├── Recurrent PPO (truncated BPTT)
└── Episode lifecycle (prepare_episode, learn, post_process_episode)

QRHQLSTMBrain(ReservoirLSTMBase)
└── _create_reservoir() → QRHBrain(config)

CRHQLSTMBrain(ReservoirLSTMBase)
└── _create_reservoir() → CRHBrain(config)
```

### 7. Config structure

**Decision**: `QRHQLSTMBrainConfig` contains both reservoir params (from `QRHBrainConfig`) and LSTM readout params. Uses Pydantic composition (flat fields, not nested configs).

**Key fields**:

- Reservoir: `num_reservoir_qubits`, `reservoir_depth`, `reservoir_seed`, `use_random_topology` (same as QRH)
- LSTM: `lstm_hidden_dim` (64), `bptt_chunk_length` (32)
- QLIF gates: `shots`, `membrane_tau`, `refractory_period`, `use_quantum_gates`
- PPO: `actor_lr`, `critic_lr`, `gamma`, `gae_lambda`, `clip_epsilon`, `entropy_coef`, `num_epochs`, `rollout_buffer_size`, `max_grad_norm`
- LR schedule: `lr_warmup_episodes`, `lr_warmup_start`, `lr_decay_episodes`, `lr_decay_end`
- Critic: `critic_hidden_dim`, `critic_num_layers`
- Sensory: `sensory_modules`

`CRHQLSTMBrainConfig` mirrors this with CRH reservoir params (`num_reservoir_neurons`, `spectral_radius`, etc.).

### 8. Feature normalization

**Decision**: Apply `LayerNorm` to reservoir features before feeding to LSTM (same as QRH's current normalization).

**Rationale**: Reservoir features have varying scales (X/Y/Z expectations in [-1,1], ZZ correlations in [-1,1]). LayerNorm stabilizes LSTM input. QRH already uses this — we maintain consistency.

## Risks / Trade-offs

**[High-dim LSTM input]** 52-75 reservoir features into a 64-dim LSTM creates large gate projection matrices (4 × Linear(128→64) = 32K params for gates alone). → Mitigation: This is comparable to QLIF-LSTM's current setup (57-dim input after actor fix). Monitor for slow convergence; if needed, add an optional linear projection layer to compress reservoir features before LSTM.

**[Quantum execution cost]** QRH-QLSTM with `use_quantum_gates=true` has two quantum components: reservoir (statevector simulation) + QLIF gates (shot-based measurement). Training will be slower than either QRH or QLIF-LSTM alone. → Mitigation: Start with classical gate ablation (`use_quantum_gates=false`) for rapid iteration, same as QLIF-LSTM evaluation strategy.

**[BPTT through frozen reservoir]** The reservoir features are computed fresh each timestep but gradients don't flow through the reservoir. The LSTM must learn purely from the readout side. → This is by design (reservoir is fixed) and matches how QRH's MLP readout trains. Not a risk, just a constraint.

**[Code duplication]** The recurrent PPO buffer and training loop are copied from `qliflstm.py` rather than shared. → Acceptable for now. A future refactor could extract a shared `RecurrentPPOTrainer` utility, but premature abstraction before validation.
