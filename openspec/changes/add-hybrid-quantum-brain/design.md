## Context

After 64+ experiment sessions across 5 quantum architectures, we've established that:

- **QSNN-REINFORCE** produces strong reactive policies (73.9% foraging) but fails at multi-objective tasks (predator evasion never improved across 16 optimization rounds)
- **QSNN-PPO** is fundamentally incompatible with surrogate gradients (importance ratio always 1.0)
- **A2C critic** fails on quantum features (explained variance ~0 across 4 rounds; architectural mismatch with non-stationary hidden spikes, short GAE windows, bootstrap death spiral)
- **Classical MLP-PPO** achieves 83-98% on the same tasks with simple architecture + larger rollout buffers

The hierarchical hybrid approach combines the proven QSNN reflex with classical PPO strategy. External research validates this pattern: QA2C (arXiv:2401.07043) showed hybrid approaches succeed where pure quantum fails; PPO-Q (arXiv:2501.07085) demonstrated pre/post-processing NNs around PQCs work across 8 environments.

### Current codebase patterns

- All QSNN brains extend `ClassicalBrain` (not `QuantumBrain`) to get the `learn()` interface — `QuantumBrain` requires `build_brain()` / `inspect_circuit()` which don't apply to QLIF architectures
- `_qlif_layers.py` provides shared QLIF components: `QLIFSurrogateSpike`, `build_qlif_circuit`, `execute_qlif_layer_*`, `encode_sensory_spikes` (~523 lines)
- `brain_factory.py` uses `isinstance` checks + lazy imports per brain type
- `config_loader.py` maps brain name strings to config classes in `BRAIN_CONFIG_MAP`
- PPO rollout buffer, GAE, clipped surrogate loss are implemented in `mlpppo.py` (~747 lines)

## Goals / Non-Goals

**Goals:**

- Create a `HybridQuantumBrain` that combines QSNN reflex (REINFORCE) + classical cortex (PPO) with mode-gated fusion
- Support three training stages: QSNN-only, cortex-only (QSNN frozen), joint fine-tune
- Reuse `_qlif_layers.py` for the quantum component and PPO patterns from `mlpppo.py` for the classical component
- Provide sensory-only critic (8-dim input) based on A2C failure analysis
- Register as a new brain type in the existing plugin system
- Enable experimentation on foraging (stage 1) and pursuit-predator (stage 2/3) environments

**Non-Goals:**

- Modifying `_qlif_layers.py` or any existing brain architecture
- Supporting local/Hebbian learning mode (REINFORCE + PPO only)
- Implementing new sensory modules or environment changes
- Quantum circuit execution on real QPU hardware (simulator only for now)
- Automatic stage transitions (manual config change between stages)

## Decisions

### Decision 1: Extend `ClassicalBrain`, not `QuantumBrain`

**Choice:** `HybridQuantumBrain(ClassicalBrain)`

**Rationale:** Both `QSNNReinforceBrain` and `QSNNPPOBrain` extend `ClassicalBrain` because they need the `learn(params, reward, episode_done)` interface for step-by-step online learning. `QuantumBrain` requires `build_brain()` / `inspect_circuit()` which return a single `QuantumCircuit` — not applicable to QLIF architectures that run circuits internally via `_qlif_layers.py`. Following the established pattern ensures consistency with the runner loop in `runners.py` (lines 654-662).

**Alternatives considered:**

- `QuantumBrain`: Would require implementing `build_brain()` / `inspect_circuit()` as no-ops. Breaks semantic contract.
- New `HybridBrain` protocol: Over-engineering for a single brain type.

### Decision 2: Fusion mechanism — mode-gated additive

**Choice:** `final_logits = reflex_logits * qsnn_trust + cortex_action_biases`

Where `qsnn_trust = softmax(mode_logits)[0]` (forage mode probability) from cortex output.

**Rationale:** The cortex learns WHEN to trust the QSNN (high trust during food-seeking, low during evasion) rather than HOW to replace it. This preserves the quantum reflex's proven foraging behaviour while allowing the cortex to override during danger. The mode gating is differentiable end-to-end for the cortex (cortex gradients flow through `mode_logits` and `action_biases`).

**Alternatives considered:**

- Pure additive (`reflex + cortex`): No mechanism to suppress QSNN during evasion. QSNN's food-seeking reflex would interfere with predator avoidance.
- Multiplicative gating on both: More complex, risk of vanishing gradients through double gating.
- Separate action heads with arbitration: More complex, doesn't leverage the proven QSNN policy directly.

### Decision 3: Cortex outputs 7 values (4 action biases + 3 mode logits)

**Choice:** Single cortex MLP with 7-dim output head, split into action biases and mode logits.

**Rationale:** Keeps the cortex simple (one forward pass). The 3 modes (forage, evade, explore) provide interpretable state for logging and debugging. Action biases allow the cortex to directly influence direction preference independent of mode.

**Alternatives considered:**

- Separate mode and action networks: Doubles cortex parameters and forward passes. No benefit for a 64-hidden MLP.
- 2 modes (forage/evade): Missing exploration mode. Agent may get stuck when neither food nor predator gradient is strong.

### Decision 4: Sensory-only critic (8-dim input)

**Choice:** Critic receives only raw sensory features, NOT hidden spike rates.

**Rationale:** A2C experiments (rounds 0-3) conclusively showed that hidden spikes are harmful to the critic:

- Non-stationary: hidden spikes shift as W_sh grows 4-8x during training
- A2C-3 with sensory-only input was the final test — EV still negative, but the lesson is clear: quantum-derived features should not enter the critic
- Classical PPO's critic works well with sensory-only input (83-98% success)

### Decision 5: Two separate training loops, not interleaved

**Choice:** Stage-based training controlled by `training_stage` config parameter:

- Stage 1: QSNN REINFORCE trains, cortex is untouched (not even instantiated in forward pass contributes to action)
- Stage 2: QSNN frozen (forward pass only), cortex PPO trains
- Stage 3: Both train with separate LRs

**Rationale:** Isolating training stages prevents interference between REINFORCE and PPO gradient signals. Stage 1 validates the quantum reflex works within the hybrid wrapper. Stage 2 tests whether the cortex can learn to modulate the frozen reflex. Stage 3 is optional and can be skipped if stage 2 works.

**Alternatives considered:**

- Simultaneous training from the start: Risk of PPO gradients destabilizing QSNN. No way to diagnose which component fails.
- Automatic stage progression: Requires convergence detection logic. Manual switching is simpler and gives more experimental control.

### Decision 6: PPO rollout buffer for cortex (not REINFORCE window)

**Choice:** 512-step rollout buffer with 4 epochs and 4 minibatches (matching `mlpppo.py` defaults).

**Rationale:** Classical PPO achieves 83-98% with buffer=2048. Using a rollout buffer (rather than QSNN's 20-step REINFORCE window) provides sufficient trajectory diversity for multi-objective learning. Buffer size of 512 balances memory cost with trajectory diversity — can be increased to 2048 if needed.

### Decision 7: QSNN forward pass runs every step regardless of stage

**Choice:** QSNN always computes reflex logits in `run_brain()`.

**Rationale:** In stage 2+, the cortex needs QSNN reflex logits for fusion even though QSNN weights are frozen. In stage 1, the cortex is bypassed (action comes directly from QSNN). The QSNN forward pass cost is constant regardless of training stage.

### Decision 8: Weight persistence for stage transitions via `.pt` files

**Choice:** Add `qsnn_weights_path` config param. Stage 1 auto-saves QSNN weights to `exports/<session>/qsnn_weights.pt` at end of training. Stage 2/3 configs reference this path to load pre-trained QSNN weights at init.

**Rationale:** No existing weight save/load infrastructure exists in the codebase — all `state_dict` usage is internal (target network sync, brain cloning). A minimal per-component approach is sufficient: save only QSNN raw tensors (W_sh, W_hm, theta_hidden, theta_motor) as a dict via `torch.save`. This keeps stage transitions explicit (user specifies the path) and avoids building a full checkpoint system.

**What gets saved:** `{"W_sh": tensor, "W_hm": tensor, "theta_hidden": tensor, "theta_motor": tensor}` — ~212 float32 values, \<1KB file.

**What gets loaded:** Stage 2/3 init checks `qsnn_weights_path`, loads the dict, assigns to QSNN tensors, and freezes them (stage 2) or sets reduced LR (stage 3).

**Alternatives considered:**

- Full brain checkpoint (all components): Over-engineering. Cortex starts fresh in stage 2, only QSNN weights carry over.
- Automatic stage progression within a single run: Requires convergence detection. Manual is simpler and gives more experimental control.

## Risks / Trade-offs

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cortex overrides QSNN entirely (`qsnn_trust` → 0) | Medium | Log `qsnn_trust` mean per episode. If < 0.1 consistently, add minimum trust floor (e.g., 0.2) or entropy bonus on mode distribution |
| Stage 3 joint fine-tune destroys QSNN | Medium | Use 10x lower LR for QSNN in stage 3. Compare against stage 2 checkpoint. Skip stage 3 if stage 2 succeeds |
| Fusion adds noise that degrades QSNN foraging in stage 1 | Low | Stage 1 bypasses cortex entirely (action from QSNN only). No fusion in stage 1 |
| PPO rollout buffer memory overhead | Low | 512 steps × ~30 floats per step ≈ 60KB. Negligible |
| QSNN state (membrane potential, refractory) complicates episode boundaries | Low | QSNN state already resets per episode in `QSNNReinforceBrain`. Follow same pattern |

## Open Questions

1. **Should stage 1 use full REINFORCE features (multi-epoch, adaptive entropy, reward normalization)?** Likely yes — these are all validated improvements. But the hybrid wrapper adds overhead. Start with full features and simplify if needed.

2. **Should the cortex receive QSNN reflex logits as additional input?** Currently no — the cortex sees only raw sensory input. Adding reflex logits (4-dim) could help the cortex learn what the QSNN "wants to do" and make better mode decisions. This is a natural extension if initial results are promising.

3. **Buffer size 512 vs 2048?** Classical PPO uses 2048. Starting with 512 to reduce experiment time (faster buffer fills). Increase if cortex learning is sample-starved.
