## Context

QSNNReinforce already has actor-critic infrastructure: `CriticMLP`, GAE advantage computation (`_compute_gae`), deferred bootstrap for intra-episode window updates, `use_critic` config toggle, and a separate critic optimizer. This was added during the predator optimization work (P0-PP9) but never validated experimentally because QSNN-PPO was pursued as the primary variance reduction approach.

QSNN-PPO has been halted (4 rounds, 16 sessions, zero policy_loss due to surrogate gradient forward-pass disconnect). The existing QSNNReinforce critic code compiles and passes tests but has known gaps compared to the proven QSNN-PPO implementation:

1. **Critic input is sensory-only** — QSNNReinforce's critic sees only raw sensory features (8-dim). QSNN-PPO's critic sees sensory features + hidden spike rates (8+16=24-dim), giving it access to the agent's internal processing state.
2. **No orthogonal initialization** — QSNNReinforce's `CriticMLP` uses PyTorch default init. QSNN-PPO's `QSNNPPOCritic` uses orthogonal init with gain=sqrt(2), which is standard for value networks.
3. **No critic update logging** — QSNNReinforce has a debug-level log in `_update_critic()` but doesn't surface critic diagnostics to the experiment tracker.
4. **Hidden spike rates not stored** — The episode buffer stores features but not hidden spike rates, so the critic can't use them even if wanted.

## Goals / Non-Goals

**Goals:**

- Enrich critic input with hidden spike rates (sensory features + hidden spikes) for better state representation
- Add orthogonal initialization to `CriticMLP` for stable value estimation
- Store hidden spike rates in the episode buffer for critic input construction
- Add critic diagnostics to the PPO-style logging (value_loss, predicted vs actual returns, explained variance)
- Create experiment configs for A2C predator pursuit validation
- Validate that A2C mode produces non-degenerate critic learning and reduced gradient variance vs vanilla REINFORCE

**Non-Goals:**

- Changing the QLIF quantum circuit or surrogate gradient mechanism (stays identical)
- Adding PPO-style importance sampling (architecturally incompatible, as PPO-3 proved)
- Creating a separate `qsnna2c` brain type (this is an enhancement to `qsnnreinforce`, toggled via `use_critic`)
- Modifying the Hebbian/local learning mode (A2C only applies to surrogate gradient mode)
- Multi-epoch critic updates (critic updates once per REINFORCE update, after all actor epochs)

## Decisions

### 1. Enrich critic input: sensory features + detached hidden spike rates

**Decision**: Concatenate raw sensory features with hidden spike rates (detached from autograd graph) as critic input, matching QSNN-PPO's approach.

**Rationale**: The critic needs to estimate V(s) — the value of the current state. Raw sensory features alone (8-dim) provide the external observation, but the hidden layer spike rates (16-dim) capture the agent's internal processing state after quantum circuit execution. Two agents with the same sensory input but different hidden states may be in very different value situations (e.g., one has momentum toward food, another toward a predator). QSNN-PPO demonstrated that the concatenated input provides a 24-dim representation.

**Alternative considered**: Sensory features only (current implementation). Rejected because predator evasion requires the critic to distinguish subtle state differences that raw sensory features alone cannot capture — particularly when nociception gradients from two nearby predators produce similar aggregate sensory vectors but different hidden layer activations.

**Implementation**: Store `hidden_spike_rates` (detached numpy array) alongside `episode_features` in the episode buffer. In `_update_critic()`, concatenate them to form the critic input batch. The critic's `input_dim` becomes `num_sensory_neurons + num_hidden_neurons`.

### 2. Add orthogonal initialization to CriticMLP

**Decision**: Initialize `CriticMLP` linear layers with orthogonal weights (gain=sqrt(2)) and zero biases, matching QSNN-PPO's `QSNNPPOCritic`.

**Rationale**: Orthogonal initialization preserves gradient magnitudes through deep networks and is standard practice for value networks in RL (PPO/A2C papers, CleanRL, Stable-Baselines3). The current PyTorch default (Kaiming uniform) works but produces higher initial value prediction variance.

### 3. Keep `use_critic` as opt-in config toggle (default False)

**Decision**: Do not change the default. Users explicitly enable A2C mode via `use_critic: true` in config YAML.

**Rationale**: Existing QSNNReinforce configs and experiment results should not change behavior. The foraging task already achieves 67% success with vanilla REINFORCE. A2C adds computational overhead (critic forward/backward per step) that is only justified for harder tasks (predator evasion). Experiment configs for predator tasks will set `use_critic: true`.

**Alternative considered**: Default `use_critic: true` for all configs. Rejected because it would change behavior for existing foraging experiments and add unnecessary overhead.

### 4. Single-pass critic updates (no multi-epoch on critic)

**Decision**: The critic updates once per REINFORCE update call, after all actor epochs complete. This is the current behavior — keep it.

**Rationale**: Multi-epoch REINFORCE replays the actor forward pass with updated weights while reusing cached quantum spike probabilities. The critic is a classical MLP that doesn't benefit from quantum caching. Running the critic update once per training cycle with the GAE returns is standard A2C practice and avoids overfitting the critic to a small batch.

### 5. Critic diagnostics in experiment logging

**Decision**: Add per-update logging of: `value_loss`, `critic_pred_mean`, `critic_pred_std`, `target_return_mean`, `explained_variance` (1 - Var(returns - predicted) / Var(returns)).

**Rationale**: PPO-3 analysis showed that logging diagnostics are essential for debugging. The critic's learning progress (or lack thereof) is the key signal for whether A2C is working. Explained variance is the standard metric — values near 1.0 indicate good value estimation, near 0.0 indicates random predictions.

### 6. No new config fields needed

**Decision**: All necessary config fields already exist: `use_critic`, `critic_hidden_dim`, `critic_num_layers`, `critic_lr`, `gae_lambda`, `value_loss_coef`, `advantage_clip`. No new parameters required.

**Rationale**: The existing config schema was designed with A2C in mind. The only implementation gaps are in the runtime code (critic input enrichment, init, logging), not configuration.

## Risks / Trade-offs

**[Critic input stale by 1 step]** The hidden spike rates stored in the buffer come from `run_brain()`, which executed before `learn()` processes the reward. The critic sees hidden spikes from step t when estimating V(s_t), which is correct — V(s_t) should use the state representation at time t. No mitigation needed; this is standard.

**[Critic overhead on quantum-limited training]** Each step now requires a critic forward pass (cheap: ~5K params MLP) in addition to the QLIF circuit execution (expensive: quantum simulation). Risk: negligible. The QLIF circuit is >99% of step time. Mitigation: none needed.

**[GAE lambda sensitivity]** GAE with lambda=0.95 may be suboptimal for the short episodes typical of predator deaths (5-20 steps). Low-lambda (0.5-0.8) provides lower variance but higher bias. Mitigation: `gae_lambda` is already configurable; start with 0.95 and tune if needed.

**[Critic learning rate vs actor learning rate]** If the critic learns too fast, advantages become noisy (overfitting). If too slow, advantages are uninformative (underfitting). Current defaults: `critic_lr=0.001`, `actor_lr=0.01`. Mitigation: these are independently tunable. QSNN-PPO used the same defaults successfully (critic loss decreased).

**[Hidden spike rates may not help for simple tasks]** The enriched critic input (24-dim vs 8-dim) may overfit on small episode batches for simple foraging. Mitigation: foraging configs will keep `use_critic: false`. Only predator configs enable A2C.
