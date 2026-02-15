## Context

QSNNReinforceBrain demonstrated that QLIF quantum neurons with surrogate gradients can learn foraging (67% success), but failed on predator evasion across 16 rounds of optimization. Root cause analysis identified three issues:

1. **No critic** — REINFORCE uses episode-level return as the only learning signal; no per-step value estimate to distinguish "good state, bad action" from "bad state"
2. **High variance** — Single-sample policy gradient from one episode before update; advantage estimates are noisy
3. **Insufficient gradient passes** — Even with multi-epoch REINFORCE, each data collection yields limited parameter updates

QSNN-PPO addresses all three by combining the proven QSNN actor (QLIF circuits with surrogate gradients, from `_qlif_layers.py`) with a classical MLP critic and PPO's clipped surrogate objective. This follows the project convention of separate brain files per algorithm (cf. `mlpreinforce.py` / `mlpppo.py`).

## Goals / Non-Goals

**Goals:**

- Pair QSNN actor with classical MLP critic for value-based advantage estimation
- Implement PPO with GAE advantages and clipped surrogate objective
- Implement quantum output caching for efficient multi-epoch PPO updates
- Reuse shared QLIF infrastructure from `_qlif_layers.py`
- Integrate with existing brain factory and CLI

**Non-Goals:**

- Hebbian/local learning mode (PPO only)
- Adaptive entropy regulation (use fixed entropy coefficient; PPO's clipped objective provides sufficient stability)
- Exploration schedule (epsilon/temperature decay — PPO's entropy bonus handles exploration)
- Hardware QPU optimization
- Recurrent critic or attention-based architectures

## Decisions

### Decision 1: Separate File for QSNN-PPO

**Choice:** Create `qsnnppo.py` as a new brain file, not extend `qsnnreinforce.py`

**Rationale:**

- Project convention: MLP brains separate by algorithm (`mlpreinforce.py`, `mlpppo.py`, `mlpdqn.py`)
- `qsnnreinforce.py` is already 2,141 lines — adding PPO infrastructure would push past 3,000
- REINFORCE (per-episode updates) and PPO (fixed-size rollout buffer + minibatch epochs) are fundamentally different training loops
- Shared QLIF code is already extracted into `_qlif_layers.py`

### Decision 2: Classical MLP Critic

**Choice:** Classical MLP critic with input = raw sensory features + hidden spike rates (detached)

**Rationale:**

- Critic input is 24-dimensional: 8 raw sensory features + 16 hidden spike rates
- Hidden spike rates are detached from the autograd graph — no gradient flow from critic through quantum circuits
- This is 6x richer than the previous failed QSNN-AC attempt (QSNNReinforce's `use_critic` mode) which only saw 4 raw features
- Orthogonal weight initialization for stable early training
- 2 hidden layers of 64 units, ~5K classical params

**Alternatives Considered:**

- Quantum critic: Would double quantum circuit cost for marginal benefit; critic doesn't need quantum expressivity
- Shared-backbone actor-critic: Gradient interference between actor and critic destabilizes quantum parameter updates

### Decision 3: Separate Optimizers

**Choice:** Independent Adam optimizers for actor and critic

**Rationale:**

- Actor parameters are raw PyTorch tensors (W_sh, W_hm, theta_hidden, theta_motor) — not nn.Module parameters
- Critic is a standard nn.Module with its own parameter groups
- Actor uses L2 weight decay (lambda=0.001) to address unbounded weight growth; critic does not
- Different learning rates: actor_lr=0.01, critic_lr=0.001 (critic learns faster as it has more classical capacity)

### Decision 4: Rollout Buffer with Quantum Spike Caches

**Choice:** Custom `QSNNRolloutBuffer` extending standard PPO buffer with `hidden_spike_rates` and `spike_caches`

**Rationale:**

- Standard PPO buffer stores (state, action, log_prob, value, reward, done) — insufficient for QSNN
- `hidden_spike_rates`: numpy arrays needed for critic input reconstruction during PPO updates
- `spike_caches`: per-step lists of `{"hidden": [...], "motor": [...]}` dicts storing quantum spike probabilities for multi-epoch reuse
- Buffer size of 512 steps matches MLPPPOBrain convention

### Decision 5: Multi-Epoch PPO with Quantum Caching

**Choice:** Epoch 0 runs quantum circuits and caches spike probs; epochs 1+ reuse cached probs

**Rationale:**

- Quantum circuit execution is the dominant cost (~95% of forward pass time)
- Caching spike probabilities after epoch 0 means epochs 1-3 only recompute RY angles from updated weights, then apply the surrogate gradient using cached spike probs
- This gives 4x the gradient passes for ~1.3x the compute cost (vs 4x without caching)
- Adapted from QSNNReinforce's multi-epoch REINFORCE caching, proven to work with surrogate gradients

### Decision 6: Per-Step Forward Passes in Minibatches

**Choice:** Iterate over individual buffer steps within each minibatch (not batched forward pass)

**Rationale:**

- Quantum circuits are not batchable — each QLIF neuron runs a separate circuit
- Each step requires its own refractory state reset and multi-timestep integration
- Unlike MLPPPOBrain where a batched `model(states)` is possible, QSNN must loop over steps
- This is the same approach used in QSNNReinforce's multi-epoch updates

### Decision 7: No Exploration Schedule

**Choice:** Omit epsilon-greedy and temperature decay; rely on PPO's entropy bonus

**Rationale:**

- PPO's clipped surrogate objective naturally balances exploration and exploitation
- Fixed entropy coefficient (0.01-0.02) provides stable exploration pressure
- Simplifies the architecture compared to QSNNReinforce's dual exploration mechanisms
- If entropy collapse occurs, the coefficient can be tuned without code changes

### Decision 8: Inherit ClassicalBrain Protocol

**Choice:** Inherit from ClassicalBrain, same as QSNNReinforceBrain

**Rationale:**

- QSNN-PPO trains classical weights; quantum is only for neuron forward dynamics
- ClassicalBrain protocol provides the required interface (run_brain, learn, prepare_episode, etc.)
- Consistent with all other QSNN variants

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Per-step circuit execution in minibatches is slow | Quantum caching reduces circuit calls to 1x per buffer fill (not per epoch) |
| Critic overfitting on small rollout buffer | L2 regularization implicit in Adam; Huber loss for robustness to extreme penalties |
| Actor-critic gradient interference | Separate optimizers with independent LR; hidden spike rates detached from critic graph |
| PPO ratio clipping too aggressive for quantum noise | Default clip_epsilon=0.2 (standard); can increase if quantum noise causes excessive clipping |
| Weight growth in actor parameters | L2 weight decay (0.001) on actor optimizer + weight clamping ([-3, 3]) |
| Theta motor divergence | Theta motor norm clamping (max L2 norm 2.0), same as QSNNReinforce |
| Stale spike caches in later epochs | Caches store spike *probabilities*, not discrete spikes; RY angles recomputed from updated weights each epoch |
