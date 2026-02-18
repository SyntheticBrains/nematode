## Context

The HybridQuantum brain (96.9% post-convergence) uses a QSNN reflex (~212 quantum params) + classical cortex MLP (~5K params) + classical critic (~5K params) + mode-gated fusion. Only ~1% of total parameters are quantum. Classical ablation (HybridClassical: 96.3%) proved the QSNN quantum component is not the performance driver — the three-stage curriculum and fusion architecture are.

We want significantly more quantum involvement at the strategic decision-making layer. This design replaces the classical cortex MLP with a QSNN-based cortex using grouped QLIF neurons per sensory modality, raising the quantum fraction from ~1% to ~11%.

**Key constraint**: PPO is incompatible with surrogate gradients (proven across 16 sessions — forward pass returns constant, ratio always 1.0). The cortex QSNN must use REINFORCE. However, REINFORCE alone has high variance. The innovation here is feeding critic-provided GAE advantages into the REINFORCE loss — combining stable value estimation with surrogate gradient compatibility.

## Goals / Non-Goals

**Goals:**

- Replace the classical cortex MLP with a QSNN cortex (~350-500 quantum params) using grouped QLIF neurons per sensory modality
- Train the QSNN cortex with REINFORCE using GAE advantages from the classical critic (not PPO)
- Support four-stage curriculum: reflex alone → cortex alone → joint fine-tune → multi-sensory scaling
- Provide a clear ablation path (same-sized classical MLP cortex as control)
- Handle multi-sensory environments natively through modality-specific QLIF neuron groups

**Non-Goals:**

- Replacing the classical critic with a quantum critic (proven to fail — EV: -0.620 across 16 sessions, external research confirms pure quantum A2C fails)
- Replacing the mode-gated fusion with a quantum circuit (margin for improvement is tiny at 96.9%, and fusion sees pre-computed logits not raw sensory data)
- Using PPO for the QSNN cortex (architecturally incompatible with surrogate gradients)
- Achieving real QPU execution (simulator-first; hardware mapping is a future concern)

## Decisions

### Decision 1: QSNN Cortex with Grouped Sensory QLIF Neurons

**Choice**: Build the cortex as a 3-layer QLIF network with modality-specific input groups feeding into a shared hidden layer, rather than a single flat QLIF network or a PQC/VQC approach.

**Architecture**:

```text
Multi-sensory Input (8+ dim, from sensory modules)
         |
         v
Modality-Specific QLIF Groups (sensory layer)
  food_chemotaxis: 2 features → 4 QLIF neurons
  nociception:     2 features → 4 QLIF neurons
  mechanosensation: 3 features → 4 QLIF neurons
  (thermotaxis:    3 features → 4 QLIF neurons — future)
         |
         v  (W_cortex_sh: grouped, not fully connected)
Shared Hidden Layer: 12 QLIF neurons
  (cross-modality integration)
         |
         v  (W_cortex_hm: 12 × 8)
Output Layer: 8 QLIF neurons
  → 4 action biases + 3 mode logits + 1 trust modulation
```

**Why grouped over flat**: A flat QLIF network (all features concatenated into all neurons) loses modality-specific processing. Grouped connectivity mirrors C. elegans neurobiology where distinct neuron classes (AWC/AWA for food, ASH/ADL for nociception) process specific sensory modalities before interneuron integration. This also creates a natural extension point for adding new modalities.

**Why QLIF over PQC/VQC**: The surrogate gradient framework (`QLIFSurrogateSpike` in `_qlif_layers.py`) is the only proven gradient method in this project. PQC approaches would require parameter-shift gradients (~1000x weaker) or a tangential DNN approximation (exponential classical shadow, untested infrastructure). QLIF lets us reuse 100% of the existing quantum circuit infrastructure.

**Alternatives considered**:

- *PQC with data re-uploading (PPO-Q style)*: Requires parameter-shift or tangential DNN; exponential classical shadow (2^(N+1) neurons) undermines quantum advantage; entirely new gradient infrastructure needed
- *Flat QLIF network*: Loses modality-specific processing; all-to-all connectivity at sensory layer wastes parameters on cross-modality connections that should happen in hidden layer
- *Quantum attention/transformer*: No proven implementation in Qiskit; theoretical appeal but no experimental evidence of trainability

### Decision 2: REINFORCE with Critic-Provided GAE Advantages

**Choice**: Train the QSNN cortex using REINFORCE policy gradient, but replace the noisy self-computed returns with GAE advantages from the classical critic. The critic trains via standard MSE/Huber loss against observed returns (no gradient flow through the quantum circuit).

**Mechanism**:

```python
# During cortex PPO buffer update (existing infrastructure)
# 1. Collect (state, action, reward, value, log_prob) in rollout buffer
# 2. Compute GAE advantages using classical critic V(s)
# 3. Instead of PPO clipped surrogate, use REINFORCE:

# QSNN cortex forward pass (surrogate gradient)
cortex_motor_spikes = multi_timestep_differentiable(cortex_features)
cortex_logits = (cortex_motor_spikes - 0.5) * logit_scale
action_probs = softmax(cortex_logits / temperature)
log_prob = log(action_probs[action])

# REINFORCE loss with GAE advantages (not self-computed returns)
loss = -log_prob * gae_advantage.detach() - entropy_coef * entropy
loss.backward()  # surrogate gradients flow through QLIFSurrogateSpike
cortex_optimizer.step()
```

**Why this works**: REINFORCE only needs `d(log_prob)/d(theta)` (backward pass), which surrogate gradients provide. The GAE advantages come from the classical critic and are detached from the cortex graph — no importance sampling ratio needed (unlike PPO). This combines the proven REINFORCE+surrogate approach with the stable advantage estimation that makes PPO effective.

**Why not pure REINFORCE with self-computed returns**: High variance. The standalone QSNN achieved only 1.25% on pursuit predators with REINFORCE alone. The critic provides dense per-step advantage signals that dramatically reduce variance.

**Why not A2C**: A2C requires the critic loss to flow through the actor's forward pass for proper value function fitting. With surrogate gradients, the forward pass is a constant (quantum measurement), so the critic can't learn to predict the actor's state representations. Our A2C experiments confirmed this (EV: -0.620). Here, the critic learns independently from its own sensory input.

**Alternatives considered**:

- *PPO for cortex QSNN*: Proven incompatible — ratio = 1.0 always with surrogate gradients
- *REINFORCE without critic*: High variance, 1.25% on pursuit predators after 60 sessions
- *A2C with shared features*: Critic can't learn V(s) under partial observability with quantum forward pass

### Decision 3: Four-Stage Curriculum

**Choice**: Extend the proven three-stage curriculum to four stages:

| Stage | QSNN Reflex | QSNN Cortex | Classical Critic | Task |
|-------|-------------|-------------|-----------------|------|
| 1 | Train (REINFORCE) | Frozen/unused | Unused | Foraging only |
| 2 | Frozen | Train (REINFORCE+GAE) | Train (MSE) | Pursuit predators |
| 3 | Train (low LR) | Train (low LR) | Train | Pursuit predators |
| 4 | Train (low LR) | Train (low LR) | Train | + thermotaxis |

**Why 4 stages**: Stage 4 is the multi-sensory scaling test. The cortex's grouped QLIF architecture makes adding new modalities a config change (add a QLIF group), not a code change. This is the key advantage over the classical cortex MLP.

**Why separate stage 2 for cortex**: Training both QSNN components simultaneously from scratch would create competing gradient signals. Freezing the reflex in stage 2 lets the cortex learn to complement the reflex's proven foraging behaviour. This pattern is validated by HybridQuantum's success.

### Decision 4: Weight Connectivity — Grouped Sensory, Dense Hidden

**Choice**: The sensory layer uses **block-diagonal** weight connectivity (each modality group connects only to its own QLIF neurons). The hidden and output layers use **dense** (fully connected) weights.

```text
W_cortex_sensory: block-diagonal
  [W_food  0      0     ]     food features → food QLIF group
  [0       W_noci 0     ]     nociception features → noci QLIF group
  [0       0      W_mech]     mechano features → mechano QLIF group

W_cortex_sh: dense (12 sensory QLIF → 12 hidden QLIF)
W_cortex_ho: dense (12 hidden QLIF → 8 output QLIF)
```

**Why block-diagonal sensory**: Forces modality-specific processing before cross-modality integration. This is biologically motivated (C. elegans sensory neurons are modality-specific) and prevents early-layer cross-contamination that could slow learning.

**Implementation**: Use `execute_qlif_layer_differentiable` separately per modality group in the sensory layer, then concatenate spike outputs before passing to the shared hidden layer. This reuses the existing QLIF infrastructure without modification.

### Decision 5: Reuse Classical Critic and Rollout Buffer

**Choice**: Keep the classical critic MLP and `_CortexRolloutBuffer` from HybridQuantum unchanged. The critic receives raw sensory features (not QSNN cortex hidden spikes) and trains via Huber loss.

**Why sensory-only critic input**: The A2C experiments proved that feeding QSNN hidden spike rates to the critic causes non-stationarity (spikes change as actor learns). Using raw sensory features (stable, policy-independent) gives the critic the best chance of learning V(s). The HybridQuantum critic already works this way and achieved EV +0.29.

### Decision 6: Cortex Output Mapping

**Choice**: The cortex output layer has 8 QLIF neurons. Their spike probabilities are mapped to cortex outputs as follows:

- Neurons 0-3: action biases (mapped via `(spike_prob - 0.5) * logit_scale`)
- Neurons 4-6: mode logits (mapped via `(spike_prob - 0.5) * mode_logit_scale`)
- Neuron 7: trust modulation (mapped via `spike_prob` directly, used as scaling factor)

This keeps the fusion mechanism identical to HybridQuantum — the cortex still produces action biases and mode logits, just from QLIF neurons instead of an MLP.

### Decision 7: Extract Shared Hybrid Brain Infrastructure

**Choice**: Before building HybridQuantumCortexBrain, extract shared code from `hybridquantum.py` and `hybridclassical.py` into a new `_hybrid_common.py` module. Both existing brains will be refactored to import from the shared module, and the new brain will build on top of it.

**What to extract** (~370 lines of duplicated code):

- `_CortexRolloutBuffer` class (rollout storage, GAE computation, returns calculation)
- `_fuse()` mode-gated fusion logic (`final_logits = reflex_logits * qsnn_trust + action_biases`)
- `_cortex_forward()` and `_cortex_value()` classical cortex MLP forward passes
- `_init_cortex()` classical cortex + critic MLP initialization with orthogonal init
- `_get_cortex_lr()` and `_update_cortex_learning_rate()` LR scheduling
- Cortex weight persistence (`_save_cortex_weights`, `_load_cortex_weights`)
- PPO update logic (`_perform_ppo_update`)
- Shared constants and defaults

**Why extract first**: `hybridclassical.py` (1,489 lines) is a near-complete copy of `hybridquantum.py` (1,869 lines) with only the reflex layer changed. Adding a third copy would triple ~370 lines of identical infrastructure. Extracting first follows the existing project pattern (`_qlif_layers.py` already extracts shared QLIF infrastructure) and makes the new brain cleaner to implement.

**Why not a base class**: A mixin or utility module is more flexible than a shared base class. The three hybrid brains have different reflex implementations (QSNN, classical MLP, QSNN) and different cortex types (classical MLP, classical MLP, QSNN). A base class would require complex template method patterns. A utility module lets each brain import only what it needs.

**Alternatives considered**:

- *Copy-paste from hybridquantum.py*: Creates a third copy of ~370 lines, making future maintenance error-prone. The existing duplication between hybridquantum.py and hybridclassical.py is already technical debt.
- *Shared base class (AbstractHybridBrain)*: Over-constrains the architecture. The three brains have sufficiently different initialization and training flows that a shared base class would need many abstract methods and template method overrides.

## Risks / Trade-offs

**REINFORCE + external GAE advantages is untested** → Start with pure REINFORCE (self-computed returns) for the cortex in initial experiments. Add GAE advantages incrementally once basic training works. The rollout buffer already computes GAE — switching the loss function is a config toggle.

**QSNN cortex may be too small (~400 params vs ~5K MLP)** → If insufficient capacity, increase hidden layer width from 12 to 16-24 neurons. The fan-in-aware scaling (`tanh(w·x / sqrt(fan_in))`) handles wider layers automatically. The QSNN reflex succeeded with just 92 params on foraging.

**2-3x quantum circuit cost per decision** → The cortex can use fewer integration timesteps than the reflex (4 vs 10) since it outputs biases/modes, not primary action logits. Also, cortex circuits only run during stage 2+ (stage 1 is reflex-only, no cortex cost).

**Classical ablation may still show parity** → This is valuable scientific data. Build `HybridClassicalCortex` (replace QSNN cortex with equally-sized classical MLP, ~400 params with matching topology) as the ablation control. If the classical version matches, we learn that the cortex scale/structure matters more than its quantum nature.

**Grouped sensory QLIF adds implementation complexity** → The block-diagonal weight structure requires per-group QLIF execution. This is straightforward using existing `execute_qlif_layer_differentiable` called once per group, but it's more code than a flat network. The biological motivation and multi-sensory extensibility justify this.

**Gradient interference between reflex and cortex REINFORCE** → In stage 3, both QSNN components train simultaneously with REINFORCE on the same final action probabilities. Use the proven mitigation: cortex QSNN trains with reduced LR (`joint_finetune_lr_factor = 0.1`), and gradients are clipped independently per component.

## Open Questions

1. **Cortex integration timesteps**: Should the cortex QSNN use the same number of integration timesteps as the reflex (10), or fewer (4-6) since it produces biases not primary logits? Fewer timesteps reduce circuit cost but increase shot noise.

2. **Trust modulation neuron**: Is a dedicated QLIF neuron for trust modulation (output neuron 7) better than deriving trust purely from mode logits (current HybridQuantum approach)? The dedicated neuron gives the cortex a direct, continuous-valued trust signal rather than a softmax-derived one.

3. **Stage 2 training algorithm**: Should stage 2 start with pure REINFORCE (proven safe, validated on foraging) and add GAE advantages only after basic learning is confirmed? Or go directly to REINFORCE+GAE since the infrastructure exists?
