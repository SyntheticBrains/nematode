# Design: Spiking Neural Network with Surrogate Gradient Descent

## Context

The current SpikingBrain implementation uses STDP (Spike-Timing Dependent Plasticity), a local learning rule inspired by biological synaptic plasticity. However, experimental results show complete learning failure across 400 episodes with 0% success rate.

**Root Cause Analysis** identified:
1. **Critical bugs**: Wrong angle preprocessing, broken credit assignment, missing weight updates
2. **Architectural limitations**: STDP's local learning cannot solve RL's global credit assignment problem
3. **Sparse signals**: STDP only updates weights when spikes coincide (~60ms window), creating very sparse learning
4. **Stochastic noise**: Poisson spike encoding reduces signal quality

**State-of-the-Art Approach**: Modern neuromorphic research (SpikingJelly, snnTorch, Norse) uses **surrogate gradient descent**, which:
- Keeps spiking neurons for biological plausibility
- Enables gradient-based learning through differentiable spike approximation
- Works with standard RL algorithms (policy gradients, PPO, Actor-Critic)
- Has proven success on complex tasks (ImageNet classification, RL navigation)

## Goals / Non-Goals

**Goals:**
- Implement a working spiking neural network that learns successfully
- Match MLPBrain's learning paradigm (policy gradients with discounted returns)
- Maintain biological plausibility (LIF neurons, spike-based computation)
- Use state-of-the-art surrogate gradient techniques
- Achieve comparable performance to MLP within reasonable hyperparameter tuning

**Non-Goals:**
- Perfect biological realism (e.g., detailed synaptic dynamics, ion channels)
- Backward compatibility with STDP implementation (different algorithm entirely)
- Matching MLP performance exactly (spiking adds complexity)
- Hardware deployment (focus on simulation first)

## Decisions

### Decision 1: Surrogate Gradient Method - Sigmoid Derivative

**Choice**: Use sigmoid-based surrogate gradient:
```text
forward: spike = H(v - v_th)  where H is Heaviside step function
backward: ∂spike/∂v ≈ α·σ(α(v - v_th))·(1 - σ(α(v - v_th)))
```

**Rationale**:
- **Smooth and differentiable**: Provides stable gradients
- **Widely used**: Proven in SpikingJelly, snnTorch literature
- **Tunable**: Parameter α controls gradient smoothness (default 10.0)
- **Simple**: Single parameter, easy to implement and debug

**Alternatives Considered**:
- **Rectangular**: `max(0, 1 - |v - v_th|)` - Simple but sharp corners can cause gradient issues
- **Triangle**: Piecewise linear - More complex, similar performance
- **Arc-tan**: `1/π · atan(α(v - v_th))` - Good but less standard

### Decision 2: Learning Algorithm - REINFORCE (Policy Gradient)

**Choice**: Use REINFORCE with baseline, exactly like MLPBrain

**Rationale**:
- **Proven to work**: MLP achieves high success rates with this algorithm
- **Simple**: Episode-level updates, easy to implement
- **Minimal changes**: Same preprocessing, same reward structure
- **Fair comparison**: Isolates spiking dynamics as the only variable

**Key Components**:
1. **Discounted returns**: `G_t = r_t + γ·G_{t+1}` backward through episode
2. **Baseline**: Running average of episode returns for variance reduction
3. **Advantage**: `A_t = G_t - baseline`
4. **Policy loss**: `-Σ log_prob(a_t) · A_t`
5. **Gradient clipping**: `max_norm=0.5` to prevent explosion
6. **Adam optimizer**: Adaptive learning rates

**Alternatives Considered**:
- **Actor-Critic**: More complex, requires value network
- **PPO**: Clipping mechanism, more stable but heavier implementation
- **A3C**: Requires async workers, overkill for this task

### Decision 3: Input Encoding - Constant Current (Not Poisson Spikes)

**Choice**: Encode state features as constant current input over simulation timesteps

**Rationale**:
- **Deterministic**: No random spike generation, reduced variance
- **Simple**: Direct mapping from features to currents
- **Dense signal**: Every timestep has input, not sparse Poisson events
- **Gradient flow**: Differentiable through current computation

**Implementation**:
```text
1. Preprocess state → [grad_strength, rel_angle] (same as MLP)
2. Pass through linear layer → input_current [batch, hidden_dim]
3. For each timestep: feed same input_current to LIF layers
4. Accumulate spikes over time → action logits
```

**Alternatives Considered**:
- **Poisson spikes**: Stochastic, adds noise, current implementation's failure mode
- **Latency coding**: Complex, unclear benefit for this task
- **Burst coding**: More biological but harder to implement

### Decision 4: Network Architecture - Multi-Layer LIF with Recurrence

**Choice**:
- Input: Linear layer (state → hidden current)
- Hidden: 1-2 LIF layers (hidden → hidden)
- Output: Linear layer (total spikes → action logits)

**Rationale**:
- **Temporal dynamics**: LIF layers process information over time
- **Recurrence**: Each LIF layer maintains membrane state across timesteps
- **Depth**: 1-2 hidden layers balances expressiveness and complexity
- **Modularity**: PyTorch modules enable easy modification

**Architecture Flow**:
```text
state [2]
  ↓ Linear
input_current [hidden_dim]
  ↓ LIF (t=0..T)  ← membrane state persists
hidden_spikes [hidden_dim]
  ↓ LIF (t=0..T)  ← membrane state persists
hidden_spikes [hidden_dim]
  ↓ Sum over time
total_spikes [hidden_dim]
  ↓ Linear
action_logits [4]
```

**Parameters**:
- `num_timesteps=100`: Simulation steps per decision (100 needed for temporal integration)
- `hidden_size=256`: Balance capacity and efficiency
- `num_hidden_layers=2`: Sufficient depth for learning

**Alternatives Considered**:
- **Readout at each timestep**: Too computationally expensive
- **Single hidden layer**: May lack capacity
- **3+ layers**: Vanishing gradients, slower convergence

### Decision 5: Configuration Schema - Remove STDP, Add Policy Gradient Params

**Remove** (STDP-specific):
- `simulation_duration, time_step` → replaced by `num_timesteps`
- `max_rate, min_rate` → no Poisson encoding
- `tau_plus, tau_minus, a_plus, a_minus` → no STDP
- `reward_scaling` → handled by baseline

**Add** (Policy gradient):
- `num_timesteps: int = 100` - Simulation duration (100 needed for temporal integration)
- `hidden_size: int = 256` - Hidden layer size
- `num_hidden_layers: int = 2` - Network depth
- `gamma: float = 0.99` - Discount factor
- `baseline_alpha: float = 0.05` - Baseline update rate
- `entropy_beta: float = 0.3` - Exploration bonus
- `entropy_beta_final: float = 0.3` - Final entropy after decay
- `entropy_decay_episodes: int = 50` - Episodes over which to decay entropy
- `surrogate_alpha: float = 1.0` - Gradient smoothness (10.0 caused explosion)
- `lr_decay_rate: float = 0.005` - Learning rate decay per episode
- `update_frequency: int = 10` - Intra-episode update frequency (0 = end of episode only)
- `weight_init: str = "orthogonal"` - Weight initialization method
- `population_coding: bool = true` - Use population coding for inputs
- `neurons_per_feature: int = 8` - Neurons per input feature for population coding

**Keep** (LIF neuron):
- `tau_m: float = 20.0` - Membrane time constant
- `v_threshold: float = 1.0` - Spike threshold
- `v_reset: float = 0.0` - Reset potential
- `learning_rate: float = 0.0001` - Optimizer LR (lower than MLP due to more parameters)

## Risks / Trade-offs

### Risk 1: Slower Convergence Than MLP
**Mitigation**:
- Start with small environment (`spiking_foraging_small`)
- Tune `num_timesteps` (more steps = more computation but richer dynamics)
- Adjust `hidden_dim` and `num_hidden_layers` if underfitting
- Use learning rate scheduling if needed

### Risk 2: Gradient Explosion Through Long Spike Sequences
**Mitigation**:
- Gradient clipping (`max_norm=1.0` combined with value clipping at 1.0)
- Use `surrogate_alpha=1.0` (10.0 causes gradient explosion)
- Layer normalization in output layer
- Monitor gradient norms during training

### Risk 3: Hyperparameter Sensitivity
**Mitigation**:
- Start with MLPBrain's proven hyperparameters
- Systematic grid search: `num_timesteps ∈ {30, 50, 100}`, `hidden_dim ∈ {64, 128, 256}`
- Document sensitivity in benchmarks
- Provide recommended configs for each environment size

### Risk 4: Biological Plausibility vs Performance Tradeoff
**Trade-off**: Surrogate gradients are not biologically plausible (brain doesn't do backprop). However:
- LIF dynamics remain biologically realistic
- Spike-based computation preserved
- This is **standard practice** in neuromorphic ML research
- Priority is demonstrating spiking networks **can** learn, then explore more biological learning rules

## Migration Plan

### Phase 1: Implementation (New Code)
1. Create `_spiking_layers.py` with surrogate gradient classes
2. Rewrite `SpikingBrain` class in `spiking.py`
3. Update `SpikingBrainConfig` schema
4. Update all 4 configuration files

### Phase 2: Validation (Testing)
1. Unit test: Gradient flow through LIFLayer (finite differences)
2. Integration test: 10 episodes, check loss decreases
3. Smoke test: Run all 4 configs for 5 episodes each

### Phase 3: Benchmarking (Evaluation)
1. Run 100 episodes on `spiking_foraging_small`
2. Compare learning curve to MLP baseline
3. If successful, run medium and large environments
4. Document performance, hyperparameter sensitivity

### Phase 4: Documentation (Completion)
1. Update README with new implementation approach
2. Add configuration guide for surrogate gradient parameters
3. Document breaking changes from STDP version
4. Archive OpenSpec change

### Rollback Plan
If surrogate gradients don't work after reasonable tuning:
1. Keep old STDP code in `spiking_stdp_legacy.py` for reference
2. Document findings: why STDP failed, why surrogate gradient failed (if applicable)
3. Consider this a research result (negative result is still valuable)

## Open Questions

1. **Should we support both STDP and surrogate gradient implementations?**
   - **Decision**: No, complete replacement. STDP is fundamentally broken for this task.
   - Keeping both adds maintenance burden
   - Can preserve old code in legacy file for research comparison

2. **What initial learning rate to use?**
   - **Decision**: Start with `lr=0.001` (same as MLP)
   - If convergence issues, try `lr=0.0003` or add scheduling

3. **Should we implement entropy regularization?**
   - **Decision**: Add parameter but make optional (`entropy_beta=0.01`)
   - Can be disabled by setting `entropy_beta=0.0`
   - Useful if agent gets stuck in local minima

4. **How many timesteps per decision?**
   - **Decision**: `num_timesteps=100` as default
   - Trade-off: More steps = richer dynamics but slower
   - Can tune per environment: small=30, medium=50, large=100
