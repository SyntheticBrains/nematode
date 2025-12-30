# Design Document: Trajectory Learning and Modular Brain Architecture

## Architecture Overview

This change introduces two major capabilities that work independently but complement each other:

1. **Trajectory Learning**: Episode-level REINFORCE for quantum ModularBrain
2. **Appetitive/Aversive Modules**: Specialized behavioral circuits with separate gradients

## Design Decisions

### 1. Trajectory Learning Architecture

#### Why Episode Buffering for Quantum Brain?

**Decision**: Store episode trajectories and compute gradients at episode end, not per-step.

**Rationale**:

- Parameter-shift rule is **linear in reward**: can use discounted returns instead of immediate rewards
- MLP already uses this pattern successfully (mlp.py lines 421-465)
- Enables temporal credit assignment: "approaching danger at t=95 caused death at t=100"
- No architectural redesign needed - quantum circuits remain stateless

**Trade-offs**:

- **Pro**: Mathematically sound, proven effective in classical REINFORCE
- **Pro**: Aligns quantum and classical learning mechanisms
- **Con**: Delayed learning signal (episode must complete)
- **Con**: Memory overhead for episode buffer (~500 steps × params)

#### Learning Trigger Point

**Decision**: Update parameters at episode end via `post_process_episode()` hook.

**Rationale**:

- Natural episode boundary already exists in agent/runners.py
- Avoids mid-episode parameter updates that could destabilize learning
- Matches MLP's `_perform_policy_update()` pattern

#### Backward Compatibility

**Decision**: Config flag `use_trajectory_learning: bool` (default: False)

**Rationale**:

- Preserves existing single-step behavior for comparison
- Allows A/B testing of trajectory vs immediate learning
- Doesn't break existing benchmarks or configs

**Implementation**:

```python
# In ModularBrain.__init__
if self.config.use_trajectory_learning:
    self.episode_buffer = EpisodeBuffer()
else:
    self.episode_buffer = None  # Single-step mode
```

### 2. Appetitive/Aversive Module Architecture

#### Module Naming Choice

**Decision**: Add `appetitive` and `aversive` as new modules alongside the existing `chemotaxis` module.

**Rationale**:

- **Backward compatibility**: Existing configs using `chemotaxis` continue to work unchanged
- **Biological accuracy**: Both appetitive and aversive are chemotaxis behaviors (attractive vs repulsive)
- **Neuroscience terminology**: Widely used in C. elegans and behavioral literature
- **Distinct use cases**:
  - `chemotaxis`: Uses combined/superposed gradient (food - predator), simpler 2-qubit setup
  - `appetitive`: Uses separated food-only gradient for approach behavior
  - `aversive`: Uses separated predator-only gradient for avoidance behavior

**References**:

- C. elegans appetitive learning: food-seeking, olfactory attraction
- C. elegans aversive learning: pathogen avoidance, toxin repulsion

#### Gradient Separation Strategy

**Decision**: The system SHALL support config-toggleable gradient mode: `unified` (default) or `split`.

**Rationale**:

- **Unified mode**: Current behavior, food+predator gradients superpose
- **Split mode**: Separate `food_gradient` and `predator_gradient` for distinct modules
- Allows testing impact of gradient separation independently
- Backward compatible with existing single-module configs

**Trade-offs**:

- **Pro**: Clean abstraction, easy to toggle
- **Pro**: Minimal environment changes (conditional in get_state())
- **Con**: Requires environment to track gradients separately (moderate refactor)

#### Qubit Allocation

**Decision**: 4-qubit architecture with 2 qubits per module.

**Rationale**:

- **Appetitive module [0, 1]**: Food-seeking circuit
- **Aversive module [2, 3]**: Predator-avoidance circuit
- Maintains symmetry (both modules have equal representation)
- 4 qubits is CPU-feasible, user confirmed strong machine

**Impact**:

- 2× circuit complexity (4 qubits vs 2)
- Recommend shots: 5000 (up from 3000) for stability
- 2× slower quantum execution (acceptable for CPU)

### 3. Feature Extraction Design

#### Appetitive Features (Food-Seeking)

**Inputs**:

- `food_gradient_strength`: Magnitude of attractive gradient
- `food_gradient_direction`: Direction to nearest food

**Encoding** (same as current chemotaxis):

```python
RX: food_gradient_strength scaled to [-π/2, π/2]
RY: relative_angle to food
RZ: 0.0 (unused)
```

#### Aversive Features (Predator Avoidance)

**Biological Principle**: C. elegans detects repulsive chemicals through the same chemosensory mechanism as attractive ones - amphid sensory neurons detecting concentration gradients. The aversive module mirrors appetitive chemotaxis but for repellent chemical signals (predator odors, toxins).

**Inputs** (biologically realistic sensory data only):

- `predator_gradient_strength`: Magnitude of repulsive chemical gradient (negative)
- `predator_gradient_direction`: Direction to repellent source

**Encoding** (mirrors appetitive exactly):

```python
RX: abs(predator_gradient_strength) scaled to [-π/2, π/2]
RY: relative_angle to predator (to know which way to turn away)
RZ: 0.0 (unused, same as appetitive)
```

**Why no "predator proximity" or "danger flag"?**

- ❌ `predator_proximity`: This is environment knowledge, not sensed by the nematode
- ❌ `danger_flag`: This would be external state, not biological sensing
- ✅ The gradient strength already encodes proximity naturally (exponential decay with distance)
- ✅ Stronger gradient = closer threat = more urgent to flee (biologically accurate)

### 4. Configuration Schema Extensions

#### Trajectory Learning Config

```yaml
brain:
  name: modular
  config:
    use_trajectory_learning: true  # NEW
    gamma: 0.99  # Discount factor (NEW)
    baseline_alpha: 0.1  # Baseline update rate (optional, NEW)
    normalize_returns: true  # Return normalization (optional, NEW)
```

#### Modular Architecture Config

```yaml
brain:
  name: modular
  config:
    modules:
      appetitive: [0, 1]  # NEW module (food-only gradient)
      aversive: [2, 3]    # NEW module (predator-only gradient)
      # chemotaxis: [0, 1]  # RETAINED for backward compat (combined gradient)
  qubits: 4  # INCREASED from 2 (when using appetitive+aversive)

environment:
  gradient_mode: split  # NEW: "unified" (default) or "split"
```

**BrainParams Extensions for Split Mode**:
When `gradient_mode: split`, BrainParams will contain:

- `food_gradient_strength` and `food_gradient_direction` (for appetitive module)
- `predator_gradient_strength` and `predator_gradient_direction` (for aversive module)

When `gradient_mode: unified` (default), BrainParams contains:

- `gradient_strength` and `gradient_direction` (superposed, current behavior)

## Implementation Sequence

### Phase 1: Trajectory Learning (Priority)

1. Add `EpisodeBuffer` to ModularBrain
2. Implement `compute_discounted_returns()`
3. Modify `parameter_shift_gradients()` to accept returns
4. Add learning trigger in `post_process_episode()`
5. Add config schema for gamma, use_trajectory_learning
6. Write tests for return computation and trajectory gradients

**Estimated effort**: 1-2 days
**Files**: modular.py (~200 lines), dtypes.py (~10 lines), runners.py (~20 lines)

### Phase 2: Appetitive/Aversive Modules (After evaluation)

1. Add `ModuleName.APPETITIVE` (keep `CHEMOTAXIS` for backward compatibility)
2. Add `ModuleName.AVERSIVE`
3. Implement `appetitive_features()` extraction (uses food-only gradient)
4. Implement `aversive_features()` extraction (uses predator-only gradient)
5. Update environment to track food/predator gradients separately
6. Add `use_separated_gradients` config and conditional gradient computation
7. Write tests for gradient separation and feature extraction

**Estimated effort**: 1-2 days
**Files**: modules.py (~100 lines), agent.py (~30 lines), config_loader.py (~5 lines)

## Testing Strategy

### Trajectory Learning Tests

- `test_discounted_returns_computation()`: Verify backward accumulation
- `test_trajectory_gradient_linearity()`: Verify grad(G_t) = grad(r_t) summation
- `test_episode_buffer_accumulation()`: Verify correct trajectory storage
- `test_single_step_fallback()`: Verify backward compatibility when disabled

### Module Architecture Tests

- `test_appetitive_features_extraction()`: Verify food gradient encoding
- `test_aversive_features_extraction()`: Verify predator gradient encoding
- `test_gradient_mode_unified()`: Verify superposition behavior (current)
- `test_gradient_mode_split()`: Verify separate gradient computation
- `test_module_name_backward_compatibility()`: Verify config migration

### Integration Tests

- `test_trajectory_learning_convergence()`: Benchmark with trajectory learning enabled
- `test_modular_architecture_predator_avoidance()`: Benchmark with 4-qubit split mode
- `test_combined_features()`: Benchmark with both trajectory + modules

## Performance Considerations

### Memory Overhead

- Episode buffer: ~500 steps × (params + actions + rewards)
- Typical: 500 × (12 floats + 1 enum + 1 float) ≈ 7KB per episode
- Negligible compared to quantum circuit memory

### Computational Cost

- **Trajectory learning**: Same gradient computations, just delayed
- **4-qubit circuits**: 2× slower than 2-qubit (linear in qubit count for CPU)
- **Gradient separation**: Minimal (separate vector sums in environment)

### Quantum Shot Requirements

- 2 qubits @ 3000 shots: Current baseline
- 4 qubits @ 5000 shots: Recommended for aversive module stability
- Rationale: Shot noise scales with circuit complexity

## Migration Path

### For Existing Users

1. **No action required**: Default configs use single-step learning, unified gradients, chemotaxis module
2. **Opt-in trajectory learning**: Set `use_trajectory_learning: true` in config
3. **Opt-in modular architecture**: Replace `chemotaxis` with `appetitive` + `aversive`, set `use_separated_gradients: true`

### Config Migration

No migration script needed - existing configs using `chemotaxis` module continue to work unchanged. Users can optionally switch to appetitive+aversive for new experiments.

## Open Questions

### Baseline Subtraction

**Question**: Should trajectory learning include baseline subtraction (variance reduction)?

**Options**:

1. **Yes (like MLP)**: Reduces variance, faster convergence
2. **No (simpler)**: Keep initial implementation minimal

**Recommendation**: Start without baseline, add as optional enhancement later.

### Return Normalization

**Question**: Should we normalize returns like MLP does?

**Options**:

1. **Yes**: `(G_t - mean) / (std + eps)` per episode
2. **No**: Use raw returns

**Recommendation**: Add as optional config flag, default OFF for simplicity.

### Module Entanglement

**Question**: Should appetitive and aversive modules be entangled (CZ gates between them)?

**Current behavior**: Full entanglement between all qubits (lines 303-318 in qmodular.py)

**Options**:

1. **Keep full entanglement**: Modules can influence each other
2. **Separate entanglement**: Modules are independent circuits
3. **Configurable**: Add `inter_module_entanglement: bool`

**Recommendation**: Keep full entanglement (current behavior) unless benchmarks show degradation.

Note: The biological accuracy constraint (whether aversive features should include "predator proximity" or "danger flag") was resolved during design - see Section 3 "Aversive Features" for the rationale. The decision was to use only biologically realistic sensory inputs (gradient strength and direction).
