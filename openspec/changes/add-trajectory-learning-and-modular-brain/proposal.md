# Change Proposal: add-trajectory-learning-and-modular-brain

## Summary
Enable quantum ModularBrain to learn from complete episode trajectories with temporal credit assignment, and introduce appetitive/aversive modular architecture for specialized behavioral learning.

## Why
Quantum ModularBrain currently achieves 83% success in non-predator foraging but only 26.5% in predator environments, compared to classical MLP's 85% success. The performance gap is caused by two fundamental limitations:

1. **No temporal credit assignment**: Single-step learning prevents the brain from connecting "died to predator at step 100" to "approached danger zone at steps 95-99"
2. **Gradient cancellation**: Superposing food (attractive) and predator (repulsive) gradients causes them to cancel when both are nearby, confusing the navigation signal

These changes bring quantum ModularBrain to architectural parity with classical MLP (which already has trajectory learning) and enable specialized behavioral circuits that mirror C. elegans biology.

## Motivation
Current quantum ModularBrain uses single-step immediate reward learning, preventing temporal credit assignment critical for complex foraging tasks with predators. The brain cannot connect "died to predator at step 100" to "approached danger zone at steps 95-99". Additionally, using a single chemotaxis module for both food-seeking and predator-avoidance causes gradient cancellation when food and predators are nearby.

### Current Performance Gap
- **Quantum ModularBrain (predator environments)**: 26.5% success rate
- **Classical MLP**: 85% success rate
- **Quantum ModularBrain (non-predator)**: 83% success rate (after recent optimizations)

### Root Causes
1. **No temporal credit assignment**: Immediate rewards only, no discounted returns
2. **Gradient superposition**: Food (attractive) and predator (repulsive) gradients cancel
3. **Architectural mismatch**: MLP has trajectory learning, quantum brain doesn't

## Proposed Solution

### Part 1: Trajectory Learning for ModularBrain
Add episode buffering and discounted return computation to quantum ModularBrain, bringing it to feature parity with classical MLP's REINFORCE implementation.

**Key capabilities**:
- Episode buffer for params, actions, and rewards
- Discounted return computation (backward through time)
- Trajectory-aware parameter-shift gradients
- Config-toggleable (backward compatible)

**Mathematical validity**: Parameter-shift gradients are linear in reward:
- Single-step: `grad = 0.5 * (P_+ - P_-) * r`
- Multi-step: `grad = 0.5 * (P_+ - P_-) * G_t` where `G_t = sum(gamma^k * r_{t+k})`

### Part 2: Appetitive/Aversive Modular Architecture
Rename existing `chemotaxis` module to `appetitive` (approach behavior) and add `aversive` module (avoidance behavior) with dedicated qubits and separate gradient processing.

**Key capabilities**:
- Biologically-inspired module naming (appetitive/aversive)
- Separate gradient encoding (split vs unified mode)
- 4-qubit architecture (2 qubits per module)
- Config-toggleable gradient separation
- **Biologically accurate sensing**: Both modules use only chemical gradient information (strength + direction)

**Biological justification**:
- C. elegans exhibits both appetitive chemotaxis (toward food odors like diacetyl) and aversive chemotaxis (away from repellents like octanol, CO2, pathogen signals)
- Both behaviors use the same sensory mechanism: amphid chemosensory neurons detecting concentration gradients
- The aversive module mirrors appetitive exactly, just for repulsive rather than attractive chemicals
- No "proximity" or "danger flags" - only gradient strength (which naturally encodes proximity via exponential decay)

## Impact
- **Expected improvement**: +30-50% success rate from trajectory learning alone
- **Combined improvement**: 60-80% success rate target with both features
- **Backward compatibility**: Config flags enable/disable both features independently

## Dependencies
- Builds on recent ModularBrain optimizations (norm clipping, adaptive momentum)
- No breaking changes to existing brain types (MLP, QModularBrain, SpikingBrain)

## Out of Scope
- Classical MLP modifications (already has trajectory learning)
- Changes to reward penalties (user testing separately)
- QPU-specific optimizations (CPU-focused development)
