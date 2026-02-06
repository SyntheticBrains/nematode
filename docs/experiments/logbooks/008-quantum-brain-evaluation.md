# 008: Quantum Brain Architecture Evaluation

**Status**: `in_progress`

**Branch**: `feature/add-qrc-brain`

**Date Started**: 2026-02-05

## Objective

Evaluate and benchmark novel quantum brain architectures against classical baselines. This experiment covers:

- **QRCBrain** (Quantum Reservoir Computing): Fixed quantum reservoir + trainable classical readout
- **QSNN** (Quantum Spiking Neural Network): Planned - QLIF neurons with local learning rules
- **QVarCircuitBrain**: Existing - Variational quantum circuit with trainable parameters

Goal: Identify which quantum architectures are viable for nematode navigation tasks.

## Background

Classical baselines are well-established (see Logbook 004, 007):

- **MLPReinforceBrain**: ~70-85% success on foraging
- **MLPPPOBrain**: 84-98% post-convergence across all thermotaxis configs

Quantum approaches face unique challenges:

- Barren plateaus in variational circuits
- Measurement noise
- Limited qubit counts
- Gradient estimation overhead

______________________________________________________________________

## QRCBrain Evaluation

### Architecture

```text
Input (2-4 features) → Fixed Quantum Reservoir → Classical Readout → Actions (4)
                       [H → [Input → Random → CZ] × depth]    [MLP/Linear]
```

Key design choices:

- **Fixed reservoir**: Random rotation angles seeded for reproducibility
- **Data re-uploading**: Input encoded before each reservoir layer
- **REINFORCE learning**: Only readout network trains

### Configuration (Final Tuned)

```yaml
brain:
  name: qrc
  config:
    num_reservoir_qubits: 4      # Reduced from 8 (16-dim vs 256-dim output)
    reservoir_depth: 3
    readout_type: mlp
    readout_hidden: 64
    learning_rate: 0.01          # 10x baseline (weak gradients)
    entropy_coef: 0.005
    shots: 1024
```

### Results Summary

| Task | Runs | Success | Chemotaxis | Status |
|------|------|---------|------------|--------|
| Foraging (2 inputs) | 1600+ | 0% | -0.13 to -0.23 | ❌ Failed |
| Predators (4 inputs) | 25 | 0% | -0.131 | ❌ Failed |

### Key Experiments

| Session | Config Changes | Success | CI | Finding |
|---------|----------------|---------|----|---------|
| 20260204_122807-122818 | Baseline (8 qubits) | 0% | -0.14 | Input encoding sparse |
| 20260204_131441-131456 | Dense encoding + entropy | 0% | -0.10 | Marginal improvement |
| 20260204_135543-135557 | 4 qubits, linear readout | 0% | -0.10 | Simpler, same result |
| 20260204_220604 | LR=0.01, MLP readout | 0% | -0.15 | Learning signal stronger |
| 20260204_222450 | Data re-uploading | 0% | -0.21 | Worsened |
| 20260204_231515 | Multi-sensory (predators) | 0% | -0.13 | Slightly better CI |

### Root Cause Analysis

Debug logging revealed:

1. **Learning IS happening**: Policy updates execute, entropy decreases (1.386 → 1.360)
2. **Gradients are weak**: Norm ~0.02-0.14, need many episodes to converge
3. **Reservoir outputs are non-discriminative**: High entropy (~2.4/2.77), similar for different inputs
4. **Policy converges to wrong behavior**: Negative chemotaxis = moving AWAY from food

### Conclusion

**QRCBrain is not viable for chemotaxis/foraging tasks.**

The fixed random reservoir doesn't create representations that distinguish "toward food" from "away from food". The architecture may suit:

- Time-series prediction (memory effects)
- Tasks with richer input signals
- Pre-trained readout networks

______________________________________________________________________

## QSNN Evaluation

**Status**: Evaluated (0% success, same as QRC)

### Architecture

```text
Sensors → QLIF (sensory) → QLIF (hidden) → QLIF (motor) → Actions
          [RY(θ_membrane + input) → RX(θ_leak) → Measure]
```

Key design choices (based on Brand & Petruccione 2024):

- **QLIF neurons**: Quantum Leaky Integrate-and-Fire with minimal 2-gate circuit
- **Network topology**: 6 sensory → 4 hidden → 4-5 motor neurons
- **Local learning**: 3-factor Hebbian rule (pre × post × reward)
- **Trainable parameters**: Membrane potential (θ_membrane), weight matrices (W_sh, W_hm)
- **Spike encoding**: Continuous inputs → spike probabilities via sigmoid

### QLIF Neuron Circuit

```python
# Minimal circuit per Brand & Petruccione (2024)
|0⟩ → RY(θ_membrane + weighted_input) → RX(θ_leak) → Measure

# θ_membrane: trainable membrane potential parameter
# weighted_input: sum(w_ij * spike_j) for all presynaptic neurons
# θ_leak: leak rate = (1 - membrane_tau) * π
```

### Local Learning Rule (3-Factor Hebbian)

```python
# Eligibility trace accumulation
eligibility += pre_spike × post_spike

# Weight update (end of episode)
Δw = learning_rate × eligibility × total_reward

# Advantages:
# - Local: only uses info available at synapse
# - No global backprop → avoids barren plateaus
# - Reward-modulated → learns from sparse signals
```

### Configuration

```yaml
brain:
  name: qsnn
  config:
    num_sensory_neurons: 6
    num_hidden_neurons: 4
    num_motor_neurons: 4
    membrane_tau: 0.9           # Leak time constant
    threshold: 0.5              # Spike threshold
    refractory_period: 2        # Steps after spike
    use_local_learning: true    # 3-factor Hebbian
    shots: 1024
    gamma: 0.99
    learning_rate: 0.01
    entropy_coef: 0.01
    weight_clip: 5.0
```

### Hypothesis

QSNN should outperform QRC because:

1. **Trainable quantum parameters** (vs fixed reservoir)
2. **Local learning rules** avoid vanishing gradients and barren plateaus
3. **Spike timing** provides richer temporal signal
4. **Biological alignment** with C. elegans neuroscience

### Experiments

| Session | Config | Episodes | Success | CI | Avg Foods | Notes |
|---------|--------|----------|---------|-----|-----------|-------|
| 20260206_122703 | qsnn_foraging_small.yml | 200 | 0% | -0.154 | 1.24 | Baseline run |

### Results Summary

QSNN achieves 0% success rate on foraging after 200 episodes, same as QRC. Key observations:

1. **Negative chemotaxis** (-0.154): Moving away from food, same pathological behavior as QRC
2. **Low food collection**: Only 1.24 foods per episode on average (need 10 to win)
3. **Learning not visible**: No improvement trend over 200 episodes

### Root Cause Analysis

Unlike QRC where the fixed reservoir was the problem, QSNN has trainable parameters but:

1. **Learning rate may be too low**: Local Hebbian learning with lr=0.01 may be too slow
2. **Eligibility traces decay**: Episode-level updates dilute learning signal
3. **Spike encoding may be suboptimal**: Sigmoid on raw features may not preserve gradient info
4. **Network too small**: 6→4→4 neurons may lack capacity for navigation

### Potential Improvements

1. **Higher learning rate** (0.1 or higher) for faster weight updates
2. **Per-step learning** instead of episode-level (reduce eligibility decay)
3. **Different spike encoding** (e.g., rate coding with multiple timesteps)
4. **Larger network** (more hidden neurons)
5. **Hybrid approach**: Add classical value function for reward prediction

### Comparison with Classical Spiking

SpikingReinforceBrain (classical) achieves 73.3% on foraging with surrogate gradients. QSNN's local learning:

- Avoids backprop through quantum circuits (good for barren plateaus)
- But may be too weak a learning signal for this task
- Classical spiking uses dense gradient information that QSNN lacks

### File Locations

- QSNN implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qsnn.py`
- QSNN tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qsnn.py`
- QSNN configs: `configs/examples/qsnn_*.yml`

______________________________________________________________________

## QVarCircuitBrain Comparison

**Status**: Existing baseline

QVarCircuit uses trainable rotation angles in a variational ansatz. Prior results show it achieves ~30-40% success on foraging with gradient-based learning.

| Metric | QRC | QSNN | QVarCircuit |
|--------|-----|------|-------------|
| Success Rate | 0% | 0% | 30-40% |
| Chemotaxis | -0.13 to -0.23 | -0.15 | ~0.1-0.3 |
| Trainable Params | Readout only | Weights + θ | Quantum + Readout |
| Gradient Issue | Weak signal | Weak local learning | Barren plateaus |

______________________________________________________________________

## Analysis

### Quantum Architecture Comparison

| Architecture | Trainable | Gradient Issue | Best Success | Viable? |
|--------------|-----------|----------------|--------------|---------|
| QRC | Readout only | Weak signal, no convergence | 0% | ❌ No |
| QSNN | Local Hebbian | Learning signal too weak | 0% | ❌ No (as implemented) |
| QVarCircuit | Full circuit | Barren plateaus | ~40% | ⚠️ Marginal |

### Key Learnings

1. **Fixed reservoirs don't work**: Random quantum circuits don't preserve input structure
2. **Multi-sensory helps marginally**: 4 inputs give slightly stronger gradients than 2
3. **REINFORCE is too slow**: Sparse rewards + weak gradients = no convergence
4. **Classical baselines are strong**: MLP achieves 70-85% easily

______________________________________________________________________

## Next Steps

- [x] Implement QSNNBrain with QLIF neurons
- [x] Run QSNN benchmark (200 episodes on foraging) - 0% success
- [ ] Tune QSNN hyperparameters (higher LR, larger network)
- [ ] Compare QSNN vs SpikingReinforceBrain (classical spiking)
- [ ] Evaluate QVarCircuit with actor-critic (lower variance)
- [ ] Consider hybrid approaches (quantum reflex + classical planning)

______________________________________________________________________

## Data References

### QRC Sessions

| Config | Sessions | Notes |
|--------|----------|-------|
| Foraging baseline | 20260204_122807-122818 | 4×200 runs, 0% success |
| Dense encoding | 20260204_131441-131456 | 4×200 runs, 0% success |
| Reduced qubits | 20260204_135543-135557 | 4×200 runs, 0% success |
| High LR + MLP | 20260204_220604-222819 | Various fixes, 0% success |
| Multi-sensory | 20260204_231515 | 25 runs with predators, 0% success |

### File Locations

- QRC implementation: `packages/quantum-nematode/quantumnematode/brain/arch/qrc.py`
- QRC configs: `configs/examples/qrc_*.yml`
