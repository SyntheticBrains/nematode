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

**Status**: Planned

### Proposed Architecture

```text
Input → Quantum LIF Neurons → Spike-based Readout → Actions
        [|ψ⟩ = membrane potential]  [STDP-like learning]
```

Design considerations:

- Leaky Integrate-and-Fire (LIF) with quantum membrane state
- Local learning rules (avoid barren plateaus)
- Alignment with biological spiking (SpikingReinforceBrain exists)

### Hypothesis

QSNN may outperform QRC because:

1. Trainable quantum parameters (vs fixed reservoir)
2. Local learning rules avoid vanishing gradients
3. Spike timing provides richer temporal signal
4. Biological alignment with C. elegans neuroscience

### Experiments

*To be conducted*

______________________________________________________________________

## QVarCircuitBrain Comparison

**Status**: Existing baseline

QVarCircuit uses trainable rotation angles in a variational ansatz. Prior results show it achieves ~30-40% success on foraging with gradient-based learning.

| Metric | QRC | QVarCircuit |
|--------|-----|-------------|
| Success Rate | 0% | 30-40% |
| Chemotaxis | -0.13 to -0.23 | ~0.1-0.3 |
| Trainable Params | Readout only | Quantum + Readout |
| Gradient Issue | Weak signal | Barren plateaus |

______________________________________________________________________

## Analysis

### Quantum Architecture Comparison

| Architecture | Trainable | Gradient Issue | Best Success | Viable? |
|--------------|-----------|----------------|--------------|---------|
| QRC | Readout only | Weak signal, no convergence | 0% | ❌ No |
| QVarCircuit | Full circuit | Barren plateaus | ~40% | ⚠️ Marginal |
| QSNN | Local rules | TBD | TBD | ? |

### Key Learnings

1. **Fixed reservoirs don't work**: Random quantum circuits don't preserve input structure
2. **Multi-sensory helps marginally**: 4 inputs give slightly stronger gradients than 2
3. **REINFORCE is too slow**: Sparse rewards + weak gradients = no convergence
4. **Classical baselines are strong**: MLP achieves 70-85% easily

______________________________________________________________________

## Next Steps

- [ ] Implement QSNNBrain with QLIF neurons
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
