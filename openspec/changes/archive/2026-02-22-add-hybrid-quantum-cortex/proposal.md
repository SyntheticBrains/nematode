## Why

The HybridQuantum brain achieves 96.9% post-convergence on pursuit predators but only ~1% of its parameters are quantum (92 of ~10K). The classical ablation (HybridClassical: 96.3%) proved the QSNN quantum reflex is not the key performance driver — the three-stage curriculum and mode-gated fusion architecture are what matter. We need an architecture with significantly more quantum involvement at the strategic decision-making layer to test whether quantum circuits can provide genuine advantage beyond reflexive behaviour, particularly for multi-sensory environments (thermotaxis, mechanosensation, larger grids).

## What Changes

- Add a new `HybridQuantumCortex` brain architecture that replaces the classical cortex MLP (~5K params) with a QSNN-based cortex (~350-500 quantum params) using grouped QLIF neurons per sensory modality and a shared hidden layer for cross-modality integration
- The QSNN cortex uses surrogate gradient REINFORCE with critic-provided GAE advantages (not PPO, which is incompatible with surrogate gradients — proven across 16 sessions)
- QSNN reflex, classical critic, and mode-gated fusion all remain from HybridQuantum — only the cortex changes
- Quantum fraction increases from ~1% to ~11% of total parameters
- Sensory-modality-grouped QLIF neurons mirror C. elegans neurobiology (AWC/AWA for food chemotaxis, ASH/ADL for nociception, ALM/PLM for mechanosensation)
- Four-stage curriculum: (1) QSNN reflex on foraging, (2) QSNN cortex on multi-objective with frozen reflex, (3) joint fine-tune, (4) optional multi-sensory scaling
- Classical ablation variant (`HybridClassicalCortex`) to test whether the QSNN cortex provides genuine quantum contribution vs an equally-sized classical MLP cortex

## Capabilities

### New Capabilities

- `hybrid-quantum-cortex-brain`: The HybridQuantumCortex brain architecture — QSNN reflex + QSNN cortex (grouped sensory QLIF neurons with shared hidden layer) + classical critic + mode-gated fusion. Includes four-stage curriculum, REINFORCE with critic-provided GAE advantages for the cortex, and weight persistence across stages.

### Modified Capabilities

- `brain-architecture`: Add BrainType registration for `hybridquantumcortex` and its config/factory integration

## Impact

- **New files**: `hybridquantumcortex.py` (brain implementation), tests, 6 YAML configs (per curriculum stage/round)
- **Modified files**: `dtypes.py` (BrainType enum), `__init__.py` (exports), brain factory (registration)
- **Reused infrastructure**: `_qlif_layers.py` (QLIFSurrogateSpike, execute_qlif_layer_differentiable, encode_sensory_spikes — 100% reuse), `modules.py` (sensory module system — 100% reuse), `hybridquantum.py` (~70% reuse as template for rollout buffer, fusion, weight persistence, episode management)
- **No breaking changes**: Existing architectures are unaffected
- **Computational impact**: ~2-3x quantum circuit cost per decision vs HybridQuantum (two QSNN components instead of one)

## Experimental Outcomes

**Status: HALTED** — Architecture implemented and evaluated across 9 rounds (32 sessions, 14,600 episodes). Implementation is complete and functional, but the QSNN cortex under REINFORCE with surrogate gradients cannot match HybridQuantum's performance on the 2-predator environment.

**Key results**:

| Stage | Best Result | Finding |
|-------|-------------|---------|
| Stage 1 (reflex foraging) | 82.5% mean, 95.1% post-conv | QSNN reflex validated |
| Stage 2a (cortex foraging) | 88.8% mean, 95.2% post-conv | Cortex exceeds reflex baseline (+6.3pp) |
| Stage 2b (1 predator) | 96.8% mean, 97.2% post-conv | Zero deaths; zero starvation |
| Stage 2c (2 predators) | 40.9% mean (best 42.8%) | ~40-45% ceiling; halted |
| Stage 3 (joint fine-tune) | 19.3% mean (declining) | Catastrophic forgetting; abandoned |

**Root causes for 2-predator ceiling**: Vanishing gradients after LR decay (norms 0.04-0.07), ineffective critic (EV ~0.10), frozen mode distributions, and insufficient gradient signal from REINFORCE with ~252 quantum parameters.

**Comparison**: HybridQuantum (classical cortex MLP, PPO) achieves 96.9% on the same task. The PPO training method provides 40x more gradient passes per buffer than REINFORCE, which is the primary advantage.

See: [hybridquantumcortex-optimization.md](../../docs/experiments/logbooks/supporting/008/hybridquantumcortex-optimization.md)
