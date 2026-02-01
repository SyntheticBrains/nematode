# Change: Add Ablation Toolkit

> **Note (February 2026)**: This change has been moved from Phase 1 to Phase 2 (Architecture Analysis & Standardization). Brain architecture names have been updated to the new naming convention. See [WORKPLAN.md](../../../docs/WORKPLAN.md) for current status.

## Why

Phase 1 introduces multiple sensory modalities and architectural components. To understand which features are critical for performance, we need systematic ablation studies - removing components one at a time and measuring performance degradation.

This toolkit enables:

1. Identifying which sensory modules are most important for each task
2. Understanding differences between quantum and classical architectures
3. Finding minimal sufficient architectures for each behavior
4. Generating interpretable insights for research publications
5. Meeting Phase 2 exit criteria: "Ablation toolkit operational with automated feature importance ranking"

## What Changes

### 1. Ablation Framework

Create a systematic ablation study framework:

- Define ablation configurations (which modules/layers to remove)
- Run baseline (all components) and ablated variants
- Compute performance degradation metrics
- Generate automated reports

### 2. QVarCircuitBrain Ablation

Support module removal for quantum QVarCircuitBrain (formerly ModularBrain):

- Zero out specific module qubits (set rotation angles to 0)
- Remove entanglement layers
- Reduce circuit depth (fewer layers)
- Track which qubits/modules are ablated

### 3. MLPPPOBrain Ablation

Support component removal for classical MLPPPOBrain (formerly PPOBrain):

- Bypass hidden layers (direct input-to-output)
- Remove critic network (policy-only)
- Reduce hidden layer size
- Zero out specific input features

### 4. MLPPPOBrain Sensory Configuration Ablation

Support sensory module configuration variations for MLPPPOBrain:

- **Legacy vs Unified Mode**: Compare 2-feature legacy preprocessing against unified modular features
- **Sensory Module Subsets**: Test individual modules and combinations (e.g., food_chemotaxis only, nociception only, both)
- **Module Combination Study**: Enumerate all meaningful module combinations to find minimal sufficient sets

### 5. Gradient Mode Ablation

Support gradient computation mode comparison:

- **Combined vs Separated Gradients**: Compare chemotaxis (combined) vs food_chemotaxis + nociception (separated)
- **Module Remapping**: Automatically remap modules when switching gradient modes
- Tests whether gradient separation improves predator avoidance behavior

### 6. Feature Normalization Ablation

Support feature scaling mode comparison:

- **Classical vs Quantum Normalization**: Compare [-1, 1] (classical) vs [-π/2, π/2] (quantum) scaling
- Validates that classical normalization is necessary for PPO performance
- Records feature value distribution statistics for analysis

### 7. Feature Importance Analysis

Compute feature importance from ablation results:

- `importance = (baseline_score - ablated_score) / baseline_score`
- Rank modules by importance
- Identify critical vs redundant components
- Cross-architecture comparison

### 8. Automated Reporting

Generate markdown reports:

- Performance comparison tables
- Feature importance rankings
- Architecture-specific insights
- Visualizations (if matplotlib available)

## Impact

**New Specs:**

- `ablation-analysis`: NEW capability for systematic architecture analysis

**Affected Code:**

- `packages/quantum-nematode/quantumnematode/analysis/ablation.py` - NEW: Ablation framework
- `packages/quantum-nematode/quantumnematode/analysis/__init__.py` - NEW: Analysis module
- `scripts/run_ablation.py` - NEW: CLI for ablation studies

**Dependencies:**

- Planned for Phase 2 (Architecture Analysis & Standardization)
- Uses existing brain architectures (no modifications required)
- Benefits from but doesn't require multi-sensory environment

**Breaking Changes:**

- None. This is a new analysis capability.
