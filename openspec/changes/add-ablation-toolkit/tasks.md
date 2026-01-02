# Tasks: Add Ablation Toolkit

## Overview

Implementation tasks for systematic ablation analysis framework. This can be developed in parallel with other Phase 1 proposals.

______________________________________________________________________

## 1. Ablation Framework

### 1.1 Create Analysis Module

- [ ] Create `packages/quantum-nematode/quantumnematode/analysis/__init__.py`
- [ ] Create `packages/quantum-nematode/quantumnematode/analysis/ablation.py`

### 1.2 Define Ablation Configuration

- [ ] Create `AblationConfig` dataclass:
  - `base_config_path: str` - Path to baseline brain config
  - `modules_to_ablate: list[ModuleName]` - Modules to test removing
  - `layers_to_ablate: list[int]` - Layer indices to test removing
  - `num_runs_per_condition: int` - Runs for statistical significance (default 50)
  - `output_dir: str` - Where to save results

### 1.3 Define Ablation Study Class

- [ ] Create `AblationStudy` class with methods:
  - `__init__(config: AblationConfig)`
  - `run_baseline() -> ConvergenceMetrics`
  - `run_ablated(ablation: AblationType) -> ConvergenceMetrics`
  - `run_all() -> dict[str, ConvergenceMetrics]`
  - `compute_feature_importance() -> dict[str, float]`
  - `generate_report() -> str`

### 1.4 Define Ablation Types

- [ ] Create `AblationType` enum or union:
  - `ModuleAblation(module: ModuleName)`
  - `LayerAblation(layer_index: int)`
  - `EntanglementAblation()` (remove all entanglement)
  - `HiddenLayerAblation()` (for classical)
  - `CriticAblation()` (for PPO)

**Validation**: Framework can enumerate and run ablation conditions

______________________________________________________________________

## 2. ModularBrain Ablation

### 2.1 Module Zeroing

- [ ] Add `ablated_modules: list[ModuleName]` parameter to ModularBrain
- [ ] When module is ablated, set its rotation angles to 0
- [ ] Ablated modules still exist in circuit but contribute no signal
- [ ] Track which modules are ablated in experiment metadata

### 2.2 Entanglement Ablation

- [ ] Add `ablate_entanglement: bool` parameter
- [ ] When True, skip CZ gate application between qubits
- [ ] Qubits operate independently (no quantum correlation)

### 2.3 Layer Ablation

- [ ] Add `ablated_layers: list[int]` parameter
- [ ] Skip specified layer indices during circuit construction
- [ ] Reduces effective circuit depth

**Validation**: Ablated ModularBrain runs without errors, produces different behavior

______________________________________________________________________

## 3. PPOBrain Ablation

### 3.1 Hidden Layer Bypass

- [ ] Add `bypass_hidden: bool` parameter to PPOBrain
- [ ] When True, use direct linear projection from input to output
- [ ] Tests whether hidden layers are necessary

### 3.2 Critic Ablation

- [ ] Add `ablate_critic: bool` parameter
- [ ] When True, use fixed baseline instead of learned value function
- [ ] Tests value function importance for PPO

### 3.3 Input Feature Masking

- [ ] Add `masked_features: list[str]` parameter
- [ ] Zero out specified feature groups in input vector
- [ ] Enables testing individual sensory modality importance

**Validation**: Ablated PPOBrain runs without errors, produces different behavior

______________________________________________________________________

## 4. PPOBrain Sensory Configuration Ablation

### 4.1 Legacy vs Unified Mode Switching

- [ ] Add `SensoryModeAblation` type:
  - `legacy` - Use 2-feature preprocessing (gradient_strength, rel_angle_norm)
  - `unified` - Use modular feature extraction with sensory_modules
- [ ] Implement mode switching in ablation framework
- [ ] Track input dimension changes in results

### 4.2 Sensory Module Subset Testing

- [ ] Implement `SensoryModuleSubsetAblation(modules: list[ModuleName])`
- [ ] Generate all meaningful module subsets:
  - Single modules: [food_chemotaxis], [nociception], [mechanosensation]
  - Pairs: [food_chemotaxis, nociception], [food_chemotaxis, mechanosensation], etc.
  - Full: [food_chemotaxis, nociception, mechanosensation]
- [ ] Auto-compute input_dim for each subset (3 features per module)

### 4.3 Module Combination Enumeration

- [ ] Create `enumerate_module_combinations(base_modules: list[ModuleName])` helper
- [ ] Filter out empty and single-element subsets for efficiency (optional)
- [ ] Generate ablation study conditions for all combinations

**Validation**: Can compare performance across all sensory module configurations

______________________________________________________________________

## 5. Gradient Mode Ablation

### 5.1 Gradient Mode Switching

- [ ] Add `GradientModeAblation` type:
  - `combined` - Use chemotaxis module with combined gradient
  - `separated` - Use food_chemotaxis + nociception with separated gradients
- [ ] Implement environment config switching (use_separated_gradients flag)
- [ ] Implement module remapping:
  - combined: [chemotaxis, mechanosensation] (6 features)
  - separated: [food_chemotaxis, nociception, mechanosensation] (9 features)

### 5.2 Gradient Mode Comparison

- [ ] Run baseline with separated gradients
- [ ] Run ablated condition with combined gradients
- [ ] Compare predator avoidance metrics specifically

**Validation**: Can switch gradient modes and compare performance

______________________________________________________________________

## 6. Feature Normalization Ablation

### 6.1 Normalization Mode Switching

- [ ] Add `NormalizationAblation` type:
  - `classical` - Scale features to [-1, 1] (normalize_for_classical=True)
  - `quantum` - Scale features to [-π/2, π/2] (normalize_for_classical=False)
- [ ] Implement normalization mode override in PPOBrain preprocess

### 6.2 Normalization Impact Analysis

- [ ] Record feature value distributions for each mode
- [ ] Compute statistics: mean, std, min, max for each feature
- [ ] Include convergence rate comparison
- [ ] Add warning note that quantum normalization is expected to degrade PPO performance

**Validation**: Can compare normalization modes and measure impact

______________________________________________________________________

## 7. Feature Importance Analysis

### 7.1 Importance Calculation

- [ ] Implement `compute_feature_importance()`:

  ```python
  def compute_feature_importance(self) -> dict[str, float]:
      baseline = self.results["baseline"].composite_score
      importance = {}
      for ablation, metrics in self.results.items():
          if ablation != "baseline":
              drop = (baseline - metrics.composite_score) / baseline
              importance[ablation] = max(0, drop)  # Clip negative
      return importance
  ```

### 7.2 Importance Ranking

- [ ] Sort modules/components by importance score
- [ ] Identify critical components (importance > 0.1)
- [ ] Identify redundant components (importance < 0.01)

### 7.3 Cross-Architecture Comparison

- [ ] Compare module importance between quantum and classical
- [ ] Identify architecture-specific critical components
- [ ] Flag modules that are critical for one but not the other

**Validation**: Importance scores computed correctly for test cases

______________________________________________________________________

## 8. Automated Reporting

### 8.1 Markdown Report Generation

- [ ] Implement `generate_report() -> str`:
  - Summary table of all ablation conditions
  - Performance metrics for each condition
  - Feature importance ranking
  - Key findings bullet points

### 8.2 Report Template

```markdown
# Ablation Study Report

## Configuration
- Base config: {config_path}
- Runs per condition: {num_runs}
- Date: {timestamp}

## Results Summary

| Condition | Score | Success Rate | Importance |
|-----------|-------|--------------|------------|
| Baseline  | 0.82  | 95%          | -          |
| -chemotaxis | 0.45 | 60%         | 0.45       |
| -thermotaxis | 0.78 | 92%        | 0.05       |

## Feature Importance Ranking

1. chemotaxis (0.45) - CRITICAL
2. nociception (0.12) - Important
3. thermotaxis (0.05) - Minor
4. mechanosensation (0.01) - Redundant

## Key Findings

- Chemotaxis is the most critical module for foraging tasks
- Removing entanglement reduces performance by 15%
- Classical and quantum architectures show similar module importance
```

### 8.3 Optional Visualization

- [ ] If matplotlib available, generate bar charts of importance
- [ ] Performance degradation curves
- [ ] Architecture comparison plots

**Validation**: Report generates correctly with all sections

______________________________________________________________________

## 9. CLI Integration

### 9.1 Ablation CLI Script

- [ ] Create `scripts/run_ablation.py`
- [ ] Accept config path and ablation parameters
- [ ] Run ablation study and generate report
- [ ] Save results to specified output directory

### 9.2 CLI Arguments

```bash
python scripts/run_ablation.py \
  --config configs/examples/modular_foraging_small.yml \
  --ablate-modules chemotaxis thermotaxis nociception \
  --ablate-entanglement \
  --runs 50 \
  --output ablation_results/
```

### 9.3 Integration with Existing CLI

- [ ] Add `--ablation` mode to run_simulation.py (optional)
- [ ] Or keep as separate script for clarity

**Validation**: CLI runs ablation study end-to-end

______________________________________________________________________

## 10. Documentation

### 10.1 Usage Documentation

- [ ] Document ablation framework usage
- [ ] Provide example ablation configurations
- [ ] Explain importance calculation methodology

### 10.2 Research Documentation

- [ ] Document how to interpret ablation results
- [ ] Guidance for publication-ready ablation studies
- [ ] Recommendations for statistical significance

______________________________________________________________________

## Dependencies

```text
1. Ablation Framework (core classes)
         │
         ├──► 2. ModularBrain Ablation
         │
         ├──► 3. PPOBrain Ablation ──────────────────┐
         │                                           │
         │    ┌──────────────────────────────────────┤
         │    │                                      │
         │    ├──► 4. PPOBrain Sensory Config        │
         │    │    (legacy vs unified, module subsets)
         │    │                                      │
         │    ├──► 5. Gradient Mode Ablation         │
         │    │    (combined vs separated)           │
         │    │                                      │
         │    └──► 6. Normalization Ablation         │
         │         (classical vs quantum scaling)    │
         │                                           │
         └──► 7. Feature Importance ◄────────────────┘
                    │
                    v
              8. Reporting
                    │
                    v
              9. CLI + 10. Docs
```

Work streams 2-6 can proceed in parallel after framework is complete.
Work streams 4, 5, 6 depend on PPOBrain ablation infrastructure.

______________________________________________________________________

## Success Criteria

- [ ] Can ablate individual modules from ModularBrain
- [ ] Can ablate hidden layers from PPOBrain
- [ ] Can switch between legacy and unified sensory modes for PPO
- [ ] Can switch between combined and separated gradient modes
- [ ] Can switch between classical and quantum normalization
- [ ] Can test all sensory module combinations systematically
- [ ] Feature importance scores correctly identify critical modules
- [ ] Automated reports are readable and informative
- [ ] CLI allows running ablation studies without code changes
- [ ] Results are reproducible with seed control
