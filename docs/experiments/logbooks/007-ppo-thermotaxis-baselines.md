# 007: PPO Thermotaxis Baseline Validation

**Status**: `active`

**Branch**: `feature/optimize-thermotaxis`

**Date Started**: 2026-01-21

**Date Completed**: (pending)

## Objective

Validate PPO baselines across all thermotaxis configurations to establish reliable benchmarks for future quantum agent comparisons. This experiment covers:

- 3 environment sizes: Small (20×20), Medium (50×50), Large (100×100)
- 3 task variants: Foraging-only, Pursuit Predators, Stationary Predators
- Total: 9 configurations × 4 sessions each = 36 benchmark runs

## Background

Thermotaxis adds temperature sensing (AFD neurons) to the agent's sensory repertoire, creating a multi-objective learning challenge:

1. **Foraging**: Collect target foods while navigating temperature zones
2. **Pursuit Predators**: Balance food collection, predator evasion, and temperature safety
3. **Stationary Predators**: Navigate around fixed toxic zones while managing temperature

Prior tuning work (see Appendix A) established optimal reward shaping:

- `comfort_reward: 0.0` - Prevents freeze behavior
- `discomfort_penalty: -0.05` - Mild signal for tolerable zones
- `danger_penalty: -0.5` - Strong avoidance signal
- `reward_discomfort_food: 0.3` - Brave foraging bonus for collecting food in risky zones
- `safe_zone_food_bias: 0.8` - Biases food spawning toward safe temperature zones

## Hypothesis

With the optimized reward shaping applied consistently across all environment sizes:

1. **Foraging configs** should achieve ≥80% post-convergence success
2. **Pursuit predator configs** should achieve ≥75% post-convergence success
3. **Stationary predator configs** should achieve ≥70% post-convergence success
4. All configs should converge within their allocated episode counts

## Method

### Episode Counts (Based on Convergence Findings)

| Environment | Foraging | Pursuit | Stationary |
|-------------|----------|---------|------------|
| Small (20×20) | 500 | 500 | 500 |
| Medium (50×50) | 500 | 500 | 500 |
| Large (100×100) | 500 | 500 | 700 |

### Configuration Files

| Config | File |
|--------|------|
| Small Foraging | `ppo_thermotaxis_foraging_small.yml` |
| Small Pursuit | `ppo_thermotaxis_pursuit_predators_small.yml` |
| Small Stationary | `ppo_thermotaxis_stationary_predators_small.yml` |
| Medium Foraging | `ppo_thermotaxis_foraging_medium.yml` |
| Medium Pursuit | `ppo_thermotaxis_pursuit_predators_medium.yml` |
| Medium Stationary | `ppo_thermotaxis_stationary_predators_medium.yml` |
| Large Foraging | `ppo_thermotaxis_foraging_large.yml` |
| Large Pursuit | `ppo_thermotaxis_pursuit_predators_large.yml` |
| Large Stationary | `ppo_thermotaxis_stationary_predators_large.yml` |

### Key Configuration Parameters

All configs use consistent reward shaping:

```yaml
thermotaxis:
  comfort_reward: 0.0        # Prevents freeze behavior
  discomfort_penalty: -0.05  # Mild - discomfort is tolerable
  danger_penalty: -0.5       # Significant - clear avoid signal
  reward_discomfort_food: 0.3  # Brave foraging bonus

foraging:
  safe_zone_food_bias: 0.8   # Biases food toward safe zones
```

Environment-specific parameters scale with grid size:

| Parameter | Small (20×20) | Medium (50×50) | Large (100×100) |
|-----------|---------------|----------------|-----------------|
| `gradient_strength` | 1.5°C/cell | 0.6°C/cell | 0.15°C/cell + spots |
| `danger_hp_damage` | 2.0 | 1.5 | 0.5 |
| `lethal_hp_damage` | 10.0 | 8.0 | 6.0 |
| `foods_on_grid` | 6 | 15 | 40-45 |
| `target_foods` | 10 | 20 | 20-25 |

### Sensory Modules

| Task Type | Modules |
|-----------|---------|
| Foraging | food_chemotaxis, thermotaxis |
| Pursuit/Stationary | food_chemotaxis, nociception, mechanosensation, thermotaxis |

______________________________________________________________________

## Results

### Summary Table

| Config | Sessions | Avg Success | Post-Conv | HP Deaths | Converged | Status |
|--------|----------|-------------|-----------|-----------|-----------|--------|
| **Small Foraging** | - | - | - | - | - | Pending |
| **Small Pursuit** | - | - | - | - | - | Pending |
| **Small Stationary** | - | - | - | - | - | Pending |
| **Medium Foraging** | - | - | - | - | - | Pending |
| **Medium Pursuit** | - | - | - | - | - | Pending |
| **Medium Stationary** | - | - | - | - | - | Pending |
| **Large Foraging** | - | - | - | - | - | Pending |
| **Large Pursuit** | - | - | - | - | - | Pending |
| **Large Stationary** | - | - | - | - | - | Pending |

### Detailed Results by Configuration

#### Small Environment (20×20)

##### Small Foraging

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Avg Foods | Converged |
|---------|---------|-----------|-----------|-----------|-----------|
| - | - | - | - | - | - |

##### Small Pursuit Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

##### Small Stationary Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

#### Medium Environment (50×50)

##### Medium Foraging

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Avg Foods | Converged |
|---------|---------|-----------|-----------|-----------|-----------|
| - | - | - | - | - | - |

##### Medium Pursuit Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

##### Medium Stationary Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

#### Large Environment (100×100)

##### Large Foraging

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Avg Foods | Converged |
|---------|---------|-----------|-----------|-----------|-----------|
| - | - | - | - | - | - |

##### Large Pursuit Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

##### Large Stationary Predators

**Sessions**: (pending)

| Session | Success | Post-Conv | HP Deaths | Evasion Rate | Converged |
|---------|---------|-----------|-----------|--------------|-----------|
| - | - | - | - | - | - |

______________________________________________________________________

## Analysis

(To be completed after experiments)

### Learning Trajectory Comparison

### Environment Size Scaling

### Predator Type Comparison

______________________________________________________________________

## Conclusions

(To be completed after experiments)

______________________________________________________________________

## Next Steps

- [ ] Run all 36 sessions
- [ ] Analyze results and update tables
- [ ] Compare with hypotheses
- [ ] Document any config adjustments needed
- [ ] Prepare for quantum agent comparison experiments

______________________________________________________________________

## Data References

- Session IDs: (to be added)
- Config files: `configs/examples/ppo_thermotaxis_*.yml`
- Artifacts: `artifacts/experiments/007-ppo-thermotaxis-baselines/`

______________________________________________________________________

## Appendix A: Configuration Evolution History

This appendix summarizes the key iterations and learnings that led to the current baseline configurations.

### A.1 Small/Medium Environment Tuning (Experiments 001-006 Era)

Early thermotaxis experiments on small (20×20) and medium (50×50) grids established foundational learnings:

#### Foraging Baseline

- **Finding**: Simple foraging with thermotaxis converges quickly (~34-53 runs)
- **Key insight**: Temperature comfort score of ~50% is acceptable given food placement constraints
- **Result**: 89-92% success rate

#### Pursuit Predators Evolution

| Round | Config Change | Success | Post-Conv | Issue/Fix |
|-------|---------------|---------|-----------|-----------|
| R1 | dmg=20, radius=6 | 16-26% | ~50% | Too harsh - predators kill too fast |
| R2 | dmg=15, radius=5 | 31-43% | 80% | Better but not converging |
| **R3** | **dmg=12, radius=5** | **24-55%** | **87.9%** | **✓ All sessions converged** |

**Key learning**: Predator damage of 12 (9 hits to kill) provides optimal challenge.

#### Stationary Predators Evolution

| Round | Config Change | Success | Post-Conv | Issue/Fix |
|-------|---------------|---------|-----------|-----------|
| Baseline | count=3, dmg=15 | 0-7% | 0% | Too many toxic zones |
| Adjusted | count=2, dmg=8 | 11-35% | 60% | Learnable but gradient issue |
| **decay=4** | sharper gradient | **32-47%** | **89.5%** | **✓ Clear toxic zone detection** |

**Key learning**: Sharper gradient decay (4.0 vs 10.0) improves toxic zone detection.

### A.2 Large Environment Tuning (100×100 Grid)

Large environment required extensive iteration due to scattered temperature zones (hot/cold spots) vs simple gradients.

#### The Freeze Behavior Problem

**Observation**: With `comfort_reward: 0.05`, agents learned to stay in comfort zones rather than forage.

**Solution**: Set `comfort_reward: 0.0` - no reward for staying still, only penalties for bad zones.

#### Iteration History Summary

| Iteration | Key Change | Success | Issue |
|-----------|------------|---------|-------|
| 1-4 | Parameter exploration | 0-15% | Lethal zones too harsh |
| 5 | Reduced HP damage | 11% | Agents ignore temperature |
| 6 | Increased temp signals | 0% | Too harsh - can't forage |
| 7-10 | Balanced signals | ~15% | Still too harsh |
| 11-13 | Safe zone food bias | 29-38% | Improving but variance |
| 14-16 | Reward differentiation | 45-66% | Close but not stable |
| 17 | comfort_reward: 0.0 | 72% | Freeze behavior fixed |
| **18** | **+ brave foraging bonus** | **83.3%** | **✓ Baseline achieved** |

#### Final Reward Shaping (Iteration 18)

```yaml
thermotaxis:
  comfort_reward: 0.0        # CRITICAL: Prevents freeze behavior
  discomfort_penalty: -0.05  # Mild - allows passage
  danger_penalty: -0.5       # Significant avoidance
  reward_discomfort_food: 0.3  # Brave foraging bonus
```

**Rationale**:

- Zero comfort reward forces agents to seek food, not comfort
- Differentiated penalties create clear zone hierarchy
- Brave foraging bonus (+0.3) rewards collecting food in risky zones
- Combined with `safe_zone_food_bias: 0.8`, creates optimal challenge

### A.3 Large Predator Configuration Validation

Building on foraging baseline, predator configs were validated with extended training:

#### Pursuit Predators (4 predators, speed=0.5)

| Training | Sessions | Success | Post-Conv | Evasion | Status |
|----------|----------|---------|-----------|---------|--------|
| 200 eps | 4 | 41.2% | 64.0% | 80.9% | Learning |
| **500 eps** | **4** | **73.0%** | **98.8%** | **90.2%** | **✓ Converged** |

**Convergence point**: ~350 runs

**Key insight**: 84 percentage point improvement (15% → 99%) - largest learning gain observed

#### Stationary Predators (5 predators, speed=0)

| Training | Sessions | Success | Post-Conv | Evasion | Status |
|----------|----------|---------|-----------|---------|--------|
| 500 eps | 1 | 64.8% | 91.0% | 94.3% | Learning |
| **700 eps** | **4** | **76.3%** | **92.0%** | **96.4%** | **✓ Converged** |

**Convergence point**: ~350 runs

**Key insight**: Higher evasion rate (96.4%) but lower success (92%) than pursuit due to fixed blocking positions

### A.4 Key Learnings Applied to All Configs

1. **Reward shaping is critical**: Same parameters work across all environment sizes
2. **comfort_reward must be 0**: Any positive value causes freeze behavior
3. **Brave foraging bonus helps**: Encourages risk-taking for food collection
4. **Safe zone food bias**: Reduces unfair spawns in dangerous areas
5. **Gradient decay matters**: Sharper decay (4.0-6.5) for better zone detection
6. **Episode counts vary**: Large predator configs need 500-700 episodes

### A.5 Configuration Changes Applied (2026-01-21)

Small and medium configs were updated to apply large environment learnings:

**Before (small/medium)**:

```yaml
comfort_reward: 0.05
discomfort_penalty: -0.1
danger_penalty: -0.3
# No reward_discomfort_food
# No safe_zone_food_bias (foraging)
```

**After (all sizes)**:

```yaml
comfort_reward: 0.0          # Prevents freeze
discomfort_penalty: -0.05    # Mild
danger_penalty: -0.5         # Significant
reward_discomfort_food: 0.3  # Brave bonus
safe_zone_food_bias: 0.8     # (foraging configs)
```

These changes ensure consistent methodology across the entire thermotaxis configuration suite.
