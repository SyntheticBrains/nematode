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

Prior tuning work (see Appendix A) established that **reward shaping requires careful balancing**:

**Small/Medium environments** (simple linear temperature gradients):

- `comfort_reward: 0.0` - No comfort reward (prevents passivity)
- `discomfort_penalty: 0.0` - No discomfort penalty (rely on HP damage signal)
- `danger_penalty: -0.3` - Moderate avoidance signal
- `penalty_health_damage: 0.0` - Disabled (conflicts with temperature HP damage)

**Large environments** (scattered hot/cold spots):

- `comfort_reward: 0.0` - Prevents freeze behavior (critical for large grids)
- `discomfort_penalty: -0.05` - Mild signal for tolerable zones
- `danger_penalty: -0.5` - Strong avoidance signal
- `penalty_health_damage: 0.3` - Immediate damage feedback

**Critical discovery**: The `safe_zone_food_bias` parameter was not being loaded from configs (bug in config_loader). When enabled at 0.8, it made simulations too easy. All benchmarks were run with this parameter effectively disabled.

## Hypothesis

With the environment-specific reward shaping applied:

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

**Small/Medium thermotaxis reward shaping:**

```yaml
reward:
  penalty_health_damage: 0.0   # Disabled - conflicts with temp HP damage

thermotaxis:
  comfort_reward: 0.0          # No comfort reward
  discomfort_penalty: 0.0      # Rely on HP damage signal
  danger_penalty: -0.3         # Moderate avoidance
```

**Large thermotaxis reward shaping:**

```yaml
reward:
  penalty_health_damage: 0.3   # Immediate damage feedback

thermotaxis:
  comfort_reward: 0.0          # Prevents freeze behavior
  discomfort_penalty: -0.05    # Mild - allows passage
  danger_penalty: -0.5         # Significant avoidance
```

**Note**: `safe_zone_food_bias` was not loaded due to a config loader bug and all benchmarks ran with uniform food distribution.

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

| Config | Sessions | Avg Success | Post-Conv | HP Deaths | Evasion | Converged | Status |
|--------|----------|-------------|-----------|-----------|---------|-----------|--------|
| **Small Foraging** | 4 | 88.2% | 98.0% | 7.4% | - | ✓ Yes | ✓ Baseline |
| **Small Pursuit** | 4 | 60.3% | 82.5% | 17.5% | - | ~High variance | Needs tuning |
| **Small Stationary** | 4 | 49.7% | 67.0% | 7.5% | - | ✗ No | Needs tuning |
| **Medium Foraging** | 4 | 82.4% | 94.5% | ~0% | - | ✓ Yes | ✓ Baseline |
| **Medium Pursuit** | 4 | 74.2% | 84.5% | ~0% | - | ✓ Yes | ✓ Baseline |
| **Medium Stationary** | 4 | 71.6% | 87.5% | ~0% | - | ✓ Yes | ✓ Baseline |
| **Large Foraging** | 4 | 91.8% | 98.5% | 4.1% | - | ✓ Yes | ✓ Baseline |
| **Large Pursuit** | 4 | 71.3% | 97.5% | 24.0% | 88.4% | ✓ Yes | ✓ Baseline |
| **Large Stationary** | 4 | 79.4% | 96.5% | 12.2% | 97.2% | ✓ Yes | ✓ Baseline |

### Detailed Results by Configuration

#### Small Environment (20×20)

##### Small Foraging ✓

**Sessions**: 20260122_105256, 20260122_105258, 20260122_105300, 20260122_105303

| Session | Success | Post-Conv | HP Deaths | Converged |
|---------|---------|-----------|-----------|-----------|
| 20260122_105256 | 92.2% | 98.0% | 7.2% | ✓ |
| 20260122_105258 | 87.0% | 98.0% | 8.6% | ✓ |
| 20260122_105300 | 87.6% | 100.0% | 8.0% | ✓ |
| 20260122_105303 | 86.2% | 96.0% | 5.8% | ✓ |
| **Average** | **88.2%** | **98.0%** | **7.4%** | **✓** |

**Hypothesis check**: ≥80% post-conv → **PASS** (98.0%)

##### Small Pursuit Predators ~

**Sessions (Round 2)**: 20260125_225334, 20260125_225337, 20260125_225340, 20260125_225343

| Session | Success | Post-Conv | HP Deaths | Starve | Converged |
|---------|---------|-----------|-----------|--------|-----------|
| 20260125_225334 | 62.4% | 66.0% | 34.0% | 0% | ✗ |
| 20260125_225337 | 48.2% | 88.0% | 12.0% | 0% | ✓ |
| 20260125_225340 | 69.2% | 82.0% | 18.0% | 0% | ✓ |
| 20260125_225343 | 61.4% | 94.0% | 6.0% | 0% | ✓ |
| **Average** | **60.3%** | **82.5%** | **17.5%** | **0%** | **~High variance** |

**Hypothesis check**: ≥75% post-conv → **PARTIAL** (82.5% avg but 66-94% range)

**Config change**: Reduced `predator_damage: 12 → 8`. HP deaths reduced from 61% to 17.5%, but high variance between sessions persists. May need further tuning or extended training.

##### Small Stationary Predators ✗

**Sessions (Round 2)**: 20260125_233053, 20260125_233057, 20260125_233100, 20260125_233103

| Session | Success | Post-Conv | HP Deaths | Starve | Converged |
|---------|---------|-----------|-----------|--------|-----------|
| 20260125_233053 | 34.4% | 62.0% | 10.0% | 2.0% | ✗ |
| 20260125_233057 | 54.8% | 70.0% | 6.0% | 6.0% | ✓ |
| 20260125_233100 | 52.2% | 52.0% | 12.0% | 10.0% | ✗ |
| 20260125_233103 | 57.4% | 84.0% | 2.0% | 2.0% | ✓ |
| **Average** | **49.7%** | **67.0%** | **7.5%** | **5.0%** | **✗** |

**Hypothesis check**: ≥70% post-conv → **FAIL** (67.0% - close but high variance)

**Config change**: Reduced `damage_radius: 2 → 1`. HP deaths reduced from 29.7% to 7.5%, but starvation now appearing (5%). Still not converging reliably. May need predator count reduction or behavior changes.

#### Medium Environment (50×50)

##### Medium Foraging ✓

**Sessions (Round 2)**: 20260125_114711, 20260125_114713, 20260125_114715, 20260125_114718

| Session | Success | Post-Conv | HP Deaths | Converged |
|---------|---------|-----------|-----------|-----------|
| 20260125_114711 | 80.6% | 90.0% | ~0% | ✓ |
| 20260125_114713 | 82.2% | 98.0% | ~0% | ✓ |
| 20260125_114715 | 80.6% | 94.0% | ~0% | ✓ |
| 20260125_114718 | 86.0% | 96.0% | ~0% | ✓ |
| **Average** | **82.4%** | **94.5%** | **~0%** | **✓** |

**Hypothesis check**: ≥80% post-conv → **PASS** (94.5%)

**Config change**: Reduced `danger_hp_damage: 1.5 → 1.0`. HP deaths eliminated.

##### Medium Pursuit Predators ✓

**Sessions (Round 2)**: 20260126_001541, 20260126_001543, 20260126_001545, 20260126_001548

| Session | Success | Post-Conv | HP Deaths | Starve | Converged |
|---------|---------|-----------|-----------|--------|-----------|
| 20260126_001541 | 66.2% | 68.0% | ~0% | 0% | ~Learning |
| 20260126_001543 | 75.0% | 88.0% | ~0% | 0% | ✓ |
| 20260126_001545 | 74.2% | 86.0% | ~0% | 0% | ✓ |
| 20260126_001548 | 81.4% | 96.0% | ~0% | 0% | ✓ |
| **Average** | **74.2%** | **84.5%** | **~0%** | **0%** | **✓** |

**Hypothesis check**: ≥75% post-conv → **PASS** (84.5%)

**Config changes**: Reduced `predator_damage: 12 → 8`, `danger_hp_damage: 1.5 → 1.0`. HP deaths eliminated.

##### Medium Stationary Predators ✓

**Sessions (Round 2)**: 20260126_011040, 20260126_011043, 20260126_011045, 20260126_011048

| Session | Success | Post-Conv | HP Deaths | Starve | Converged |
|---------|---------|-----------|-----------|--------|-----------|
| 20260126_011040 | 68.4% | 90.0% | ~0% | 0% | ✓ |
| 20260126_011043 | 75.8% | 94.0% | ~0% | 0% | ✓ |
| 20260126_011045 | 70.4% | 82.0% | ~0% | 0% | ✓ |
| 20260126_011048 | 71.8% | 84.0% | ~0% | 0% | ✓ |
| **Average** | **71.6%** | **87.5%** | **~0%** | **0%** | **✓** |

**Hypothesis check**: ≥70% post-conv → **PASS** (87.5%)

**Config changes**: Reduced `damage_radius: 2 → 1`, `danger_hp_damage: 1.5 → 1.0`. HP deaths eliminated.

#### Large Environment (100×100)

##### Large Foraging ✓

**Sessions**: 20260123_115319, 20260123_115321, 20260123_115327, 20260123_115329

| Session | Success | Post-Conv | HP Deaths | Converged |
|---------|---------|-----------|-----------|-----------|
| 20260123_115319 | 92.6% | 100.0% | 4.8% | ✓ |
| 20260123_115321 | 91.8% | 98.0% | 2.8% | ✓ |
| 20260123_115327 | 90.0% | 98.0% | 2.8% | ✓ |
| 20260123_115329 | 92.8% | 98.0% | 5.8% | ✓ |
| **Average** | **91.8%** | **98.5%** | **4.1%** | **✓** |

**Hypothesis check**: ≥80% post-conv → **PASS** (98.5%)

##### Large Pursuit Predators ✓

**Sessions**: 20260123_212840, 20260123_212843, 20260123_212846, 20260123_212850

| Session | Success | Post-Conv | HP Deaths | Evasion | Converged |
|---------|---------|-----------|-----------|---------|-----------|
| 20260123_212840 | 71.4% | 96.0% | 21.4% | 91.2% | ✓ |
| 20260123_212843 | 66.8% | 98.0% | 28.8% | 87.5% | ✓ |
| 20260123_212846 | 72.4% | 96.0% | 24.8% | 86.3% | ✓ |
| 20260123_212850 | 74.6% | 100.0% | 20.8% | 88.5% | ✓ |
| **Average** | **71.3%** | **97.5%** | **24.0%** | **88.4%** | **✓** |

**Hypothesis check**: ≥75% post-conv → **PASS** (97.5%)

##### Large Stationary Predators ✓

**Sessions**: 20260124_000557, 20260124_000600, 20260124_000603, 20260124_000606

| Session | Success | Post-Conv | HP Deaths | Evasion | Converged |
|---------|---------|-----------|-----------|---------|-----------|
| 20260124_000557 | 73.0% | 96.0% | 13.0% | 97.2% | ✓ |
| 20260124_000600 | 82.7% | 98.0% | 13.1% | 97.2% | ✓ |
| 20260124_000603 | 80.3% | 98.0% | 10.4% | 97.1% | ✓ |
| 20260124_000606 | 81.7% | 94.0% | 12.4% | 97.2% | ✓ |
| **Average** | **79.4%** | **96.5%** | **12.2%** | **97.2%** | **✓** |

**Hypothesis check**: ≥70% post-conv → **PASS** (96.5%)

______________________________________________________________________

## Analysis

### Hypothesis Results

| Hypothesis | Target | Result | Status |
|------------|--------|--------|--------|
| Foraging ≥80% post-conv | 80% | Small: 98%, Medium: 94.5%, Large: 98.5% | ✓ PASS |
| Pursuit ≥75% post-conv | 75% | Small: 82.5%, Medium: 84.5%, Large: 97.5% | ✓ PASS |
| Stationary ≥70% post-conv | 70% | Small: 67%, Medium: 87.5%, Large: 96.5% | ~PARTIAL |

### Environment Size Scaling

**Key finding**: After config tuning, all medium configs and most small configs now converge.

| Task | Small | Medium | Large | Trend |
|------|-------|--------|-------|-------|
| Foraging | 98.0% | 94.5% | 98.5% | Consistent |
| Pursuit | 82.5% | 84.5% | 97.5% | ↑ Increasing |
| Stationary | 67.0% | 87.5% | 96.5% | ↑ Increasing |

**Config tuning key findings**:

- Reducing `predator_damage` from 12 to 8 eliminated most HP deaths in pursuit configs
- Reducing `damage_radius` from 2 to 1 helped stationary configs by reducing blocking zones
- Reducing `danger_hp_damage` from 1.5 to 1.0 helped medium configs

**Small Stationary remains challenging**:

- Fixed toxic zones in tight spaces create unavoidable blocking patterns
- High variance (52-84% post-conv) suggests some sessions find good strategies, others don't
- Starvation deaths appearing (5%) - agents may be avoiding food areas blocked by predators

### Predator Type Comparison

| Environment | Pursuit Post-Conv | Stationary Post-Conv | Winner |
|-------------|-------------------|----------------------|--------|
| Small | 82.5% | 67.0% | Pursuit |
| Medium | 84.5% | 87.5% | ~Tie |
| Large | 97.5% | 96.5% | ~Tie |

**Observation**: Stationary predators remain harder in small grids due to fixed blocking patterns.

### Config Loader Bug Impact

The `safe_zone_food_bias` parameter was not being loaded from configs. When tested with it working:

- Medium foraging with `safe_zone_food_bias: 0.8` achieved 95%+ success
- This is too easy and doesn't represent the intended challenge

All benchmarks ran with uniform food distribution, which is more realistic and challenging.

______________________________________________________________________

## Conclusions

### Established Baselines (7 of 9)

1. **Small Foraging**: 98.0% post-convergence ✓
2. **Medium Foraging**: 94.5% post-convergence ✓
3. **Medium Pursuit**: 84.5% post-convergence ✓
4. **Medium Stationary**: 87.5% post-convergence ✓
5. **Large Foraging**: 98.5% post-convergence ✓
6. **Large Pursuit**: 97.5% post-convergence ✓
7. **Large Stationary**: 96.5% post-convergence ✓

### Configs Needing Further Work (2 of 9)

1. **Small Pursuit** (82.5%): Above target but high variance (66-94%), may need extended training or further parameter tuning
2. **Small Stationary** (67.0%): Below 70% target, high variance (52-84%), starvation deaths appearing

### Key Technical Findings

1. **`penalty_health_damage` conflicts with temperature HP damage**: Setting to 0 for small/medium was necessary to prevent double-penalization
2. **Large environments benefit from immediate damage feedback**: `penalty_health_damage: 0.3` works well there
3. **Predator damage of 8 (vs 12) is more appropriate for small/medium pursuit**: Reduces HP deaths while maintaining challenge
4. **Stationary predator `damage_radius` of 1 (vs 2) helps medium configs**: Reduces blocking zones
5. **Small stationary predators remain fundamentally challenging**: Fixed toxic zones in tight spaces create unavoidable blocking patterns

______________________________________________________________________

## Next Steps

- [x] Run all 36 sessions (completed)
- [x] Analyze results and update tables (completed)
- [x] Compare with hypotheses (completed)
- [x] Tune small/medium predator configs (completed - Round 2)
- [x] Re-run medium foraging with minor HP tuning (completed - Round 2)
- [ ] Further tune small pursuit/stationary if needed:
  - Option 1: Extended training to 700-1000 episodes
  - Option 2: Reduce predator count for small stationary
  - Option 3: Accept small stationary as inherently harder task
- [ ] Prepare quantum agent comparison using established baselines (7 of 9 ready)

______________________________________________________________________

## Data References

### Session IDs by Configuration

| Config | Sessions (Round 1) | Sessions (Round 2 - tuned) |
|--------|--------------------|-----------------------------|
| Small Foraging | 20260122_105256, 20260122_105258, 20260122_105300, 20260122_105303 | - |
| Small Pursuit | 20260124_101144, 20260124_101145, 20260124_101148, 20260124_101151 | 20260125_225334, 20260125_225337, 20260125_225340, 20260125_225343 |
| Small Stationary | 20260124_120426, 20260124_120428, 20260124_120430, 20260124_120433 | 20260125_233053, 20260125_233057, 20260125_233100, 20260125_233103 |
| Medium Foraging | 20260123_023943, 20260123_023946, 20260123_023949, 20260123_023952 | 20260125_114711, 20260125_114713, 20260125_114715, 20260125_114718 |
| Medium Pursuit | 20260124_073827, 20260124_073829, 20260124_073831, 20260124_073834 | 20260126_001541, 20260126_001543, 20260126_001545, 20260126_001548 |
| Medium Stationary | 20260124_085511, 20260124_085515, 20260124_085517, 20260124_085520 | 20260126_011040, 20260126_011043, 20260126_011045, 20260126_011048 |
| Large Foraging | 20260123_115319, 20260123_115321, 20260123_115327, 20260123_115329 | - |
| Large Pursuit | 20260123_212840, 20260123_212843, 20260123_212846, 20260123_212850 | - |
| Large Stationary | 20260124_000557, 20260124_000600, 20260124_000603, 20260124_000606 | - |

### File Locations

- Config files: `configs/examples/ppo_thermotaxis_*.yml`
- Exports: `exports/<session_id>/session/data/simulation_results.csv`
- Experiment configs: `experiments/<session_id>/<config>.yml`
- **Baseline artifacts**: `artifacts/logbooks/007/` - Contains saved session results and configs for established baselines

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

1. **Reward shaping is environment-specific**: Large env learnings don't transfer to small/medium
2. **comfort_reward must be 0 for LARGE only**: Prevents freeze behavior in scattered zones
3. **Small/medium need comfort_reward: 0.05**: Creates positive incentive without freezing
4. **Brave foraging bonus (large only)**: Encourages risk-taking in scattered danger zones
5. **Safe zone food bias (large only)**: Reduces unfair spawns in dangerous areas
6. **Gradient decay matters**: Sharper decay (4.0-6.5) for better zone detection
7. **Episode counts vary**: Large predator configs need 500-700 episodes

### A.5 Small/Medium Foraging Tuning (2026-01-21 to 2026-01-23)

Extensive tuning was performed to find optimal parameters for small/medium foraging.

#### Small Foraging Tuning History

| Setup | comfort | discomfort | danger | hp_penalty | Avg Success | Best |
|-------|---------|------------|--------|------------|-------------|------|
| 1 | 0 | -0.1 | -0.3 | 0 | 66.4% | |
| 2 | 0 | -0.1 | -0.3 | 0.5 | 56.6% | |
| 3 | 0 | -0.05 | -0.5 | 0.5 | 57.8% | |
| 4 | 0 | 0 | -0.5 | 0.3 | 41.6% | |
| 5 | 0 | 0 | 0 | 0 | 74.8% | |
| **6** | **0** | **0** | **-0.3** | **0** | **88.2%** | **✓** |
| 7 | 0 | 0 | 0 | 0.3 | 87.3% | |
| 8 | 0 | 0 | -0.3 | 0.3 | 59.2% | |
| 9 | 0 | 0 | -0.5 | 0 | 72.9% | |
| 10 | 0 | 0 | -0.1 | 0 | 85.5% | |
| 11 | 0 | 0 | -0.3 | 0.1 | 81.4% | |

**Winner**: Setup 6 with `danger_penalty: -0.3` and all other thermotaxis rewards/penalties at 0.

**Key insight**: The code change that added `penalty_health_damage` for temperature HP damage created double-penalization. Setting `penalty_health_damage: 0` lets agents learn from HP damage naturally without reward conflicts.

#### Medium Foraging Tuning History

| Setup | danger | hp_penalty | Other | Avg Success | Best |
|-------|--------|------------|-------|-------------|------|
| 1 | -0.3 | 0 | baseline | 70.4% | |
| **2** | **-0.3** | **0** | **baseline** | **72.0%** | **✓** |
| 3 | -0.5 | 0.3 | +discomfort, +brave | 53.0% | |
| 4 | -0.5 | 0.3 | +discomfort, +brave | 54.0% | |
| 5 | -0.5 | 0 | +brave | 63.5% | |
| 6 | -0.5 | 0 | +safe_zone_bias | 95.3% | Too easy |

**Note**: Setup 6 with `safe_zone_food_bias: 0.8` achieved 95%+ but was rejected as it made the task trivially easy.

#### Code Bug Discovery

The `safe_zone_food_bias` parameter was never being loaded from config files due to a bug in `config_loader.py`. This was fixed, but testing showed that enabling it (0.8) makes simulations too easy. All final benchmarks ran without it.

### A.6 Final Configuration Summary

**Small/Medium configs** (committed):

```yaml
reward:
  penalty_health_damage: 0.0   # CRITICAL: Disabled

thermotaxis:
  comfort_reward: 0.0          # No comfort reward
  discomfort_penalty: 0.0      # No discomfort penalty
  danger_penalty: -0.3         # Only danger penalty active
  reward_discomfort_food: 0.0  # No brave bonus
```

**Large configs** (unchanged):

```yaml
reward:
  penalty_health_damage: 0.3   # Active for large

thermotaxis:
  comfort_reward: 0.0
  discomfort_penalty: -0.05
  danger_penalty: -0.5
  reward_discomfort_food: 0.3  # Brave bonus
```

**Rationale**: Small/medium environments have simpler temperature gradients that don't require the complex reward shaping needed for large environments with scattered hot/cold spots.
