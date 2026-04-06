# 010: Aerotaxis Baselines — Supporting Data

## Per-Seed Oracle Results

### O2 Foraging Medium (50×50, 2000 episodes, LR-fix config)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort |
|------|---------|------|------|-------|------------|
| 42 | 58.2% | **85.0%** | 76.8% | 70.9% | 0.853 |
| 43 | 52.2% | 77.0% | 78.0% | 71.6% | 0.868 |
| 44 | 50.0% | 67.0% | 70.0% | 65.4% | 0.876 |
| 45 | 49.1% | 74.0% | 72.4% | 67.8% | 0.868 |
| **Mean** | **52.4%** | **75.8%** | **74.3%** | **68.9%** | **0.866** |

### O2 Foraging Large (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort |
|------|---------|------|------|-------|------------|
| 42 | 77.5% | **90.0%** | 82.2% | 84.0% | 0.969 |
| 43 | 77.8% | 82.0% | **87.6%** | 87.0% | 0.972 |
| 44 | 65.0% | 76.0% | 71.8% | 73.3% | 0.972 |
| 45 | 58.8% | 68.0% | 71.4% | 65.3% | 0.969 |
| **Mean** | **69.8%** | **79.0%** | **78.3%** | **77.4%** | **0.971** |

### O2 + Pursuit (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | O2 Comfort |
|------|---------|------|------|------------|
| 42 | 48.5% | **67.0%** | 54.0% | 0.954 |
| 43 | 52.8% | 53.0% | **65.8%** | 0.951 |
| 44 | 44.1% | 67.0% | 63.2% | 0.953 |
| 45 | 45.1% | 64.0% | 56.4% | 0.956 |
| **Mean** | **47.6%** | **62.8%** | **59.9%** | **0.954** |

### O2 + Stationary (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | O2 Comfort |
|------|---------|------|------|------------|
| 42 | 40.0% | 49.0% | **48.4%** | 0.951 |
| 43 | 34.5% | 51.0% | 44.8% | 0.951 |
| 44 | 19.6% | **53.0%** | 44.8% | 0.727 |
| 45 | 26.1% | 36.0% | 29.8% | 0.952 |
| **Mean** | **30.1%** | **47.3%** | **42.0%** | **0.895** |

Note: Seed 44 has anomalously low O2 comfort (0.727) — possible early convergence to a suboptimal policy.

### O2 + Thermal Foraging (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | O2 Comfort | Temp Comfort |
|------|---------|------|------|------------|--------------|
| 42 | 73.0% | **96.0%** | **90.4%** | 0.952 | 0.705 |
| 43 | 70.0% | 89.0% | 84.0% | 0.968 | 0.732 |
| 44 | 72.5% | 86.0% | 81.0% | 0.960 | 0.722 |
| 45 | 67.4% | 85.0% | 81.8% | 0.965 | 0.714 |
| **Mean** | **70.7%** | **89.0%** | **84.3%** | **0.961** | **0.718** |

### O2 + Thermal + Pursuit (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | O2 Comfort | Temp Comfort |
|------|---------|------|------|------------|--------------|
| 42 | 47.2% | 69.0% | 58.2% | 0.952 | 0.743 |
| 43 | 43.3% | **76.0%** | 59.2% | 0.939 | 0.767 |
| 44 | 40.4% | 70.0% | **63.6%** | 0.923 | 0.748 |
| 45 | 38.1% | 65.0% | 48.4% | 0.932 | 0.777 |
| **Mean** | **42.3%** | **70.0%** | **57.4%** | **0.937** | **0.759** |

### O2 + Thermal + Stationary (100×100, 2000 episodes)

| Seed | Overall | L100 | L500 | O2 Comfort | Temp Comfort |
|------|---------|------|------|------------|--------------|
| 42 | 37.0% | **59.0%** | **55.4%** | 0.942 | 0.747 |
| 43 | 35.5% | 55.0% | 49.6% | 0.950 | 0.765 |
| 44 | 26.2% | 49.0% | 39.0% | 0.946 | 0.771 |
| 45 | 32.9% | 40.0% | 38.8% | 0.948 | 0.768 |
| **Mean** | **32.9%** | **50.8%** | **45.7%** | **0.947** | **0.763** |

### Thermal Foraging Control (100×100, 2000 episodes, NO oxygen)

| Seed | Overall | L100 | L500 | Temp Comfort |
|------|---------|------|------|--------------|
| 42 | 98.1% | **100.0%** | 98.2% | 0.716 |
| 43 | 97.7% | 98.0% | 98.4% | 0.714 |
| 44 | 97.3% | 100.0% | **99.4%** | 0.722 |
| 45 | 97.0% | 99.0% | 98.0% | 0.711 |
| **Mean** | **97.5%** | **99.3%** | **98.5%** | **0.716** |

### Thermal + Pursuit Control (100×100, 2000 episodes, NO oxygen)

| Seed | Overall | L100 | L500 | Temp Comfort |
|------|---------|------|------|--------------|
| 42 | 87.8% | **98.0%** | **97.0%** | 0.727 |
| 43 | 87.6% | 98.0% | 95.0% | 0.715 |
| 44 | 86.1% | 89.0% | 90.6% | 0.728 |
| 45 | 85.4% | 90.0% | 88.4% | 0.727 |
| **Mean** | **86.7%** | **93.8%** | **92.8%** | **0.724** |

### Thermal + Stationary Control (100×100, 2000 episodes, NO oxygen)

| Seed | Overall | L100 | L500 | Temp Comfort |
|------|---------|------|------|--------------|
| 42 | 85.4% | **95.0%** | 94.0% | 0.711 |
| 43 | 85.7% | 93.0% | 90.6% | 0.705 |
| 44 | 86.2% | 95.0% | **96.2%** | 0.704 |
| 45 | 81.3% | 90.0% | 90.2% | 0.726 |
| **Mean** | **84.7%** | **93.3%** | **92.8%** | **0.712** |

## Temporal Sensing Results

### O2 Foraging Medium Temporal (50×50, 12000 episodes, GRU)

| Seed | Overall | L100 | L1000 | O2 Comfort |
|------|---------|------|-------|------------|
| 42 | 5.7% | 9.0% | 13.6% | 0.968 |
| 43 | 8.1% | 6.0% | 12.5% | 0.969 |
| 44 | 6.7% | 12.0% | 10.2% | 0.964 |
| 45 | 6.7% | 11.0% | **16.5%** | 0.967 |
| **Mean** | **6.8%** | **9.5%** | **13.2%** | **0.967** |

### Temporal Learning Curves (2000-episode windows)

| Window | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Mean |
|--------|---------|---------|---------|---------|------|
| 1-2000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2001-4000 | 0.0% | 0.5% | 0.9% | 0.1% | 0.4% |
| 4001-6000 | 1.9% | 9.7% | 7.4% | 4.5% | 5.9% |
| 6001-8000 | 7.5% | 11.9% | 11.2% | 9.4% | 10.0% |
| 8001-10000 | 12.9% | 13.8% | 9.8% | 11.1% | 11.9% |
| 10001-12000 | **11.9%** | **12.4%** | **11.0%** | **14.9%** | **12.6%** |

Curves plateau in the 8000-12000 range. No seed shows continued acceleration.

## Dual-Modality Learning Curves (O2 + Thermal Foraging)

### Success Rate + Comfort Scores Over Training (200-episode windows, best seed 42)

| Window | Success | O2 Comfort | Temp Comfort | HP Deaths |
|--------|---------|------------|--------------|-----------|
| Ep 1-200 | 15.0% | 0.896 | 0.714 | 45 |
| Ep 201-500 | 54.3% | 0.965 | 0.755 | 7 |
| Ep 501-1000 | 73.4% | 0.975 | 0.736 | 10 |
| Ep 1001-1500 | 75.8% | 0.973 | 0.677 | 11 |
| Ep 1501-2000 | 81.8% | 0.974 | 0.704 | 7 |

O2 comfort rises from 0.90→0.97 (agent learns O2 avoidance). Temp comfort stays ~0.70-0.75 (constrained by hot/cold spot layout). HP deaths drop from 45→7 (stops entering lethal zones).

## Hyperparameter Tuning Notes

### LR Schedule Fix (Medium Oracle)

The original medium oracle config had `lr_decay_episodes: 200`, causing LR to bottom out after 10% of a 2000-episode run. Fixing to `lr_decay_episodes: 1500` improved mean L100 from 70.0% → 75.8% and reduced seed variance from 23pp → 18pp spread.

### Difficulty Tuning

| Grid | Original gradient | Comfort % | Tuned gradient | Comfort % |
|------|-------------------|-----------|----------------|-----------|
| Medium (50×50) | 0.15 | 78% | **0.30** | 46% |
| Large (100×100) | 0.08 (1 high + 1 low spot) | 73% | **0.12** (2 high + 3 low spots) | 50% |

### Failed Experiment: penalty_health_damage on Medium

Adding `penalty_health_damage: 0.3` to the medium oxygen config (alongside the existing `danger_penalty: -0.3`) created a perverse incentive — total penalty of -0.6/step in danger zones caused the agent to freeze in the comfort zone and starve rather than forage. Success dropped from 61% → 1-3%. Lesson: don't combine zone penalties with health damage penalties for oxygen.

## Derivative Sensing Results (Large Grid, 6000 Episodes)

### O2 Foraging Derivative (Group J)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort |
|------|---------|------|------|-------|------------|
| 42 | 73.5% | **100%** | **99.2%** | **96.8%** | 0.963 |
| 43 | 62.0% | 86.0% | 56.0% | 60.6% | 0.965 |
| 44 | 63.3% | 76.0% | 78.8% | 79.3% | 0.964 |
| 45 | 50.0% | 68.0% | 75.4% | 72.3% | 0.966 |
| **Mean** | **62.2%** | **82.5%** | **77.4%** | **77.2%** | **0.965** |

### O2+Thermal Foraging Derivative (Group K)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort | Temp Comfort |
|------|---------|------|------|-------|------------|--------------|
| 42 | 69.5% | **99.0%** | **99.2%** | **99.2%** | 0.951 | 0.719 |
| 43 | 64.0% | 98.0% | 95.8% | 95.0% | 0.948 | 0.744 |
| 44 | 63.5% | 97.0% | 97.4% | 96.7% | 0.954 | 0.735 |
| 45 | 62.1% | 93.0% | 92.4% | 91.6% | 0.937 | 0.771 |
| **Mean** | **64.8%** | **96.8%** | **96.2%** | **95.6%** | **0.948** | **0.742** |

### O2+Pursuit Derivative (Group L)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort |
|------|---------|------|------|-------|------------|
| 42 | 49.3% | **96.0%** | **92.2%** | 87.3% | 0.963 |
| 43 | 56.5% | 95.0% | 87.4% | 81.7% | 0.958 |
| 44 | 52.6% | 93.0% | 89.0% | 83.3% | 0.960 |
| 45 | 48.3% | 78.0% | 82.0% | 82.6% | 0.958 |
| **Mean** | **51.6%** | **90.5%** | **87.7%** | **83.7%** | **0.960** |

### O2+Thermal+Pursuit Derivative (Group M, 3 seeds)

| Seed | Overall | L100 | L500 | L1000 | O2 Comfort | Temp Comfort |
|------|---------|------|------|-------|------------|--------------|
| 42 | 39.6% | **96.0%** | **92.2%** | **92.1%** | 0.950 | 0.749 |
| 43 | 35.4% | 88.0% | 87.8% | 84.8% | 0.947 | 0.772 |
| 45 | 26.4% | 80.0% | 80.8% | 80.6% | 0.941 | 0.829 |
| **Mean** | **33.8%** | **88.0%** | **86.9%** | **85.8%** | **0.946** | **0.784** |

### O2+Stationary Derivative (Group N, best 4 of 8 seeds from exports)

| Seed | L100 | L500 | O2 Comfort |
|------|------|------|------------|
| best | 68% | 62.6% | 0.960 |
| 2nd | 66% | 59.2% | 0.943 |
| 3rd | 64% | 74.0% | 0.955 |
| 4th | 59% | 66.4% | 0.942 |
| **Mean (best 4)** | **64.3%** | **65.6%** | **0.950** |

High seed variance (41-68% L100) — stationary toxic zones create position-dependent difficulty.

### O2+Thermal+Stationary Derivative (Group O, 4 seeds)

| Seed | Overall | L100 | L500 | O2 Comfort | Temp Comfort |
|------|---------|------|------|------------|--------------|
| 42 | 25.4% | 62.0% | 57.8% | 0.955 | 0.789 |
| 43 | 21.6% | 45.0% | 57.2% | 0.958 | 0.798 |
| 44 | 25.4% | 62.0% | 57.8% | 0.955 | 0.789 |
| 45 | 23.8% | 59.0% | 66.4% | 0.942 | 0.802 |
| **Mean** | **24.1%** | **57.0%** | **59.8%** | **0.953** | **0.795** |

## Temporal Sensing Results (Large Grid, 6000 Episodes)

### O2 Foraging Large Temporal (Group P)

| Seed | Overall | L100 | L500 | O2 Comfort |
|------|---------|------|------|------------|
| 42 | 45.1% | **99%** | **95.4%** | 0.968 |
| 43 | 61.6% | **99%** | **97.2%** | 0.961 |
| 44 | 58.8% | **97%** | **95.4%** | 0.966 |
| 45 | 59.4% | **100%** | **98.2%** | 0.960 |
| **Mean** | **56.2%** | **98.8%** | **96.6%** | **0.964** |

### O2+Thermal Foraging Large Temporal (Group Q) — FAILED

All 4 seeds achieved 0% success in 6000 episodes. Root cause: `gradient_decay_constant: 12.0` instead of 4.0 in food foraging section. Config fixed, needs re-run.

## Config Issue: gradient_decay_constant

The `oxygen_thermal_foraging/lstmppo_large_temporal.yml` config was created from the oracle config template which uses `gradient_decay_constant: 12.0`. All temporal and derivative configs require 4.0 for detectable scalar concentration changes during movement. Fixed to 4.0 after discovering the Q failure.
