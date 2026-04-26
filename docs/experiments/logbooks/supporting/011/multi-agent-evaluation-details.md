# 011: Multi-Agent Evaluation — Supporting Detail

## Bug Timeline

| Date | Bug | Impact | Fix |
|------|-----|--------|-----|
| 2026-04-06 | #112 Sensing position | All agents read sensory data from default agent | PR #113 |
| 2026-04-11 | #115 Reward calculator | All agents receive rewards from default agent position | Commit a6c86fca |
| 2026-04-11 | STAM dimension mismatch | Brain input_dim wrong with pheromone temporal modules | Commit a1b47493 |
| 2026-04-11 | food_marking_buffer ungated | food_sharing_events fires without pheromones | Commit edab23f4 |

## Campaign A-v2 Per-Session Data (Post-Fix)

Cannot distinguish baseline/pheromone from social/full-stack within CSV. Grouped by social_feeding_events presence.

| Group | Sessions | Food/Ep | Steps | Agg Index | Gini | F100 | L100 |
|-------|----------|---------|-------|-----------|------|------|------|
| No social (A1+A2) | 8 | 48.58 | 227.2 | 0.801 | 0.056 | 4568 | 4840 |
| Social (A3+A4) | 8 | 48.51 | 227.6 | 0.798 | 0.063 | 4527 | 4775 |

## Campaign B1-v2 Per-Seed Data

### 1-Agent (4000 eps)

| Seed | Food/Ep | Success | Steps |
|------|---------|---------|-------|
| 42 | 9.99 | 99.8% | 149.9 |
| 43 | 9.99 | 99.8% | 152.8 |
| 44 | 9.99 | 99.8% | 154.1 |
| 45 | 9.98 | 99.6% | 150.2 |

### 2-Agent (4000 eps)

| Seed | Food/Ep | Food/Agent | Steps |
|------|---------|-----------|-------|
| 42 | 19.97 | 9.99 | 179.9 |
| 43 | 19.97 | 9.99 | 179.9 |
| 44 | 19.97 | 9.99 | 177.0 |
| 45 | 19.98 | 9.99 | 179.5 |

### 5-Agent (4000 eps)

| Seed | Food/Ep | Food/Agent | Steps |
|------|---------|-----------|-------|
| 42 | 48.27 | 9.65 | 232.5 |
| 43 | 48.22 | 9.64 | 232.7 |
| 44 | 47.97 | 9.59 | 233.9 |
| 45 | 48.02 | 9.60 | 234.7 |

### 10-Agent (4000 eps)

| Seed | Food/Ep | Food/Agent | Steps |
|------|---------|-----------|-------|
| 42 | 80.89 | 8.09 | 264.8 |
| 43 | 80.25 | 8.03 | 265.6 |
| 44 | 79.47 | 7.95 | 265.3 |
| 45 | 80.12 | 8.01 | 264.8 |

## Campaign B2+D-v2 Per-Session Data

| Config | Seed | Food/Ep | Food/Agent | F100 | L100 |
|--------|------|---------|-----------|------|------|
| 1-agent temporal | 42 | 7.99 | 7.99 | 324 | 1000 |
| 1-agent temporal | 43 | 8.67 | 8.67 | 421 | 968 |
| 5-agent no-pheromone | 42 | 47.38 | 9.48 | 1871 | 4998 |
| 5-agent no-pheromone | 43 | 47.77 | 9.55 | 1971 | 4998 |
| 5-agent pheromone | 42 | 47.21 | 9.44 | 1989 | 4952 |
| 5-agent pheromone | 43 | 48.19 | 9.64 | 2236 | 4981 |

## Campaign E Per-Session Data

| Config | Seed | Food/Ep | Food/Agent | Steps | Agg Index | Gini |
|--------|------|---------|-----------|-------|-----------|------|
| No social | 42 | 16.94 | 3.39 | 152.4 | 0.871 | 0.501 |
| No social | 43 | 17.32 | 3.46 | 153.6 | 0.874 | 0.496 |
| Social ON | 42 | 23.06 | 4.61 | 221.5 | 0.903 | 0.412 |
| Social ON | 43 | 23.20 | 4.64 | 224.4 | 0.907 | 0.406 |

## Campaign F Per-Agent Data

### Session 1 (seed 42)

| Agent | Phenotype | Food/Ep | Steps | Reward | Satiety |
|-------|-----------|---------|-------|--------|---------|
| agent_0 | social | 9.46 | 231.5 | 71.17 | 191.2 |
| agent_1 | social | 9.75 | 228.6 | 73.04 | 196.2 |
| agent_2 | social | 9.60 | 232.4 | 75.26 | 193.6 |
| agent_3 | solitary | 9.66 | 229.7 | 75.20 | 182.1 |
| agent_4 | solitary | 9.66 | 226.0 | 76.08 | 183.0 |

### Session 2 (seed 43)

| Agent | Phenotype | Food/Ep | Steps | Reward | Satiety |
|-------|-----------|---------|-------|--------|---------|
| agent_0 | social | 9.69 | 226.3 | 70.97 | 194.9 |
| agent_1 | social | 8.45 | 247.5 | 72.37 | 172.0 |
| agent_2 | social | 9.72 | 226.2 | 73.29 | 195.7 |
| agent_3 | solitary | 9.66 | 227.2 | 75.20 | 182.8 |
| agent_4 | solitary | 9.57 | 234.9 | 77.32 | 179.8 |

## Campaign G Per-Session Data

| Agents | Food Grid | Seed | Food/Ep | Food/Agent | Steps | Gini |
|--------|----------|------|---------|-----------|-------|------|
| 1 | 3 | 42 | 9.98 | 9.98 | 153.2 | — |
| 1 | 3 | 43 | 9.98 | 9.98 | 153.6 | — |
| 2 | 6 | 42 | 19.97 | 9.99 | 139.6 | 0.001 |
| 2 | 6 | 43 | 19.99 | 9.99 | 138.0 | 0.000 |
| 5 | 15 | 42 | 49.97 | 9.99 | 124.6 | 0.000 |
| 5 | 15 | 43 | 49.98 | 10.00 | 125.4 | 0.000 |

## Cross-Campaign Comparison: Oracle vs Temporal Multi-Agent Effect

| Mode | 1-Agent Food/Ep | 5-Agent Food/Agent/Ep | Multi-Agent Effect |
|------|----------------|----------------------|-------------------|
| Oracle (3 food) | 9.99 | 9.62 | -3.7% (competition) |
| Temporal (5 food) | 8.33 | 9.52 | +14.3% (collective benefit) |
| Oracle proportional (15 food) | 9.98 | 10.00 | +0.2% (no overhead) |

______________________________________________________________________

## Klinotaxis Era Campaign — Per-Seed Detail

All Klinotaxis Era sessions used LSTM PPO GRU (Phase 3 brain hyperparams) with klinotaxis sensing, 4 seeds (42-45) per config, 1000 episodes per session unless stated. Variance reported as standard deviation across the 4 seeds.

### K1 — Single-Cluster Pheromone Evaluation (5 and 10 agents)

100×100 grid, 1 food cluster at (50,50), `no_respawn=true`, food gradient_decay=2.0, hotspot_decay=2.0, strong food-marking pheromones (emission=3.0, spatial_decay=12.0, temporal_half_life=100). 5 agents target 1 food/agent (foods_on_grid=5); 10 agents target 1 food/agent (foods_on_grid=10).

#### K1 5-agent pheromone — per-seed agents-fed and all-fed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 83.4% | 90.8% | 65.4% | 81.0% |
| 43 | 82.8% | 90.4% | 63.6% | 70.0% |
| 44 | 81.9% | 90.0% | 62.3% | 74.0% |
| 45 | 80.9% | 85.8% | 58.1% | 66.0% |
| **Mean** | **82.2 ± 1.1%** | **89.2 ± 2.3%** | **62.4 ± 3.1%** | **72.8 ± 6.4%** |

#### K1 5-agent control (no pheromones) — per-seed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 41.7% | 51.2% | 0.5% | 3.0% |
| 43 | 41.0% | 52.0% | 0.8% | 2.0% |
| 44 | 42.0% | 49.2% | 0.5% | 0.0% |
| 45 | 41.1% | 49.0% | 0.6% | 3.0% |
| **Mean** | **41.4 ± 0.5%** | **50.4 ± 1.5%** | **0.6 ± 0.1%** | **2.0 ± 1.4%** |

#### K1 10-agent pheromone — per-seed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 93.0% | 92.2% | 69.3% | 66.0% |
| 43 | 91.8% | 92.5% | 66.3% | 68.0% |
| 44 | 92.7% | 93.5% | 70.4% | 73.0% |
| 45 | 92.0% | 92.9% | 64.5% | 62.0% |
| **Mean** | **92.4 ± 0.6%** | **92.8 ± 0.6%** | **67.6 ± 2.7%** | **67.2 ± 4.6%** |

#### K1 10-agent control — per-seed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 45.5% | 51.9% | 0.0% | 0.0% |
| 43 | 45.8% | 50.4% | 0.0% | 0.0% |
| 44 | 46.0% | 54.4% | 0.0% | 0.0% |
| 45 | 45.9% | 54.3% | 0.0% | 0.0% |
| **Mean** | **45.8 ± 0.2%** | **52.8 ± 1.9%** | **0.0 ± 0.0%** | **0.0 ± 0.0%** |

### K2A — Pheromone-Only Mechanism Isolation

Same K1 environment but `food_chemotaxis` removed from sensory_modules. K2A control further removes `pheromone_food` (no food-related sensing at all).

#### K2A pheromone_only — per-seed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 40.2% | 40.6% | 12.6% | 17.0% |
| 43 | 38.1% | 43.6% | 10.9% | 18.0% |
| 44 | 41.4% | 52.0% | 13.7% | 22.0% |
| 45 | 37.6% | 47.2% | 10.2% | 16.0% |
| **Mean** | **39.3 ± 1.8%** | **45.9 ± 4.9%** | **11.8 ± 1.6%** | **18.2 ± 2.6%** |

#### K2A no-food-sensing — per-seed

| Seed | Agents-fed Overall | Agents-fed L100 | All-fed Overall | All-fed L100 |
|------|--------------------|------------------|-----------------|---------------|
| 42 | 16.0% | 20.0% | 0.0% | 0.0% |
| 43 | 15.9% | 17.2% | 0.0% | 0.0% |
| 44 | 16.2% | 21.0% | 0.0% | 0.0% |
| 45 | 16.3% | 17.2% | 0.0% | 0.0% |
| **Mean** | **16.1 ± 0.2%** | **18.9 ± 1.9%** | **0.0 ± 0.0%** | **0.0 ± 0.0%** |

#### K2A learning-curve detail (seed 42, agents-fed % per 100-run window)

- Pheromone_only: 26.8 → 36.8 → 40.2 → 39.4 → 44.0 → 40.4 → 46.2 → 39.6 → 48.0 → 40.6
- No-food-sensing: 15.0 → 13.2 → 15.2 → 15.2 → 16.4 → 14.4 → 16.8 → 17.0 → 17.2 → 20.0

### K2B — Klinotaxis Pursuit Alarm

20×20 grid, 5 agents, 2 pursuit predators, klinotaxis nociception. 1000 episodes, max_steps=1000.

| Seed | Foods/agent L100 | Death rate L100 | Completed L100 | Steps L100 |
|------|------------------|-----------------|----------------|------------|
| Alarm 42 | 9.86 | 2.0% | 97.8% | 248 |
| Alarm 43 | 9.65 | 3.6% | 94.2% | 277 |
| Alarm 44 | 9.76 | 1.6% | 94.2% | 278 |
| Alarm 45 | 9.79 | 2.2% | 96.2% | 247 |
| **Alarm mean** | **9.77 ± 0.09** | **2.35 ± 0.87%** | **95.6 ± 1.7%** | **263 ± 17** |
| No-alarm 42 | 9.65 | 2.2% | 92.4% | 279 |
| No-alarm 43 | 9.88 | 0.8% | 96.2% | 260 |
| No-alarm 44 | 9.77 | 2.6% | 96.4% | 246 |
| No-alarm 45 | 9.72 | 2.0% | 94.0% | 292 |
| **No-alarm mean** | **9.76 ± 0.10** | **1.90 ± 0.77%** | **94.8 ± 1.9%** | **269 ± 20** |

### K3 — Mixed-Phenotype Evaluation

K1 environment with heterogeneous brains: followers have `pheromone_food` perception, loners do not. STAM dim auto-resolution fix (commit `bbfb3293`) enabled this campaign.

#### K3 mixed_3F_2L — per-seed L100 (3 followers + 2 loners)

| Seed | Followers L100 | Loners L100 | All-fed L100 |
|------|---------------|-------------|--------------|
| 42 | 93.0% | 46.0% | 20.0% |
| 43 | 89.0% | 44.0% | 14.0% |
| 44 | 93.0% | 48.0% | 17.0% |
| 45 | 87.0% | 44.5% | 14.0% |
| **Mean** | **90.5 ± 3.0%** | **45.6 ± 1.8%** | **16.2 ± 2.9%** |

#### K3 mixed_1F_4L — per-seed L100 (1 follower + 4 loners)

| Seed | Followers L100 | Loners L100 | All-fed L100 |
|------|---------------|-------------|--------------|
| 42 | 88.0% | 44.2% | 1.0% |
| 43 | 94.0% | 48.2% | 4.0% |
| 44 | 96.0% | 52.8% | 6.0% |
| 45 | 95.0% | 49.2% | 0.0% |
| **Mean** | **93.2 ± 3.6%** | **48.6 ± 3.5%** | **2.8 ± 2.8%** |

#### K3 mixed_4F_1L — per-seed L100 (4 followers + 1 loner)

| Seed | Followers L100 | Loners L100 | All-fed L100 |
|------|---------------|-------------|--------------|
| 42 | 93.0% | 56.0% | 45.0% |
| 43 | 91.0% | 51.0% | 45.0% |
| 44 | 91.0% | 47.0% | 42.0% |
| 45 | 84.8% | 50.0% | 47.0% |
| **Mean** | **89.9 ± 3.6%** | **51.0 ± 3.7%** | **44.8 ± 2.1%** |

#### K3 weak-pheromone control (5 followers, original pre-strengthened pheromone params)

| Seed | Agents-fed L100 | All-fed L100 |
|------|-----------------|--------------|
| 42 | 59.6% | 16.0% |
| 43 | 63.4% | 12.0% |
| 44 | 64.8% | 17.0% |
| 45 | 60.8% | 6.0% |
| **Mean** | **62.1 ± 2.4%** | **12.8 ± 5.0%** |

### K4 — Mechanism Dissection (single-cluster pheromone parameter ablations)

K1 environment, varying one parameter at a time on the strong-pheromone recipe.

| Config | Agents-fed L100 | All-fed L100 |
|--------|-----------------|--------------|
| K1 strong (reference, food + aggregation pheromones) | 89.2 ± 2.3% | 72.8 ± 6.4% |
| K1 nopher (reference) | 50.4 ± 1.5% | 2.0 ± 1.4% |
| **food_marking_only** (no aggregation perception) | **92.3 ± 1.9%** | **79.5 ± 3.1%** |
| **aggregation_only** (no food-marking perception) | **47.2 ± 2.4%** | **2.0 ± 1.6%** |
| half_life_25 (vs 100) | 81.7 ± 3.9% | 48.2 ± 9.2% |
| spatial_decay_6 (vs 12) | 66.8 ± 2.7% | 14.5 ± 4.4% |

Per-seed K4 detail available in `artifacts/logbooks/011/klinotaxis_configs/k4_*/sessions/*.csv`.

### K5 — Stationary Predator Alarm Pheromones (3 variants)

#### K5 v1 (small 20×20, 3 stationary predators, alarm spatial_decay=12, detection_radius=3)

Small grid was too cramped — 85% of grid covered by predator gradient, alarm pheromone reaches whole grid. Death rate ~52% indicates env over-determined.

| Variant | Foods/agent L100 | Death rate L100 | Completed L100 |
|---------|------------------|-----------------|----------------|
| Alarm | 6.76 ± 0.08 | 52.55 ± 2.59% | 45.25 ± 3.12% |
| No-alarm | 6.83 ± 0.08 | 52.15 ± 1.54% | 46.20 ± 2.54% |

#### K5 v2 (medium 50×50, 3 stationary predators, alarm spatial_decay=6, detection_radius=6)

Re-tuned env. Healthy training: 88% completion, 10% death rate, 0% starvation.

| Variant | Foods/agent L100 | Death rate L100 | Completed L100 |
|---------|------------------|-----------------|----------------|
| Alarm | 9.51 ± 0.11 | 10.10 ± 2.07% | 88.80 ± 1.64% |
| No-alarm | 9.46 ± 0.02 | 10.60 ± 0.59% | 88.60 ± 0.33% |

#### K5 v3 (medium 50×50, biologically faithful — no nociception module, alarm spatial_decay=8)

Removes nociception entirely from the brain to match C. elegans biology where threat detection is contact-based and alarm signals are the natural long-range threat info channel.

| Variant | Foods/agent L100 | Death rate L100 | Completed L100 |
|---------|------------------|-----------------|----------------|
| Alarm-only | 9.00 ± 0.07 | 19.25 ± 1.48% | 80.50 ± 1.43% |
| No-threat-sensing | 8.99 ± 0.07 | 18.75 ± 1.09% | 80.95 ± 0.90% |

All three variants null. Alarm channel is informationally inert across cramped, well-tuned, and biologically faithful conditions.

### K6 — Social Feeding Klinotaxis Port

20×20, 5 agents, 1 food, satiety_decay_rate=2.0, initial_satiety=200.

| Variant | Foods/agent L100 | Steps L100 | Completed L100 |
|---------|------------------|------------|----------------|
| Social ON | 5.07 ± 0.11 | 239.9 ± 3.4 | 31.10 ± 1.25% |
| Social OFF | 3.43 ± 0.12 | 155.3 ± 2.5 | 14.80 ± 0.52% |
| **Δ vs OFF** | **+47.8%** | **+54.5%** | **+110%** |
| Campaign E (oracle ref) | +35% | +46% | n/a |

### K7 — Multi-Cluster + Aggregation 2×2 (Gap-Closing Round)

#### K7a/b — Multi-cluster (2 hotspots at (25,25) and (75,75), 6 food, 5 agents)

| Variant | Agents-fed L100 | All-fed L100 |
|---------|-----------------|--------------|
| Multi-cluster pheromone | 81.2 ± 1.3% | 40.8 ± 1.9% |
| Multi-cluster control | 78.5 ± 2.9% | 24.8 ± 4.8% |
| **K1 single-cluster reference (pher)** | 89.2 ± 2.3% | 72.8 ± 6.4% |
| **K1 single-cluster reference (control)** | 50.4 ± 1.5% | 2.0 ± 1.4% |

Multi-cluster pheromone benefit: +2.7pp on agents-fed (vs +38.8pp single-cluster), +16pp on all-fed (vs +70.8pp single-cluster). Generalisation partial — the all-fed metric still shows ~1.6× lift.

#### K7c/d — Scarcity 2×2 factorial (social feeding × aggregation perception)

K6 environment plus `pheromone_aggregation` in sensory_modules (K7c, K7d).

| Cell | Social | Aggregation | Foods/agent L100 | Completed L100 | Steps L100 |
|------|--------|-------------|------------------|----------------|------------|
| K6c (R4c) | ON | OFF | 5.07 ± 0.11 | 31.10 ± 1.25% | 239.9 |
| K6d (R4d) | OFF | OFF | 3.43 ± 0.12 | 14.80 ± 0.52% | 155.3 |
| K7c | ON | ON | 5.09 ± 0.03 | 32.35 ± 1.37% | 237.7 |
| K7d | OFF | ON | 3.53 ± 0.07 | 16.55 ± 1.48% | 155.2 |

Marginal effects:

- Social feeding (with agg OFF): +47.8% food
- Social feeding (with agg ON): +44.2% food (essentially same)
- Aggregation (with social ON): +0.4% food (null)
- Aggregation (with social OFF): +2.9% food (within noise)
- Interaction: ~0pp

### Klinotaxis Era Cross-Campaign Summary

#### Pheromone channel value matrix

| Channel | Single-cluster food | Multi-cluster food | Scarcity (with social feeding) | Stationary predator |
|---------|---------------------|--------------------|--------------------------------|---------------------|
| Food-marking pheromone | **+77pp / +36×** (K1) | +2.7pp / +16pp (K7) | n/a | n/a |
| Aggregation pheromone | neutral (K4) | (untested separately) | **0pp** (K7) | n/a |
| Alarm pheromone | n/a | n/a | n/a | **0pp** (K5 v1/v2/v3) |

#### Mechanism criteria for collective pheromone benefit (refined across K1–K7)

1. Signal source spatially persistent
2. Direct sensing insufficient at long range
3. Most of grid has no signal
4. Emission location coincides with signal-of-interest location
5. Channel is food-marking specifically (not aggregation, not alarm)
6. Single dominant attractor rather than multiple competing ones

#### Aggregate session count

- 104 multi-agent sessions across K1-K7
- 4 seeds per config, 1000 eps per session
- 28 distinct configs in `artifacts/logbooks/011/klinotaxis_configs/`
