# 011: Multi-Agent Phase 4 Evaluation — Supporting Detail

## Bug Timeline

| Date | Bug | Impact | Fix |
|------|-----|--------|-----|
| 2026-04-06 | #112 Sensing position | All agents read sensory data from default agent | PR #113 |
| 2026-04-11 | #115 Reward calculator | All agents receive rewards from default agent position | Commit a6c86fca |
| 2026-04-11 | STAM dimension mismatch | Brain input_dim wrong with pheromone temporal modules | Commit a1b47493 |
| 2026-04-11 | food_sharing_buffer ungated | food_sharing_events fires without pheromones | Commit edab23f4 |

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
