## Overview

This deliverable adds the final metrics and evaluation campaign needed to close Phase 4. The infrastructure (D1-D3) is complete — what remains is measuring competitive dynamics, collective predator response, and classical coordination difficulty through a comprehensive evaluation campaign.

## Goals / Non-Goals

**Goals:**

- Two new per-episode metrics: territorial_index and alarm_response_rate
- Nine scenario configs designed for evaluation (20x20 grids, oracle + temporal modes)
- Four-campaign evaluation producing data for all Phase 4 exit criteria
- Game-theoretic indicators computed from existing + new metrics
- Logbook 011 documenting findings, Phase 4 conclusion, quantum checkpoint assessment

**Non-Goals:**

- Nash equilibrium computation (requires stable converged strategies; deferred unless agents converge at 4000 eps)
- New reward shaping or training algorithms (evaluate what exists, don't force behaviors)
- Thermal/aerotaxis multi-agent (requires 100x100 grids; deferred to Phase 8)
- New pheromone types or sensing modules
- New brain architectures

## Design Decisions

### Decision 1: territorial_index via Spatial Gini

```python
def _compute_territorial_index(
    per_agent_food_positions: dict[str, list[tuple[int, int]]],
    grid_size: int,
) -> float:
```

For each agent, compute the mean distance of their food collection positions from their centroid (foraging spread). Then compute a Gini coefficient across agents' spreads. High Gini = some agents forage in tight zones while others range widely = territorial specialization.

Requires tracking `_per_agent_food_positions: dict[str, list[tuple[int, int]]]` in `_resolve_food_step()`. Populated alongside existing `_per_agent_food` counter. Agents that collect no food contribute 0 spread.

Edge cases:

- < 2 agents with food: return 0.0 (no territorial comparison possible)
- Agent with 1 food item: spread = 0 (single point has no spread)

### Decision 2: alarm_response_rate via Direction Change Tracking

When an alarm pheromone is emitted (agent takes predator damage):

1. Record `(position, step, emitter_id)` in a buffer
2. For each non-emitter agent within `social_detection_radius` at emission time, record their current direction
3. Over the next `ALARM_RESPONSE_WINDOW = 5` steps, check if that agent's direction changed
4. Rate = direction changes / opportunities

```python
ALARM_RESPONSE_WINDOW = 5
```

This measures *causal* response to alarm signals — did the alarm emission cause other agents to react? The existing `alarm_evasion_events` metric counts zone exits but doesn't establish causality (agents may exit alarm zones without having been influenced by the alarm).

Edge cases:

- No alarm emissions: rate = 0.0
- No nearby agents at emission time: no opportunities, rate = 0.0

### Decision 3: Evaluation Campaign Structure

**Campaign A (2000 eps, 4 seeds)** — Post-bug-fix validation on 20x20:

| Config | Purpose |
|--------|---------|
| mlpppo_small_5agents_competition_oracle | Baseline (no features) |
| mlpppo_small_5agents_competition_pheromone_oracle | Pheromones vs none |
| mlpppo_small_5agents_competition_social_oracle | Social feeding vs none |
| mlpppo_small_5agents_competition_full_oracle | Full stack |

**Campaign B1 (4000 eps, 4 seeds)** — Oracle classical strain:

| Config | Purpose |
|--------|---------|
| mlpppo_small_1agent_foraging_oracle | Single-agent ceiling |
| mlpppo_small_2agents_competition_oracle | 2-agent scaling |
| mlpppo_small_5agents_competition_oracle | 5-agent scaling |
| mlpppo_small_10agents_competition_oracle | 10-agent scaling |

**Campaign B2 (4000 eps, 4 seeds)** — Temporal classical strain:

| Config | Purpose |
|--------|---------|
| lstmppo_small_1agent_foraging_temporal | Single-agent temporal ceiling |
| lstmppo_small_5agents_competition_temporal | 5-agent temporal scaling (no pheromones) |

Note: `lstmppo_small_5agents_competition_temporal` is the **no-pheromone temporal baseline** — temporal chemotaxis + STAM only, no pheromone modules. It is reused in Campaign D as the control against the pheromone-enabled variant (same config, different comparison partner).

**Campaign C (2000 eps, 4 seeds)** — Collective predator response:

| Config | Purpose |
|--------|---------|
| mlpppo_small_5agents_pursuit_oracle | Alarm-enabled |
| mlpppo_small_5agents_pursuit_no_alarm_oracle | No-alarm control |

**Campaign D (4000 eps, 4 seeds)** — Temporal pheromone value:

| Config | Purpose |
|--------|---------|
| lstmppo_small_5agents_competition_temporal | No-pheromone temporal control (same as B2) |
| lstmppo_small_5agents_competition_pheromone_temporal | Full pheromone stack, temporal |

Campaign D reuses the B2 no-pheromone run data — no duplicate execution needed. The comparison is: does adding pheromone modules (food-marking, alarm, aggregation) improve temporal-mode multi-agent foraging?

### Decision 4: Game-Theoretic Indicators

Most indicators computed from CSV data — no new runtime code. One (best-response) requires additional simulation runs.

- **Food Gini trajectory**: Plot `food_gini_coefficient` from multi_agent_summary.csv over episodes. Increasing trend = competitive dynamics emerging. *(CSV post-hoc)*
- **Cooperation/competition ratio**: `food_sharing_events / max(food_competition_events, 1)` from summary CSV. Ratio > 1 = cooperative tendency, < 1 = competitive. *(CSV post-hoc)*
- **Territorial index trajectory**: Plot `territorial_index` over episodes. Increasing trend = spatial specialization. *(CSV post-hoc)*
- **Best-response analysis**: After Campaign B1, load trained 5-agent weights and run a modified session: 4 agents with `termination_policy: freeze` terminated at step 0 (present but inert), 1 agent active. Compare the active agent's food collection rate to its rate in the full 5-agent run. If rate changes significantly, agents have learned strategic interaction. *(Requires additional simulation runs with saved weights)*

### Decision 5: Logbook 011 Structure

Follow the template established by Logbooks 007-010:

```text
# Logbook 011: Multi-Agent Phase 4 Evaluation
## Study Overview
## Campaign A: Post-Bug-Fix Validation
## Campaign B: Classical Strain Measurement
## Campaign C: Collective Predator Response
## Campaign D: Temporal Pheromone Value
## Game-Theoretic Analysis
## Phase 4 Exit Criteria Assessment
## Quantum Checkpoint Assessment
## Go/No-Go Decision
## Conclusions
```

## Risks / Trade-offs

- **PPO exploration collapse persists**: If all agents collapse to the same behavior by episode 500, features appear neutral. This is a valid finding — the training algorithm is the bottleneck.
- **4000 episodes insufficient**: Learning curves may still be noisy. 4 seeds provide variance estimates; overlapping confidence intervals = "no statistically significant strain" (reported as limitation).
- **Pursuit + multi-agent untested on 20x20**: D1 validated multi-agent predator targeting on 50x50. The 20x20 pursuit configs are new and may reveal integration issues during sanity checks.
