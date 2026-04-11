# 011: Multi-Agent Phase 4 — Coordination, Social Dynamics, and Pheromone Evaluation

**Status**: `in-progress`

**Branch**: `feat/phase4-evaluation-campaign`

**Date Started**: 2026-04-11

**Date Completed**: —

## Objective

Evaluate the complete Phase 4 multi-agent infrastructure (Deliverables 1-3) through a comprehensive campaign measuring: classical coordination strain, social feeding effectiveness, pheromone communication value, collective predator response, and emergent behavior potential. Assess Phase 4 exit criteria and quantum checkpoint trigger.

## Background

Phase 4 delivered multi-agent infrastructure in three deliverables:

- **D1**: Multi-agent orchestrator (2-10 agents, food competition, termination policies)
- **D2**: Pheromone communication (food-marking, alarm, oracle/temporal sensing, STAM 6-channel)
- **D3**: Social dynamics (social feeding via satiety decay reduction, aggregation pheromones, collective metrics, STAM 7-channel)

Two critical bugs were discovered and fixed during evaluation:

- **#112/PR #113**: All agents read sensory data from default agent's position (fixed before D3)
- **#115**: Reward calculator used default agent's position for distance reward, exploration bonus, goal bonus, anti-dithering, predator evasion, boundary collision, and temperature avoidance (fixed during this evaluation)

A third bug (STAM dimension mismatch with pheromone temporal modules) was found and fixed: `get_classical_feature_dimension()` now auto-infers correct STAM memory dimension from pheromone temporal modules in the sensory list.

**Prior work**: Scratchpad evaluations in D1-D3 (pre-bug-fix, largely invalidated). Logbook 009 (temporal sensing baselines). Logbook 010 (aerotaxis baselines).

## Hypotheses

1. Multi-agent coordination creates measurable performance strain versus single-agent baselines (Phase 4 exit criterion)
2. Social feeding (satiety decay reduction) improves survival when food is scarce
3. Pheromone communication (alarm, aggregation) improves foraging or survival in temporal sensing mode
4. Alarm pheromones improve collective predator survival
5. Emergent social behaviors arise from multi-agent training

## Method

### Bug Fixes Applied During Evaluation

All campaigns below were run after fixing the reward calculator (#115). Campaigns A and B1 were run twice: pre-fix data was invalidated and re-run.

Additional fixes applied mid-evaluation:

- `food_marking_buffer.append()` gated on `pheromones_enabled` (food_sharing_events no longer fires without pheromones)
- `config_name` column added to both multi-agent CSV exports
- STAM dimension auto-inference in `get_classical_feature_dimension()`
- Proper temporal configs using Phase 3 proven brain hyperparams (LR warmup/decay, entropy decay, critic_hidden=128, 400 satiety, 1000 max_steps)

### Campaigns

| Campaign | Configs | Episodes | Seeds | Grid | Question |
|----------|---------|----------|-------|------|----------|
| A-v2 | 4 × 5-agent oracle (baseline/pheromone/social/full) | 2000 | 4 | 20×20 | Feature comparison post-bug-fix |
| B1-v2 | 1/2/5/10-agent oracle | 4000 | 4 | 20×20 | Classical strain scaling |
| B2+D-v2 | 1-agent/5-agent temporal ± pheromones | 2000 | 2 | 20×20 | Temporal strain + pheromone value |
| C | 5-agent pursuit ± alarm pheromone | 2000 | 2 | 20×20 | Collective predator response |
| E | 5-agent extreme scarcity ± social feeding | 2000 | 2 | 20×20 | Social feeding under starvation |
| F | 5-agent mixed phenotype (3 social + 2 solitary) | 2000 | 2 | 20×20 | Phenotype competition |
| G | 1/2/5-agent proportional food | 2000 | 2 | 20×20 | True coordination overhead |

## Results

### Campaign B1-v2: Classical Strain Scaling (Oracle)

| Agents | Food/Agent/Ep | vs Single-Agent | L100 Trend |
|--------|--------------|-----------------|------------|
| 1 | 9.99 | baseline | ↑ |
| 2 | 9.99 | 0% | →/↑ |
| 5 | 9.62 | -3.7% | →/↑ |
| 10 | 8.02 | -19.7% | →/↑ |

### Campaign G: Proportional Food Scaling (Oracle)

| Agents | Food on Grid | Food/Agent/Ep | Steps |
|--------|-------------|---------------|-------|
| 1 | 3 | 9.98 | 153 |
| 2 | 6 | 9.99 | 139 |
| 5 | 15 | 10.00 | 125 |

### Campaign B2+D-v2: Temporal Sensing

| Config | Food/Agent/Ep | F100 | L100 | Trend |
|--------|--------------|------|------|-------|
| 1-agent temporal | 8.33 | 373 | 984 | ↑↑ |
| 5-agent no-pheromone | 9.52 | 1921 | 4998 | ↑↑ |
| 5-agent pheromone (alarm+agg) | 9.54 | 2113 | 4967 | ↑↑ |

### Campaign E: Social Feeding Under Extreme Scarcity

| Config | Food/Agent/Ep | Survival Steps | Agg Index | Gini |
|--------|--------------|----------------|-----------|------|
| No social (control) | 3.43 | 153 | 0.872 | 0.499 |
| Social feeding ON | **4.63 (+35%)** | **223 (+46%)** | **0.905** | **0.409** |

### Campaign C: Collective Predator Response (Oracle)

| Config | Food/Agent/Ep | Predator Deaths | Alarm Response Rate |
|--------|--------------|-----------------|---------------------|
| Alarm ON | 8.49 | 19.1% | 81.6% |
| Alarm OFF | 8.43 | 20.0% | — |

### Campaign A-v2: Feature Comparison (Oracle, 5-agent)

| Group | Food/Ep | Agg Index |
|-------|---------|-----------|
| No social | 48.6 | 0.801 |
| Social | 48.5 | 0.798 |

### Campaign F: Mixed Phenotype

| Phenotype | Food/Ep | Steps | Satiety Remaining |
|-----------|---------|-------|-------------------|
| Social (3 agents) | 9.44 | 232 | 190.6 |
| Solitary (2 agents) | 9.64 | 230 | 181.9 |

## Analysis

### Finding 1: Temporal Collective Exploration Advantage

The most scientifically novel result. Multi-agent effect reverses between sensing modes:

- **Oracle**: 5-agent is -3.7% vs single-agent (mild resource competition)
- **Temporal**: 5-agent is **+14.3%** vs single-agent (collective exploration benefit)

Mechanism: more agents executing independent random walks increases collective probability of encountering food in information-scarce environments. The temporal sensing gap nearly closes at multi-agent scale (9.52 vs oracle 9.62 — only 1% difference). This is a statistical parallel-search effect, not learned coordination.

**Significance**: Demonstrates that multi-agent value depends critically on sensing fidelity. Information-rich environments create competition; information-poor environments create collective benefit.

### Finding 2: Social Feeding Works Under Scarcity Pressure

Campaign E produced the clearest positive result: **+35% food, +46% survival, +3.8% aggregation** under extreme scarcity (1 food item, 2x satiety decay). The npr-1-inspired decay reduction (0.7x near conspecifics) provides genuine survival advantage when starvation is the dominant threat. Under abundant food (Campaign A), the effect vanishes — agents don't need energy conservation when food is plentiful.

**Significance**: The biologically-motivated social feeding mechanic works as designed. The conditions must match the biological scenario (scarce resources, starvation pressure) for the effect to manifest.

### Finding 3: Zero Coordination Overhead With Proportional Resources

Campaign G definitively showed that with food scaled proportionally to agent count (3 food per agent), MLP PPO achieves identical per-agent performance at 1, 2, and 5 agents (9.98-10.00 food/agent/ep). Episodes actually get shorter with more agents (more food items = shorter paths).

**Significance**: The B1 "classical strain" (19.7% at 10 agents) is entirely resource scarcity, not coordination difficulty. Classical MLP PPO has no computational overhead from multi-agent coordination in oracle mode. The quantum checkpoint trigger ("ceiling <85%") is not met by genuine coordination complexity.

### Finding 4: Pheromones Are Consistently Neutral

Across all campaigns and sensing modes, pheromone signals (alarm, aggregation) showed no measurable benefit:

- Oracle foraging: neutral (food gradients make pheromone trails redundant)
- Temporal foraging: neutral (collective exploration already compensates)
- Oracle pursuit: neutral (nociception provides direct predator sensing)

Food-marking pheromones could not be evaluated: uniform random food respawn makes trails point to stale locations (issue #116). This is a design limitation requiring food spatial persistence for meaningful evaluation.

**Significance**: Pheromone infrastructure works correctly but the current environment design doesn't create conditions where indirect chemical communication adds value beyond direct sensing or collective exploration.

### Finding 5: Two Critical Bugs Dominated Early Results

Issues #112 and #115 (same bug class: backward-compat properties in multi-agent context) caused all agents to receive wrong sensory inputs and wrong rewards. Pre-fix evaluations showed "1000x performance gap" that was entirely artificial. Post-fix, multi-agent agents learn effectively (9.62 food/agent/ep at 5 agents vs 9.99 single-agent ceiling).

## Hypothesis Outcomes

| # | Hypothesis | Result |
|---|-----------|--------|
| 1 | Multi-agent creates measurable coordination strain | **Partially supported** — strain exists with scarce resources (-19.7% at 10 agents) but is zero with proportional resources. Strain is resource competition, not coordination complexity. |
| 2 | Social feeding improves survival under scarcity | **Supported** — +35% food, +46% survival under extreme scarcity (Campaign E) |
| 3 | Pheromones improve temporal foraging/survival | **Not supported** — neutral across all tested scenarios |
| 4 | Alarm pheromones improve collective predator survival | **Not supported** — neutral in oracle pursuit mode |
| 5 | Emergent social behaviors arise | **Not supported** — mechanical effects observed (clustering from social feeding), but no learned social strategies |

## Phase 4 Exit Criteria Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| ≥5 agents running stably | ✅ Met | 5 and 10-agent configs run reliably across all campaigns |
| ≥1 emergent behavior documented | ⚠️ Partially met | Temporal collective exploration advantage is emergent (not programmed). Social feeding clustering is mechanical, not learned. |
| Pheromone communication functional | ✅ Met | Infrastructure works correctly. Neutral results are valid findings. |
| Classical strain on coordination | ⚠️ Partially met | Strain from resource scarcity (-19.7% at 10 agents) but zero genuine coordination overhead with proportional resources |

## Quantum Checkpoint Assessment

**Trigger condition**: Classical approaches show measurable difficulty on coordination tasks (aspirational: ceiling <85%).

**Result**: The 80.2% of ceiling at 10 agents (B1-v2) meets the numerical threshold, but Campaign G proved this is resource allocation difficulty, not computational complexity. With proportional resources, classical MLP PPO achieves 100% of ceiling at all scales. There is no exponential state space explosion or coordination-specific difficulty for quantum approaches to exploit.

**Recommendation**: Do not trigger quantum multi-agent evaluation at this time. The multi-agent environment does not yet create genuine coordination complexity. Conditions that could create it: food spatial persistence requiring learned communication (issue #116), satiety-dependent foraging requiring learned strategy switching, or partial observability requiring information sharing.

## Conclusions

1. **Multi-agent infrastructure works correctly** (post-bug-fix) — agents learn, forage, and survive effectively at scales up to 10 agents.

2. **Temporal collective exploration advantage** is the most novel finding — multi-agent improves per-agent performance in information-scarce environments through parallel random search, with the temporal sensing gap nearly closing at multi-agent scale.

3. **Social feeding provides genuine survival advantage under scarcity** — +35% food, +46% survival with 0.7x decay reduction when food is scarce and starvation pressure is high.

4. **No genuine coordination overhead exists** in oracle mode with proportional resources — the "classical strain" finding was entirely resource scarcity.

5. **Pheromone communication is neutral** across all tested scenarios — the environment doesn't create conditions where indirect chemical signals add value beyond direct sensing.

6. **No emergent learned social strategies** observed — all positive effects are mechanical (decay reduction) or statistical (parallel search), not learned coordination.

7. **Two critical bugs** (#112, #115) invalidated pre-fix evaluation data and highlight the importance of correctness verification before performance evaluation.

## Future Work (Open)

The following items are planned for a subsequent PR to enable deeper multi-agent evaluation:

- **Food spatial persistence** (issue #116): Food patches/hotspots or biased respawn to make food-marking pheromone trails valid signals
- **Satiety-dependent foraging**: Agents learn to stop seeking food when sated, enabling patch-sharing dynamics and preventing local food monopoly
- **Re-evaluation of pheromone food-marking** once food spatial persistence is implemented
- **Temporal nociception + alarm pheromones**: Test alarm pheromone value when predator sensing is indirect
- **Formal Nash/game-theoretic analysis**: Deferred until agents demonstrate stable converged strategies

## Data References

- Scratchpad: `tmp/evaluations/multi_agent_infrastructure/multi_agent_scratchpad.md`
- Campaign data: `exports/20260410_*` (Campaign A-v2, B1-v2), `exports/20260411_*` (B2+D-v2, C, E, F, G)
- Config files: `configs/scenarios/multi_agent_foraging/`, `configs/scenarios/multi_agent_pursuit/`, `configs/scenarios/foraging/`
- Supporting detail: `docs/experiments/logbooks/supporting/011/`
- Issues: #112 (sensing position), #115 (reward calculator), #116 (food spatial persistence)
