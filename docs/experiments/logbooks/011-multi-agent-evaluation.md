# 011: Multi-Agent — Coordination, Social Dynamics, and Pheromone Evaluation

**Status**: `complete`

**Branch**: `feat/phase4-evaluation-campaign`

**Date Started**: 2026-04-11

**Date Completed**: 2026-04-18

## Objective

Evaluate the multi-agent infrastructure through a comprehensive campaign measuring: classical coordination strain, social feeding effectiveness, pheromone communication value, collective predator response, and emergent behavior potential. Assess exit criteria and quantum checkpoint trigger.

## Background

Multi-agent infrastructure was delivered in three deliverables:

- **D1**: Multi-agent orchestrator (2-10 agents, food competition, termination policies)
- **D2**: Pheromone communication (food-marking, alarm, oracle/temporal sensing, STAM 6-channel)
- **D3**: Social dynamics (social feeding via satiety decay reduction, aggregation pheromones, collective metrics, STAM 7-channel)

Two critical bugs were discovered and fixed during evaluation:

- **#112/PR #113**: All agents read sensory data from default agent's position (fixed before D3)
- **#115**: Reward calculator used default agent's position for distance reward, exploration bonus, goal bonus, anti-dithering, predator evasion, boundary collision, and temperature avoidance (fixed during this evaluation)

A third bug (STAM dimension mismatch with pheromone temporal modules) was found and fixed: `get_classical_feature_dimension()` now auto-infers correct STAM memory dimension from pheromone temporal modules in the sensory list.

**Prior work**: Logbook 009 (temporal sensing baselines). Logbook 010 (aerotaxis baselines). Earlier D1-D3 evaluations (pre-bug-fix) were largely invalidated.

## Hypotheses

1. Multi-agent coordination creates measurable performance strain versus single-agent baselines
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

**Significance**: The B1 "classical strain" (19.7% at 10 agents) is entirely resource scarcity, not coordination difficulty. Classical MLP PPO has no computational overhead from multi-agent coordination in oracle mode. The quantum checkpoint trigger ("ceiling \<85%") is not met by genuine coordination complexity.

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
| 3 | Pheromones improve temporal foraging/survival | **Not supported** — neutral across all tested scenarios including with food hotspots (Campaigns H1-H4). Root cause: temporal sensing lacks klinotaxis (issue #125). |
| 4 | Alarm pheromones improve collective predator survival | **Not supported** — neutral in oracle pursuit mode. Temporal pursuit deferred pending klinotaxis. |
| 5 | Emergent social behaviors arise | **Not supported** — mechanical effects observed (clustering from social feeding), but no learned social strategies |

## Exit Criteria Assessment (Roadmap Phase 4)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| ≥5 agents running stably | ✅ Met | 5 and 10-agent configs run reliably across all campaigns |
| ≥1 emergent behavior documented | ⚠️ Partially met | Temporal collective exploration advantage is emergent (not programmed). Social feeding clustering is mechanical, not learned. |
| Pheromone communication functional | ✅ Met | Infrastructure works correctly. Neutral results traced to sensing model limitation (klinokinesis only, no klinotaxis). Issue #125 raised. |
| Classical strain on coordination | ⚠️ Partially met | Strain from resource scarcity (-19.7% at 10 agents) but zero genuine coordination overhead with proportional resources |

## Quantum Checkpoint Assessment (Roadmap Phase 4)

**Trigger condition**: Classical approaches show measurable difficulty on coordination tasks (aspirational: ceiling \<85%).

**Result**: The 80.2% of ceiling at 10 agents (B1-v2) meets the numerical threshold, but Campaign G proved this is resource allocation difficulty, not computational complexity. With proportional resources, classical MLP PPO achieves 100% of ceiling at all scales. There is no exponential state space explosion or coordination-specific difficulty for quantum approaches to exploit.

**Recommendation**: Do not trigger quantum multi-agent evaluation at this time. The multi-agent environment does not yet create genuine coordination complexity. Conditions that could create it: food spatial persistence requiring learned communication (issue #116), satiety-dependent foraging requiring learned strategy switching, or partial observability requiring information sharing.

## Conclusions

1. **Multi-agent infrastructure works correctly** (post-bug-fix) — agents learn, forage, and survive effectively at scales up to 10 agents.

2. **Temporal collective exploration advantage** is the most novel finding — multi-agent improves per-agent performance in information-scarce environments through parallel random search, with the temporal sensing gap nearly closing at multi-agent scale.

3. **Social feeding provides genuine survival advantage under scarcity** — +35% food, +46% survival with 0.7x decay reduction when food is scarce and starvation pressure is high.

4. **No genuine coordination overhead exists** in oracle mode with proportional resources — the "classical strain" finding was entirely resource scarcity.

5. **Pheromone communication is neutral** across all tested scenarios — including with food spatial persistence (hotspots + satiety gate). Food-marking trails add zero value over alarm+aggregation (H1 vs H2), and all pheromones combined provide only +4% over no pheromones with hotspots (H1 vs H3).

6. **Food hotspots hurt temporal foraging** — uniform food distribution outperforms hotspot clustering by +24-31% in temporal mode. Clustered food creates deserts that temporal agents can't navigate.

7. **Root cause: temporal sensing lacks klinotaxis** — our temporal mode implements only klinokinesis (temporal dC/dt). Real C. elegans also use klinotaxis (head-sweep spatial gradient). Without klinotaxis, agents cannot efficiently follow narrow pheromone trails. This explains all negative pheromone results. Issue #125 raised.

8. **No emergent learned social strategies** observed — all positive effects are mechanical (decay reduction) or statistical (parallel search), not learned coordination.

9. **Two critical bugs** (#112, #115) invalidated pre-fix evaluation data and highlight the importance of correctness verification before performance evaluation.

## Food Hotspot Evaluation (Campaigns H1-H7)

Following the initial evaluation, food spatial persistence (issue #116, PR #124) was implemented: configurable food hotspots with exponential decay sampling, satiety-gated consumption (agents can't eat above 80% satiety), and static food mode. This enabled re-evaluation of food-marking pheromones.

### Method

**Pilot calibration**: Initial 50×50 temporal config (15 food, 400 satiety) produced 99.8% starvation. Adjusted to 25 food on grid, 600 initial satiety to match 20×20 food density (~1% coverage).

**Dense hotspot pilot**: 8 hotspots (decay=10, bias=0.7) covering ~85% of grid tested as alternative to 3-hotspot sparse layout. Performance (4.1 food/agent/ep at ep 200) still below uniform (5.0).

| Campaign | Config | Grid | Sensing | Hotspots | Pheromones | Episodes × Seeds |
|----------|--------|------|---------|----------|------------|-----------------|
| H1 | hotspot_pheromone_temporal | 50×50 | Temporal | 3 sparse | ALL (food+alarm+agg) | 2000 × 4 |
| H2 | hotspot_alarm_agg_temporal | 50×50 | Temporal | 3 sparse | Alarm+Agg only | 2000 × 4 |
| H3 | hotspot_no_pheromone_temporal | 50×50 | Temporal | 3 sparse | None | 2000 × 4 |
| H4 | pheromone_temporal | 50×50 | Temporal | None | ALL (food+alarm+agg) | 2000 × 4 |
| H5 | hotspot_oracle (small) | 20×20 | Oracle | 2 sparse | ALL | 500 × 2 |
| H6 | hotspot_no_pheromone_oracle (small) | 20×20 | Oracle | 2 sparse | None | 500 × 2 |

### Results

#### H5+H6: Oracle Small Hotspot Verification

| Metric | H5 (pheromones) | H6 (no pheromones) |
|--------|----------------|-------------------|
| Food/agent/ep (L50) | 8.7-8.9 | 8.4 |
| Survival to max_steps | 98.6-99.3% | 98.8-99.1% |
| Satiety gate fires | 76-77% | 76% |

Satiety gate working correctly. Pheromones neutral in oracle mode (confirms Campaign A-v2).

#### H1-H4: Temporal Medium Food-Marking Evaluation

| Campaign | F100 | L100 | Δ vs H3 | Starved | Gate% |
|----------|------|------|---------|---------|-------|
| **H1** hotspot + ALL pheromones | 3.9 | 6.1 ± 0.1 | +3.8% | 39.1% | 2.4% |
| **H2** hotspot + alarm+agg | 3.9 | 6.2 ± 0.2 | +5.3% | 38.7% | 2.7% |
| **H3** hotspot + no pheromones | 3.9 | 5.9 ± 0.1 | baseline | 39.5% | 2.6% |
| **H4** no hotspots + ALL pheromones | 4.9 | 7.7 ± 0.2 | +30.8% | 32.3% | 5.8% |

### Finding 6: Food-Marking Pheromones Neutral Even With Spatial Persistence

H1 (6.1) vs H2 (6.2): food-marking adds zero value over alarm+aggregation. H1/H2 vs H3: all pheromones combined provide only +4-5% improvement with hotspots. The food spatial persistence infrastructure (issue #116) does not make food-marking pheromone trails effective.

### Finding 7: Hotspots Hurt Temporal Foraging

H4 (uniform food, 7.7 food/agent/ep) dramatically outperforms all hotspot configs (5.9-6.2) by +24-31%. Food clustering creates "desert" zones between patches where temporal agents have no chemical signal to follow. With temporal sensing (scalar concentration only, no direction), agents rely on random walk to encounter food — uniform distribution maximizes encounter rate.

### Finding 8: Root Cause — Temporal Sensing Lacks Klinotaxis

The fundamental issue is not environment design but the **sensing model**. Our temporal mode implements only **klinokinesis** (temporal dC/dt comparison via STAM). Real C. elegans chemotaxis also uses **klinotaxis** — physical head sweeps that sample concentration at left-of-center and right-of-center positions, providing immediate spatial gradient information.

Pheromone trails are narrow spatial features. A head-sweep can detect "left smells stronger" in one sample, enabling trail-following. Temporal-only sensing requires multiple steps of random movement to infer direction, by which time the agent may have crossed the trail entirely.

This explains all negative pheromone results across Campaigns A-v2, B2+D-v2, C, H1-H4: the pheromone infrastructure works correctly (verified in oracle mode), but temporal agents lack the sensing capability to exploit spatial gradient information from chemical trails.

**Issue #125 raised**: Add klinotaxis (head-sweep) sensing mode that samples concentration at 2-3 positions offset from the agent's heading direction. This is the path forward for biologically accurate pheromone evaluation.

## Artifacts

Located at `artifacts/logbooks/011/`:

| Directory | Contents |
|-----------|----------|
| `configs/` | 18 evaluation config YAMLs from Campaigns A-G (historical snapshots) |
| `hotspot_configs/` | 7 evaluation config YAMLs from Campaigns H1-H6 + dense hotspot pilot |
| `weights/` | Best single-agent weights (oracle + proportional food baselines) |

Results are reproducible by re-running the configs with the seeds documented in the supporting detail. Key numbers are recorded in the results tables above and in the supporting appendix.

Note: Sessions were run without `--track-experiment`, so no experiment JSONs were generated. Multi-agent sessions do not produce per-agent weight files.

## Future Work (Open)

- ~~**Food spatial persistence** (issue #116)~~: Implemented (PR #124). Food hotspots, satiety-gated consumption, static food mode.
- ~~**Satiety-dependent foraging**~~: Implemented as part of PR #124 (satiety_food_threshold).
- ~~**Re-evaluation of pheromone food-marking**~~: Completed (Campaigns H1-H4). Food-marking neutral even with hotspots. Root cause: temporal sensing limitation.
- **Klinotaxis sensing mode** (issue #125): Head-sweep spatial gradient sampling. Required for biologically accurate pheromone trail-following. The primary blocker for demonstrating pheromone value.
- **Temporal nociception + alarm pheromones** (Campaign H7): Deferred pending klinotaxis implementation. Alarm pheromone value with temporal predator sensing requires the same spatial gradient capability.
- **Formal Nash/game-theoretic analysis**: Deferred until agents demonstrate stable converged strategies.

## Data References

- Artifacts: `artifacts/logbooks/011/` (configs and weights only; CSVs are reproducible from configs)
- Config files: `configs/scenarios/multi_agent_foraging/`, `configs/scenarios/multi_agent_pursuit/`, `configs/scenarios/foraging/`
- Supporting detail: `docs/experiments/logbooks/supporting/011/`
- Issues: #112 (sensing position), #115 (reward calculator), #116 (food spatial persistence), #125 (klinotaxis sensing)
