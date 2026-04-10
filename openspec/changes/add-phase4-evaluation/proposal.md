## Why

Phase 4 infrastructure is complete (Deliverables 1-3 merged) but exit criteria are not met:

1. **"≥1 emergent behavior documented"** — Social feeding creates mechanical clustering (25% survival extension, 7.4% higher aggregation index) but agents haven't *learned* to exploit it. Longer training is needed to determine if emergent social strategies develop.

2. **"Classical approaches show measurable strain on coordination"** — No single-agent vs multi-agent comparison exists. This is the critical missing measurement for the quantum checkpoint trigger.

Additionally, D1/D2 evaluation data was collected before the multi-agent sensing position bug fix (PR #113), so learning-related findings need re-validation with correct agent sensing.

The remaining roadmap deliverables (D4: Competitive Foraging, D5: Collective Predator Response) are analysis-heavy, not infrastructure-heavy. "Territorial behavior" and "Nash equilibria" are patterns to measure. "Coordinated evasion" emerges from alarm pheromones that already exist. What's missing is targeted metrics, evaluation scenarios, and a rigorous campaign producing the data needed to close Phase 4.

## What Changes

### 1. Two New Metrics on MultiAgentEpisodeResult

**territorial_index (float)**: Spatial Gini coefficient of per-agent food collection positions. Measures whether agents specialize in foraging regions. 0 = all agents forage everywhere equally, 1 = each agent monopolizes a distinct zone.

**alarm_response_rate (float)**: When an alarm pheromone is emitted by a damaged agent, what fraction of nearby agents (within `social_detection_radius`) change movement direction within 5 steps? Measures causal alarm response — distinct from existing `alarm_evasion_events` which counts zone exits regardless of cause.

### 2. Game-Theoretic Indicators (Computed from Existing + New Metrics)

No new code — these are analysis patterns applied to CSV data during evaluation:

- **Food Gini trajectory**: Resource inequality over training episodes
- **Best-response analysis**: Does a single agent's food collection change when other agents are frozen vs active?
- **Cooperation/competition ratio**: food_sharing_events / food_competition_events over training
- **Territorial index trajectory**: Spatial specialization over training

### 3. Nine New Scenario Configs

Evaluation-focused configs on 20x20 grids (where agents actually interact):

- Single-agent oracle/temporal baselines for classical strain measurement
- 2/5/10-agent oracle scaling series
- 5-agent pursuit with/without alarm pheromones
- 5-agent full-stack competition (all features enabled)
- LSTM PPO temporal variants for pheromone value assessment

### 4. Comprehensive Evaluation Campaign (Campaigns A-D)

- **A**: Post-bug-fix validation (2000 eps, 4 seeds)
- **B**: Classical strain — single vs multi-agent scaling in oracle (B1) and temporal (B2) modes (4000 eps, 4 seeds)
- **C**: Collective predator response — alarm-enabled vs disabled (2000 eps, 4 seeds)
- **D**: Temporal pheromone value — no-pheromone vs full-stack in temporal mode (4000 eps, 4 seeds)

### 5. Logbook 011 & Phase 4 Conclusion

Formal evaluation logbook documenting: learning curves, classical strain quantification, game-theoretic indicators, emergent behavior assessment, collective predator response. Roadmap update with exit criteria evidence and quantum checkpoint Go/No-Go.

## Capabilities

**Modified**: `multi-agent` (2 new metrics, 2 new CSV columns), `configuration-system` (9 new scenario configs).

## Impact

**Core code:**

- `quantumnematode/agent/multi_agent.py` — territorial_index, alarm_response_rate tracking + MultiAgentEpisodeResult
- `quantumnematode/report/csv_export.py` — 2 new summary CSV columns
- `scripts/run_simulation.py` — Pass new fields to summary writer

**Configs:**

- `configs/scenarios/multi_agent_foraging/` — 4 new configs (2-agent, 10-agent, full-stack, temporal pheromone)
- `configs/scenarios/multi_agent_pursuit/` — 2 new configs (alarm-enabled, alarm-disabled)
- `configs/scenarios/foraging/` — 2 new single-agent baselines (oracle, temporal) with environment params matching multi-agent competition configs
- `configs/scenarios/multi_agent_foraging/` — 1 new temporal no-pheromone config

**Docs:**

- `docs/experiments/logbooks/011-multi-agent-evaluation.md` — Evaluation results
- `docs/roadmap.md` — Phase 4 conclusion, exit criteria evidence

## Breaking Changes

None. New metrics have default values (0.0) — existing configs and CSV parsers unaffected.

## Backward Compatibility

When territorial_index and alarm_response_rate are not applicable (e.g., single-agent mode), they default to 0.0. Existing multi_agent_summary.csv consumers can ignore the new columns.

## Dependencies

None beyond existing infrastructure (D1-D3).
