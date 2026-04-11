## Why

Multi-agent infrastructure is complete (Deliverables 1-3 merged) but exit criteria are not met. No single-agent vs multi-agent comparison exists for classical strain measurement, pheromone value is untested with correct agent sensing, and no emergent behaviors have been documented.

Additionally, D1/D2 evaluation data was collected before the multi-agent sensing position bug fix (PR #113), so learning-related findings need re-validation.

## What Changes

### 1. Two New Metrics on MultiAgentEpisodeResult

**territorial_index (float)**: Spatial Gini coefficient of per-agent food collection positions. Measures whether agents specialize in foraging regions.

**alarm_response_rate (float)**: When an alarm pheromone is emitted, what fraction of nearby agents change movement direction within 5 steps? Measures causal alarm response.

### 2. Bug Fixes Discovered During Evaluation

- **#115**: Reward calculator used default agent position for all reward computation (same bug class as #112). Fixed with `agent_id` parameter throughout.
- **STAM dimension mismatch**: `get_classical_feature_dimension()` used static STAM dim (11) regardless of pheromone channels. Fixed with `_infer_stam_dim()` auto-detection.
- **food_sharing_buffer ungated**: `food_marking_buffer.append()` fired without pheromones enabled. Gated on `pheromones_enabled`.

### 3. CSV Export Enhancements

- `config_name` column added to both multi-agent CSV exports (session identification)
- Per-agent CSV expanded with `total_reward`, `success`, `satiety_remaining`, `foods_available`

### 4. Eighteen Evaluation Scenario Configs

Configs for 7 evaluation campaigns on 20×20 grids:

- Single-agent oracle/temporal baselines
- 2/5/10-agent oracle scaling series
- 5-agent pursuit with/without alarm pheromones
- 5-agent full-stack competition
- LSTM PPO temporal variants (proper Phase 3 brain hyperparams)
- Extreme scarcity ± social feeding
- Mixed phenotype (3 social + 2 solitary)
- Proportional food scaling (3 food per agent)

### 5. Seven Evaluation Campaigns

- **A-v2**: Post-bug-fix feature comparison (2000 eps, 4 seeds)
- **B1-v2**: Oracle classical strain scaling 1/2/5/10-agent (4000 eps, 4 seeds)
- **B2+D-v2**: Temporal strain + pheromone value with proper configs (2000 eps, 2 seeds)
- **C**: Collective predator response, alarm ± (2000 eps, 2 seeds)
- **E**: Social feeding under extreme scarcity (2000 eps, 2 seeds)
- **F**: Mixed phenotype competition (2000 eps, 2 seeds)
- **G**: Proportional food scaling (2000 eps, 2 seeds)

### 6. Logbook 011 & Roadmap Update

Evaluation logbook documenting all campaign results, exit criteria assessment, quantum checkpoint recommendation (do not trigger), and Go/No-Go decision.

### 7. Food Spatial Persistence (Deferred)

Food-marking pheromone evaluation deferred: uniform random food respawn makes trails point to stale locations. Issue #116 raised for food patches/hotspots. Added to roadmap as deferred deliverable.

## Capabilities

**Modified**: `multi-agent` (2 new metrics, bug fixes, CSV enhancements), `configuration-system` (18 scenario configs).

## Impact

**Core code:**

- `quantumnematode/agent/multi_agent.py` — territorial_index, alarm_response_rate, per_agent_reward, per_agent_satiety
- `quantumnematode/agent/reward_calculator.py` — agent_id parameter throughout (bug fix #115)
- `quantumnematode/brain/modules.py` — `_infer_stam_dim()`, `stam_dim_override` parameter
- `quantumnematode/env/env.py` — `get_nearest_food_distance_for()`, `get_nearest_predator_distance_for()`
- `quantumnematode/report/csv_export.py` — config_name column, per-agent CSV expansion, 2 new summary columns
- `scripts/run_simulation.py` — config_name passthrough, new CSV fields

**Configs:**

- `configs/scenarios/multi_agent_foraging/` — 12 new configs
- `configs/scenarios/multi_agent_pursuit/` — 2 new configs
- `configs/scenarios/foraging/` — 4 new single-agent baselines

**Docs:**

- `docs/experiments/logbooks/011-multi-agent-evaluation.md` — Evaluation results
- `docs/experiments/logbooks/supporting/011/` — Per-seed detail
- `docs/roadmap.md` — Exit criteria evidence, quantum checkpoint, Go/No-Go
- `artifacts/logbooks/011/` — CSV data, configs, weights

## Breaking Changes

None. New metrics have default values (0.0). New CSV columns are additive.

## Backward Compatibility

All new fields default to 0.0 or empty string. Existing configs and CSV parsers unaffected. Reward calculator `agent_id` defaults to "default" for single-agent backward compatibility.
