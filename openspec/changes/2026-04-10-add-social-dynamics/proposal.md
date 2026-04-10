## Why

Deliverables 1 (multi-agent infrastructure) and 2 (pheromone communication) are merged. Evaluation showed pheromones are neutral in oracle mode — direct food gradients make pheromone trails redundant, a valid scientific finding. The remaining Phase 4 exit criteria are: (1) at least one emergent behavior documented, and (2) classical approaches show measurable strain on coordination tasks. Neither can be met with the current feature set because agents have no incentive to aggregate and no metrics to measure collective behavior.

Real C. elegans exhibits social feeding mediated by the npr-1 neuropeptide receptor: social animals (npr-1 loss-of-function) reduce locomotion and increase pharyngeal pumping on bacterial lawns when near conspecifics, effectively conserving energy. They also continuously emit ascaroside pheromones that attract nearby worms — the biological basis for aggregation on food patches. These two mechanisms create the aggregation pressure needed for emergent social behaviors.

This deliverable adds three features: (1) social feeding via satiety decay reduction when near conspecifics, (2) aggregation pheromones as a biologically honest "where are others" signal, and (3) collective behavior metrics for measuring emergent phenomena (aggregation index, alarm evasion, food sharing).

## What Changes

### 1. Social Feeding (Satiety Decay Reduction)

New `SocialFeedingParams` dataclass on `DynamicForagingEnvironment`:

- When enabled and `nearby_agents_count > 0`, the per-step satiety decay rate is multiplied by a configurable reduction factor (default 0.7 — 30% slower energy burn)
- Detection radius configurable (default 5 Manhattan distance)
- Per-agent phenotype support: `social` (gets decay reduction) vs `solitary` (configurable neutral or penalty multiplier) — models npr-1 genetic variation
- This is an environmental mechanic, not a reward signal — agents must learn that staying near others extends survival

### 2. Aggregation Pheromone

Third `PheromoneType.AGGREGATION` extending the existing pheromone field system:

- Continuous emission: every alive agent emits every step (unlike event-driven food-marking/alarm)
- Models ascaroside pheromones that attract conspecifics to bacterial lawns
- Defaults tuned for "current presence": `emission_strength=0.5, spatial_decay_constant=10.0, temporal_half_life=10.0, max_sources=200`
- Short half-life ensures field reflects where agents are now, not where they were
- Two new sensing modules following established pattern: `PHEROMONE_AGGREGATION` (oracle: gradient strength + direction) and `PHEROMONE_AGGREGATION_TEMPORAL` (scalar concentration + dC/dt)
- STAM extends to 7 channels when aggregation enabled (MEMORY_DIM 17)

### 3. Collective Behavior Metrics

Four new fields on `MultiAgentEpisodeResult` for measuring emergent phenomena:

- `social_feeding_events`: count of step-agent pairs where decay reduction was applied
- `aggregation_index`: mean normalized inverse pairwise distance across all steps (0=dispersed, 1=clustered)
- `alarm_evasion_events`: count of agents that moved away from alarm pheromone gradient
- `food_sharing_events`: count of non-emitter agents approaching a food-marking pheromone source within N steps

### 4. Configuration

```yaml
environment:
  social_feeding:
    enabled: true
    decay_reduction: 0.7
    detection_radius: 5
    solitary_decay: 1.0

  pheromones:
    enabled: true
    aggregation:
      emission_strength: 0.5
      spatial_decay_constant: 10.0
      temporal_half_life: 10
      max_sources: 200

sensing:
  pheromone_aggregation_mode: oracle
```

Per-agent phenotype override in multi-agent config:

```yaml
multi_agent:
  agents:
    - brain_type: mlpppo
      social_phenotype: social
    - brain_type: mlpppo
      social_phenotype: solitary
```

## Capabilities

**New**: `social-dynamics` — social feeding mechanic, aggregation pheromones, collective behavior metrics, npr-1 phenotype variation.

**Modified**: `environment-simulation` (SocialFeedingParams, aggregation pheromone field), `brain-architecture` (BrainParams aggregation fields, 2 sensing modules), `multi-agent` (decay reduction in step loop, continuous emission, collective metrics), `configuration-system` (SocialFeedingConfig, aggregation pheromone config, phenotype support).

## Impact

**Core code:**

- `quantumnematode/env/env.py` — SocialFeedingParams, aggregation pheromone field + methods
- `quantumnematode/env/pheromone.py` — AGGREGATION added to PheromoneType enum
- `quantumnematode/env/__init__.py` — Export SocialFeedingParams
- `quantumnematode/agent/multi_agent.py` — Decay reduction, aggregation emission, collective metrics, MultiAgentEpisodeResult extension
- `quantumnematode/agent/agent.py` — Aggregation pheromone in _create_brain_params + _compute_temporal_data
- `quantumnematode/agent/stam.py` — 7-channel mode (MEMORY_DIM 17)
- `quantumnematode/brain/arch/_brain.py` — 4 new BrainParams fields
- `quantumnematode/brain/modules.py` — PHEROMONE_AGGREGATION + PHEROMONE_AGGREGATION_TEMPORAL modules
- `quantumnematode/utils/config_loader.py` — SocialFeedingConfig, aggregation config, phenotype support
- `quantumnematode/report/csv_export.py` — New metric columns in multi_agent_summary.csv

**Configs:**

- `configs/scenarios/multi_agent_foraging/` — Social and aggregation variants

**Tests:**

- New tests for social feeding, aggregation pheromone, collective metrics

## Breaking Changes

- STAM `MEMORY_DIM` changes from 15 to 17 when aggregation pheromones enabled (dynamic). Existing 6-channel pheromone configs remain at MEMORY_DIM=15. Trained models from 6-channel STAM are incompatible with 7-channel mode.

## Backward Compatibility

When `social_feeding.enabled` is false (default) and no aggregation pheromone config is present, all behavior is identical to current code. STAM remains at 4 or 6 channels based on existing pheromone config. No existing configs are affected.

## Dependencies

None beyond existing NumPy. Aggregation pheromone uses existing PheromoneField class.
