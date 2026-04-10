## Why

The multi-agent infrastructure (Deliverable 1) provides oracle social proximity sensing — agents receive exact nearby-agent counts, which is biologically dishonest. Real C. elegans detects conspecifics via ascaroside pheromone concentration gradients using chemosensory neurons (ASK, ADL, ASI). Pheromone communication is the biologically honest replacement and the key enabler for emergent multi-agent behaviors.

Two roadmap exit criteria depend on this: (1) pheromone communication functional with at least alarm + food-marking, and (2) at least one emergent behavior documented. Food-marking pheromones enable information sharing (agents that find food "tell" others), and alarm pheromones enable collective threat avoidance (agents flee areas where others were damaged). These are the two most impactful C. elegans social behaviors for foraging and survival.

This change also bundles three deferred items from the infrastructure deliverable: CSV export with agent_id column, weight save/load round-trip test, and end-of-session summary table.

## What Changes

### 1. PheromoneField Class

New file `quantumnematode/env/pheromone.py` implementing a dynamic chemical field using point-source decay:

- `PheromoneType` enum: `FOOD_MARKING`, `ALARM`
- `PheromoneSource` dataclass: `position`, `pheromone_type`, `strength`, `emission_step`, `emitter_id`
- `PheromoneField` class managing active sources per type
- Concentration at any position via superposition of exponential decay: `C(P) = Σ strength_i * exp(-dist / spatial_decay) * exp(-age / temporal_decay)`
- Gradient via central differences (same pattern as OxygenField)
- Source pruning: expired sources removed each step (age > max_age)
- Same math as food gradients — O(S) per query where S = active sources

### 2. Pheromone Emission in MultiAgentSimulation

Event-driven emission integrated into the orchestrator step loop:

- **Food-marking**: deposited after food consumption in `_resolve_food_step()` at the consumed food position
- **Alarm**: emitted after predator damage in the predator phase when agent HP decreases
- Per-step field update: prune expired sources, advance step counter
- Configurable emission strength per pheromone type

### 3. Pheromone Sensing Modules

Four new sensory modules following the established oracle/temporal pattern:

- `PHEROMONE_FOOD` (oracle): gradient strength + direction toward food-marking pheromones (classical_dim=2)
- `PHEROMONE_ALARM` (oracle): gradient strength + direction toward alarm pheromones (classical_dim=2)
- `PHEROMONE_FOOD_TEMPORAL`: scalar concentration + dC/dt (classical_dim=2)
- `PHEROMONE_ALARM_TEMPORAL`: scalar concentration + dC/dt (classical_dim=2)

### 4. BrainParams Extension

Eight new optional fields for pheromone sensing:

- Oracle: `pheromone_food_gradient_strength`, `pheromone_food_gradient_direction`, `pheromone_alarm_gradient_strength`, `pheromone_alarm_gradient_direction`
- Temporal: `pheromone_food_concentration`, `pheromone_alarm_concentration`
- Derivative: `pheromone_food_dconcentration_dt`, `pheromone_alarm_dconcentration_dt`

### 5. STAM Extension (4 → 6 channels)

Extend the Short-Term Associative Memory buffer to support pheromone channels:

- New channels: pheromone_food (index 4), pheromone_alarm (index 5)
- Dynamic `MEMORY_DIM`: 11 when pheromones disabled (4 channels), 15 when enabled (6 channels)
- Dynamic `num_channels`: 4 or 6 based on pheromone config
- Memory layout (6 channels): 6 weighted means + 6 derivatives + 2 position deltas + 1 action entropy
- Backward compatible: existing 4-channel configs produce identical 11-dim output

### 6. Environment Integration

`DynamicForagingEnvironment` gains pheromone field support:

- `PheromoneParams` dataclass with enabled flag, per-type config (emission_strength, spatial_decay_constant, temporal_half_life, max_sources)
- `pheromone_field_food: PheromoneField | None` and `pheromone_field_alarm: PheromoneField | None` instances
- Concentration/gradient methods: `get_pheromone_food_concentration()`, `get_pheromone_alarm_gradient()`, etc.
- `*_for(agent_id)` variants for multi-agent
- `emit_food_pheromone()` and `emit_alarm_pheromone()` methods

### 7. Configuration Schema

```yaml
environment:
  pheromones:
    enabled: true
    food_marking:
      emission_strength: 1.0
      spatial_decay_constant: 8.0
      temporal_half_life: 50
      max_sources: 100
    alarm:
      emission_strength: 2.0
      spatial_decay_constant: 5.0
      temporal_half_life: 20
      max_sources: 50

sensing:
  pheromone_food_mode: oracle
  pheromone_alarm_mode: oracle
```

### 8. CSV Export with agent_id (deferred from Deliverable 1)

- Add `agent_id` column to `_SIMULATION_RESULTS_FIELDNAMES` in `report/csv_export.py`
- Write per-agent rows from `MultiAgentEpisodeResult` in `_run_multi_agent`
- New `multi_agent_summary.csv` with per-episode aggregate metrics (total food, competition events, proximity events, Gini coefficient)
- End-of-session summary table printed to console

### 9. Weight Save/Load Round-Trip Test (deferred from Deliverable 1)

- Dedicated test verifying multi-agent weight persistence: save 3 agents → 3 .pt files → load back → verify model state matches

## Capabilities

**New**: `pheromone-communication` — pheromone field system, event-driven emission, biologically honest pheromone sensing, STAM pheromone channels, multi-agent CSV export.

**Modified**: `environment-simulation` (PheromoneField, pheromone methods), `brain-architecture` (BrainParams pheromone fields, 4 sensing modules), `configuration-system` (PheromoneConfig, SensingConfig pheromone modes), `multi-agent` (emission integration, CSV export).

## Impact

**Core code:**

- `quantumnematode/env/pheromone.py` — NEW: PheromoneSource, PheromoneType, PheromoneField
- `quantumnematode/env/env.py` — PheromoneParams, pheromone methods + \*\_for variants, emit methods
- `quantumnematode/brain/arch/_brain.py` — 8 new BrainParams pheromone fields
- `quantumnematode/brain/modules.py` — 4 new sensing modules (PHEROMONE_FOOD, PHEROMONE_ALARM + temporal variants)
- `quantumnematode/agent/stam.py` — Dynamic num_channels/MEMORY_DIM for pheromone support
- `quantumnematode/agent/multi_agent.py` — Pheromone emission in step loop, field update
- `quantumnematode/agent/agent.py` — Pheromone concentration/gradient in \_create_brain_params
- `quantumnematode/utils/config_loader.py` — PheromoneConfig, SensingConfig pheromone modes
- `quantumnematode/report/csv_export.py` — agent_id column, multi_agent_summary.csv
- `scripts/run_simulation.py` — CSV writes in \_run_multi_agent, session summary

**Configs:**

- `configs/scenarios/multi_agent_foraging/` — Pheromone-enabled variants

**Tests:**

- `tests/.../env/test_pheromone.py` — NEW: PheromoneField unit tests
- `tests/.../brain/test_pheromone_modules.py` — NEW: Pheromone sensing module tests
- `tests/.../agent/test_multi_agent.py` — Extended with pheromone emission tests

## Breaking Changes

- STAM `MEMORY_DIM` changes from 11 to 15 when pheromones enabled (dynamic). Existing configs without pheromones are unchanged (MEMORY_DIM remains 11). Trained models from 4-channel STAM are incompatible with 6-channel STAM.

## Backward Compatibility

When `pheromones.enabled` is false (default), all behavior is identical to current code. STAM uses 4 channels and MEMORY_DIM=11. No pheromone fields are created. All existing configs work unchanged. The CSV export changes are additive (new column, new file) and don't affect single-agent exports.

## Dependencies

None beyond existing NumPy. PheromoneField uses the same math as food gradients (already validated).
