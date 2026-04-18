## Why

Our temporal sensing mode implements only klinokinesis — temporal concentration comparison via STAM dC/dt. Real C. elegans chemotaxis uses two complementary strategies simultaneously:

1. **Klinokinesis** (what we have): temporal dC/dt modulating turning rate
2. **Klinotaxis** (what we need): physical head sweeps sampling concentration at left and right offsets from the heading direction. ASE neurons (ASEL/ASER) compare near-simultaneous readings for immediate spatial gradient information.

Without klinotaxis, agents cannot efficiently follow narrow pheromone trails or navigate spatial gradients using local information. This was identified as the root cause of all negative pheromone results in logbook 011 evaluation campaigns (A-H): the pheromone infrastructure works correctly (verified in oracle mode), but temporal agents lack the sensing capability to exploit spatial gradient information from chemical trails.

Klinotaxis applies to ALL environmental gradient modalities — not just chemical sensing. C. elegans uses head sweeps for thermotaxis (AFD neurons), aerotaxis (URX/BAG neurons), and all chemosensory modalities (ASE, AWC, AWA, ASH neurons).

Klinotaxis is the most biologically accurate sensing mode — it provides exactly what real C. elegans receive, using only local spatial information (±1 cell), not global gradient direction like oracle mode.

## What Changes

### 1. New Sensing Mode: `SensingMode.KLINOTAXIS`

Fourth sensing mode added to `SensingMode` enum. Each modality can independently use oracle, temporal, derivative, or klinotaxis mode.

Klinotaxis is a superset of temporal/derivative — for each modality the agent receives:

- **strength**: scalar concentration at agent position (same as temporal)
- **angle**: lateral gradient = `tanh((right - left) * lateral_scale)` — spatial gradient from head sweep
- **binary**: temporal derivative = `tanh(dC/dt * derivative_scale)` — temporal comparison from STAM

This gives `classical_dim=3` per module (vs 2 for temporal/derivative).

### 2. Head-Sweep Geometry

Sample 1 cell perpendicular to the agent's heading direction:

- UP heading: left=(x-1,y), right=(x+1,y)
- RIGHT heading: left=(x,y+1), right=(x,y-1)
- DOWN heading: left=(x+1,y), right=(x-1,y)
- LEFT heading: left=(x,y-1), right=(x,y+1)
- STAY: use last non-STAY heading (tracked on agent)
- Clamp to grid bounds at edges

1-cell offset matches C. elegans head width (~50μm) at the grid scale implied by gradient decay constants.

### 3. Seven Klinotaxis Sensory Modules

All modalities that have temporal variants get klinotaxis variants:

1. `food_chemotaxis_klinotaxis` — food (ASE/AWC neurons)
2. `nociception_klinotaxis` — predator (ASH neurons)
3. `thermotaxis_klinotaxis` — temperature (AFD neurons)
4. `aerotaxis_klinotaxis` — oxygen (URX/BAG neurons)
5. `pheromone_food_klinotaxis` — food-marking pheromone
6. `pheromone_alarm_klinotaxis` — alarm pheromone
7. `pheromone_aggregation_klinotaxis` — aggregation pheromone

### 4. Configuration

```yaml
sensing:
  chemotaxis_mode: klinotaxis
  nociception_mode: klinotaxis
  thermotaxis_mode: klinotaxis
  aerotaxis_mode: klinotaxis
  pheromone_food_mode: klinotaxis
  pheromone_alarm_mode: klinotaxis
  pheromone_aggregation_mode: klinotaxis
  lateral_scale: 50.0
  stam_enabled: true    # auto-enabled for klinotaxis (needed for dC/dt)
```

### 5. Evaluation Configs

Single-agent benchmarks (LSTMPPO GRU) for comparison against logbook 009/010 baselines:

- `foraging/lstmppo_small_klinotaxis.yml`
- `pursuit/lstmppo_small_klinotaxis.yml`
- `thermal_pursuit/lstmppo_large_klinotaxis.yml`
- `thermal_stationary/lstmppo_large_klinotaxis.yml`
- `oxygen_foraging/lstmppo_large_klinotaxis.yml`
- `oxygen_thermal_foraging/lstmppo_large_klinotaxis.yml`

Multi-agent pheromone evaluation configs:

- `multi_agent_foraging/lstmppo_medium_5agents_hotspot_pheromone_klinotaxis.yml`
- `multi_agent_foraging/lstmppo_medium_5agents_hotspot_no_pheromone_klinotaxis.yml`
- `multi_agent_foraging/lstmppo_medium_5agents_pheromone_klinotaxis.yml`

## Capabilities

**Modified**: `brain-architecture` (BrainParams lateral gradient fields, 7 new sensory modules), `environment-simulation` (lateral concentration queries), `configuration-system` (SensingMode.KLINOTAXIS, lateral_scale).

## Impact

**Core code:**

- `quantumnematode/utils/config_loader.py` — `SensingMode.KLINOTAXIS`, `lateral_scale` field, `apply_sensing_mode()` refinement
- `quantumnematode/brain/arch/_brain.py` — 7 lateral gradient fields + `lateral_scale` on BrainParams
- `quantumnematode/agent/agent.py` — `_compute_lateral_offsets()`, lateral sampling in `_compute_temporal_data()`, last heading tracking
- `quantumnematode/brain/modules.py` — 7 `ModuleName` entries + extraction functions + registration

**Documentation:**

- `README.md` — perception modes table, feature description
- `AGENTS.md` — `_klinotaxis` in sensing suffixes
- `CONTRIBUTING.md` — testing examples
- `docs/roadmap.md` — Phase 4 sensing notes
- `.claude/skills/nematode-run-experiments/skill.md` — episode guidance

**Configs:**

- 6 single-agent benchmark configs across foraging, pursuit, thermal, oxygen scenarios
- 3 multi-agent pheromone evaluation configs

**Tests:**

- Module registration and feature extraction tests
- Lateral offset computation tests (all directions + edge clamping)
- Integration tests (klinotaxis populates BrainParams correctly)
- Config loader tests (mode substitution, STAM auto-enable)

## Breaking Changes

None. All defaults unchanged. New mode must be explicitly selected via `*_mode: klinotaxis` in config.

## Backward Compatibility

Existing configs produce identical behavior. The `apply_sensing_mode()` refinement changes `!= ORACLE` checks to explicit mode matching, but TEMPORAL and DERIVATIVE continue mapping to `*_temporal` modules as before.

## Dependencies

None. Uses existing environment concentration query methods.
