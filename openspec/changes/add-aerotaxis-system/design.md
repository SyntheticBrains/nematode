## Context

The simulation currently supports three gradient-based sensory modalities: chemotaxis (food), nociception (predators), and thermotaxis (temperature). Oxygen sensing was listed as a Phase 3 deliverable but deferred when temporal sensing proved sufficient for exit criteria. All temporal sensing infrastructure is now operational — oracle, temporal, and derivative modes with STAM memory buffers — making oxygen integration straightforward.

The thermotaxis implementation (completed March 2026) established the pattern for adding new environmental fields: a standalone field class (`temperature.py`), environment integration (`env.py` with params/methods/effects), sensory modules (`modules.py` with core extractors), config classes (`config_loader.py`), and visualization (`sprites.py`, `pygame_renderer.py`, `theme.py`). Oxygen sensing follows this pattern with one key architectural difference: the zone system is asymmetric (a preferred range of 5-12% O2 rather than symmetric deltas from a cultivation point).

The user also requires combined thermal+oxygen environments where both fields coexist with food gradients, creating three-way multi-objective trade-offs.

### Current State

- **TemperatureField** (`env/temperature.py`): Linear gradient + hot/cold spots with exponential decay. 7 symmetric zones around Tc.
- **ThermotaxisParams** (`env/env.py`): Config dataclass with rewards/penalties/HP damage per zone.
- **Sensory modules** (`brain/modules.py`): `_thermotaxis_core()` (oracle, 3 dims) and `_thermotaxis_temporal_core()` (temporal/derivative, 3 dims).
- **STAM** (`agent/stam.py`): 3 channels (food=0, temp=1, pred=2), 9-dim memory state.
- **SensingConfig** (`utils/config_loader.py`): Per-modality mode selection (oracle/temporal/derivative) for chemotaxis, thermotaxis, nociception.
- **Pixel theme** (`env/pygame_renderer.py`, `env/sprites.py`): Temperature zone overlays (blue→transparent→red) with layered compositing.

## Goals / Non-Goals

**Goals:**

- Implement biologically accurate oxygen sensing with URX/AQR/PQR (hyperoxia) and BAG (hypoxia) neuron-inspired zones
- Support all three sensing modes (oracle, temporal, derivative) from day one
- Enable combined thermal+oxygen+food environments on large grids
- Extend STAM to 4 channels for oxygen temporal integration
- Visualize oxygen zones across all themes (pixel, rich, terminal)
- Update skills, docs, and specs to reflect the new modality

**Non-Goals:**

- Dynamic oxygen consumption/production (oxygen field is static per episode, like temperature)
- Metabolic oxygen requirements (no ATP model — oxygen is a comfort/avoidance signal only)
- Multi-agent oxygen competition (Phase 4 scope)
- Oxygen-mediated social behavior (npr-1 bordering, Phase 4)
- New brain architectures for oxygen — existing brains handle it via the modular sensory system

## Decisions

### 1. Asymmetric Zone System with Absolute Thresholds

**Decision**: Use absolute O2 percentage thresholds rather than symmetric deltas from a reference point.

**Rationale**: Temperature zones are symmetric around a cultivation temperature (Tc ± delta). Oxygen preference is a biological range (5-12% O2), not a point. There is no single "cultivation oxygen" analogous to Tc. The asymmetry between hypoxia danger (low end, bacteria-dense environments) and hyperoxia danger (high end, exposed surfaces) requires independent thresholds.

**Design**:

```python
@dataclass
class OxygenZoneThresholds:
    lethal_hypoxia_upper: float = 2.0    # <2% is lethal
    danger_hypoxia_upper: float = 5.0    # 2-5% is dangerous
    comfort_lower: float = 5.0           # 5-12% is comfortable
    comfort_upper: float = 12.0
    danger_hyperoxia_upper: float = 17.0 # 12-17% is dangerous
    # >17% is lethal hyperoxia (implicit from danger_hyperoxia_upper)
```

**Alternative considered**: Symmetric zones around 8.5% (midpoint). Rejected because biological thresholds are asymmetric — hypoxia becomes dangerous faster than hyperoxia — and this would obscure the distinct URX vs. BAG neuron pathways.

### 2. Brain Deviation Signal: Midpoint Normalization

**Decision**: Normalize oxygen deviation from the comfort midpoint (8.5%) by maximum realistic deviation (12.5%), yielding `(O2 - 8.5) / 12.5` clipped to [-1, 1].

**Rationale**: The brain needs a single scalar indicating "how far from ideal" that has the same [-1, 1] range as the thermotaxis deviation signal `(T - Tc) / 15.0`. The midpoint 8.5% and normalization factor 12.5% (8.5% to edge of [0, 21] range) produce a semantically consistent signal: negative = too little O2, positive = too much O2, zero = ideal.

**Alternative considered**: Normalize relative to comfort boundaries (0 inside comfort, scaled outside). Rejected for adding non-linearity that obscures gradient information.

### 3. OxygenField Mirrors TemperatureField Architecture

**Decision**: Create `OxygenField` as a near-identical dataclass to `TemperatureField` with `get_oxygen()`, `get_gradient()`, `get_gradient_polar()` methods, using the same linear-gradient-plus-spots model.

**Rationale**: Proven pattern, consistent API, minimal new concepts. The spots model maps well to oxygen ecology: low-oxygen spots = dense bacterial lawns consuming O2, high-oxygen spots = ventilation points or surface exposure.

**Key differences from TemperatureField**:

- Field name: `high_oxygen_spots` / `low_oxygen_spots` (vs hot/cold)
- Values clamped to [0.0, 21.0] (vs unclamped temperature)
- Default `base_oxygen: 10.0` (center of comfort, vs `base_temperature: 20.0`)
- Default `gradient_strength: 0.1` (O2 % per cell, tuned so medium/large grids span meaningful range)

### 4. STAM: Hardcoded 4 Channels, MEMORY_DIM=11

**Decision**: Bump `num_channels` from 3 to 4 and `MEMORY_DIM` from 9 to 11. Keep the hardcoded approach.

**Rationale**: The current STAM is hardcoded to 3 channels with a validation assertion. Adding one more channel is simple, and no 5th channel is planned. Making STAM fully dynamic would add complexity to the memory state layout, quantum compression, and `stam` sensory module — all for a hypothetical future need.

**New layout (11-dim)**:

```text
[0:4]  Weighted scalar means: food, temp, pred, oxygen
[4:8]  Temporal derivatives:  d(food)/dt, d(temp)/dt, d(pred)/dt, d(oxygen)/dt
[8:10] Position deltas:       dx_deviation, dy_deviation
[10]   Action entropy
```

**Index shifts**: `IDX_POS_DELTA_X` 6→8, `IDX_POS_DELTA_Y` 7→9, `IDX_ACTION_ENTROPY` 8→10.

**Quantum compression update**: Currently compresses [0:3]→mean, [3:6]→mean, [8]→entropy. Update to [0:4]→mean, [4:8]→mean, [10]→entropy. Output remains 3 floats.

**Breaking change mitigation**: All brain architectures compute `input_dim` from `get_classical_feature_dimension()` which reads `classical_dim` from the module. Updating `STAMSensoryModule.classical_dim` from 9→11 automatically propagates. No saved model weights depend on STAM dimension (STAM was added in Phase 3, all prior models predate it).

### 5. Combined Environment Sizing and Gradient Orientation

**Decision**: Minimum medium (50×50) for oxygen-only, large (100×100) for combined oxygen+thermal. Orthogonal gradient directions.

**Rationale**: Two overlapping gradient fields must leave sufficient "habitable overlap" — cells where both temperature and oxygen are in comfort zones — for foraging and predator evasion. With orthogonal gradients (e.g., temperature east at 0 rad, oxygen north at π/2 rad), the comfort overlap forms a rectangular region. On a 100×100 grid with moderate gradient strengths, this overlap region covers ~40-60% of cells, providing ample foraging space.

**Combined environment template**:

```yaml
thermotaxis:
  gradient_direction: 0.0     # Temperature increases east
  gradient_strength: 0.15     # Mild linear (same as thermal_foraging/large)
aerotaxis:
  gradient_direction: 1.5708  # O2 increases north (π/2)
  gradient_strength: 0.08     # O2 % per cell
  base_oxygen: 10.0           # Center of comfort range at grid center
```

Oxygen sinks placed near food-dense areas (bacteria consume O2, creating realistic trade-off: best food = lowest oxygen). High-oxygen spots at exposed grid edges (surface ventilation = predation risk).

### 6. Scenario Naming: Alphabetized Modalities

**Decision**: Directory names use alphabetized modality prefixes before the task name: `oxygen_thermal_foraging/`, not `thermal_oxygen_foraging/`.

**Rationale**: Consistent, deterministic ordering that scales to future modalities. "oxygen" < "thermal" alphabetically. Future: `oxygen_pheromone_thermal_foraging/`.

### 7. Visualization Layer Ordering

**Decision**: Oxygen zones render between temperature zones and toxic (predator) zones in the pixel theme compositing stack.

**Rationale**: Temperature and oxygen are both environmental field properties (background layer), while toxic zones are entity-specific (foreground). Rendering order: background → temperature → oxygen → toxic → entities. When both fields overlap, the alpha-blended oxygen overlay tints on top of the temperature overlay, creating a distinct visual blend.

**Color palette** (complementary to temperature's blue-red spectrum):

- Lethal hypoxia: dark red (180, 40, 40, 90) — distinct from temperature's blue-cold
- Danger hypoxia: red-brown (200, 80, 60, 70)
- Comfort: transparent (0, 0, 0, 0)
- Danger hyperoxia: light cyan (80, 200, 220, 70)
- Lethal hyperoxia: bright cyan (40, 180, 220, 90)

This creates a red-to-cyan spectrum for oxygen that is visually distinguishable from temperature's blue-to-red spectrum when both are overlaid.

### 8. Reward Structure: No Comfort Reward

**Decision**: Default `comfort_reward: 0.0` for oxygen (matching the thermotaxis lesson).

**Rationale**: Early thermotaxis experiments discovered that positive comfort rewards cause "freeze behavior" — the agent stops foraging and sits in the comfort zone. The solution (Logbook 007) was setting `comfort_reward: 0.0` and using only penalties for discomfort/danger. Oxygen follows the same pattern: penalize being outside comfort, don't reward being inside it.

Default penalties: `danger_penalty: -0.5`, `danger_hp_damage: 0.5`, `lethal_hp_damage: 6.0`. Note: unlike the 7-zone thermotaxis system, oxygen uses 5 zones (no discomfort tier) because URX/BAG neurons have relatively sharp activation thresholds — the transition from comfort to danger is biologically abrupt. The `reward_discomfort_food` field (default 0.0) provides a "brave foraging" bonus for collecting food in danger zones, matching the thermotaxis pattern in runners.py.

## Risks / Trade-offs

**[Combined environment complexity may exceed agent capacity]** → Mitigation: Start evaluation with oxygen-only scenarios to establish baselines before testing combined environments. If combined scenarios prove too hard for MLP PPO, increase grid size or reduce penalty severity.

**[STAM dimension change breaks existing temporal configs]** → Mitigation: The change is transparent — `classical_dim` auto-propagates to `input_dim`. Existing temporal configs will work but produce slightly different training dynamics due to the extra 2 STAM dimensions (oxygen weighted mean + derivative will be zero when aerotaxis is disabled). Run regression tests on existing temporal scenarios.

**[Oxygen + temperature overlay visual confusion]** → Mitigation: Use a complementary color spectrum (red-amber-cyan for oxygen vs blue-red for temperature). Provide a status bar readout for both values. If overlay blending is confusing, consider a toggle or separate rendering mode in future.

**[Grid size requirements limit experiment iteration speed]** → Mitigation: Medium (50×50) for oxygen-only provides 2.5× faster episodes than large. Combined scenarios on large grids will need proportionally more training episodes — document expected episode counts in config headers.

## Open Questions

None — all design decisions are resolved. Implementation can proceed.
