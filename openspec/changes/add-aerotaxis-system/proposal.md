## Why

Oxygen sensing (aerotaxis) is one of the best-characterized C. elegans sensory behaviors — worms actively prefer 5-12% O2 using URX/AQR/PQR neurons (hyperoxia detection) and BAG neurons (hypoxia detection), avoiding both extremes. This was originally a Phase 3 deliverable deferred when temporal sensing proved sufficient. Implementing it now adds a third environmental constraint alongside food and temperature, creates richer multi-objective trade-offs (food quality vs. temperature comfort vs. oxygen safety), and increases observation dimensionality — all advancing the project's quantum advantage investigation. The temporal sensing infrastructure (oracle/temporal/derivative modes, STAM) is already in place, so oxygen sensing can integrate cleanly.

## What Changes

### 1. OxygenField Class

Create a new class to define spatial oxygen concentration distributions:

- Linear O2 gradients (configurable direction and strength)
- High-oxygen spots (ventilation/surface points) and low-oxygen spots (bacterial consumption sinks) with exponential decay
- On-demand O2 computation at any position (O(1) storage)
- Gradient vector computation via central difference
- O2 values clamped to [0.0, 21.0] (atmospheric maximum)

### 2. Asymmetric Oxygen Zone System

Unlike temperature (symmetric zones around cultivation temperature), oxygen uses absolute percentage thresholds for an asymmetric zone system:

- `LETHAL_HYPOXIA` (\<2% O2) — anaerobic, lethal
- `DANGER_HYPOXIA` (2-5% O2)
- `COMFORT` (5-12% O2) — preferred range
- `DANGER_HYPEROXIA` (12-17% O2)
- `LETHAL_HYPEROXIA` (>17% O2)

Zone-based rewards/penalties and HP damage parallel the thermotaxis system.

### 3. Aerotaxis Sensory Modules

Implement `_aerotaxis_core()` and `_aerotaxis_temporal_core()` using the unified SensoryModule architecture:

- Oracle mode (3 features): gradient strength, egocentric angle to higher O2, comfort deviation
- Temporal/derivative mode (3 features): absolute deviation, dO2/dt signal, signed deviation
- Comfort deviation normalized from range midpoint (8.5%) by `(O2 - 8.5) / 12.5`

### 4. STAM Expansion (3 → 4 Channels)

- **BREAKING**: STAM `MEMORY_DIM` changes from 9 → 11, `num_channels` from 3 → 4
- New layout: 4 weighted means + 4 derivatives + 2 position deltas + 1 action entropy
- Adds oxygen as channel index 3 (food=0, temperature=1, predator=2, oxygen=3)
- Safe breaking change: all configs derive input_dim dynamically, no trained models depend on exact STAM dimensions

### 5. Combined Oxygen + Thermal Environments

- Environments with both thermal and oxygen fields coexisting alongside food gradients
- Orthogonal gradient directions (e.g., temperature east, O2 north) creating 2D optimization landscapes
- Oxygen sinks near food clusters (realistic: bacteria consume O2) for ecological trade-offs
- Minimum medium (50×50) for oxygen-only, large (100×100) for combined scenarios

### 6. Scenario Configurations

New scenario directories (alphabetized modality naming):

- `oxygen_foraging/`, `oxygen_pursuit/`, `oxygen_stationary/` — oxygen-only
- `oxygen_thermal_foraging/`, `oxygen_thermal_pursuit/`, `oxygen_thermal_stationary/` — combined

### 7. Visualization Support

- Pixel theme: oxygen zone overlays (hypoxia=red, hyperoxia=cyan) rendered between temperature and toxic layers
- Rich/terminal themes: oxygen zone symbols and background colors
- Status bar: oxygen reading displayed alongside temperature

### 8. Temporal Sensing Integration

All three sensing modes supported from day one:

- Oracle: spatial O2 gradient (strength + direction)
- Temporal (Mode A): scalar O2 only, brain uses STAM memory
- Derivative (Mode B): scalar O2 + dO2/dt pre-computed

Config: `aerotaxis_mode: oracle|temporal|derivative` in sensing section.

## Capabilities

### New Capabilities

- `aerotaxis`: Oxygen field generation, asymmetric zone classification, gradient computation, zone-based rewards/HP damage, and visualization support

### Modified Capabilities

- `environment-simulation`: Add OxygenField integration, `get_oxygen()`/`get_oxygen_gradient()` methods, `apply_oxygen_effects()`, `AerotaxisParams` dataclass, oxygen in `get_separated_gradients()`
- `brain-architecture`: Add oxygen BrainParams fields (`oxygen_concentration`, `oxygen_gradient_strength`, `oxygen_gradient_direction`, `oxygen_dconcentration_dt`), `aerotaxis`/`aerotaxis_temporal` sensory modules in ModuleName and SENSORY_MODULES
- `configuration-system`: Add `AerotaxisConfig` class, `aerotaxis_mode` to SensingConfig, aerotaxis in `apply_sensing_mode()` and `validate_sensing_config()`
- `temporal-sensing`: Add `aerotaxis_mode` as fourth independently configurable modality
- `short-term-associative-memory`: Expand from 3 channels/9-dim to 4 channels/11-dim with oxygen as channel 3

## Impact

**Affected Code:**

Core implementation:

- `quantumnematode/env/oxygen.py` — NEW: OxygenField, OxygenZone, OxygenZoneThresholds
- `quantumnematode/env/env.py` — AerotaxisParams, oxygen methods, apply_oxygen_effects()
- `quantumnematode/brain/modules.py` — aerotaxis modules, ModuleName entries, STAM dim update
- `quantumnematode/brain/arch/_brain.py` — BrainParams oxygen fields
- `quantumnematode/agent/stam.py` — 3→4 channels, MEMORY_DIM 9→11
- `quantumnematode/agent/agent.py` — oxygen BrainParams population, STAM channel 3
- `quantumnematode/agent/runners.py` — apply_oxygen_effects() in step loop
- `quantumnematode/utils/config_loader.py` — AerotaxisConfig, SensingConfig extensions
- `quantumnematode/dtypes.py` — OxygenSpot type alias

Visualization:

- `quantumnematode/env/sprites.py` — oxygen zone overlay colors
- `quantumnematode/env/pygame_renderer.py` — oxygen zone rendering, status bar
- `quantumnematode/env/theme.py` — oxygen zone symbols/colors for all themes

Configs:

- `configs/scenarios/oxygen_*/` — 3 new oxygen-only scenario directories
- `configs/scenarios/oxygen_thermal_*/` — 3 new combined scenario directories

Documentation:

- `README.md` — aerotaxis feature entry, pixel theme docs
- `AGENTS.md` — new scenario directories
- `docs/roadmap.md` — Phase 3 oxygen status update
- `.claude/skills/nematode-evaluate/skill.md` — aerotaxis diagnostics

Tests:

- `tests/.../env/test_oxygen.py` — NEW: OxygenField, zones, gradients
- `tests/.../agent/test_stam.py` — 4-channel STAM tests
- `tests/.../brain/test_modules.py` — aerotaxis module tests

**Breaking Changes:**

- STAM state dimension changes from 9 → 11 (safe: input_dim derived dynamically)

**Backward Compatibility:**

- All existing configs work unchanged (aerotaxis disabled by default)
- Existing brains receive None for oxygen fields when disabled
- STAM channel expansion is transparent to configs using `stam_enabled: true` (dimension auto-computed)

**Dependencies:**

- None. Uses existing Pydantic, NumPy, Pygame infrastructure.
