## Context

The observation pipeline currently flows: **Environment** (computes spatial gradients from global state) → **Agent** (`_create_brain_params()` assembles `BrainParams`) → **SensoryModule registry** (transforms `BrainParams` → `CoreFeatures`) → **Brain** (receives features via `extract_classical_features()` or `to_quantum()`).

Oracle gradient computation in `env.py`:

- Food: `_compute_food_gradient_vector()` (lines 1111-1149) — sums exponential decay vectors from all food positions
- Predator: `_compute_predator_gradient_vector()` (lines 1151-1193) — same pattern, negative/repulsive
- Temperature: `temperature.py:get_gradient()` (lines 163-194) — central difference T(x+1,y)-T(x-1,y)/2

The `SensoryModule` registry in `modules.py` maps `ModuleName` → `SensoryModule` with a `CoreFeatures` extractor. All 18 brains consume features through this registry — either `extract_classical_features(params, modules)` returning a flat `np.ndarray`, or `to_quantum(params)` returning `[rx, ry, rz]` gate angles.

`BrainParams` (`_brain.py:126-213`) is a Pydantic-like dataclass with all sensory fields as optional (`| None`).

`EnvironmentConfig` (`config_loader.py:493-524`) holds nested config sections (`foraging`, `predators`, `health`, `thermotaxis`).

## Goals / Non-Goals

**Goals:**

- Replace oracle spatial gradient sensing with biologically-accurate temporal sensing for chemotaxis, thermotaxis, and nociception
- Implement STAM memory buffers for temporal integration
- Make sensing mode configurable per modality via YAML
- Maintain full backward compatibility — existing configs and tests unchanged
- Work with all 18 brain architectures without modifying any brain implementation

**Non-Goals:**

- Oxygen sensing (Phase 3, Deliverable 3 — separate change)
- ITAM/LTAM longer-term memory (Phase 3, Deliverable 4 — conditional on STAM success)
- Associative learning paradigms (Phase 3, Deliverable 5)
- Modifying any brain architecture code
- Performance optimization of STAM (premature — profile first)

## Decisions

### 1. Intercept at the SensoryModule registry, not at individual brains

**Decision**: Add new `ModuleName` entries (`FOOD_CHEMOTAXIS_TEMPORAL`, `NOCICEPTION_TEMPORAL`, `THERMOTAXIS_TEMPORAL`, `STAM`) to the existing registry in `modules.py`. When a non-oracle sensing mode is configured, the config loader auto-substitutes the temporal module name for the oracle one.

**Rationale**: The registry is the single choke point — all 18 brains call `extract_classical_features()` or `to_quantum()`. Adding modules to the registry changes what every brain sees without touching any brain code. The alternative (adding sensing mode logic to each brain's `preprocess()`) would require modifying 18 files.

**Alternative considered**: A middleware layer between environment and agent that transforms observations. Rejected because it would add a new abstraction layer when the existing `SensoryModule` pattern already provides the right interception point.

### 2. STAM lives in the agent layer, not brain or environment

**Decision**: New file `agent/stam.py` with `STAMBuffer` class. The agent owns the buffer lifecycle (create, record, reset). STAM state is passed to brains via a new `stam_state` field on `BrainParams`.

**Rationale**: STAM mediates between environment observations and brain inputs — it needs access to raw scalars from the environment and must deliver processed memory state to the brain. The agent layer (`agent.py:_create_brain_params()`) is exactly where this mediation happens. Putting STAM in the brain would require each brain to implement it; putting it in the environment would conflate sensing with memory.

### 3. Fixed-size memory state vector (9 floats)

**Decision**: `get_memory_state()` returns a fixed-size 9-element `np.ndarray` regardless of buffer fill level (zero-padded when underfull):

- 3 weighted scalar means (food, temperature, predator)
- 3 temporal derivatives (dC/dt per channel)
- 2 position deltas (dx, dy from weighted mean)
- 1 action variety metric

**Rationale**: Classical brains need `input_dim` at construction time. A fixed dimension means `get_classical_feature_dimension()` can compute the total without runtime state. The 9-float summary captures the essential temporal information without exposing raw buffer contents (which would vary in size and be harder for networks to learn from).

**Alternative considered**: Variable-length buffer exposure (pass last N readings). Rejected because it changes input dimension based on buffer fill, complicates quantum transforms (which expect exactly 3 values), and makes the learning problem harder without clear benefit.

### 4. STAM module uses a custom subclass for variable classical_dim

**Decision**: Create a `STAMSensoryModule` subclass that overrides `to_classical()` to return the full 9-float memory state, and `to_quantum()` to compress to a 3-float summary (mean scalar, mean derivative, action entropy). Set `classical_dim = 9`.

**Rationale**: The standard `SensoryModule` outputs 2-3 features via `CoreFeatures(strength, angle, binary)`. STAM needs 9 features. Subclassing keeps the registry interface consistent while allowing the higher-dimensional output. The `classical_dim` field already exists and is used by `get_classical_feature_dimension()` to compute total input size.

### 5. Temporal derivative computation uses weighted finite difference

**Decision**: `dC/dt = Σ(w[i] * (C[i] - C[i+1])) / Σ(w[i])` where `w[i] = exp(-decay_rate * i)` and `i=0` is most recent. Returns 0.0 when buffer has < 2 entries.

**Rationale**: Exponential weighting emphasizes recent changes (matching biological sensory neuron adaptation). The weighted average smooths noise without requiring a fixed window size. Biological basis: AFD neurons respond to temperature *changes* with ~0.01°C sensitivity — a weighted finite difference models this temporal comparison.

### 6. Config-level automatic module translation

**Decision**: Add `_apply_sensing_mode(config)` to `config_loader.py` that replaces oracle module names with temporal variants based on `SensingConfig`. Users set `chemotaxis_mode: temporal` and the system injects `food_chemotaxis_temporal` in place of `food_chemotaxis`.

**Rationale**: Keeps brain configs simple — users don't need to know about temporal module names. The translation is deterministic and happens once at config load time. If `stam_enabled: true`, the `stam` module is appended to the sensory modules list automatically.

**Alternative considered**: Require users to manually list temporal module names in brain config. Rejected because it's error-prone (easy to forget to change one module) and creates coupling between environment sensing config and brain module config.

### 7. Scalar concentration reuses existing decay math

**Decision**: `get_food_concentration(position)` sums `base_strength * exp(-distance / decay_constant)` from all food sources — the same formula as `_compute_food_gradient_vector()` but returning the scalar sum of magnitudes instead of the vector.

**Rationale**: Reusing the existing decay model ensures consistency between oracle and temporal modes — the scalar value at a position is exactly what the gradient was computed from. This means the temporal derivative of concentration values will be physically consistent with the spatial gradients in oracle mode.

## Risks / Trade-offs

**[Risk] STAM memory dimension may be too small or too large for effective learning** → Start with 9 floats (the minimal useful summary). Monitor learning curves. If networks struggle, we can increase the summary size or expose raw buffer windows in a follow-up. The fixed-size design makes this easy to change.

**[Risk] Temporal derivative is noisy at early timesteps** → Returns 0.0 when buffer has \<2 entries. For the first few steps, Mode B effectively degrades to Mode A (scalar only). This is biologically plausible — sensory neurons need a brief adaptation period.

**[Risk] Mode A (raw scalar, no derivative) may be too hard for current brains** → This is expected and scientifically interesting. The performance gap between oracle and temporal modes IS the measure of how much we were cheating. Mode B provides a middle ground. Both modes can be tested independently per modality.

**[Risk] STAM classical_dim=9 changes input dimension for existing configs if accidentally enabled** → Mitigated by defaulting `stam_enabled: false`. STAM module is only added to the sensory modules list when explicitly enabled. Existing configs never reference the `stam` module name.

**[Trade-off] Fixed 9-float summary vs raw buffer** → We lose fine-grained temporal information but gain stable input dimensions and simpler learning. The temporal derivatives capture the most decision-relevant signal. If raw buffer access proves necessary (e.g., for recurrent architectures), it can be added as an additional module without breaking the existing one.

**[Trade-off] Auto module translation vs explicit config** → Auto translation is convenient but "magical." Mitigated by logging the translation at config load time so users can see what modules are actually active.
