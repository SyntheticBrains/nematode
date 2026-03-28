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

### 7. STAM resets each episode

**Decision**: The STAM buffer resets at the start of each episode via `prepare_episode()`. No STAM state carries across episodes.

**Rationale**: Food and predator positions are randomized each episode, so concentration values from a previous episode are meaningless — the agent starts at a new position in a different arrangement. STAM models short-term biological memory (cAMP/calcium, minutes to ~30 min), not cross-episode learning; that role belongs to the brain's trained weights. Temperature zones are fixed across episodes, but STAM still needs to rebuild temporal context from the agent's *current trajectory* each episode regardless, since dT/dt depends on the agent's movement path, not just the field itself.

### 8. No environment configuration changes required; start with existing small configs

**Decision**: Use existing small (20×20, 500-step) environment configs for initial temporal sensing experiments. No changes to grid size, max steps, food count, or respawn logic.

**Rationale**: On a 20×20 grid with `gradient_decay_constant: 8.0` and 5 food sources, the concentration signal is detectable from most positions — gradients are steep enough that moving even one cell produces a measurable dC/dt. The 500-step budget is generous relative to grid traversal (~20 steps edge-to-edge), leaving headroom for the "orientation overhead" temporal sensing introduces (agents need a few initial steps to build STAM context before derivatives become informative).

**Considerations for follow-up**:

- If small grids prove too easy even with temporal sensing (concentration gradients are trivially informative from everywhere), medium (50×50) environments become the scientifically interesting regime — flatter gradients at distance make temporal derivatives near-zero and demand more sophisticated exploration.
- If agents can't converge at 500 steps, a modest step budget increase (750-1000) may be warranted, but this should be data-driven, not preemptive.
- Food respawn logic is unaffected — new food spawning mid-episode is fine because STAM naturally adapts to concentration changes in the environment.

### 9. Scalar concentration reuses existing decay math and normalization

**Decision**: `get_food_concentration(position)` sums `base_strength * exp(-distance / decay_constant)` from all food sources — the same formula as `_compute_food_gradient_vector()` but returning the scalar sum of magnitudes instead of the vector. The raw scalar is normalized via `tanh(raw * GRADIENT_SCALING_TANH_FACTOR)` to [0, 1], matching the oracle module normalization (env.py:1294).

**Rationale**: Reusing the existing decay model and normalization ensures consistency between oracle and temporal modes — the scalar value at a position uses the same scale as gradient magnitudes. This means temporal derivatives computed from normalized concentrations will have comparable magnitudes to the features brains are already trained on, and switching between oracle and temporal modes changes the information content (directional vs scalar) without also changing the value range.

### 10. Derivative mode implicitly requires STAM

**Decision**: When any modality is set to `derivative` mode, the system requires STAM to be enabled (since temporal derivatives are computed from STAM history). If `derivative` mode is configured without `stam_enabled: true`, the config loader SHALL auto-enable STAM with default parameters and log an info message.

**Rationale**: Derivative mode's temporal derivative computation (`compute_temporal_derivative()`) operates on the STAM buffer. Without STAM, there is no history from which to compute dC/dt. Rather than failing validation (which would confuse users), auto-enabling STAM is the most ergonomic approach — derivative mode implies temporal history.

### 11. Position deltas use step-to-step differences, not absolute coordinates

**Decision**: The STAM position delta components (2 of the 9 memory state floats) are computed as the displacement from the weighted mean of recent step-to-step position *changes* (dx[i] = pos[i] - pos[i-1]) to the most recent step change. The buffer stores position *differences* per step, not absolute grid coordinates.

**Rationale**: Absolute grid position is god-like knowledge — a real worm doesn't know where it is on the petri dish. But a worm does have proprioceptive awareness of its own movement (which direction and how far it moved each step). Step-to-step deltas are biologically legitimate proprioceptive signals. The weighted mean of recent deltas captures the agent's recent movement trend, and deviation from that trend captures changes in movement pattern.

### 12. Combined chemotaxis module not supported for temporal mode

**Decision**: The legacy combined `chemotaxis` module (which encodes food+predator as a single superposed gradient) is not compatible with temporal sensing. If `chemotaxis_mode` is set to `temporal` or `derivative`, the module translation MUST replace `chemotaxis` with `food_chemotaxis_temporal` and also add `nociception_temporal` if not already present (since the combined signal included predator information). If `nociception_mode` is still `oracle`, `nociception` (oracle) is added instead.

**Rationale**: The combined chemotaxis module is a legacy shortcut — it merges food attraction and predator repulsion into one gradient vector. In temporal mode, there is no directional gradient to combine. Food and predator signals must be separated into independent scalar channels, each with its own temporal derivative. This separation is also more biologically accurate: real C. elegans uses distinct sensory neurons for food (AWC, AWA) and predator chemicals (ASH, ADL).

### 13. Biological calibration as documentation, not enforcement

**Decision**: Biological reference values (AFD sensitivity ~0.01°C, ASE concentration comparisons over ~1-second head sweep timescales, STAM decay ~minutes to 30 minutes) are documented as comments in code and config files. No runtime enforcement or unit conversion system is added. The `stam_decay_rate: 0.1` default with `buffer_size: 30` approximates a half-life of ~7 steps; if each step represents ~1-2 seconds of biological time, this maps to ~10-15 seconds of salient memory, appropriate for the short-term chemotaxis comparisons ASE neurons perform.

**Rationale**: The simulation operates in discrete grid steps, not biological time. Introducing a formal time-unit mapping would add complexity without improving the computational experiment — what matters is whether temporal sensing creates measurably harder problems, not whether the exact timescales match biology. The biological reference values serve as calibration guidance for parameter tuning, not as constraints. Comments in the STAM module and config files will document the intended biological mapping.

### 14. Action variety metric is computational, not biological

**Decision**: The action variety component of the STAM memory state vector (1 of 9 floats) is documented as a computational convenience for learning, not a biological feature. It captures exploration diversity (entropy of recent actions) which helps brains distinguish "stuck in a loop" from "actively exploring."

**Rationale**: There is no direct C. elegans neural correlate for "action diversity awareness." However, it provides useful signal for RL agents learning in temporal mode, where movement strategy is critical for gradient inference. Documenting it as non-biological prevents over-claiming.

## Risks / Trade-offs

**[Risk] STAM memory dimension may be too small or too large for effective learning** → Start with 9 floats (the minimal useful summary). Monitor learning curves. If networks struggle, we can increase the summary size or expose raw buffer windows in a follow-up. The fixed-size design makes this easy to change.

**[Risk] Temporal derivative is noisy at early timesteps** → Returns 0.0 when buffer has \<2 entries. For the first few steps, Mode B effectively degrades to Mode A (scalar only). This is biologically plausible — sensory neurons need a brief adaptation period.

**[Risk] Mode A (raw scalar, no derivative) may be too hard for current brains** → This is expected and scientifically interesting. The performance gap between oracle and temporal modes IS the measure of how much we were cheating. Mode B provides a middle ground. Both modes can be tested independently per modality.

**[Risk] STAM classical_dim=9 changes input dimension for existing configs if accidentally enabled** → Mitigated by defaulting `stam_enabled: false`. STAM module is only added to the sensory modules list when explicitly enabled. Existing configs never reference the `stam` module name.

**[Trade-off] Fixed 9-float summary vs raw buffer** → We lose fine-grained temporal information but gain stable input dimensions and simpler learning. The temporal derivatives capture the most decision-relevant signal. If raw buffer access proves necessary (e.g., for recurrent architectures), it can be added as an additional module without breaking the existing one.

**[Trade-off] Auto module translation vs explicit config** → Auto translation is convenient but "magical." Mitigated by logging the translation at config load time so users can see what modules are actually active.
