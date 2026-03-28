## Why

The simulation currently provides "oracle" spatial gradients to agents — computing central differences from adjacent cells and superposing exponential decays from all food/predator positions. A ~1mm worm at position (x,y) cannot access T(x+1,y) or the locations of distant food sources. This constitutes environmental cheating that makes the computational problem artificially easy, keeping classical success rates at 94-98% and leaving no headroom for quantum advantage. Replacing oracle sensing with biologically-accurate temporal sensing (Phase 3, Deliverables 1 & 2) is the single most impactful fidelity upgrade on the roadmap and directly addresses two quantum advantage thresholds: non-Markovian dependencies and partial observability.

## What Changes

- **New sensing modes**: Three modes configurable per sensory modality (chemotaxis, thermotaxis, nociception):
  - **Mode A (temporal)**: Agent receives only the scalar value at its current position (concentration, temperature). No gradient or directional information. The brain must use STAM memory to infer gradients from its own movement history.
  - **Mode B (derivative)**: Agent receives the scalar value plus its temporal derivative (dC/dt, dT/dt). Models what sensory neurons actually output (e.g., AFD signals "warming" or "cooling"). Still no directional information.
  - **Legacy (oracle)**: Existing spatial gradient behavior, retained for backward compatibility and as a comparison baseline.
- **STAM (Short-Term Associative Memory)**: Exponential-decay memory buffers storing recent sensory readings, positions, and actions. Provides the temporal context needed for Mode A sensing. Biologically modeled on cAMP/calcium signaling (minutes to ~30 min timescale).
- **Scalar concentration methods**: New environment methods returning the scalar signal strength at a position (food concentration, predator concentration) without directional information.
- **Temporal sensory modules**: New entries in the SensoryModule registry (`food_chemotaxis_temporal`, `nociception_temporal`, `thermotaxis_temporal`, `stam`) that replace oracle modules transparently — all 18 brain architectures work without modification.
- **Sensing configuration**: Per-modality mode selection and STAM parameters via YAML config, with automatic sensory module translation.
- **Example configs**: New temporal-mode configs for foraging, thermotaxis, and predator scenarios.
- Mechanosensation (boundary/predator contact) is already biologically accurate and remains unchanged.

## Capabilities

### New Capabilities

- `temporal-sensing`: Biologically-accurate sensing modes that replace oracle spatial gradients with scalar-at-position readings and optional temporal derivatives, configurable per sensory modality (chemotaxis, thermotaxis, nociception).
- `short-term-associative-memory`: Exponential-decay memory buffers (STAM) that store recent sensory history, enabling agents to infer gradients from temporal comparisons of their own movement and sensory experience.

### Modified Capabilities

- `environment-simulation`: New scalar concentration methods (`get_food_concentration`, `get_predator_concentration`) and sensing mode integration into the observation pipeline.
- `configuration-system`: New `SensingConfig` model with per-modality mode selection and STAM parameters; automatic sensory module translation based on sensing mode.

## Impact

- **Environment** (`env.py`): New scalar concentration methods alongside existing gradient methods.
- **Brain params** (`_brain.py`): New optional fields for scalar concentrations, temporal derivatives, and STAM state.
- **Sensory modules** (`modules.py`): New temporal module entries and STAM module in the registry. Existing oracle modules unchanged.
- **Agent** (`agent.py`): STAM buffer lifecycle and conditional observation assembly in `_create_brain_params()`.
- **Config** (`config_loader.py`): New `SensingConfig` Pydantic model and module translation logic.
- **New file** (`agent/stam.py`): STAM buffer implementation.
- **Backward compatibility**: All defaults are oracle mode with STAM disabled. Existing configs and tests are unaffected.
- **No new dependencies**: Uses only numpy and stdlib (collections.deque).
