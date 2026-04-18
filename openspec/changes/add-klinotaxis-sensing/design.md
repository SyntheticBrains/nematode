## Context

The current sensing architecture supports three modes per modality: ORACLE (full spatial gradient), TEMPORAL (scalar concentration, STAM provides dC/dt), and DERIVATIVE (scalar + explicit dC/dt). Temporal and derivative share the same `*_temporal` sensory modules — the difference is whether STAM computes the derivative explicitly. The `apply_sensing_mode()` function in config_loader.py substitutes module names using `!= SensingMode.ORACLE` pattern.

BrainParams has fields for scalar concentrations (`food_concentration`, etc.), temporal derivatives (`food_dconcentration_dt`, etc.), and oracle gradients (`food_gradient_strength`, `food_gradient_direction`, etc.). Sensory modules extract `CoreFeatures(strength, angle, binary)` with `classical_dim=2` for temporal or `classical_dim=3` for thermotaxis temporal.

The Direction enum (UP/DOWN/LEFT/RIGHT/STAY) maps to radians and is stored on AgentState. All concentration query methods accept arbitrary grid positions.

## Goals

- Add klinotaxis sensing mode that provides local spatial gradient + temporal derivative
- Cover all 7 gradient modalities (food, predator, temperature, oxygen, 3 pheromones)
- Maintain full backward compatibility with existing modes
- Enable biologically accurate pheromone trail-following
- Provide evaluation configs for benchmarking against established baselines

## Non-Goals

- Biological noise on lateral readings (future: `lateral_noise_std` param)
- Variable offset distance (fixed at 1 cell for v1)
- Quantum sensory module variants of klinotaxis (can be added later)
- Full pheromone evaluation campaign (follow-up work)

## Decisions

### Decision 1: Klinotaxis as superset of temporal+derivative

Klinotaxis modules provide 3 features per modality: center concentration (strength), lateral gradient (angle), and temporal derivative (binary). This is `classical_dim=3`. Klinotaxis requires STAM for the dC/dt component — STAM is auto-enabled when any modality uses klinotaxis.

**Rationale:** Biologically, real worms use both strategies simultaneously. Providing both spatial and temporal gradient information in one module matches the biology and gives the learning algorithm the most useful signal.

### Decision 2: STAM unchanged — no lateral gradient channels

Lateral gradients are instantaneous spatial measurements, not temporal signals. STAM continues recording scalar concentrations and computing dC/dt. No new STAM channels needed for klinotaxis.

**Rationale:** The lateral gradient changes meaning as the agent turns — it's heading-relative, not absolute. Recording it in STAM would conflate spatial and temporal information.

### Decision 3: lateral_scale as separate config parameter

`lateral_scale: float = 50.0` on SensingConfig, separate from `derivative_scale`. Both use tanh normalization but scale different physical quantities (spatial concentration difference vs temporal rate of change).

**Rationale:** The magnitudes of spatial and temporal gradients can differ by orders of magnitude depending on environment configuration. Independent tuning is necessary.

### Decision 4: 1-cell offset distance

Sample concentration at ±1 cell perpendicular to heading. At the biological scale implied by gradient decay constants (decay=4-10 cells maps to ~200-500μm), 1 cell ≈ 50μm — matching C. elegans head width between left and right amphid sensilla.

**Rationale:** Biologically accurate and computationally simple. Larger offsets would give stronger signals but less realistic sensing.

### Decision 5: Track last non-STAY heading for STAY direction

When Direction is STAY (agent didn't move), use the last non-STAY heading for lateral offset computation. Store `_last_heading` on the agent, updated whenever direction != STAY.

**Rationale:** An agent that chooses STAY should still be able to "sweep its head" based on its facing direction. Using a consistent fallback avoids zero-gradient artifacts at episode start.

### Decision 6: apply_sensing_mode() refinement

Change from `!= SensingMode.ORACLE` to explicit mode matching:

- `== KLINOTAXIS` → `*_klinotaxis` module
- `!= ORACLE and != KLINOTAXIS` → `*_temporal` module (TEMPORAL + DERIVATIVE)
- `== ORACLE` → original module

**Rationale:** The existing `!= ORACLE` pattern would incorrectly map klinotaxis to `*_temporal` modules. Explicit matching is safer and more readable.

### Decision 7: Temperature and oxygen klinotaxis

Temperature klinotaxis: strength = |temp_deviation / 15.0|, angle = lateral thermal gradient (normalized temp difference), binary = dT/dt. The lateral gradient uses raw temperature values at left/right positions, with the same deviation normalization as the center value.

Oxygen klinotaxis: strength = O2 concentration normalized, angle = lateral O2 gradient, binary = dO2/dt.

**Rationale:** Head-sweep sensing is biologically documented for thermotaxis (AFD neurons) and aerotaxis (URX/BAG neurons). The same mechanical behavior (lateral head movement) applies to all gradient-sensing modalities.

### Decision 8: All 7 modalities get klinotaxis, not just chemical

Klinotaxis applies to food, predator, temperature, oxygen, and all 3 pheromone types. Each modality can independently select its sensing mode.

**Rationale:** C. elegans uses head sweeps for ALL gradient navigation, not just chemotaxis. The underlying behavioral mechanism is the same across modalities.

## Risks / Trade-offs

1. **Grid edge effects**: At position x=0 with UP heading, left sample clamps to x=0 (same as center), making lateral gradient zero on one side. Biologically reasonable (worm at boundary) but could cause learning artifacts near edges.

2. **Concentration query cost**: 7 modalities × 2 extra queries = 14 additional concentration evaluations per step. Each involves exponential decay sums over food/predator/pheromone sources. Acceptable for current grid sizes but could become noticeable on very large grids with many sources.

3. **classical_dim=3 vs 2**: Klinotaxis modules have larger feature dimension than temporal modules. Configs cannot be swapped in-place during training without resetting weights (different network input size).

4. **apply_sensing_mode behavioral change**: Refining the `!= ORACLE` pattern to explicit matching. Must verify no existing temporal/derivative configs break.
