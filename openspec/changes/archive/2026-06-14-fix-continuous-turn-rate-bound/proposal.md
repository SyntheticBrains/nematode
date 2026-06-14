## Why

On the continuous-2D substrate the worm's action is `(speed, turn)`. Continuous-action brains emit a
**substrate-independent normalized** action (`speed ∈ [0,1]`, `turn ∈ [-1,1]`); the **environment**
rescales it to physical units in `move_agent_normalized`. The turn rescale is `turn_rad = turn_norm * π`
(`continuous_2d.py`), so the effective per-step turn is bounded to **±π rad ≈ ±180° — a full heading
reversal every step.** Even a *converged* policy therefore visibly **"helicopter"-spins** (a 360° spin in
~2 steps), which is not biologically plausible: real *C. elegans* crawls with gradual curvature and
reorients in bounded sharp turns (~15–30°/step), not at 180°/step.

This matters beyond aesthetics: T7's **headline external anchor is the real-worm behavioural-chemotaxis
validation** (turn-rate vs dC/dt, curving-rate vs bearing — `T7.validation`). A worm with a 180°/step
turn-rate distribution would directly weaken that comparison — the one thing that gives the continuous
substrate its biological meaning. Surfaced during the T7 C1 foraging-convergence visual inspection.

## What Changes

- **Bounded turn rate as an env-side kinematic parameter.** Replace the hard-coded `* π` turn rescale
  with a configurable `Continuous2DParams.max_turn_rad` (a maximum per-step angular velocity in radians,
  the rotational analogue of `max_step_mm`): `turn_rad = turn_norm * max_turn_rad`. The default is a
  biologically realistic value (re-validated against C1 foraging convergence + visual inspection), well
  below the previous π. The brains stay **env-agnostic** — they keep emitting the normalized
  `turn ∈ [-1, 1]`; the physical turn scale lives in the environment alongside `max_step_mm`.
- **Configurable per scenario.** `max_turn_rad` is exposed on `Continuous2DConfig` (the YAML
  `continuous:` block) and threaded through the factory, so the turn realism can be tuned per scenario
  (e.g. for matching real-worm turn-rate distributions in `T7.validation`).
- **No brain change; grid path unchanged.** The 5 continuous brains' normalized `[-1, 1]` turn bound is
  untouched. The discrete grid environment's cardinal-action movement is unaffected and byte-stable.
- **Re-validation required (downstream).** Bounding the physical turn changes how the worm moves, so the
  T7 C1 foraging convergence (just established) is re-validated against the new bound — and the
  helicopter spin confirmed fixed — before the foraging recipe/substrate is locked.

## Capabilities

### New Capabilities

<!-- None — this parameterises the existing continuous-2D movement model. -->

### Modified Capabilities

- `continuous-2d-environment`: the "Kinematic continuous movement" requirement specifies the
  `(speed, turn)` → position update but fixes the physical turn scale at `π` (±180°/step). This change
  makes the per-step turn bounded to a configurable biologically realistic maximum angular velocity
  (`max_turn_rad`), applied where the normalized action is rescaled to physical units. Grid byte-stability
  preserved.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — add
    `Continuous2DParams.max_turn_rad` (default realistic value); in `move_agent_normalized` rescale
    `turn = turn_norm * self.continuous.max_turn_rad` (was `* math.pi`).
  - `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — add
    `Continuous2DConfig.max_turn_rad` (`Field(default=…, gt=0.0)`) and thread it through
    `create_env_from_config` into `Continuous2DParams`.
- **Tests:** `move_agent_normalized` rescales the turn by `max_turn_rad` (turn_norm=1 → heading delta =
  max_turn_rad); the default is a realistic bound (`0 < default < π`); the config field parses + the
  factory wires it; grid/discrete path unaffected.
- **Downstream:** C1 foraging convergence re-validated against the new bound; the real-worm turn-rate
  validation (`T7.validation`) compares a realistic turn-rate distribution. No new dependencies.
