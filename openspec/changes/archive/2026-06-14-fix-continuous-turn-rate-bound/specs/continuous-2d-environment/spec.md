## MODIFIED Requirements

### Requirement: Kinematic continuous movement

The continuous-2D environment SHALL translate a continuous action `(speed, turn)` into a kinematic position update — a heading rotation by the turn angle followed by a forward displacement proportional to speed — bounded by the world extent, replacing the discrete one-cell cardinal-step movement. Continuous-action brains emit a **normalized** action (`speed ∈ [0, 1]`, `turn ∈ [-1, 1]`); the environment SHALL rescale it to physical units — `speed_mm = speed_norm · max_step_mm` and **`turn_rad = turn_norm · max_turn_rad`**, where `max_turn_rad` is a configurable **maximum per-step angular velocity** (the rotational analogue of `max_step_mm`, default `0.5` rad ≈ 29°/step (real C. elegans reorients ~15–30°/step)). The resulting heading SHALL be wrapped to `[−π, π]`. So no single step rotates the heading by more than `max_turn_rad`, and the worm reorients in bounded sharp turns rather than rotating continuously ("helicopter" spinning). The discrete grid environment's cardinal-action movement SHALL remain unchanged and byte-stable.

#### Scenario: Heading-and-displacement update

- **WHEN** an agent emits a normalized continuous action `(speed_norm, turn_norm)`
- **THEN** the agent's heading rotates by `turn_norm · max_turn_rad` (the resulting heading wrapped to `[−π, π]`) and the agent advances by `speed_norm · max_step_mm` (clamped to `[0, max_step_mm]`) along the new heading

#### Scenario: Turn rate is bounded to a realistic maximum

- **WHEN** a continuous-action brain emits a normalized turn of magnitude up to `1.0`
- **THEN** the physical heading change for that step is at most `max_turn_rad` radians (default `0.5` rad ≈ 29°/step), so no single step performs a near-full heading reversal

#### Scenario: World-bound clamping

- **WHEN** a movement would take the agent outside the world bounds
- **THEN** the agent's new position is clamped per-axis to `[0, world_size_mm]` — it advances as far as the bound allows (partial movement), the step is **not** rejected, no error is raised, and the episode continues
