## Context

The continuous-2D substrate (`Continuous2DEnvironment`, a subclass of
`DynamicForagingEnvironment`) overrides the worm's movement, food placement, food
capture, and field sampling to be real-valued, but overrides **nothing**
predator-related. Predators therefore inherit the grid model in three places:

- **Movement** — `Predator.update_position` drives a cardinal `PredatorAction`
  (`UP/DOWN/LEFT/RIGHT`) one integer cell at a time, with a `movement_accumulator`
  for fractional/multi-step `speed`. Pursuit picks the greedy cardinal step toward the
  nearest agent each call; positions are `tuple[int, int]` on the integer lattice.
- **Detection / damage** — `is_agent_in_danger_for` and `is_agent_in_damage_radius_for`
  compute **Manhattan** distance (`abs(dx)+abs(dy)`) between a predator's integer
  position and the agent's **discretised integer** `position` (not `pos_continuous`).
- **Contact zone** — `get_agent_predator_contact_zone_for` likewise uses Manhattan
  distance and classifies the approach cone against the discrete `Direction`.

Food capture is already Euclidean (`math.hypot` vs `capture_radius_mm`), so on the
continuous substrate predators are the lone integer-grid island. The result is visible
quantised predator hops next to a smoothly-moving worm and a Manhattan-diamond danger
zone on a plate the worm navigates continuously.

Constraints: the **discrete grid environment must stay byte-stable** (the predator
byte-equivalence regression suite locks its trajectories), the **predator reward
formula is frozen** (RQ5 — only the distance *metric* may change), and the change must
not introduce a new config dependency or external package.

## Goals / Non-Goals

**Goals:**

- Predators on the continuous-2D substrate move with a continuous `(speed, heading)`
  kinematic update and carry a real-valued `pos_continuous` truth position.
- Detection, damage, and contact-zone classification on the continuous substrate use
  true Euclidean distance against the agent's `pos_continuous` and the worm's
  continuous `heading_rad`.
- Predators spawn at float coordinates within the world bounds, keeping the existing
  Euclidean min-separation spawn check.
- The discrete grid predator model (movement, Manhattan detection/damage, byte-stable
  trajectories) is completely unchanged.

**Non-Goals:**

- A learned/brained continuous predator policy. The cardinal `PredatorBrain` pipeline
  stays for the grid; continuous predators use a simple analytic kinematic rule. A
  continuous predator brain is a possible later follow-up.
- Turn-rate-limited "realistic" pursuit dynamics. Default pursuit orients fully toward
  the target each step (the continuous analogue of greedy cardinal pursuit); a bounded
  turn rate is noted as an optional future knob, not shipped here.
- Re-tuning predator radii or the predator-evasion reward (RQ5 guardrail — the metric
  shift is validated, not retuned, downstream in the T7 C2 smoke).
- Dynamic-diffusion predator fields / source dynamics (separate gated stretch,
  `T6.gradients.2`).

## Decisions

### D1 — Behaviour change lives in `Continuous2DEnvironment` overrides, not the predator brain

The continuous kinematics, Euclidean detection/damage, and float spawn are implemented
as **overrides in `Continuous2DEnvironment`** plus **additive optional fields on
`Predator`**. The grid `DynamicForagingEnvironment` path and the cardinal
`PredatorBrain` are untouched.

*Why:* the predator brain pipeline emits cardinal actions and is locked by a
byte-equivalence regression; threading a continuous action mode through it is heavy and
risks the grid contract. Predator behaviour is simple (pursue / wander / stationary), so
an analytic env-level kinematic rule is sufficient and keeps the blast radius inside the
subclass. *Alternative considered:* a continuous `PredatorBrain` action mode — rejected
as disproportionate for analytic pursuit and a byte-stability hazard.

### D2 — Predator gains an additive `pos_continuous` truth + synced integer view

Mirror the `AgentState` pattern: add optional `pos_continuous: tuple[float, float] | None`
(and a continuous `heading_rad`) to `Predator`. On the continuous substrate
`pos_continuous` is the truth; the integer `position` is set as `round()`-and-clamped
after each move so inherited grid-coupled readers (renderer sprites, metrics, logging)
keep working unchanged. On the grid these fields stay `None`/default and are never read.

*Why:* additive-optional fields preserve the grid `Predator`'s type contract and
byte-stability while giving the continuous path a real-valued position. *Alternative:*
a separate `ContinuousPredator` subclass — rejected as over-engineering for two fields
and an awkward fit with the shared `predators` list and `_make_predator` factory.

### D3 — Continuous pursuit = orient-to-bearing + speed-scaled advance (mirrors the worm)

Each step, for a **pursuit** predator with an agent inside `detection_radius`
(Euclidean): set heading toward the bearing to the nearest agent's `pos_continuous`,
then advance by `speed · max_step_mm` along the new heading, clamping the new position
per-axis to `[0, world_size_mm]` (the same world-bound clamp the worm uses). This is the
continuous analogue of the grid's greedy cardinal pursuit (which also re-orients toward
the target every step). **Wandering** (no agent in range, or a non-stationary predator
with no target): perturb heading by a bounded random angle drawn from `self.rng`, then
advance. **Stationary** predators never move (unchanged). The grid `movement_accumulator`
multi-step logic is dropped on the continuous path — fractional/large `speed` simply
scales the per-step displacement, which is the natural continuous reading of `speed`.

*Why:* full orient-then-advance keeps continuous pursuit at least as effective as the
grid's greedy pursuit (so predator pressure isn't accidentally weakened) while producing
smooth motion; reusing `max_step_mm` and the worm's clamp keeps predator and worm
kinematics on one consistent scale with no new config. *Alternative:* a bounded turn
rate per step (more biologically realistic curved pursuit) — deferred as an optional
knob (Non-Goal) to avoid changing predator pressure and to keep this change minimal.

### D4 — Euclidean detection / damage / contact-zone overrides on the continuous substrate

Override `is_agent_in_danger_for`, `is_agent_in_damage_radius_for`, and
`get_agent_predator_contact_zone_for` in `Continuous2DEnvironment` to compute
`math.hypot` between the predator's `pos_continuous` and the agent's `pos_continuous`,
comparing against `detection_radius` / `damage_radius` as **Euclidean-mm** thresholds.
The contact-zone approach angle is taken between the predator→agent unit vector and the
worm's continuous forward unit vector (the same `heading_rad` convention `_kinematic_move`
and the renderer use), retaining the existing ±45° anterior/lateral/posterior cones and
nearest-predator-within-damage-radius selection.

*Why:* matches the already-Euclidean food-capture semantics and removes the
discretised-position read; keeping the same cone thresholds means only the metric and
the heading source change, not the classification policy.

### D5 — Float predator spawn within world bounds

Override `_initialize_predators` on the continuous substrate to sample candidate
predator positions at **real-valued** coordinates in `[0, world_size_mm)` (via
`self.rng`), retaining the existing Euclidean min-separation-from-agent validity check
(it is already Euclidean). Set `pos_continuous` and the synced integer `position`.

### D6 — Radius semantics shift is documented and validated, not retuned

A configured radius `r` previously bounded a **Manhattan diamond** of "radius" `r` on
the grid; on the continuous substrate the same `r` bounds a **Euclidean disc** of radius
`r`. With identical config values the danger/damage zone changes shape and area even
though no parameter changed. Per the RQ5 guardrail the predator-evasion reward formula
stays frozen; this metric shift is a **validate-don't-retune** item for the downstream
T7 C2 predator smoke (it is called out in `phase6-tracking` T7.prep).

## Risks / Trade-offs

- **Grid byte-stability regression** → the change is additive-optional on `Predator` and
  override-only in the subclass; the existing predator byte-equivalence suite
  (`test_predator_brain_byte_equivalence.py`) must stay green and is the gate.
- **Predator pressure inadvertently changed** (the Euclidean disc is smaller than the
  inscribed Manhattan diamond of equal `r` along the diagonals) → mitigated by D3's
  full-orient pursuit (keeps pursuit effective) and by surfacing the metric shift for
  explicit validation in the T7 C2 smoke rather than silently absorbing it.
- **Heading-convention mismatch** between predator contact-zone trig and the worm's
  `heading_rad` → mitigation: reuse the exact forward-unit-vector convention used by
  `_kinematic_move`/the renderer, and unit-test a known geometry (predator dead ahead →
  ANTERIOR; behind → POSTERIOR; abeam → LATERAL).
- **Coordinate-system foot-guns** keeping `position` (integer view) in sync with
  `pos_continuous` → set the integer view from the float truth in one helper after every
  move/spawn, exactly mirroring the agent's `_apply_movement`/`_discretise` pattern.

## Migration Plan

No data migration. Behaviour change is gated on the continuous-2D environment type;
existing grid configs and runs are byte-identical. Continuous predator configs (authored
in `T7.prep.continuous_behaviours`) automatically pick up the new kinematics. Rollback is
reverting the subclass overrides and the additive `Predator` fields.

## Open Questions

- Should the wander heading-perturbation magnitude be configurable, or is a fixed
  default sufficient for the T7 bring-up? (Lean: fixed default now; expose only if the
  predator smoke shows wander behaviour matters.)
- `PredatorParams.detection_radius` / `damage_radius` are typed `int`. On the continuous
  substrate they are read as Euclidean-mm thresholds, which integers satisfy (e.g. an
  `8` value → an 8 mm disc on a 50 mm plate). Fractional/sub-mm radii would require
  widening these to `float` in the config + dataclass — **out of scope** here; flag it
  only if the T7 predator smoke needs sub-mm precision.
