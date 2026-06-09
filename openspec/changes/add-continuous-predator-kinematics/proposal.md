## Why

The continuous-2D substrate made the worm, food capture, and chemosensory fields
fully continuous (float positions, Euclidean capture, real-valued field sampling),
but **predators were left on the inherited integer grid**: `Continuous2DEnvironment`
overrides no predator behaviour, so pursuit predators still move in whole 1 mm
Manhattan steps and their detection / damage / contact-zone checks measure
**Manhattan distance against the worm's *discretised* integer `position`**, not its
real-valued `pos_continuous`. This is the last substrate-coherence gap on the
continuous-2D platform — pursuit predators visibly jump in quantised cells next to
the smoothly-moving worm (first surfaced by the fidelity renderer), the effective
danger zone is a Manhattan diamond on a grid the worm no longer lives on, and the
predator-survival sub-metric runs on this mismatched geometry. It does **not**
confound the architecture *ranking* (all arms share it), but it weakens the
continuous-physics platform claim and the real-worm-validation defensibility. T7's
integrated-C3 cell runs predator evasion on this substrate and the per-architecture
predator bring-up smoke depends on it, so this is the first T7-prep item.

## What Changes

- **Continuous predator kinematics.** On the continuous-2D substrate, predators move
  with a continuous `(speed, heading)` kinematic update — pursuit predators steer
  their heading toward the bearing to the nearest agent's real-valued position
  (bounded turn rate) and advance by a speed-scaled displacement along the new
  heading, replacing the inherited cardinal one-cell Manhattan step. Wandering
  (no agent within detection radius) and stationary predators are preserved.
- **Float predator position.** Predators gain a real-valued `pos_continuous` truth
  field on the continuous substrate; the integer `position` is retained as a rounded,
  clamped view so inherited grid-coupled readers (rendering, metrics) keep working.
- **Euclidean detection / damage / contact zones.** On the continuous substrate,
  `is_agent_in_danger_for`, `is_agent_in_damage_radius_for`, and
  `get_agent_predator_contact_zone_for` measure **true Euclidean distance** between
  the predator's float position and the agent's `pos_continuous`, and the
  contact-zone approach angle is taken relative to the worm's continuous
  `heading_rad`. The configured `detection_radius` / `damage_radius` become
  Euclidean-mm thresholds (a Euclidean disc, not a Manhattan diamond).
- **Float predator placement.** Predators on the continuous substrate spawn at
  real-valued coordinates within the world bounds, retaining the existing
  Euclidean min-separation-from-agent spawn validity check.
- **Grid environment unchanged and byte-stable.** All behaviour change is confined to
  `Continuous2DEnvironment` overrides plus additive optional `Predator` fields; the
  discrete grid environment's integer-Manhattan predator model is untouched. The
  predator reward *formula* is unchanged — only the underlying distance **metric**
  shifts (Manhattan → Euclidean); this is a **validate-don't-retune** coupling for
  the downstream T7 predator smoke (RQ5 reward-constant guardrail).

## Capabilities

### New Capabilities

<!-- None — this extends the existing continuous-2D environment capability. -->

### Modified Capabilities

- `continuous-2d-environment`: the "Source placement and Euclidean fields" requirement
  currently scopes float/Euclidean to **food sources** and explicitly states
  "**Predator** placement and movement remain on the integer lattice in this
  iteration ... scheduled for T7 prep." This change retires that deferral: predator
  placement becomes float, predator movement becomes continuous `(speed, heading)`
  kinematics, and predator detection/damage/contact distances become Euclidean against
  `pos_continuous`. The grid-environment byte-stability guarantee is preserved.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/env.py` — additive `Predator`
    `pos_continuous` field (+ float-aware kinematic helper); no change to the grid path.
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — new overrides:
    `_initialize_predators` (float spawn), continuous predator movement in
    `update_predators`, and Euclidean `is_agent_in_danger_for` /
    `is_agent_in_damage_radius_for` / `get_agent_predator_contact_zone_for`.
  - `packages/quantum-nematode/quantumnematode/env/pygame_renderer.py` — the
    continuous renderer can drop the visual-jank workaround once predators move smoothly
    (predator sprites + detection/damage rings already read float-capable positions).
- **Tests:** new continuous-predator movement/detection/damage tests under
  `tests/quantumnematode_tests/env/`; existing integer-grid predator byte-equivalence
  suite must stay green (grid path untouched).
- **Downstream:** consumed by `T7.prep.continuous_behaviours` (predator bring-up smoke
  runs on Euclidean, continuously-moving predators) and the T7 integrated-C3 predator
  sub-metric. No new dependencies.
