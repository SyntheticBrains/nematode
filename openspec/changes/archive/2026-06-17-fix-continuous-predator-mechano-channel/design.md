# Design — continuous predator_mechano contact-intensity channel

## Context

`_predator_contact_intensity_at` (agent.py) emits the graded `predator_mechano` contact signal
(`max(0, 1 − dist / damage_radius)`) consumed by the STAM channel fetcher and
`_create_brain_params`. It computes Manhattan distance against the integer predator `.position`
and skips predators with `damage_radius <= 0`. On the continuous substrate the default
`damage_radius` is `0` (the grid "same-cell" sentinel; the continuous env applies the
`predator_damage_radius_mm` fallback via `_effective_damage_radius` everywhere else), so the
function skips every predator and the channel is **constantly zero** — and Manhattan-vs-Euclidean
incoherent when a positive radius is set. Same bug class as
`fix-continuous-distance-reward-metric`. Affects C2 and the connectome's ALM/PLM/AVM projections.

## Decision 1 — Make it an environment method (geometry lives with the env)

Add `predator_contact_intensity_at(pos)` to the environment, mirroring the
`get_nearest_predator_distance_for` / `_from` overrides:

- **Grid base** (`DynamicForagingEnvironment`): the current logic verbatim — Manhattan against
  the integer `.position`, skip `damage_radius <= 0`, `max(0, 1 − manhattan / damage_radius)`
  over the max-intensity predator. Byte-identical to today.
- **Continuous override** (`Continuous2DEnvironment`): Euclidean (`math.hypot` against
  `_predator_xy(pred)`) and `self._effective_damage_radius(pred)` (the existing fallback that
  substitutes `predator_damage_radius_mm` when the configured radius `<= 0`, and respects an
  explicit positive radius). Same `max(0, 1 − dist / radius)` shape.

`agent.py`'s `_predator_contact_intensity_at(position, env)` becomes a thin delegator to
`env.predator_contact_intensity_at(position)`; the two consumers are untouched. This keeps the
metric/radius policy where the rest of the continuous predator geometry already lives (detection,
damage, contact-zone, nearest-distance are all env overrides) rather than branching on substrate
type inside the agent.

## Decision 2 — Scope: metric + effective radius only (not the query position)

The callers still pass the agent's rounded integer `.position`. Threading the float
`pos_continuous` into this query (and the other sensing/reward query positions) is the broader
position-representation fix (continuous audit #1–#5), handled in a separate follow-up. This change
fixes the two substrate-incoherences that make the channel **dead / Manhattan**: the
`damage_radius <= 0` skip and the metric. (Euclidean-from-an-integer-agent-position is still a
strict improvement over Manhattan-from-integer and revives the channel; the residual ≤~0.5 mm
agent-position rounding is the follow-up's concern.)

## Decision 3 — Sensor definition unchanged (RQ5-safe)

The contact-intensity formula `max(0, 1 − dist / radius)` and the channel's meaning are
unchanged; only the distance **metric** and the **effective** radius (already defined for the
continuous substrate) are applied. This is a bugfix realizing the intended continuous behaviour,
not a sensor redesign.

## Validation

- **Unit / byte-stability:** grid `predator_contact_intensity_at` matches the prior inline
  computation (Manhattan, raw radius, skip `<= 0`).
- **Continuous:** with default `damage_radius = 0`, a predator within `predator_damage_radius_mm`
  yields a **non-zero** graded intensity (channel revived); intensity uses Euclidean (an off-axis
  predator yields a different value than Manhattan would); `0.0` outside the radius and when
  predators are disabled.
- **Regression:** reward-calculator + continuous-env + agent sensing suites pass; targeted
  pre-commit.

## Risks / alternatives considered

- **Branch on substrate type inside `_predator_contact_intensity_at`** — duplicates the
  metric/radius policy in the agent layer; rejected in favour of the env-method override
  (consistent with the existing continuous predator-geometry overrides).
- **Also thread float `pos_continuous` here** — correct but couples this fix to the broader
  position-representation change; kept separate so this lands as a focused, low-risk bugfix.
