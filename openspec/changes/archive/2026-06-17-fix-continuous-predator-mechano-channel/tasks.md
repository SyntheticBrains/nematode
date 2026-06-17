# Tasks — continuous predator_mechano contact-intensity channel

## 1. Environment contact-intensity query

- [x] 1.1 Add `predator_contact_intensity_at(pos)` to the base `DynamicForagingEnvironment`
  (`env.py`): the current inline logic verbatim — Manhattan against the integer predator
  position, skip `damage_radius <= 0`, `max(0, 1 − manhattan / damage_radius)` over the
  max-intensity predator; `0.0` when predators disabled / none / none in range.
- [x] 1.2 Override on `Continuous2DEnvironment` (`continuous_2d.py`): Euclidean (`math.hypot`
  against `_predator_xy(pred)`) and `_effective_damage_radius(pred)` (the
  `predator_damage_radius_mm` fallback for `damage_radius <= 0`; explicit positive radius
  wins); same `max(0, 1 − dist / radius)` shape.

## 2. Delegate from the agent

- [x] 2.1 `agent.py` `_predator_contact_intensity_at(position, env)` delegates to
  `env.predator_contact_intensity_at(position)`; the STAM `predator_mechano` fetcher and
  `_create_brain_params` consumers are unchanged.

## 3. Tests

- [x] 3.1 Grid byte-stability: `predator_contact_intensity_at` matches the prior inline
  computation (Manhattan, raw radius, `damage_radius <= 0` → 0.0).
- [x] 3.2 Continuous: default `damage_radius = 0` + predator within `predator_damage_radius_mm`
  → non-zero graded intensity (channel revived); Euclidean (off-axis predator ≠ Manhattan);
  `0.0` outside the radius and when predators disabled.

## 4. Gates + tracking

- [x] 4.1 `openspec validate fix-continuous-predator-mechano-channel --strict`.
- [x] 4.2 Targeted `pre-commit` (ruff / pyright / markdownlint) on changed files; reward /
  continuous-env / agent-sensing suites pass.
- [x] 4.3 Note in `phase6-tracking` (Stage 2 / connectome row) that the continuous
  `predator_mechano` channel was dead pre-fix (Stage-1 predator calibration ran without it).
