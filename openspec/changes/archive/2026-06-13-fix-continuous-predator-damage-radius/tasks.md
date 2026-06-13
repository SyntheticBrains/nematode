# Tasks

## 1. Implementation

- [x] 1.1 Add `predator_damage_radius_mm: float = 1.0` to `Continuous2DParams`
  (`env/continuous_2d.py`), documented as the body/contact-scale Euclidean damage radius
  applied when the configured `damage_radius <= 0`; add the matching
  `Continuous2DConfig.predator_damage_radius_mm` pydantic field and factory wiring in
  `utils/config_loader.py` so it is settable from the YAML `continuous:` block.
- [x] 1.2 Add `_effective_damage_radius(self, pred) -> float` to `Continuous2DEnvironment`:
  `float(pred.damage_radius)` if `pred.damage_radius > 0` else `self.continuous.predator_damage_radius_mm`.
- [x] 1.3 Use `_effective_damage_radius(pred)` in `is_agent_in_damage_radius_for` and
  `get_agent_predator_contact_zone_for` (replace the bare `pred.damage_radius` comparisons).

## 2. Tests

- [x] 2.1 Continuous predator damage triggers at the body/contact scale with the default
  `damage_radius = 0`: a predator placed within `predator_damage_radius_mm` of the agent is
  in the damage radius; one just outside is not.
- [x] 2.2 Explicit positive `damage_radius` is honoured (fallback not applied) on continuous.
- [x] 2.3 Grid byte-stability: discrete-grid damage with `damage_radius = 0` retains
  same-cell semantics (existing predator suite stays green).

## 3. Validate + gate

- [x] 3.1 Behavioural confirmation: an untrained policy on a continuous predator config now
  takes predator deaths (was 0/20 at default; expect deaths to return) — records the fix end-to-end.
- [x] 3.2 `openspec validate fix-continuous-predator-damage-radius --strict`.
- [x] 3.3 Targeted `pre-commit` on changed files; full `pre-commit run -a` before push;
  `uv run pytest -m "not nightly"` for the env suite.
