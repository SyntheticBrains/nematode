# Design

A focused substrate-coherence fix. Three small decisions:

## D1 — Fallback, not a changed default

`Predator.damage_radius` is typed `int` and defaults to `0`, which the **grid** path relies on
as the "same-cell" contact rule (and changing that default would break grid byte-stability).
So rather than change the field's default, the continuous-2D substrate **falls back** to a
body/contact-scale Euclidean radius only when `damage_radius <= 0`. A positive configured
value always wins. This keeps the grid path byte-stable and honours any intentional override.

## D2 — `predator_damage_radius_mm` on `Continuous2DParams` (default 1.0 mm)

The fallback value is a new field wired through both the pydantic `Continuous2DConfig`
(`config_loader.py`, the YAML `continuous:` block) and the env-side `Continuous2DParams`
dataclass via the `create_env_from_config` factory, mirroring `capture_radius_mm` /
`world_size_mm`:

- Default **1.0 mm** = one `body_length_mm`, i.e. "predator within one body length = contact bite" —
  the same body-scale logic used for food capture, and a value the Stage-1 canary confirmed
  produces predator deaths (12/12 untrained at 1 mm).
- Configurable per scenario, so the upcoming Stage-1 predator-difficulty calibration tunes
  lethality through this knob (and `speed`) rather than through the reward (RQ5 guardrail).

## D3 — Compute effective radius in the damage/contact methods, don't mutate the predator

A `float`-typed helper `_effective_damage_radius(pred) -> float` returns
`float(pred.damage_radius)` when positive else `self.continuous.predator_damage_radius_mm`,
used by `is_agent_in_damage_radius_for` and `get_agent_predator_contact_zone_for`. Computing
the effective value locally (rather than mutating the `int`-typed `pred.damage_radius` to a
float at spawn) avoids the int/float type conflict and keeps the grid path untouched.
`detection_radius` is left alone — it is already a reachable positive value and the danger
check works; only the zero-default damage/contact radius is in scope.
