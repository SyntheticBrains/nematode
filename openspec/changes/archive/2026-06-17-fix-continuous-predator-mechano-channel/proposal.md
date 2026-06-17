## Why

The `predator_mechano` sensory channel (a graded contact intensity, biologically the
ALM/PLM/AVM mechanosensory signal) is **dead on the continuous-2D substrate**. Its source,
`_predator_contact_intensity_at` (agent.py), computes
`max(0, 1 − manhattan_dist / pred.damage_radius)` and **skips any predator with
`damage_radius <= 0`**:

```python
for pred in env.predators:
    if pred.damage_radius <= 0:
        continue
    manhattan = abs(...) + abs(...)
    ...
```

On the continuous substrate the **default `damage_radius` is the integer `0`** ("same cell" on
the grid) — the continuous env applies a body/contact-scale fallback
(`Continuous2DEnvironment._effective_damage_radius`, default `predator_damage_radius_mm = 1.0`)
everywhere else, but this function reads the raw `pred.damage_radius` and so `continue`s past
**every** predator → the channel returns `0.0` at all times. Even when an explicit positive
`damage_radius` is set, the function uses **Manhattan distance against the predator's integer
`.position`**, incoherent with the continuous substrate's Euclidean geometry (the same
metric-mismatch class fixed for the distance/evasion terms in
`fix-continuous-distance-reward-metric`).

This is consumed by the STAM `predator_mechano` channel fetcher and by `_create_brain_params`,
so on continuous a sensor the brains (and the connectome's ALM/PLM/AVM mechanosensory
projections) rely on silently emits a constant zero — it cannot contribute to predator-evasion
learning on the continuous substrate, which matters for C2 and the connectome bring-up
(Stage 2).

## What Changes

- **Move the contact-intensity computation into an environment method**
  `predator_contact_intensity_at(pos)`, mirroring the existing `get_nearest_*_distance_*`
  pattern so the metric and the effective damage radius live with the geometry:
  - **Grid base** (`DynamicForagingEnvironment`): Manhattan distance against the integer
    predator position, skipping `damage_radius <= 0` — **byte-identical** to the current
    behaviour.
  - **Continuous override** (`Continuous2DEnvironment`): **Euclidean** distance against the
    predator's real-valued position (`_predator_xy`) and the **effective** damage radius
    (`_effective_damage_radius`, which applies the `predator_damage_radius_mm` fallback when
    the configured `damage_radius <= 0`). The channel is no longer dead and is metric-coherent.
- **`agent.py` delegates.** `_predator_contact_intensity_at(position, env)` calls
  `env.predator_contact_intensity_at(position)`; the STAM fetcher and `_create_brain_params`
  consumers are unchanged.
- **Reward formula / sensor definition unchanged (RQ5-safe).** Same graded
  `max(0, 1 − dist / radius)` contact intensity; only the **metric** (Euclidean on continuous)
  and the **effective damage radius** (the existing continuous fallback) are made coherent — a
  bugfix realizing the intended behaviour on the continuous substrate.

NB this fixes the metric and the dead-channel cause; the **agent query position** is still the
rounded integer `.position` (passed by the callers). Threading the float `pos_continuous` into
this and the other sensing/reward query positions is the broader position-representation fix
tracked separately (continuous audit findings #1–#5).

## Capabilities

### Modified Capabilities

- `continuous-2d-environment`: extends the Euclidean-coherence guarantee to the predator
  **contact-intensity** query — the environment SHALL expose `predator_contact_intensity_at(pos)`
  returning the graded contact intensity in its native metric (Euclidean on continuous) using
  the effective damage radius, so the `predator_mechano` sensory channel is non-zero and
  metric-coherent on the continuous substrate. Grid behaviour is Manhattan + raw radius and
  byte-stable.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/env.py` — add base
    `predator_contact_intensity_at(pos)` (Manhattan, raw `damage_radius`, skip `<= 0` — current
    grid behaviour).
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — override with Euclidean
    (`_predator_xy`) + `_effective_damage_radius`.
  - `packages/quantum-nematode/quantumnematode/agent/agent.py` — `_predator_contact_intensity_at`
    delegates to `env.predator_contact_intensity_at(position)` (consumers unchanged).
- **Tests:** grid contact intensity byte-stable (Manhattan + raw radius); continuous contact
  intensity is **non-zero** within the effective (fallback) damage radius and uses Euclidean
  (off-axis predator distinguishes Euclidean from Manhattan); zero outside the radius / when
  predators disabled.
- **Downstream:** restores the `predator_mechano` channel for C2 (foraging+predator) and the
  connectome's mechanosensory projections (Stage 2). No new dependencies.
