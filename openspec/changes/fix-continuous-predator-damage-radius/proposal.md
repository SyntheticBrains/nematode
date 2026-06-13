## Why

`add-continuous-predator-kinematics` made the continuous-2D predator detection, damage,
and contact-zone checks measure **true Euclidean distance** against the worm's
`pos_continuous`, and specified that the configured `detection_radius` / `damage_radius`
become **Euclidean-millimetre thresholds**. But `damage_radius` defaults to integer **`0`**
(`env.py`), a grid "same-cell" concept: on the grid, `Manhattan distance <= 0` means the
predator occupies the worm's cell. As a **Euclidean** threshold, `hypot(predator, worm) <= 0.0`
is **unreachable** for real-valued float positions — so predators on the continuous-2D
substrate **move and pursue but can never land a hit**.

This silently disables predator damage across the entire continuous substrate. Surfaced by
the `T7.prep.continuous_behaviours` Stage-1 substrate canary: **0 predator deaths in 20
untrained runs** at the default (all max_steps/starved, HP never drops); setting
`damage_radius` to 1–2 mm produces **12/12 untrained deaths**. Left unfixed, the
`T7.*.c2_foraging_predator` smoke and the integrated-cell **predator-survival sub-metric**
would be vacuous — no arm ever dies, so predator evasion is unmeasurable and the RQ5
env-upgrade delta on predator behaviour would be meaningless.

## What Changes

- **Body/contact-scale continuous damage radius.** On the continuous-2D substrate, when a
  predator's configured `damage_radius` is `0` (the unreachable grid default), predator
  **damage** and **contact-zone** geometry fall back to a body/contact-scale Euclidean
  radius — a new `Continuous2DParams.predator_damage_radius_mm` (default `1.0` mm, one body
  length), mirroring how food capture uses `capture_radius_mm`. An explicit positive
  `damage_radius` still takes precedence (so existing/intentional values are honoured), and
  it is configurable per scenario for predator-difficulty calibration.
- **Detection unchanged.** `detection_radius` is already a reachable positive value (e.g. 6 mm)
  and the danger check works; only the zero-default **damage**/contact radius is fixed.
- **Grid environment unchanged and byte-stable.** The fallback lives only in the
  `Continuous2DEnvironment` damage/contact-zone path; the discrete grid keeps
  `damage_radius = 0` as the integer "same-cell" contact rule, byte-stable.
- **Reward formula unchanged (RQ5).** This fixes a geometry threshold only; the predator
  reward formula is untouched. `predator_damage_radius_mm` is a kinematics/substrate
  parameter (like `capture_radius_mm`), not a reward term.

## Capabilities

### New Capabilities

<!-- None — this fixes the existing continuous-2D environment capability. -->

### Modified Capabilities

- `continuous-2d-environment`: the "Euclidean predator detection, damage, and contact-zone
  geometry" requirement states the configured `damage_radius` is a Euclidean-mm threshold but
  does not address that its integer default `0` is an unreachable Euclidean distance. This
  change adds the body/contact-scale fallback (`predator_damage_radius_mm`, default 1.0 mm)
  applied when `damage_radius <= 0` on the continuous substrate, so predator damage is
  reachable. Grid byte-stability is preserved.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — add
    `Continuous2DParams.predator_damage_radius_mm` (default 1.0); an `_effective_damage_radius`
    helper returning `float(pred.damage_radius)` when positive else
    `predator_damage_radius_mm`; use it in `is_agent_in_damage_radius_for` and
    `get_agent_predator_contact_zone_for`.
- **Tests:** continuous predator damage triggers at body scale with the default
  `damage_radius` (regression); explicit positive `damage_radius` still honoured; grid damage
  remains integer same-cell, byte-stable.
- **Downstream:** unblocks `T7.prep.continuous_behaviours` Stage-1 predator canary and the T7
  integrated-C3 predator-survival sub-metric. No new dependencies.
- **Out of scope (documented follow-up):** the pygame renderer draws predator damage rings from
  the raw `pred.damage_radius` and skips them when `<= 0` (`pygame_renderer.py`), so on the
  continuous substrate with the default the damage *ring* is not drawn even though damage now
  occurs at the body/contact scale. The canary runs headless so this does not block; pointing the
  continuous renderer's damage ring at the effective radius is a small cosmetic follow-up, kept out
  of this minimal correctness fix.
