## Context

T6 added a mode-aware **food** field magnitude: `ForagingParams` carries
`gradient_field_mode` (`exponential` default | `fick`), `gradient_decay_constant`,
`gradient_strength`, `diffusion_coefficient`, `assay_time`, and a `fick_length()` helper;
`_food_field_magnitude(distance)` dispatches between the exponential kernel
`A·exp(−r/decay)` and the frozen Fick Gaussian `A·exp(−(r/L)²)` with `L = √(4·D·t_assay)`
(falling back to `decay` when `D` is unset).

The **predator** field is exponential-only: `PredatorParams` has only
`gradient_decay_constant` / `gradient_strength`, and `_compute_predator_gradient_vector`,
`get_predator_concentration` (+ the `get_predator_sulfolipid_concentration` alias) inline
`exp(−distance / decay)`. The `chemical-gradient-fidelity` spec explicitly excluded the
predator field because predators-on-continuous weren't exercised — no longer true (#226).

Constraints: **grid + existing configs must stay byte-stable** (the predator field feeds
sensing → training reproducibility), and the **food field must remain numerically
identical** (its Fick tests are the regression).

## Goals / Non-Goals

**Goals:**

- The predator-sulfolipid chemical field can be Fick-shaped with its **own** diffusion
  coefficient, selectable per config; exponential stays the default.
- One shared kernel implementation for food + predator (no duplicated exp-vs-fick math).
- Food field behaviour and all existing configs byte-identical.

**Non-Goals:**

- **Temperature** and the **oxygen base gradient** — imposed/boundary gradients (thermal
  conduction / plate-edge O₂), not point-source diffusion. Out of scope. (Oxygen *spot*
  falloffs are point-source but secondary to the imposed aerotaxis gradient; left as-is.)
- **Pheromone** Fick — genuinely diffusing but currently Manhattan-distance + multi-agent;
  its own follow-up change reusing this infrastructure.
- **Per-signal `D` literature calibration** (food / sulfolipid / CO₂ in agar) — phase-7
  fidelity depth (`phase6-tracking` T6.gradients.1 note). This change ships the *mechanism*
  - a sensible default, not calibrated `D` values.
- Dynamic/time-evolving diffusion (the gated stretch, T6.gradients.2).

## Decisions

### D1 — Extract one shared mode-aware kernel helper (DRY, food byte-identical)

Add a module-level pure helper, e.g.
`field_magnitude(distance, *, mode, decay, strength, fick_length)` → float, encoding the
exp-vs-fick dispatch once. `_food_field_magnitude` is refactored to call it with the
foraging params (its returned values are **numerically identical** — same formulas), and a
new `_predator_field_magnitude(distance)` calls it with the predator params.

*Why:* avoids duplicating the kernel math across food + predator (and future fields), and
keeps the food path a pure delegation (no behaviour change → byte-stable). *Alternative:*
duplicate the dispatch into a `_predator_field_magnitude` and leave `_food_field_magnitude`
untouched — slightly safer for food but duplicates the kernel; rejected since the shared
helper is a trivial pure function and the food Fick tests guard equivalence.

### D2 — Additive Fick fields on `PredatorParams`, per-signal `D`, exponential default

`PredatorParams` gains `gradient_field_mode: str = "exponential"`,
`diffusion_coefficient: float | None = None`, `assay_time: float = 1.0`, and a
`fick_length()` method identical in form to `ForagingParams.fick_length()` (so predator `D`
is independent of food `D`). `PredatorConfig` (pydantic) gains the matching fields +
`to_params`. Defaults reproduce today's exponential behaviour exactly.

*Why:* mirrors the established food pattern; per-signal `D` is the whole point ("food vs
sulfolipid diffuse differently"); the `None`→`decay` fallback preserves continuity with the
tuned exponential scale.

### D3 — Route all three predator field readers through `_predator_field_magnitude`

`get_predator_concentration` (scalar) and `_compute_predator_gradient_vector` (the
direction × magnitude "pull") both replace their inline `exp(−distance/decay)` with
`self._predator_field_magnitude(distance)`. `get_predator_sulfolipid_concentration` is a
thin alias of `get_predator_concentration` → inherits Fick for free. The `distance == 0`
boundary (returns `strength`) is preserved (both kernels give `strength` at `r=0`).

### D4 — Byte-stability is the gate

Predator default `exponential` + food path numerically identical ⇒ every existing config
(grid and continuous) produces identical field values. Gates: the existing food Fick tests

- the predator byte-equivalence / field tests stay green; a focused test asserts the
  predator field at the exponential default equals the pre-change formula.

## Risks / Trade-offs

- **Food field regression from the shared-helper refactor** → the helper returns identical
  values (same formulas); the existing food Fick tests (`test_fick_*` in `test_env.py`) are
  the guard, plus a direct equality check.
- **Predator `D` default semantics** → with `diffusion_coefficient=None`, `fick_length()`
  returns `gradient_decay_constant`, so selecting `fick` without a `D` still gives a
  sensible Gaussian at the tuned scale (matches the food fallback) — documented.
- **Scope creep into oxygen/pheromone** → explicitly fenced in Non-Goals; only the predator
  field changes here.

## Migration Plan

No data migration. Behaviour change is opt-in per config (`predator.gradient_field_mode: fick`). Existing configs are byte-identical (exponential default). Rollback = revert the
additive fields + the helper extraction. The calibrated continuous foraging+predator
configs adopt predator-Fick when authored (the `continuous_behaviours` / `continuous_tuning`
tracks); this change ships the mechanism, not a config flip.

## Open Questions

- Should predator `fick` reuse food's `assay_time` or keep an independent one? (Lean:
  independent per-field `assay_time`, consistent with per-signal `D` — different chemicals
  assayed over the same plate time, but the field exposes it per-signal for generality.)
