## Why

T6 (`add-rung2-chemical-gradients`) shipped static Fick-shaped gradient geometry but
wired it on the **food** field only, and the `chemical-gradient-fidelity` spec
explicitly excluded the predator field — *"the predator concentration field is out of
scope and retains the exponential kernel (predators-on-continuous are not yet
exercised)."* That premise no longer holds: `add-continuous-predator-kinematics` (#226)
made predators continuous with Euclidean detection/damage. The predator **sulfolipid**
distal-chemo field is a genuinely *diffusing chemical* (Liu et al. 2018 — *C. elegans*
detects it at a distance via ASH/ASI), so it should share the same physically-faithful
Fick geometry as food, with its **own** diffusion coefficient. This is the design intent
of "signal-type-specific Fick" (per-signal `D` → distinct geometries) and closes the last
fidelity-coherence gap on the chemical fields: with this, every *diffusing* chemical
signal the worm senses uses the diffusion-faithful kernel, strengthening the
continuous-physics platform claim and the T7 real-worm-validation defensibility.

## What Changes

- **Predator-sulfolipid field gains the Fick mode + per-signal `D`.** Add
  `gradient_field_mode` (`exponential` | `fick`), `diffusion_coefficient`, `assay_time`,
  and a `fick_length()` helper to the predator field config (mirroring the food field),
  so the predator chemical field can be Fick-shaped with a diffusion coefficient distinct
  from food's.
- **Predator field magnitude becomes mode-aware.** `_compute_predator_gradient_vector`,
  `get_predator_concentration` (and its alias `get_predator_sulfolipid_concentration`)
  evaluate the configured kernel — `exponential` (default) or the frozen Fick Gaussian
  `A·exp(−r²/(4·D·t_assay))` — via a **shared kernel helper** also used by the food field
  (no behaviour change to food; it just delegates the math).
- **Exponential remains the default → byte-stable.** The predator field defaults to the
  existing exponential kernel, so every existing config and the discrete-grid path are
  unchanged.
- **Scope — diffusing chemicals only.** Predator-sulfolipid is the primary (now-relevant)
  target. **Temperature** (thermotaxis) and the **oxygen base gradient** (aerotaxis) are
  **explicitly out of scope** — they are *imposed / boundary* gradients (thermal
  conduction / plate-edge O₂), not point-source diffusion. **Pheromone** is a documented
  follow-up (it is genuinely diffusing but currently uses Manhattan distance and is
  multi-agent — its own change), reusing this change's per-signal Fick infrastructure.

## Capabilities

### New Capabilities

<!-- None — this extends the existing chemical-gradient-fidelity capability. -->

### Modified Capabilities

- `chemical-gradient-fidelity`: the "Signal-type-specific static Fick-shaped gradient
  geometry" requirement currently states the predator concentration field is out of scope
  and retains the exponential kernel. This change retires that exclusion: the Fick mode +
  per-signal `D` apply to the **predator-sulfolipid** chemical field too (and, by the same
  infrastructure, any future diffusing chemical field), while **imposed/boundary** fields
  (temperature, the oxygen base gradient) remain out. Exponential stays the default;
  byte-stability is preserved.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/env.py` — additive Fick fields +
    `fick_length()` on `PredatorParams`; a shared mode-aware field-magnitude helper
    (`_food_field_magnitude` refactored to delegate to it — numerically identical); the
    three predator field methods routed through the predator magnitude.
  - `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `PredatorConfig`
    gains `gradient_field_mode` / `diffusion_coefficient` / `assay_time` (+ `to_params`).
- **Tests:** new predator Fick-kernel tests (Gaussian; differs from exponential; source =
  strength; per-signal `D` → distinct geometry; exponential-default byte-stability); the
  existing food Fick tests must stay green (the shared helper is numerically identical).
- **Downstream:** the per-signal `D` literature calibration (food / sulfolipid / CO₂) and
  pheromone Fick remain future fidelity work (`phase6-tracking` T6.gradients.1 note /
  pheromone follow-up). No new dependencies.
