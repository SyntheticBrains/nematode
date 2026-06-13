## MODIFIED Requirements

### Requirement: Signal-type-specific static Fick-shaped gradient geometry

The environment SHALL be able to compute **diffusing chemical** concentration fields using a **frozen analytic Fick (Gaussian) kernel** `C(r) = A · exp(−r² / (4·D·t_assay))` with **per-signal diffusion coefficients** `D` (e.g. food, predator sulfolipid, pheromone, CO₂), selectable as a per-field mode on the continuous-2D substrate. The pre-existing exponential-decay kernel SHALL remain the default field mode for every field so that legacy and discrete-grid configurations are unaffected and byte-stable. The Fick mode SHALL be selectable **independently per diffusing chemical field, each with its own `D`** — this includes the **food** field and the **predator sulfolipid** distal-chemo field (a diffusing chemical the worm senses at a distance; Liu et al. 2018). Fields that are **not point-source diffusion** are out of scope and retain their existing behaviour: **temperature** (an imposed thermal-conduction gradient) and the **oxygen base gradient** (an imposed/boundary aerotaxis gradient). The field SHALL remain **static** (frozen at assay time); time-evolving diffusion is explicitly out of scope (gated stretch).

#### Scenario: Fick field mode produces the Gaussian kernel

- **WHEN** a configuration selects the Fick field mode for a diffusing chemical signal (food or predator sulfolipid)
- **THEN** the concentration at distance `r` from a source SHALL be evaluated as `A · exp(−r² / (4·D·t_assay))` using that signal's configured `D`, rather than the exponential-decay kernel

#### Scenario: Per-signal diffusion coefficients set distinct geometry

- **WHEN** two diffusing chemical signals (e.g. food and predator sulfolipid) are configured with different `D` values
- **THEN** their static field geometries SHALL differ in spatial spread accordingly (larger `D` → broader gradient), independently per field

#### Scenario: Predator sulfolipid field supports the Fick mode

- **WHEN** the predator field is configured with the Fick mode
- **THEN** the predator concentration / distal-chemo (sulfolipid) field SHALL use the Fick Gaussian kernel with the predator's own configured `D`, replacing the exponential kernel for that field

#### Scenario: Imposed/boundary fields are excluded

- **WHEN** temperature or the oxygen base gradient is computed
- **THEN** it SHALL retain its existing imposed/boundary-gradient behaviour and SHALL NOT be evaluated with the Fick chemical-diffusion kernel (these are not point-source diffusion)

#### Scenario: Legacy exponential kernel remains the default

- **WHEN** a configuration does not select the Fick field mode for a given field
- **THEN** the environment SHALL use the existing exponential-decay kernel for that field, and discrete-grid field values SHALL remain byte-stable
