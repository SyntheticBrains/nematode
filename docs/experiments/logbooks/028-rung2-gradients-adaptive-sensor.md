# 028: Rung-2 Chemical Gradients + Adaptive Chemosensory Sensor (Phase 6 Tranche 6)

**Date:** 2026-06-08
**Tranche:** Phase 6 T6 (env fidelity). **Gate:** none (T6 has no decision gate; feeds T7).
**Change:** [`add-rung2-chemical-gradients`](../../../openspec/changes/add-rung2-chemical-gradients/).
**Status:** complete (gating scope) — renderer deferred as a non-gating seed.

## Objective

Deliver the Phase 6 Rung-2 env-fidelity commitment on top of the T5 continuous-2D
substrate (logbook 027): a **static signal-specific Fick-shaped gradient geometry**
paired with an **adaptive-threshold / biphasic chemosensory sensor** (the PRIMARY,
high-leverage deliverable per the 2026-06-04 checkpoint — the field invests its
fidelity budget in the *sensor*, not a live diffusion PDE), plus the deferred T5
float-source / Euclidean-field work. Feeds the T7 apples-to-apples ranking +
behavioural-chemotaxis validation.

## What shipped

| Group | Scope |
|----|-------|
| 1 | Static Fick-shaped gradient geometry — selectable `gradient_field_mode` (`exponential` default / `fick`); frozen analytic Gaussian kernel `exp(-(r/L)^2)`, `L = sqrt(4·D·t)` (or `gradient_decay_constant`). Food/chemical field only; grid byte-stable. |
| 2 | Float source placement + true Euclidean + continuous-field sensing — continuous-2D sources placed at real-valued coords, `get_nearest_food_distance_for` un-rounded, lateral klinotaxis samples real-valued points (no cell snap). |
| 3 | **Adaptive/biphasic chemosensory sensor (PRIMARY)** — leaky-integrator background + 3 readouts (fold-change / contrast / log baseline); on the agent (stateful, like STAM); disabled-by-default → byte-identical. |
| 6 | Step-input adaptation-transient validation (the load-bearing analysis check). |

## Design decisions realised

- **Adaptive sensor model (D1).** Background `B_t = (1-α)·B_{t-1} + α·C_t` via a
  shared `LeakyIntegrator` (STAM exposes no public baseline API, confirmed during
  the spec review, so the utility is authored fresh following STAM's decay pattern).
  Three config-selectable readouts; the readout fixes the **channel interaction**:
  `fold_change` reshapes the derivative/turning channel `(dC/dt)/(C+ε)` (the `+ε` is
  load-bearing — raw `/C` is singular at the common `C≈0` regime), `contrast`
  reshapes the strength channel `(C-B)/(C+B+ε)`, `log` is the `log(1+C)` baseline.
  **Both co-primary readouts pass the step-input gate** (below), so the D6 iteration
  sub-tranche was **not** needed (marked N/A).
- **Fick geometry (D3).** Gaussian kernel chosen over the singular steady-state
  forms; `decay → L` continuity anchor verified (both kernels equal `e⁻¹` at `r=L`).
- **Adaptation lives in the sensor transform, not the env field (D2)**, applied as a
  dedicated step on the chemosensory channel after the STAM derivative — composes
  with `SensingMode`, leaves the env/sensor boundary intact, and is scoped to
  chemosensory channels only (never thermo/predator).

## Validation — step-input adaptation transient (the load-bearing gate, D7)

A `background → background+step` series driven through each readout in isolation
(`validation/adaptive_sensor.py`), measuring the response-channel **peak** and
**relaxation ratio** (`|final|/|peak|`; low ⇒ adapts back), and **Weber spread** of
the peak across absolute backgrounds for the same relative step (low ⇒ invariant):

| Readout | peak | relaxation_ratio | Weber spread |
|---|---|---|---|
| contrast (adaptive) | 0.428 | **0.001** (adapts fully) | **0.0056** (invariant) |
| fold_change (adaptive) | 0.664 | 0.000 (impulse decays) | — |
| **log (baseline)** | 0.262 | **1.000** (no adaptation) | **1.639** (background-dependent) |

**Headline:** the adaptive readouts reproduce the hyper-Weber signature — a peak on
the step that relaxes back to baseline, with a peak ~invariant to background level —
while the log baseline neither relaxes nor stays invariant (Weber spread **~290× worse**).
This is the T6 acceptance evidence the adaptive sensor was scoped to produce.

## Env-fidelity geometry (Fick vs Rung-0 exponential, `L = decay = 8`)

| r | exponential `exp(-r/8)` | Fick `exp(-(r/8)²)` |
|---|---|---|
| 4 | 0.607 | 0.779 (broader near source) |
| 8 | 0.368 | 0.368 (continuity anchor, `e⁻¹`) |
| 16 | 0.135 | 0.018 (sharper far field) |

The Gaussian is flatter near the source and falls off faster in the tail — the
physical diffusion profile, vs the exponential placeholder.

End-to-end: a 3-run headless smoke on
`mlpppo_small_continuous2d_rung2_klinotaxis.yml` (fick gradients + fold-change
adaptive sensor + float sources) runs clean (no NaN/crash); behavioural performance
is untrained-smoke-level and out of scope here — multi-seed tuning of the
readout / adaptation rate against the finalized substrate is **T7-prep**.

## Deferrals & known limitations

- **Continuous-substrate fidelity renderer (group 5) — deferred (non-gating seed).**
  The plan scopes T6.render as a "seed task — flesh out at T6 scoping," non-gating
  for T6's analysis. The existing grid render was made float-safe (snaps continuous
  sources) so continuous runs work headless; the full pygame `world→pixel` +
  concentration-heatmap renderer is a follow-up (the detailed approach is recorded
  in the tracker). Static logbook figures of the step-input transient are likewise
  deferred — the quantitative tables above are the load-bearing evidence.
- **Dynamic-diffusion PDE (T6.gradients.2) — explicitly-gated stretch, not pursued.**
  Pursue only on a concrete behavioural need (depletion-driven area-restricted
  search); left as gated future scope.
- **Predator field stays exponential** (predators-on-continuous not yet exercised);
  Fick mode is chemical-only.
- **food_history** reporting record snaps continuous float sources to cells (the
  worm senses the real float field; the int-typed result schema is unchanged).

## Conclusions

T6 ships its gating scope: static signal-specific Fick gradient geometry + the
adaptive/biphasic chemosensory sensor (the PRIMARY deliverable) + the float-source /
Euclidean / continuous-field deferrals from T5, with the step-input adaptation-
transient gate passed for both co-primary readouts and the adaptive coding shown to
beat the log baseline on Weber invariance by ~290×. The substrate is ready for the
T7 L2 re-run.

## Next steps (T7-prep)

- Carry the entropy-0.10 continuous lesson (logbook 027) + multi-seed-tune the
  adaptive readout / adaptation rate (`α`) on the finalized substrate.
- Predator + thermotaxis continuous bring-up (T7.prep.continuous_behaviours).
- Flesh out the continuous fidelity renderer (deferred group 5) for the T7 figures.

## Data references

- Change: [`add-rung2-chemical-gradients`](../../../openspec/changes/add-rung2-chemical-gradients/).
- Sensor + validation: `agent/adaptive_sensor.py`, `validation/adaptive_sensor.py`.
- Sample config: `configs/scenarios/foraging/mlpppo_small_continuous2d_rung2_klinotaxis.yml`.
