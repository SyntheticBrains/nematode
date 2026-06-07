## Context

T5 delivered the continuous-2D substrate (Gate 2 GO, [logbook 027](../../../docs/experiments/logbooks/027-platform-refactor-continuous-2d.md)); T6 is the env-fidelity half of the upgrade. The current ("Rung-0") implementation:

- **Field geometry** is a superposition of exponential-decay point sources — `C_raw = Σ gradient_strength · exp(−dist / gradient_decay_constant)` (`env/env.py:2021-2060` for the scalar; `:1937-1975` for the gradient vector), defaults `strength=1.0`, `decay=10.0` grid / `8.0` continuous. Not Fick-shaped.
- **The sensed scalar** is `tanh(C_raw · 1.0)` → [0,1] (`env/env.py:70`). There is **no adaptation / background tracking / Weber transform** — plain tanh saturation. STAM supplies a weighted finite-difference temporal derivative `dC/dt` (`agent/stam.py:336-372`), and klinotaxis supplies a lateral difference `tanh((right−left)·lateral_scale)` (`brain/modules.py:1061-1074`). `SensingMode` ∈ {ORACLE, TEMPORAL, DERIVATIVE, KLINOTAXIS} (`utils/config_loader.py:725-754`).
- **Continuous-2D sources** sit on the integer lattice (`self.foods: list[tuple[int, int]]`, `env/env.py:1417`) within the continuous arena, and the "Euclidean" nearest-food distance is `round(math.hypot(...))` — int-snapped (`env/continuous_2d.py:275-282`). The spec already records these as explicit T6 deferrals.
- **Renderer**: pygame `PIXEL` theme (default), `_cell_to_pixel(grid_x:int, grid_y:int)` cell-snaps (`env/pygame_renderer.py:237-247`); zone overlays use alpha-blit (`env/sprites.py`) — a template for a concentration heatmap.

The Phase 6 Rung-2 commitment (roadmap § Continuous environment + sensory physics), rebalanced 2026-06-04, puts the fidelity budget in the **sensor** (adaptive-threshold / biphasic LN — Kato et al. 2014 *Neuron*; Levy & Bargmann 2020 *Neuron*), keeps **static** Fick-shaped gradient geometry, and demotes a live diffusion PDE to a gated stretch. **T7 depends on T6** for its ranking + behavioural-chemotaxis validation.

## Goals / Non-Goals

**Goals:**

- A configurable **adaptive-threshold / biphasic chemosensory sensor** layered on the sensed concentration, with log-concentration retained as a documented baseline special case. (PRIMARY.)
- **Static, signal-type-specific Fick-shaped gradient geometry** (per-signal D) replacing the exponential-decay kernel on the continuous substrate.
- **Float source placement + true Euclidean fields** on the continuous substrate (lift the int storage + int-rounded distance).
- A **continuous-substrate fidelity renderer** (non-gating seed) with concentration/gradient/sensor-state overlays.
- A **step-input adaptation-transient validation** as the load-bearing analysis check, plus the env-fidelity-gain quantification + T6 logbook.

**Non-Goals:**

- Dynamic diffusion (`∂C/∂t = D∇²C`) + source depletion/replenishment — gated **stretch**, only on a written behavioural justification.
- Brain-interface changes (brains consume sensor output unchanged; no per-architecture branches), discrete-grid changes (byte-stable), 3D physics, Sibernetic interop, aerotaxis/pheromone-behaviour restoration, multi-agent.
- Thermosensory (AFD) adaptation — this tranche's adaptive sensor is **chemosensory**; thermotaxis consumes the continuous Euclidean fields only.

## Decisions

### D1 — Adaptive sensor = background-tracking leaky integrator + relative (Weber/fold-change) coding; exact readout settled by the step-input gate

What is **locked**: a Linear-Nonlinear stage in which the chemosensory signal is coded **relative to a slowly-tracked background**, not as a static squash. A leaky-integrator background `B_t = (1−α)·B_{t−1} + α·C_t` (adaptation timescale τ via α) supplies the background; the readout is **biphasic relative coding** (ON when the signal rises above background, OFF when it falls below). **Why:** the current tanh has no adaptation at all; plain log-concentration is the roadmap's named under-powered special case. **Reuse, not reinvent:** STAM already maintains a per-channel leaky buffer (`agent/stam.py`) — the background tracker uses the same machinery rather than a new buffer.

What is **deliberately NOT locked** — the exact readout form, which the D7 step-input adaptation-transient gate (and, if needed, the D6 iteration sub-tranche) settles empirically. Two **co-primary** candidates, A/B'd against the step-input target:

1. **Instantaneous contrast on the magnitude channel** — `s = (C_t − B_t)/(C_t + B_t + ε)` (divisive/subtractive contrast against background), optionally squashed.
2. **Fold-change detection on the derivative channel** — normalize the temporal derivative by concentration, `(dC/dt)/C ≈ d(log C)/dt`, applied to the `dC/dt`/turning channel that klinotaxis actually acts on. This is the more literature-faithful reading of the hyper-Weber result (Kato et al. 2014; Levy & Bargmann 2020): the *derivative response scales with background*, it is not only an instantaneous magnitude contrast.

**Channel-interaction decision (must be explicit in the spec):** the system already computes `dC/dt` via STAM and klinotaxis biases turning on it, so the spec MUST state whether adaptation **reshapes the existing derivative/turning channel** (candidate 2, fold-change) or **adds a standalone contrast magnitude channel** (candidate 1) — these are different behavioural contracts, not interchangeable tunings. Default expectation: the derivative channel carries the Weber/fold-change normalization (the behaviourally load-bearing path), with the contrast magnitude as a secondary signal.

**Alternatives considered (rejected / deferred):** pure `log(1+C)` (kept as the baseline ablation mode, not the headline); pure tanh (status quo, no adaptation); full Kato-style two-state LN with explicit separate rise/fall kinetics or an incoherent-feed-forward-loop FCD circuit (deferred to the iteration sub-tranche if both co-primary forms underfit the step-input target).

### D2 — Adaptation lives in the sensor transform, not the env field; composes with SensingMode

Keep the **field generation (env) separate from the sensor transform (sensing)**: the env emits raw concentration; the adaptive stage transforms it. The transform is a config-selected layer that **composes with** the existing `SensingMode` (klinotaxis + adaptive together), applied at the `brain/modules.py` core-feature stage / a dedicated adaptive-sensor step, keyed per chemosensory channel. **Why:** preserves the `klinotaxis-sensing` capability boundary and the strict env/sensor separation; avoids a per-mode combinatorial explosion. **Alternative:** a new fused `SensingMode.ADAPTIVE` — rejected because adaptation is orthogonal to head-sweep geometry and should stack on klinotaxis.

### D3 — Static Fick geometry = frozen analytic Gaussian kernel with per-signal D, as a selectable field mode

Replace the exponential-decay kernel with the **frozen analytic Fick solution at assay time** — the instantaneous-point-source Gaussian `C(r) = A · exp(−r² / (4·D·t_assay))`, per-signal D (food / pheromone / CO₂). The existing `gradient_decay_constant` maps to `√(4·D·t_assay)`, so the field is a drop-in kernel swap in `get_food_concentration` / `_compute_*_gradient_vector`, selected by a field-mode flag. **Why Gaussian over the steady-state forms:** the 2D steady-state point-source solution (`∼ −ln r`) and 3D (`∼ 1/r`) are singular at the source; the time-frozen Gaussian is bounded and is the form the chemotaxis literature actually plots. **Backward-compat:** the exponential kernel stays the default for existing/grid configs (byte-stable); Fick is opt-in for the new continuous configs.

### D4 — Float sources + Euclidean confined to the continuous env

Lift the two deferrals: store continuous-env sources as float and drop the `round()` in `get_nearest_food_distance_for`. **Approach:** confine the float type to the continuous env (override placement/storage and the distance read) rather than rippling `self.foods` across the grid path, keeping the discrete grid `int` + byte-stable. The spec already pre-committed this; a grid byte-stability regression guards it.

### D5 — Renderer: world→pixel + heatmap overlay on the existing pygame PIXEL path (non-gating)

Adapt `_cell_to_pixel(int,int)` → a `_world_to_pixel(float,float)` map, blit sprites at sub-cell positions, add a continuous theme flag; layer a concentration-field heatmap (via `surfarray` + colormap, reusing the zone-overlay alpha-blit template), gradient vectors, and adaptive-sensor state. Reuse `scripts/export_screenshot.py` for frame export; matplotlib only for static logbook figures. **Why pygame, not a new stack:** continuous coords are simpler here than the grid path and it reuses sprites/themes/export, consistent with the batch/headless/reproducibility posture.

### D6 — Pre-structured adaptive-sensor iteration sub-tranche

If the step-input adaptation transient (D7) needs tuning (τ_adapt, ON/OFF balance, filter shape), absorb it as a **bounded** T6 sub-tranche of parameter sweeps — no Gate re-open (Gate 2 already closed at T5). The analysis gate is the acceptance criterion, not a fixed first-shot parameterisation.

### D7 — Validation: step-input adaptation transient is the load-bearing check

Present a concentration **step input** to the sensor in isolation and measure the transient — peak response on the step, then relaxation back toward baseline as the background tracker catches up — and compare its shape against the adaptation signature (Weber-like invariance to background level), quantified **against the log-concentration baseline**. This, plus a field-geometry fidelity comparison (static Fick vs Rung-0 on a smoke task), is the T6.analysis gate before T7.

## Risks / Trade-offs

- **Adaptive sensor needs M4-style iteration** → D6 bounded iteration sub-tranche; the log-concentration baseline is always available as a working fallback so T7 is never blocked on sensor tuning.
- **Float-source type change touches shared env code** → D4 confines float to the continuous env; a grid byte-equivalence regression test guards the discrete path.
- **Fick-kernel + adaptation change the absolute concentration scale** → could disturb the entropy-0.10 continuous calibration and the existing continuous configs. Mitigation: keep the exponential kernel default-off-selectable, document the scale change, and re-tune the continuous arms at **T7.prep on the finalised substrate** (already planned in the tracker) rather than now.
- **Adaptation confounds the T4→T7 env-upgrade delta (RQ5)** → adaptation *is* part of the intended substrate upgrade; the log-concentration baseline + from-scratch arm isolate its contribution, and the discrete grid is untouched so the T4 baseline is unmoved.
- **Renderer scope creep** → non-gating seed, capped at static-Fick + adaptive-state overlays; no web/game-engine stack.

## Migration Plan

- Field kernel is mode-selected: exponential (default, legacy + grid byte-stable) vs Fick (opt-in, new continuous configs). Adaptive sensor is opt-in via config; existing configs keep the tanh/log baseline behaviour.
- Float sources confined to the continuous env; grid env unchanged.
- Rollback: config flags revert any arm to Rung-0 (exponential kernel + non-adaptive readout) with no code change.

## Open Questions

- **Adaptive readout form** — the two D1 co-primary candidates (instantaneous magnitude contrast `(C−B)/(C+B)` vs derivative-channel fold-change `(dC/dt)/C`), and whether a fuller two-state LN / IFFL-FCD is needed; resolved against the D7 step-input target in the specs / iteration sub-tranche. The **channel-interaction** choice (derivative-channel vs standalone magnitude channel) is a behavioural contract the spec MUST pin, not a free tuning.
- **Per-signal D values + `t_assay`** — calibrate so the food field's spatial spread stays in the ballpark of the current tuned `decay≈8` (continuity with the T5 continuous configs); pheromone/CO₂ D only bite if those behaviours are active (deferred), so food D is the T7-relevant one.
- **Renderer overlay depth** — how much adaptive-sensor internal state to visualise; settled at T6 scoping, non-gating either way.
