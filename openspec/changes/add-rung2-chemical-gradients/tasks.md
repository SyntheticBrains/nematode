# Tasks: Rung-2 Chemical Gradients + Adaptive Chemosensory Sensor (Phase 6 Tranche 6)

Implements the `add-rung2-chemical-gradients` change. Maps to the `phase6-tracking`
tracker T6 rows (T6.sensor.1, T6.gradients.1/.2/.4, T6.physics.1, T6.render,
T6.analysis, T6.logbook). Order is dependency-first: env field + float sources
underneath the sensor, sensor next (the primary deliverable), then renderer,
validation, and logbook. Tick the corresponding tracker row as each lands.

## 1. Static Fick-shaped gradient geometry (env field)

- [x] 1.1 Add a **field-mode** selector for chemical signals (exponential-decay = default/legacy; `fick` = new) on `ForagingConfig` (`config_loader.py:276`, alongside `gradient_decay_constant`/`gradient_strength`); default preserves current behaviour. Chemical fields only — the predator concentration field stays exponential (predators-on-continuous not yet exercised).
- [x] 1.2 Implement the frozen analytic Gaussian Fick kernel `C(r) = A·exp(−r²/(4·D·t_assay))` as the `fick` kernel in the food/chemical concentration + gradient-vector computations (`env/env.py` `get_food_concentration` / `_compute_food_gradient_vector`), parameterised by per-signal `D` and `t_assay`; map the existing `gradient_decay_constant` to `√(4·D·t_assay)` for continuity.
- [x] 1.3 Per-signal `D` + `t_assay` configuration (food, and pheromone/CO₂ placeholders) on `ForagingConfig`/the relevant signal configs; document calibration so the food field's spread stays near the tuned continuous `decay≈8`.
- [x] 1.4 Grid byte-stability regression: discrete-grid + legacy (exponential) field values unchanged (test).
- [x] 1.5 Unit tests: Fick kernel shape, per-signal `D` → distinct spread, mode selection.

## 2. Float source placement + Euclidean fields (continuous-2D)

- [x] 2.1 Store continuous-2D sources as real-valued (float) coordinates, confined to the continuous env (override placement/storage), leaving the discrete grid `int` + byte-stable.
- [x] 2.2 Drop the integer rounding in `Continuous2DEnvironment.get_nearest_food_distance_for` (true Euclidean), and make lateral klinotaxis sampling read the continuous field at real-valued sample positions (no integer-cell snap).
- [x] 2.3 Tests: float source placement within bounds + min-separation; Euclidean distance un-rounded; continuous-field sampling at sub-cell positions; grid env byte-stable.

## 3. Adaptive chemosensory sensor (PRIMARY)

- [x] 3.1 Background tracker: a **shared leaky-integrator utility** `B_t = (1−α)·B_{t−1} + α·C_t` per chemosensory channel, following STAM's exponential-decay pattern (optionally factored out of `STAMBuffer`). NB: `STAMBuffer` exposes no public baseline API (the weighted mean is private to `get_memory_state`), so author/extract the utility rather than calling a STAM baseline method.
- [x] 3.2 Adaptive readout modes (config-selectable): (a) derivative-channel **fold-change** `(dC/dt)/(C+ε)` — the `+ε` (or `/(B+ε)`) is required, raw `/C` is singular at `C≈0`; (b) instantaneous **magnitude contrast** `(C−B)/(C+B+ε)`; (c) **log-concentration** baseline. Default = derivative-channel fold-change.
- [x] 3.3 Channel-interaction wiring at the klinotaxis core-feature stage (`brain/modules.py`): apply ONLY to chemosensory cores (`_food_chemotaxis_klinotaxis_core`, plus pheromone-chemo cores where active — NOT thermo/nociception/predator). The `*_klinotaxis_core` logic is duplicated across ~9 functions with no chokepoint, so add a shared helper or apply narrowly. Fold-change reshapes the `binary` (dC/dt) field; magnitude-contrast supplies the `strength` field; composes with `SensingMode` (klinotaxis + adaptive stack); disabled → current non-adaptive pipeline byte-identical. *(**As-built deviation**: applied **upstream in the agent sensory-assembly step** `agent._compute_temporal_data` — a single clean chokepoint where the stateful background lives — not inside the duplicated pure `brain/modules.py` cores; it reshapes `result["food_concentration"]`/`["food_dconcentration_dt"]` before they become `BrainParams`, so the cores stay pure. **Food channel only** this tranche; pheromone/CO₂-chemo extension is a follow-up — those behaviours are deferred. Fold-change→derivative, contrast→strength, disabled→byte-identical all hold. See design.md § Deviations.)*
- [x] 3.4 Config surface on `SensingConfig` (`config_loader.py:734`): enable flag, readout mode, channel-interaction mode (explicit), `α`/`τ`, `ε`; extend `validate_sensing_config` (`:861`) for any interlocks; sample config(s) on the continuous substrate. *(**As-built**: `adaptive_chemosensor_{enabled,readout,alpha,epsilon}` — the **single `readout` enum** (`fold_change`/`contrast`/`log`) is the explicit channel-interaction statement (one field, not two); `τ` is expressed as the leaky rate `α`. No `validate_sensing_config` interlock was added — fold-change consumes the STAM-derived derivative, so a meaningful signal relies on STAM/klinotaxis being enabled (the sample config does; otherwise the derivative is 0 and fold-change is inert — see design.md § Deviations).)*
- [x] 3.5 Tests: relative coding vs background; relaxation as background catches up; each readout mode; explicit channel-interaction applied; disabled-path equivalence.

## 4. Adaptive-sensor iteration sub-tranche (bounded; only if the step-input gate underfits)

- [x] 4.1 Decision point: after §6.1, if the step-input transient underfits both co-primary readouts, open a bounded parameter sweep (τ_adapt, ON/OFF balance, filter shape) — no Gate re-open. Record the decision + sweep results; otherwise mark N/A with the passing evidence.

## 5. Continuous-substrate fidelity renderer (non-gating seed)

> **REALIZED via the `add-continuous-fidelity-renderer` follow-up change (non-gating).**
> This seed was deferred at T6 (the existing grid render was made float-safe so continuous
> runs work headless; the T6 quantitative validation tables in logbook 028 are the
> load-bearing evidence). The full renderer has since landed as its own change — a
> `Continuous2DRenderer` with a camera-aware `world→pixel` map (full-arena + agent-follow
> toggle), grid-renderer parity, the fidelity overlays, a continuous PNG export, and the
> matplotlib logbook figures. Selected via `--theme pixel_continuous`. See that change for
> the detailed design/spec/tasks.

- [x] 5.1 `world→pixel` map replacing the cell-snapping `_cell_to_pixel` for a continuous render path (`env/pygame_renderer.py`); sub-cell sprite blit; continuous theme flag (`env/theme.py`). *(Done — `Continuous2DRenderer` + `Theme.PIXEL_CONTINUOUS` in `add-continuous-fidelity-renderer`.)*
- [x] 5.2 Fidelity overlays: concentration-field heatmap (`surfarray` + colormap, reusing the zone-overlay alpha-blit template in `env/sprites.py`), gradient vectors, sensor/contact zones, adaptive-sensor state. *(Done — heatmap, gradient quiver, klinotaxis + predator detection/damage rings, adaptive-sensor readout.)*
- [x] 5.3 Frame export via `scripts/export_screenshot.py` (gif/mp4); matplotlib only for static logbook/validation figures. Smoke-render one continuous episode. *(Done — continuous PNG export + `report/continuous_figures.py` (trajectory/heatmap/quiver) + a smoke-render test. **gif/mp4 remains a documented follow-up** — it would add an `imageio`/`ffmpeg` dependency.)*

## 6. Analysis + validation (the T6 gate)

- [x] 6.1 **Step-input adaptation-transient validation (load-bearing):** drive a chemosensory step input to the sensor in isolation; measure the transient (peak + relaxation toward baseline) and Weber-like background invariance; report against the log-concentration baseline.
- [x] 6.2 Env-fidelity-geometry comparison: static Fick vs Rung-0 exponential on a smoke task (field shape + a short behavioural smoke).
- [x] 6.3 Quantify the env-fidelity gain; figures via matplotlib (trajectory + field heatmap + quiver). *(Quantified numerically in logbook 028 — Weber spread 0.0056 adaptive vs 1.639 log (~290×); Fick-vs-exponential geometry table. Matplotlib figures deferred with the renderer per the group-5 note.)*

## 7. Documentation, tracker, logbook

- [x] 7.1 Update the `phase6-tracking` tracker T6 rows (T6.sensor.1, T6.gradients.1/.4, T6.physics.1, T6.render, T6.analysis) + `docs/roadmap.md` Phase 6 Tranche Tracker T6 row.
- [x] 7.2 Publish the T6 logbook (`docs/experiments/logbooks/0XX-rung2-gradients.md`): adaptive-sensor decision (which readout won the step-input gate + why), env-fidelity delta, deferrals. Required reading before T7.
- [x] 7.3 Record the dynamic-diffusion stretch (T6.gradients.2) status: pursued only on a written behavioural justification (depletion-driven area-restricted search) — otherwise explicitly left as gated future scope.

## 8. Pre-merge gate

- [x] 8.1 `openspec validate add-rung2-chemical-gradients --strict` clean; full test suite (`uv run pytest -m "not nightly"`) green.
- [x] 8.2 Targeted `pre-commit run --files <changed>` clean; full `pre-commit run -a` before push.
- [x] 8.3 Open PR; link the T6 logbook.
