# Design — real-worm behavioural-thermotaxis validation

## Context

Extends the archived `realworm-behavioural-validation` capability (Logbook 035) from chemotaxis to a
second modality, thermotaxis, reusing its metric / grading / harness layer verbatim. The only genuinely
new logic is (1) how a homeostatic thermal drive is captured and (2) the thermal reference set. All
downstream analysis is the committed 035 code.

## Decisions

### D1 — Reuse the 035 machinery via a setpoint-adjusted drive (not a new metric family)

The 035 bias-curve metrics operate on a per-step scalar **drive**, its temporal derivative, and a
**gradient direction/strength** (`BehaviourStep.concentration / dc_dt / grad_dir / grad_strength`).
They are agnostic to what the drive *is*. Thermotaxis is homeostatic, so define:

- `drive = −|T − Tc|` — 0 at the cultivation temperature `Tc`, more negative the further from comfort
  (monotonic "good"). Its one-step derivative `d(drive)/dt` is negative exactly when the worm moves
  **away** from `Tc` — the thermal analogue of `dC/dt < 0` (moving away from food). So the **same**
  klinokinesis curve (turn-rate elevated when `d(drive)/dt < 0`) applies.
- `toward-comfort direction` — the direction that reduces `|T − Tc|`: up the thermal gradient (toward
  warmer) when `T < Tc`, the opposite when `T > Tc`. Captured as `grad_dir`, so the **same** weathervane
  curve (curving toward `grad_dir`) applies.

**Consequence:** no change to `behavioural_curves.py` / `behavioural_agreement.py` / the harness
compute path. The setpoint math lives only in the capture; the reference set differs; the harness gains
a `--modality` selector. This is the smallest surface that adds a second modality.

**Alternative rejected** — a separate thermal metric family (turn-rate vs raw `dT/dt`, weathervane vs
the raw thermal gradient). It would double the metric code and mis-model the homeostatic setpoint
(raw `dT/dt` conflates "toward Tc" with "toward warmer"). The setpoint-drive framing is both simpler
and biologically correct.

### D2 — Capture the setpoint drive in the agent, gated by modality (byte-identical when `food`)

A `_behaviour_capture_fields(sensing, temporal, food_grad_dir, food_grad_strength, sensing_pos, temperature)` helper returns the `(drive, drive-derivative, toward-drive direction, gradient strength)` for the `BehaviourStep`. For `food` it returns exactly today's values (byte-identical). For
`thermotaxis` it computes the setpoint drive from `env.thermotaxis.cultivation_temperature`, the live
`env.get_temperature(sensing_pos)`, and the live `env.get_temperature_gradient(sensing_pos)` (which
points toward warmer, so flipped when `T > Tc`). The one-step derivative reads the previous captured
step's drive (`self.behaviour[-1].concentration`); the first step's derivative is 0. `kinematics()`
re-wraps `grad_dir − heading`, so an unwrapped toward-comfort angle is fine.

Sampling the **live** thermal gradient (not a post-hoc recompute) mirrors 035's live-food-gradient
snapshot and is necessary because the sensing pipeline pops gradient keys under non-oracle sensing.

### D3 — Thermal references are sign-only (direction, not magnitude)

Thermal bias magnitudes (turn-rate ratio, curving slope in rad/mm) are not comparable to the published
thermotactic-bias statistics (fraction-of-runs-biased, deg/mm) — the same units caveat 035 applied to
the weathervane, now for all four thermal statistics. So every thermal reference has
`magnitude_range = null` and is graded **on sign** (a significant correct-direction bias →
REPRODUCED). Citations: Luo et al. 2014 primary (it decomposes thermotaxis into klinokinesis +
klinotaxis), with Ryu & Samuel 2002 / Clark et al. 2007 supporting.
`load_bias_signatures(modality="thermotaxis")` resolves the thermal file (with a hardcoded fallback);
an explicit missing path still raises (035's review fix).

### D4 — A thermotaxis-dominant cell (the thermal analogue of the food-only klinotaxis cell)

035 captured on the food-only klinotaxis foraging cell so foraging **was** the task. The thermotaxis
arm needs an analogue where reaching / tracking `Tc` is the dominant drive (not incidental
lethal-zone avoidance while foraging), so the captured behaviour is genuinely thermotactic. The cell:
a linear thermal gradient with the spawn region off-setpoint and the reward rewarding comfort (a
`Tc`-seeking / isotherm-tracking assay), thermotaxis sensing `klinotaxis` (head-sweep), foraging
pressure minimised. The exact difficulty (gradient strength, comfort band width, reward weights) is
**calibrated at the smoke** (as 035 calibrated `θ_sharp`), then frozen for the panel.

### D5 — Same evaluation shape as 035

MLP primary (gating-arm architecture) + connectome companion, n ≥ 8, post-convergence tail, the same
θ_sharp / tail-window / curving-rate-floor robustness, the same REPRODUCED/PARTIAL/ABSENT grading with
80% bootstrap CIs.

A **specificity control is available and is the direct analogue of 035's food-derivative arm**:
`thermotaxis_mode: derivative` (temporal thermal sensing — dT/dt, no head-sweep) removes the spatial
thermal-gradient signal while the captured setpoint drive stays intact (the capture samples the true
thermal gradient regardless of the worm's sensing mode). The prediction mirrors 035: klinokinesis
persists (the temporal channel is intact) while the weathervane collapses if it is sensor-driven. Run
it **iff the panel shows a positive thermal weathervane** (as the food weathervane did) — it is the
control that would establish the thermal weathervane as sensor-driven rather than a geometry confound.
It is optional at authoring time because thermotaxis may reproduce klinokinesis only (see Risks); the
smoke/panel decides.

## Risks

- **The cell may not converge to clean thermotaxis** (the substrate under-demands it, or the reward is
  mis-weighted). Mitigated by the smoke-calibration gate before the panel — if no arch shows a thermal
  bias after honest calibration, that is itself a recordable substrate finding (as the 032 ARS null
  was), not a silent failure.
- **Setpoint-drive derivative is a one-step difference** (vs food's STAM-smoothed `dC/dt`). Acceptable
  — it is the drive's rate of change and feeds the same sign-based klinokinesis; documented in the
  logbook.
