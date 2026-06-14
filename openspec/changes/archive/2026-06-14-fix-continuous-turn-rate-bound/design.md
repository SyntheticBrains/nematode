# Design

A small env-side kinematic-fidelity fix. Three decisions:

## D1 — Bound at the env rescale, not the brain action space

The brains emit a **normalized** `(speed, turn)` so they stay substrate-independent (the docstring on
`move_agent_normalized` makes this an explicit design intent); the physical scale is the environment's
job. The speed already follows this — `speed = speed_norm * max_step_mm`. The turn did not: it was
hard-coded `turn = turn_norm * math.pi`. So the fix belongs at that exact line — replace `π` with a
physical `max_turn_rad` — **not** at the brains' `[-1, 1]` bound (bounding the normalized action would
make the brain env-aware and leave the constant's units ambiguous). This keeps speed and turn symmetric:
both are normalized in the brain and scaled by an env kinematic parameter (`max_step_mm`, `max_turn_rad`).

## D2 — `max_turn_rad` on `Continuous2DParams` (configurable), default re-validated

It is a `Continuous2DParams` / `Continuous2DConfig` field (like `max_step_mm`, `capture_radius_mm`,
`predator_damage_radius_mm`), so it is config-settable and sweepable, and the eventual `T7.validation`
need to match real-worm turn-rate distributions can be met by tuning it. The **default** is chosen by
re-validation: sweep `max_turn_rad ∈ {0.5, 0.79, 1.05}` rad (≈29°/45°/60° per step) on the C1 MLP
foraging task, pick the **tightest value that still converges** to the 10-food target (tighter = more
realistic), and confirm the spin is gone visually. The previous behaviour was `π` (180°/step); the new
default is far below it.

## D3 — Brains and grid path untouched

No brain change — the normalized `[-1, 1]` turn bound is correct and stays. The continuous sampling /
log-prob / entropy math is unaffected (it operates on the normalized action). The discrete grid
environment does not use `move_agent_normalized`, so its cardinal movement is byte-stable. A bound does
not by itself forbid a low-speed in-place rotation at the max rate; if re-validation shows that residual
matters, a follow-up (min-speed coupling or turn-change penalty) can be considered — but removing the
±180°/step reversal is the load-bearing fix.
