## Why

Phase 8's body rung is unresolved.

- **The body result.** Through the kinematic body, on hard350, the wild type led the chemical-only
  null by +0.016 [−0.012, +0.044] (C.1e, Logbook 085). That is the cell where the point worm's lead
  was already small, +0.022.
- **Where the effect is large.** The substrate's largest resolved wiring effect is on the thermal
  cell at target 35: **gap-junction placement**, +0.096 against the gap-only null on the point worm
  (M.8, Logbook 082).
- **Why not the chemical null.** On that cell the chemical wiring's lead is about zero under every
  sign model (Logbooks 078 and 091).

M.9 asks the body rung's question where the point worm's answer is clear: **does the wild type's
gap-junction placement advantage survive embodiment?**

Decided with the user on 2026-10-12:

- the gap-only null is primary, scoped to gap questions by a dated D21 note, as M.8 did;
- the chemical-only null is read beside;
- Cook 2019 is the connectome, the one M.8's effect was measured on;
- the rung runs in stages, each gated.

## What Changes

- **A thermal body cell.** The thermal t35 cell (Cook 2019, 500 steps, target 35) through the
  kinematic body:

  - `body_model: kinematic`, `allow_reversal`, `action_space: body_drive`;
  - entropy 0.004, the body drive's per-dimension pressure.

  Arms: the wild type, the gap-only null and the chemical-only null, each learning and frozen; and
  MLP-PPO, learning and frozen, as the positive control. Configs come from a generator, with loader
  tests and a test that thermal sensing reads the body's head.

- **Stage 1, a positive control** (the new component). MLP-PPO must learn the thermal cell through
  the body, under C.1d's rule: its plateau beats its frozen floor with the 80% interval above zero,
  and every seed reaches 30%.

  - **The target is chosen by a rule fixed beforehand.** The MLP runs at targets 35 and 20 in
    parallel: 35 if it passes, else 20, else M.9 closes deferred-with-destination. At 20, a
    point-worm gap-only check restores the anchor.

- **Stage 2, a connectome pilot** on disjoint seeds: the gates per null, the spreads, the cost, and
  the panel's size.

- **Stage 3, the registered panel.** One reading: the wild type's lead over the gap-only null on
  `auc_success`, classified at 0.058, M.8's minimum.

  - **Beside:** the chemical-only null's lead, the frozen floors' gap, and C.3's kinematics
    (undulation, wavelength, speed, reversals) for the wild type and the null. The kinematics show
    whether a lead is about crawling or about taxis.

- **Analysis:** `thermal_body.py` in the gate preflight's shape, with the MLP gate and the kinematic
  comparison.

- **Records:** a D21 note in the roadmap; the tracker's M.9 entry; a logbook.

## Capabilities

**Modified**:

- `continuous-2d-environment`: thermotaxis through the kinematic body reads the body's head.

## Impact

- `scripts/campaigns/generate_thermal_body_configs.py` and the configs under
  `configs/scenarios/thermal_foraging/`.
- `scripts/analysis/thermal_body.py`.
- Tests, the roadmap (D21 note, M.9), the tracker, a logbook and the CHANGELOG.
- No package code is expected to change. Thermal sensing already reads the body's head.
