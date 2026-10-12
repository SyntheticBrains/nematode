## Overview

The body rung, asked where the point worm's wiring answer is clear. Gap-junction placement leads by
+0.096 on the thermal cell (M.8). The question is whether that survives a kinematic body. The
decisions below were taken with the user on 2026-10-12.

## Decisions

### Decision A: The cell

The thermal t35 cell as M.8 and M.10 ran it, through the kinematic body:

- **Unchanged:**
  - Cook 2019;
  - settling to depth 4;
  - 500 steps, a target of 35 foods;
  - the thermotaxis block (Tc 20, gradient 1.25, the zone deltas and damage);
  - 3,000 episodes.
- **Added:** `continuous.body_model: kinematic` and `allow_reversal: true`, which the body requires;
  `action_space: body_drive`; and, for the connectome arms, `entropy_coef` 0.004. That entropy is the per-dimension pressure the
  point worm's 0.05 gives over two action numbers, summed over the body drive's 25, as C.1e set it.
  `signed_speed` is dropped, since the body drive replaces it.
- **The connectome is Cook 2019,** not C.1e's Emmons 2024. The effect under test was measured on
  Cook, and Emmons differs from Cook in exactly four gap-junction pairs, the layer under test. C.1e's
  body results are therefore a loose comparison, not a pairing.
- **Sensing, as C.1e had it.** Temperature, its time derivative and the zone are read at the body's
  head. The left–right difference comes from the synthetic head-sweep around the heading (±0.5 mm),
  not from the body's own head swing. C.3 found that weathervaning survives without the sweep.

### Decision B: The arms

| arm | wiring | learning |
|---|---|---|
| wild type | `wild_type` | PPO, and frozen |
| gap-only null | `rewired_gap_junctions_only` | PPO, and frozen |
| chemical-only null | `rewired_chemical_only` | PPO, and frozen |
| MLP-PPO | — | PPO, and frozen (stage 1 only) |

The MLP is the thermal t35 MLP (w64, two layers, thermotaxis and food sensing) with the body drive,
as C.1d derived its body MLP. Its `entropy_coef` is **0.05**, the setting at which C.1d's body
positive control passed (92.5%). The thermal point-worm MLP's 0.02 and C.1e's MLP yardstick's 0.004
are both untested as body positive controls. The connectome arms keep C.1e's 0.004.

### Decision C: Stage 1, the positive control

The generator emits every arm at targets 35 and 20, so the target rule's choice needs no new configs.

The thermal cell through a body is a new component, so it gets its own positive control before any
connectome arm runs.

- **Seeds and arms:** MLP-PPO and its frozen floor on seeds 9701–9708.
- **It passes on C.1d's rule:**
  - the plateau (last quarter of episodes, success %) beats the floor's, paired by seed, with the
    80% interval above zero;
  - every seed's plateau reaches 30%.
- **Saturation still passes.** A plateau at or above the 90% bar passes as a positive control. Only
  the connectome arms are gated on saturation.
- **A target rule, fixed before it runs** (added at the spec review). C.1d's body MLP reached only
  41.7% on 20 foods at 350 steps, and this cell asks for 35 foods in 500 steps with a body slower than
  the point worm. So stage 1 runs the MLP and its floor at **targets 35 and 20 in parallel** on the
  same seeds:
  - **35 if it passes** C.1d's rule;
  - otherwise **20 if it passes**;
  - otherwise **M.9 stops.** It closes deferred-with-destination with the reason, and no connectome
    arm runs.
- **If target 20 is taken,** the point worm's +0.096 was measured at 35 and is no longer a pairing.
  A point-worm check (the wild type and the gap-only null, learning and frozen, at target 20 on
  stage 2's seeds) restores the anchor at the cell actually run. Its lead is reported as the anchor
  beside the body reading.

### Decision D: Stage 2, the connectome pilot

- **Seeds and arms:** all six connectome arms on seeds 9711–9714, configured as the panel.

- **Gates, per null:**

  - each learner beats its floor;
  - the two learners do not both saturate at the 90% bar.

  Gates are checked with `gate_preflight.py --panel thermal_body`. A null failing either gate leaves
  its reading unreadable. A failing gap-only null leaves the rung unreadable; a failing chemical-only
  null leaves only its beside reading unreadable.

- **It sets** the spread of the primary reading, which sizes the panel between 16 and 64 seeds, and
  the cost. A size over 64 goes back to the user.

### Decision E: Stage 3, the registered panel

**One reading:** the wild type's lead over the gap-only null, `gap = wild type − gap-only null`, on
`auc_success`, paired by seed.

- It is classified by `mc.classify` at **0.058**, 2/3 of the thermal split's committed 0.0865. This
  is M.8's minimum on this cell.
- One reading, so no family correction; the q value is the two-sided Wilcoxon's.

| state | verdict |
|---|---|
| `move_wt` | **the gap advantage survives embodiment**: the wild type leads by at least the minimum |
| `move_null` | **the null leads** by at least the minimum |
| `below` | a significant lead smaller than the minimum |
| `no_move` | **no lead at the minimum**: embodiment washes the gap advantage out |
| `unresolved` | unresolved at this sensitivity |

**Reported beside, never read as verdicts:**

- the wild type against the chemical-only null, which D21 requires reported;

- the frozen floors' own gap, wild type against each null;

- `episodes_to_30pct_success`;

- the MLP's plateau against the connectome's;

- **C.3's kinematics** for the wild type and the gap-only null, on trained and frozen weights:
  undulation frequency, wavelength, speed and reversal fraction, from `body_kinematics_eval.py`.

  The gap-only null rewires the motor neurons' gap junctions too. If the null crawls differently
  even untrained, a lead is at least partly locomotion, not taxis, and the logbook says so.

- the point worm's +0.096 on this cell, as the anchor.

### Decision F: A D21 note

D21 makes the chemical-only null the primary null for wiring contrasts. A dated note scopes it:

- **The rule:** a contrast asking about gap-junction placement reads the gap-only null as primary,
  and the chemical-only null beside it.
- **Why:** the null must differ from the wild type in the layer under question.
- **Precedent:** M.8 did the same on the point worm.

## Risks

- **The positive control may fail.** Thermal sensing through a body that turns by undulating, with
  the sensor on a synthetic sweep, is untested. Stage 1 exists to find this out in hours, not days.
- **Cost.** A body learning run takes about 70–80 minutes and a frozen one about 60, roughly three
  times the point worm. A 6-arm panel at 64 seeds is about 28 hours on 16 workers. The pilot sizes it
  and may cut it.
- **Locomotion confound.** Gap rewiring touches motor-neuron coupling. The kinematic comparison is
  beside to separate that, but cannot remove it.
- **Saturation.** The point worm plateaued at 66% on this cell. Through the body it could saturate
  or floor, and the gates catch either.
