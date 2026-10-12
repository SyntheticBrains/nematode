## Overview

Record a kinematic-body run at sub-step resolution, then replay it as a worm on a plate looks.
Recording and drawing are separate, so a reviewer sees exactly the episode that ran, at any speed.

## Decisions

### Decision A: The recording

**What is recorded.** The kinematic body already offers a per-step record of its sub-steps
(`posture_log`): each sub-step's time, curvatures, head and frame angle. With `--record-replay`,
`run_simulation.py` turns the log on for each run and writes `<dir>/<session>-run<NNNN>.npz` when the
run ends:

| array | shape | content |
|---|---|---|
| `substep_time` | (S,) | worm-seconds from the episode start |
| `substep_curvature` | (S, 12) | segment curvatures, head to tail |
| `substep_head` | (S, 2) | head position (mm) |
| `substep_frame` | (S,) | frame angle (rad) |
| `step_of_substep` | (S,) | the environment step each sub-step belongs to |
| `step_reversed`, `step_intake`, `step_on_lawn`, `step_satiety` | (T,) | per step |
| `step_modulators` | (T, 2) | serotonin and PDF; NaN without the field |
| `step_lawn_density` | (T, C) | every lawn cell's density; absent without lawns |
| `lawn_cells`, `lawn_centres`, `lawn_radius_mm`, `lawn_cell_mm` | — | static lawn geometry |
| `world_size_mm`, `body_length_mm`, `step_seconds`, `meta` | — | arena, body, step length, and a JSON string with config, seed and weights |

**Recording is inert.** The posture log is an existing evaluation record; turning it on draws
nothing. It is refused without the kinematic body, and with `--manyworlds` or multi-agent runs.

### Decision B: Playback in worm time

The player keeps a clock in worm-seconds, advanced by wall time × speed (1×, 10× or 60×). At each
frame it finds the two sub-steps around the clock and interpolates between them:

- curvature, head and frame angle linearly;
- the frame angle along its shorter arc.

Sub-steps are 0.25 worm-s apart at the default 20 per 5 s step, so 1× plays smoothly at 60 fps.
Per-step quantities (lawn densities, intake, status) hold until the next step.

**Pacing** follows Wormlight's pacer: wall-clock gaps are capped at 0.1 s, so a paused or dragged
window does not jump.

**Keys:**

| key | effect |
|---|---|
| `1` `2` `3` | speed 1×, 10×, 60× |
| space | pause |
| → | step one sub-step while paused |
| `R` | restart |
| `C` | camera follow on/off (on by default) |
| `+` `−` | zoom |
| `Esc` | quit |

### Decision C: The look, bright-field

Formulas are adapted from Wormlight's renderer (Apache-2.0), with brightness inverted for
bright-field, and pre-computed with numpy where Wormlight shades per pixel.

- **Agar.** A pale warm base with value-noise mottle at three scales and fine specks, baked once to
  a surface. A darker meniscus band at the walls, `exp(−d / 0.6 mm)`.
- **Lawns.**
  - Each lawn's cell densities are upsampled to a fine raster and Gaussian-smoothed. They then set
    the opacity of a slightly darker, cooler film.
  - A rim band of extra opacity runs inside each lawn's edge, since real lawns are thickest at the
    rim.
  - Grazed cells thin the film.
- **Body.**
  - **Shape:** a Catmull–Rom spline through the 13 joints, 8 subdivisions per segment, with
    mirrored end points.
  - **Width:** an elliptical profile, `r(s) = R · sin(acos(u))` with `u` from −1 to 1 along the body
    scaled by M/2 + 0.2 so the tips stay open (Boyle, Berri & Cohen 2012). R is a fifteenth of the
    body length for a worm-like aspect.
  - **Shading:** a translucent fill with darker edges, built from nested offset polygons, and a
    faint halo.
  - **Bulbs:** two darker pharyngeal bulbs at 5.8% and 11% of the length, drawn only when the body
    spans more than about 60 pixels.
- **Label,** always on: `kinematic body · 12 segments · <speed>× · t = <worm time>`. The visuals must
  not suggest a fidelity the model lacks. Status (intake, lawn occupancy, satiety, modulators) sits
  under it.

### Decision D: Scope

- **Kinematic body only;** a recording without a body is refused at load.
- **Lawns are drawn when present.** A body-only recording, such as the thermal body cell, plays on
  bare agar.
- **Not drawn:** predators, temperature or oxygen zones, and multi-agent runs.

### Decision E: Demo configs

`generate_lawn_configs.py` gains a body variant of D.1's MLP lawn arm, learning and frozen: the
kinematic body, reversal, the body drive, and entropy 0.05 (C.1d's body MLP).

## Risks

- **Pygame polygon shading is coarser than shaders.** Nested offset polygons approximate a tube; at
  extreme zoom the facets show. Anti-aliased outlines hide most of it.
- **Recording size.** About 1.5 MB per 500-step run at 20 sub-steps, compressed. The flag's help says
  to record a few runs.
