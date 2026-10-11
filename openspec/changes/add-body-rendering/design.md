## Overview

Draw what the simulation runs, show lawns, and let a reviewer save frames. Nothing reaches a
headless run.

## Decisions

### Decision A: The body from its state, through the snapshot

The renderer reads only its snapshot and the environment, which the existing requirement mandates.
So the agent puts what the renderer needs into `ContinuousRenderState`:

- `body_midline`: the world-frame points from head to tail, 13 joints for 12 segments;
- `body_reversed`: whether the last step ran tail-to-head.

Both are `None` for the point worm.

- **The geometry is public.** `body.py` gains `midline(state, params)`, which returns those points.
  It uses the same `_segment_angles` and `_rotate` the physics uses, so what is drawn is what is
  simulated.
- **How it is drawn:** a tube tapered from head to tail through the joints, with a head marker and
  the heading line as today.
- **Reversals.** A reversing step draws the body in a distinct tint and puts a small arrow at the
  tail.
- **Omega turns** need no special case: a large curvature draws as a coiled body.
- **The point worm is unchanged:** it keeps the path-following trail.

### Decision B: The posture overlay

`P` toggles a small panel in the arena's corner:

- the 12 segment curvatures as a bar profile from head to tail;
- the first two eigenworm amplitudes, from `validation/posture.py`'s vendored basis.

It is off by default, and drawn only for the kinematic body.

### Decision C: Lawns

**The refusal is narrowed.** The environment's lawn refusal now names only the grid `pixel` theme,
which cannot be pointed at a continuous environment anyway.

**How a lawn is drawn,** under the field overlays and above the soil:

- each lawn cell is a filled square, its colour blended from the soil to a lawn green by its
  remaining density;
- each lawn's edge is a circle outline.

Densities are read from `env.lawn_field` each frame, so grazing shows as it happens.

### Decision D: Internal state in the status bar

The snapshot gains these optional fields, each `None` when it does not apply:

- `intake`: the last step's intake;
- `on_lawn`: whether the worm is on a lawn;
- `modulators`: the serotonin and PDF concentrations.

The status bar adds a line for each that is present. The agent reads the concentrations through the
brain's public `modulator_concentrations` property, so the renderer's decoupling holds.

### Decision E: Frame recording

`run_simulation.py` gains:

- `--record-frames <dir>`: implies the `pixel_continuous` theme, and is refused with `headless`;
- `--record-every <n>`, default 1.

After each rendered frame, the renderer saves `<dir>/run<run>-step<step>.png`. Files are named by
run and step so a reviewer can find an episode's moment.

- **Without a window.** Under `SDL_VIDEODRIVER=dummy` it records with no window, as the existing
  screenshot export does.
- **No animation export.** Assembling a GIF or MP4 is left to the reviewer's tools (for example
  `ffmpeg -framerate 10 -pattern_type glob -i '<dir>/run1-*.png' out.mp4`), as the user chose. No new
  dependency.

### Decision F: Headless is byte-identical

The renderer and its snapshot are built only under a pixel theme. So every change here is inert
under `--theme headless`, and the campaign runner always uses headless. The body helper is new and
only read.

## Risks

- **Rendering cost.** Drawing every lawn cell each frame is a few hundred rectangles: 4 lawns of
  about 21 cells each, or 8 of about 5. That is negligible at the renderer's 30 fps cap.
- **Frame recording fills a disk if left on for a whole run.** `--record-every` and a short `--runs`
  are the guard, and the flag's help says so.
