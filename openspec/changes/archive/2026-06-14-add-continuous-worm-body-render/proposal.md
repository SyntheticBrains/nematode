## Why

The continuous renderer draws the worm as a single filled circle plus a heading line. Two visual
issues surfaced on manual inspection (`--theme pixel_continuous`):

1. The **heading line is nearly the same colour as the body marker** (`(255,230,180)` vs
   `(220,195,160)` — both beige), so the direction indicator is hard to distinguish from the worm.
2. The worm has **no body** — it is a point with a forward line, which reads as an arrow, not a
   crawling *C. elegans*. A realistic body makes the **real-worm behavioural-chemotaxis validation**
   (`T7.validation` — turn-rate vs dC/dt, curving-rate vs bearing) far more legible, and it is the
   project's headline demo artifact.

This is a **non-gating** renderer-visual change (the continuous-fidelity renderer is an explicitly
non-gating capability); it does not touch the simulation, physics, or any brain.

## What Changes

- **Path-following undulating body.** The renderer maintains its own short deque of the worm's recent
  real-valued positions (it persists across frames and receives `pos` each frame — no render-state or
  agent change needed), and draws a tapered body backbone through them, so the body **trails the worm's
  actual path** (curving through where the head has been, like a real worm in a turn). A small sinusoidal
  lateral undulation (phase advancing per frame) is overlaid for the crawl wave; the body length scales
  with `body_length_mm`. The deque is **reset at episode boundaries**, detected by a position-jump
  discontinuity (the reset teleports the worm to centre), so the body never streaks across a reset.
- **Distinct heading indicator colour.** The heading line is recoloured to clearly contrast the
  body/head marker (a direction cue, not blended into the worm).
- **Head/tail distinction + taper.** The head end is drawn distinctly (shade/size) from the tapering
  tail, so orientation is readable.
- **Renderer-only, negligible cost.** ~15–20-point deque per agent and ~15–20 segment draws per frame —
  trivial next to the existing per-frame heatmap lattice sampling. No physics/brain/state changes; the
  worm remains a point kinematically (the body is a pure visual overlay). Single-agent continuous
  renderer only.

## Capabilities

### New Capabilities

<!-- None — this enhances the existing continuous-fidelity renderer. -->

### Modified Capabilities

- `continuous-fidelity-renderer`: the "Worm rendering with continuous heading" requirement currently
  specifies a marker + heading indicator. This change adds a path-following, undulating, tapered **body**
  trailing the head, a **distinct heading-indicator colour**, and a **head/tail distinction** — all
  derived from the worm pose the renderer already receives (no new render-state fields).

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/pygame_renderer.py` — `Continuous2DRenderer`: add a
    body-history deque + undulation phase in `__init__`; update it per frame (append `pos`, reset on a
    position-jump); draw the tapered undulating body in `_render_entities` before the head marker;
    recolour the heading indicator; add head/tail shading. New body/heading colour constants.
- **Tests:** headless (`SDL_VIDEODRIVER=dummy`) render-smoke — a few frames render without error and the
  body history accumulates then resets on a simulated position jump; the heading-indicator and
  body colours differ.
- **Downstream:** none functional. Improves the `T7.validation` behavioural legibility and the demo.
  Manual visual confirmation on `--theme pixel_continuous`.
