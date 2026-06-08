## Why

The continuous-2D substrate shipped its physics (float sources, Euclidean/continuous-field sensing, static Fick-shaped gradient geometry, the adaptive/biphasic chemosensory sensor), but the **fidelity renderer was deferred as a non-gating seed**. Today the only way to watch a continuous run is the existing pygame `PIXEL` renderer, which was merely made *float-safe* — it **snaps the continuous worm and sources to integer cells** and shows **none of the new fidelity** (no concentration field, no gradient geometry, no sensor state). So the substrate that the architecture comparison will run on cannot be visually inspected, demonstrated, or turned into the spatial figures the upcoming real-worm behavioural validation needs.

This change builds a proper continuous-substrate renderer: sub-cell worm motion on a full-arena plate view, with the chemosensory fidelity (concentration heatmap, gradient field, sensor zones, adaptive-sensor state) overlaid — and the offline matplotlib figures (trajectory / field / quiver) for logbooks.

## What Changes

- **A `Continuous2DRenderer`** (sibling of the existing `PygameRenderer`) that renders the continuous-2D arena at sub-cell fidelity via a `world→pixel` map with a configurable pixels-per-mm zoom and a **full-arena, non-scrolling** view. The worm renders as a point + heading line (continuous `heading_rad`).
- **Feature parity** with the grid renderer, ported to continuous coordinates: background, temperature zones, oxygen zones, stationary-predator toxic zones, and the full status bar — nothing the grid theme shows is lost.
- **Four new fidelity overlays:** a **concentration-field heatmap** (selectable field; matplotlib colormap via `surfarray`), a **gradient quiver** (coarse-grid arrows; keyboard-toggleable, off by default for perf), **klinotaxis sensor zones** (the head-sweep sample points) + **predator detection/damage rings**, and an **adaptive-sensor readout** (background, readout, mode) in the status bar.
- **A new `Theme.PIXEL_CONTINUOUS`** value selecting the renderer (explicit `--theme pixel_continuous`).
- **Clean state exposure:** a frozen `ContinuousRenderState` snapshot (mirroring the existing `AgentRenderState` decoupling pattern) + a public `AdaptiveChemosensor.background` accessor, so the renderer never reaches into agent internals.
- **Static figure helpers** (`report/continuous_figures.py`: trajectory / field heatmap / gradient quiver) for logbook/validation figures, and a continuous **PNG** export path in `scripts/export_screenshot.py`.
- **Out of scope (deferred):** gif/mp4 animation export (would add an `imageio`/`ffmpeg` dependency) and the **multi-agent** continuous render path (needs a parallel snapshot builder) — both documented as follow-ups.

## Capabilities

### New Capabilities

- `continuous-fidelity-renderer`: A pygame renderer for the continuous-2D substrate — `world→pixel` sub-cell mapping with a full-arena view and configurable zoom; grid-renderer feature parity (background, temperature/oxygen/toxic zones, status bar) ported to continuous coordinates; and four fidelity overlays (concentration heatmap, gradient quiver, klinotaxis + predator sensor zones, adaptive-sensor readout). Selected via `Theme.PIXEL_CONTINUOUS`; fed a frozen `ContinuousRenderState` snapshot so it stays decoupled from agent internals. Includes a single-frame PNG export and offline matplotlib figure helpers.

### Modified Capabilities

None. The pixels-per-mm zoom is renderer-local (a `Continuous2DRenderer` constructor argument), so no existing capability's requirements change. If a follow-up exposes the zoom on `Continuous2DParams`, that becomes a small `continuous-2d-environment` amendment then.

## Impact

- **Code:** new `env/pygame_renderer.py` `Continuous2DRenderer` + `ContinuousRenderState`; `env/theme.py` (`PIXEL_CONTINUOUS`); `agent/agent.py` render dispatch + `_render_step_continuous` snapshot builder; `agent/adaptive_sensor.py` public `background` + `last_readout`; `scripts/run_simulation.py` (`--theme` choice + selection); `scripts/export_screenshot.py` (continuous PNG); new `report/continuous_figures.py`.
- **Dependencies:** none new — `pygame` (the optional `pixel` extra) and `matplotlib` (main dep) are already present; `imageio`/`ffmpeg` are explicitly deferred.
- **Substrate / brains:** the existing grid `PygameRenderer` is byte-unchanged; the renderer is action-mode-agnostic (reads `pos_continuous`/`heading_rad`, populated on both discrete and continuous-action paths). Headless/batch runs are unaffected.
- **Cross-tranche:** realises the deferred `T6.render` seed and produces the spatial figures the real-worm behavioural validation (next L2 pass) references.
- **Out of scope:** gif/mp4 export, multi-agent-continuous rendering, a web/game-engine stack (against the batch/headless/reproducibility posture).
