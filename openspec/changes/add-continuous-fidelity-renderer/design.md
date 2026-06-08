## Context

The continuous-2D environment (`env/continuous_2d.py`) stores the worm's float truth in
`AgentState.pos_continuous` + `heading_rad`, float `env.foods`, and a `Continuous2DParams`
(`world_size_mm` default 50; `grid_size = round(world_size_mm)`). All field getters
(`get_food_concentration`, `get_predator_concentration`, `get_temperature`,
`get_oxygen_concentration`, `get_separated_gradients`, pheromone) now accept real-valued
positions. The adaptive sensor lives on the agent (`agent._adaptive_food`) with a private
`_integrator.background`. The klinotaxis head-sweep sample points are produced by the pure
function `agent._continuous_lateral_offsets(pos, heading_rad, sweep, grid_size)`.

The existing pygame renderer (`env/pygame_renderer.py`) is grid-coupled: `render_frame` and
every `_render_*` method use an integer **scrolling** viewport and
`_cell_to_pixel(grid_x:int, grid_y:int, viewport)` (y inverted over a cell count). It was made
float-safe for T6 only by snapping continuous state to cells. The multi-agent path already
demonstrates the clean decoupling pattern: a frozen `AgentRenderState` snapshot built by the
orchestrator, so the renderer never reads agent internals.

`pygame` (optional `pixel` extra) and `matplotlib` (main dep) are already available;
`imageio`/`ffmpeg` are not.

## Goals / Non-Goals

**Goals:**

- A continuous-substrate pygame renderer with sub-cell worm motion, a full-arena plate view,
  configurable zoom, and at least full feature-parity with the grid renderer.
- Four fidelity overlays (concentration heatmap, gradient quiver, klinotaxis + predator sensor
  zones, adaptive-sensor readout).
- Decoupling from agent internals via a frozen render-state snapshot + a public adaptive
  accessor.
- Single-frame PNG export + offline matplotlib figure helpers (trajectory / heatmap / quiver).

**Non-Goals:**

- gif/mp4 animation export (would add a media-encoding dependency) — deferred.
- Multi-agent-continuous rendering — deferred (needs a parallel snapshot builder in
  `multi_agent.py`).
- Any change to the grid `PygameRenderer` behaviour (it stays byte-unchanged), the headless
  path, 3D, or a web/game-engine stack.

## Decisions

### D1 — A sibling `Continuous2DRenderer`, not a subclass; a `world→pixel` map, not an extended `_cell_to_pixel`

`PygameRenderer`'s methods are hardwired to the integer scrolling viewport and the cell-count
y-inversion. Subclassing would force fragile overrides of nearly every method. Instead add a
**sibling** `Continuous2DRenderer` in the same module. It reuses the module-level helper
*functions* (`create_sprites`, `draw_body_segment`, and the `create_zone_overlay` zone
*colours*); the status bar and zone overlays are renderer *methods* on `PygameRenderer`
(`_render_status_bar`, `_render_temperature_zones`, …) — not free helpers — so the continuous
renderer re-implements them in continuous coordinates (the status-bar field-formatting can be
extracted to a shared free function to avoid divergence; the zone overlays become arena-region
fills rather than the grid's 32×32 per-cell blits). It provides its own `_world_to_pixel(x, y)`:

```text
px = x * pixels_per_mm
py = (world_size_mm - y) * pixels_per_mm   # y-up → y-down over the world height
```

The view is the **whole arena** (no agent-following scroll — the worm is a point on a plate),
window = `world_size_mm * pixels_per_mm` square + `STATUS_BAR_HEIGHT`. `pixels_per_mm` is a
constructor arg (default ~12 → a 600×600 plate) so zoom is tunable. **Alternative rejected:**
extending `_cell_to_pixel` to accept floats — it still carries the scrolling-viewport semantics
that are wrong for a plate view.

### D2 — Concentration heatmap via field-sampling + matplotlib colormap + `surfarray`

Sample the selected field getter over an N×N lattice spanning the arena, normalise, apply a
`matplotlib.cm` colormap → RGBA uint8 array → `pygame.surfarray.make_surface` → scale to the
window → alpha-blit. This is the dense-array analogue of the existing per-cell alpha-blit
pattern (`_render_temperature_zones`, `_render_pheromone_overlay`). N is configurable (default
~64) to bound the per-frame field-call cost. **Caching:** an N×N lattice is ~N² field calls per
frame, each summing over all sources — too much at 30 FPS to recompute every frame. The field
is static within an episode except when sources change (food consumed/respawned, predators
move), so **cache the sampled lattice and recompute only on a source-change signal** (or on a
coarse cadence / the selected-field switch). The field is selectable (food default; also
predator / temperature / oxygen / pheromone). **Note:** the heatmap (raw field value) and the
ported categorical *zone* bands (comfort/danger) are complementary and both selectable, so they
don't collide by default.

### D3 — Decouple via a frozen `ContinuousRenderState` snapshot + a public adaptive accessor

Mirror the `AgentRenderState` pattern. Add a frozen `ContinuousRenderState` dataclass holding
`pos`, `heading_rad`, `left_sample`/`right_sample` (klinotaxis), `sweep`, `adaptive_background`,
`adaptive_readout`, `adaptive_mode`, and the existing status fields. The **agent** builds it in
`_render_step_continuous` (calling `_continuous_lateral_offsets` itself, reading the adaptive
state via a new public accessor). The renderer reads only the snapshot + `env` (for
fields/foods/predators) — never `agent` privates. To avoid reaching into `_integrator`, add a
public `AdaptiveChemosensor.background` property (delegating to the already-public
`LeakyIntegrator.background`) and a `last_readout` field set in `adapt()`.

### D4 — Gradient quiver is coarse + keyboard-toggleable, off by default; the continuous renderer needs its own keyboard event handling

A full-resolution quiver would issue O(N²) field calls per frame at 30 FPS. Use a coarse
lattice (e.g. 12×12), default off, gated behind a keyboard toggle.

**Caveat:** the keyboard-toggle the grid renderer has (the pheromone `P`-key) lives only in the
**multi-agent** event pump (`_pump_multi_agent_events`); the **single-agent** path
(`render_frame` / `pump_events`) only handles window-close. So the continuous (single-agent)
renderer must **add its own keyboard event handling** — this is new code, not a reuse: a key
toggle for the quiver, and (optionally) keys to switch the heatmap field and toggle the heatmap.
The window-close handling can follow the existing `pump_events` shape.

### D5 — PNG + matplotlib figures now; gif/mp4 deferred

Ship a single-frame continuous PNG (extend `scripts/export_screenshot.py` with an
`SDL_VIDEODRIVER=dummy` headless capture) and a new `report/continuous_figures.py`
(`plot_trajectory`, `plot_field_heatmap` via `imshow`, `plot_gradient_quiver` via matplotlib
`quiver`) — the offline analogue of the live overlays, used for logbook/validation figures.
matplotlib is already a dep, so no new dependency. gif/mp4 (needs `imageio`) is deferred to a
follow-up.

### D6 — Theme selection via a new `Theme.PIXEL_CONTINUOUS` value

An explicit new enum value (plus a `--theme pixel_continuous` choice and a render-dispatch
branch in `agent.py`) is discoverable and matches the existing theme selection axis.
**Alternative rejected:** auto-detecting the continuous env under `PIXEL` — implicit, couples
renderer choice to env type. The renderer is action-mode-agnostic (reads `pos_continuous`,
populated on both discrete and continuous-action paths).

## Risks / Trade-offs

- **Heatmap/quiver per-frame cost** (N² field calls) → bound N (heatmap ~64, quiver ~12), make
  quiver toggle-off-by-default; the heatmap can also be a keyboard toggle.
- **pygame video backend in CI** → tests use `SDL_VIDEODRIVER=dummy` + the existing
  `requires_pygame` skip guard (as `test_multi_agent_renderer.py` / `export_screenshot.py` do).
- **Drift from the grid renderer** (two renderers to maintain) → reuse the shared module
  helpers (`create_sprites`/`create_zone_overlay`/`draw_body_segment`/status logic) rather than
  duplicating; the grid renderer is untouched.
- **Snapshot vs live state** → the snapshot is rebuilt every render step from current agent/env
  state, so it can't go stale within a frame.

## Migration Plan

- Purely additive: new theme value, new renderer class, new snapshot/accessor, new figure
  module. Existing `PIXEL`/`headless`/text themes and the grid renderer are unchanged.
- Rollback: `--theme pixel`/`headless` continue to work exactly as before.

## Open Questions

- **Zoom config location** — `pixels_per_mm` on the renderer constructor (renderer-local) vs on
  `Continuous2DParams` (config-exposed). Lean renderer-local with an optional config override;
  decide during implementation (affects whether `continuous-2d-environment` is a modified
  capability).
- **Heatmap default field + colormap** — food + a perceptually-uniform map (viridis);
  finalise during implementation.
