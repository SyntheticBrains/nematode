# Tasks: Continuous-Substrate Fidelity Renderer

Implements the `add-continuous-fidelity-renderer` change. Realises the deferred
`phase6-tracking` **T6.render** seed (non-gating). Single-agent only; multi-agent-continuous +
gif/mp4 export are documented follow-ups. Order is dependency-first: selection + decoupling
scaffolding, then the renderer skeleton + parity layers, then the fidelity overlays, then
export/figures, then tests.

## 1. Theme selection + decoupled state scaffolding

- [x] 1.1 Add `Theme.PIXEL_CONTINUOUS` to `env/theme.py` (+ a `THEME_SYMBOLS` entry mirroring the `PIXEL` block).
- [x] 1.2 Add `pixel_continuous` to the `scripts/run_simulation.py` `--theme` choices + the renderer-selection branch (single-agent path).
- [x] 1.3 Add a public `AdaptiveChemosensor.background` property (delegating to `LeakyIntegrator.background`) + a `last_readout` field set in `adapt()` (`agent/adaptive_sensor.py`).
- [x] 1.4 Add a frozen `ContinuousRenderState` dataclass (`env/pygame_renderer.py`): `pos`, `heading_rad`, `left_sample`/`right_sample`, `sweep`, `adaptive_background`/`adaptive_readout`/`adaptive_mode`, + status fields (HP, satiety, foods_collected/target, in_danger, temperature + zone, oxygen + zone, step/max).

## 2. Continuous renderer skeleton + grid-renderer parity

- [x] 2.1 New sibling `Continuous2DRenderer` (`env/pygame_renderer.py`): `__init__(world_size_mm, pixels_per_mm=~12, ...)`, full-arena window sizing, `_world_to_pixel(x, y)` (mm→pixel, y-inversion). Grid `PygameRenderer` byte-unchanged.
- [x] 2.2 Parity layers ported to continuous coords: background; temperature zones; oxygen zones; stationary-predator toxic zones; the full status bar. Reuse `create_zone_overlay` *colours* (zone overlays become arena-region fills, not 32×32 cell blits); the status bar + zone overlays are `PygameRenderer` *methods*, so re-implement them for continuous coords (optionally extract the status-bar field-formatting to a shared free function to avoid divergence).
- [x] 2.3 Entities: food + predator sprites blit centered at `_world_to_pixel`; the worm as a marker + heading line from `heading_rad`.
- [x] 2.4 `agent/agent.py`: render-dispatch branch for `Theme.PIXEL_CONTINUOUS` → `_get_continuous_renderer()` + `_render_step_continuous()` (builds the `ContinuousRenderState` snapshot — calls `_continuous_lateral_offsets`, reads the adaptive accessor). Mirrors `_get_pygame_renderer`/`_render_step_pygame`.

## 3. Fidelity overlays

- [x] 3.1 Concentration heatmap: sample a selectable field getter over an N×N lattice (default food, N~64) → `matplotlib.cm` colormap → `pygame.surfarray` surface → scaled alpha-blit. **Cache the sampled lattice** and recompute only on a source-change signal (food consumed/respawned, predators moved) or a field-selector switch — not every frame.
- [x] 3.2 Gradient quiver: coarse lattice (~12×12) → `get_separated_gradients` arrows scaled by strength; default off.
- [x] 3.3 Sensor zones: klinotaxis left/right sample points (from the snapshot) + sweep marker; predator detection/damage rings (`pygame.draw.circle` at radius·zoom); optional contact cone.
- [x] 3.4 Adaptive-sensor readout appended to the status bar (background, readout, mode) from the snapshot.
- [x] 3.5 **Keyboard + window event handling for the continuous renderer** (NEW — the single-agent renderer has none today; the grid pheromone `P`-toggle is multi-agent-only): window-close (follow the existing `pump_events` shape) + a quiver toggle key + optional heatmap-toggle / field-selector keys.

## 4. Export + offline figures

- [x] 4.1 `scripts/export_screenshot.py`: `_export_continuous(output_path)` (set `SDL_VIDEODRIVER=dummy`, stage a `Continuous2DEnvironment`, render one frame, `pygame.image.save`); update the module docstring + a `--continuous` flag.
- [x] 4.2 New `report/continuous_figures.py`: `plot_trajectory`, `plot_field_heatmap` (matplotlib `imshow` over the lattice), `plot_gradient_quiver` (matplotlib `quiver`) — headless static figures for logbooks.

## 5. Tests

- [x] 5.1 `tests/.../env/test_continuous_renderer.py` (guard with the existing `requires_pygame` skip + `SDL_VIDEODRIVER=dummy`): `_world_to_pixel` math (origin / corner / centre, y-inversion, zoom scaling).
- [x] 5.2 Snapshot builder: left/right klinotaxis sample points match a direct `_continuous_lateral_offsets` call for a known `(pos, heading, sweep)`; `AdaptiveChemosensor.background` returns the integrator background.
- [x] 5.3 Smoke render: build the renderer (dummy driver) + render one frame on a staged `Continuous2DEnvironment`; assert no exception + expected `_screen` dimensions.
- [x] 5.4 Offline figures: generate trajectory / heatmap / quiver PNGs headlessly; assert files written + non-empty.

## 6. Docs, tracker, pre-merge gate

- [x] 6.1 Tick `phase6-tracking` **T6.render** + the roadmap T6 row note (renderer no longer deferred); brief usage note (the `--theme pixel_continuous` command + `--continuous` screenshot).
- [ ] 6.2 `openspec validate add-continuous-fidelity-renderer --strict` clean; full suite (`uv run pytest -m "not nightly"`) green.
- [ ] 6.3 Targeted `pre-commit run --files <changed>` clean; full `pre-commit run -a` before push; open PR.
