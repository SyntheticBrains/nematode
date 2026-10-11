# continuous-fidelity-renderer Specification

## Purpose

Defines the continuous-2D substrate fidelity renderer: a pygame renderer (a sibling of the grid `PygameRenderer`) selected via `Theme.PIXEL_CONTINUOUS` that draws the continuous-2D arena at sub-cell fidelity so continuous runs can be inspected, demonstrated, and turned into spatial figures. It maps real-valued world coordinates to pixels through a camera-aware `world→pixel` transform with a configurable zoom — a full-arena plate view by default plus a keyboard-toggleable agent-following camera — and ports the grid renderer's feature parity (background, temperature/oxygen/toxic zones, status bar) to continuous coordinates. Over that it overlays the chemosensory fidelity: a concentration-field heatmap, a gradient quiver, klinotaxis + predator sensor zones, and the adaptive-sensor readout. The renderer is fed a frozen `ContinuousRenderState` snapshot so it stays decoupled from agent internals, and the capability also provides a single-frame PNG export and offline matplotlib figure helpers (trajectory / field heatmap / gradient quiver) for logbook and validation figures.

## Requirements

### Requirement: Continuous renderer selection

The system SHALL provide a `Theme.PIXEL_CONTINUOUS` theme that selects a continuous-substrate
renderer for the continuous-2D environment, alongside the existing themes. The grid `PIXEL`
renderer and the `HEADLESS` path SHALL be unaffected.

#### Scenario: Continuous theme selected

- **WHEN** a single-agent simulation on a continuous-2D environment runs with `--theme pixel_continuous`
- **THEN** the system SHALL create the continuous renderer (`Continuous2DRenderer`) and render each step against it

#### Scenario: Existing themes unchanged

- **WHEN** a simulation runs with `--theme pixel` or `--theme headless`
- **THEN** the grid `PygameRenderer` / headless behaviour SHALL be byte-unchanged from before this capability

### Requirement: World-to-pixel mapping with full-arena view and zoom

The continuous renderer SHALL map real-valued world coordinates (millimetres) to pixels via a
`world→pixel` transform with a configurable pixels-per-mm zoom, render the **whole arena** by
default (no agent-following scroll), and invert the y-axis (world y-up → screen y-down) over the
world height. The worm SHALL be drawn at its real-valued `pos_continuous` (sub-cell), not
snapped to a grid cell.

#### Scenario: Sub-cell position mapping

- **WHEN** the worm is at a fractional position (e.g. `(10.4, 20.7)` mm)
- **THEN** it SHALL be drawn at the corresponding sub-cell pixel location (not rounded to a cell)

#### Scenario: Full arena visible by default

- **WHEN** the renderer is created for a `world_size_mm` arena
- **THEN** the drawing surface SHALL span the whole arena (`world_size_mm * pixels_per_mm` per side, plus the status bar), showing the whole plate with no scrolling viewport

### Requirement: Agent-following camera toggle

The continuous renderer SHALL provide a keyboard-toggleable agent-following camera in addition
to the default full-arena view. When following, the `world→pixel` transform SHALL zoom in and
keep the worm centred, clamped so the view never shows past the plate edges; all layers (zones,
heatmap, sensor overlays, entities) SHALL transform consistently with the active camera. This
keeps the worm legible on large worlds where it is a small dot in the full-arena view.

#### Scenario: Following centres the worm

- **WHEN** the agent-following camera is toggled on and the worm is away from the plate edges
- **THEN** the worm SHALL be drawn at the centre of the arena viewport at the zoomed scale, and the other layers SHALL align to the same camera

#### Scenario: Camera clamps at the plate edge

- **WHEN** the agent-following camera is on and the worm is near a plate edge
- **THEN** the view SHALL clamp so it does not show past the arena bounds (the worm moves off-centre rather than revealing out-of-plate space)

#### Scenario: Default is full-arena

- **WHEN** the renderer starts
- **THEN** the camera SHALL be the full-arena view until the follow toggle is pressed

### Requirement: Grid-renderer feature parity

The continuous renderer SHALL render, in continuous coordinates, every layer the grid pixel
renderer shows so no information is lost: the background, the temperature comfort/danger zone
overlays, the oxygen zone overlays, the stationary-predator toxic (damage-radius) zone, the
food and predator sprites, the worm, and the full status bar (step, food, HP, satiety,
danger state, temperature + zone, oxygen + zone).

#### Scenario: Parity overlays render

- **WHEN** a continuous run has thermotaxis, aerotaxis, or stationary predators enabled
- **THEN** the corresponding temperature-zone, oxygen-zone, and toxic-zone overlays SHALL render in continuous coordinates with the same categorical colours as the grid renderer

#### Scenario: Status bar parity

- **WHEN** the continuous renderer draws the status bar
- **THEN** it SHALL show the same fields as the grid renderer's status bar (step/max, food/target, HP, satiety, danger state, temperature + zone name, oxygen + zone name)

### Requirement: Worm rendering with continuous heading

The continuous renderer SHALL draw the worm with a distinct **heading indicator** derived from the continuous pose. Under the kinematic body model, the renderer SHALL draw the **body the simulation runs**: a tapered tube through the body's world-frame midline, computed from its curvature, head position and frame angle, with the head end distinct from the tail and a step run tail-to-head marked as a reversal. Under the point worm, the renderer SHALL draw a **path-following, tapered, undulating body** through a short history of the worm's recent real-valued positions, with a small sinusoidal lateral undulation whose phase advances per frame, the body length scaled by `body_length_mm`, and the history **reset at an episode boundary** detected by a position jump larger than any single legal step; that body is a pure visual overlay and SHALL NOT affect physics, sensing, or any brain. In both cases the **head end SHALL be visually distinct** from the tail, and the **heading indicator SHALL be a colour that clearly contrasts** the body and head marker.

#### Scenario: Heading indicator follows continuous heading

- **WHEN** the worm has heading `heading_rad`
- **THEN** the heading indicator SHALL point along `heading_rad`, rotate smoothly as the heading changes, and be drawn in a colour that visibly contrasts the worm's body/head marker

#### Scenario: Body trails the worm's path

- **WHEN** the point worm moves over several frames (including through a turn)
- **THEN** the body SHALL be drawn through the worm's recent real-valued positions (tapered head→tail, with a sinusoidal undulation overlay), so it curves along the path the head travelled rather than rigidly snapping to the current heading

#### Scenario: Body history resets at an episode boundary

- **WHEN** the point worm's position jumps discontinuously (e.g. reset to the arena centre at a new episode), by more than any single legal step
- **THEN** the renderer SHALL clear the accumulated body history so the body does not streak across the reset

#### Scenario: The kinematic body is drawn from its state

- **WHEN** a kinematic body is rendered
- **THEN** the drawn midline SHALL pass through the 13 world-frame joints the body module's midline helper returns for that body state, head first

#### Scenario: A reversal is marked

- **WHEN** the kinematic body's last step ran tail-to-head
- **THEN** the body SHALL be drawn in the reversal tint with a marker at the tail

### Requirement: Concentration-field heatmap overlay

The continuous renderer SHALL provide a concentration-field heatmap that samples a selectable
field getter over a lattice spanning the arena, maps values through a colormap, and alpha-blits
the result. The default field SHALL be the food concentration; predator / temperature / oxygen
/ pheromone fields SHALL be selectable. The lattice resolution SHALL be bounded so the
per-frame cost stays acceptable.

#### Scenario: Food heatmap by default

- **WHEN** the heatmap overlay is enabled
- **THEN** it SHALL sample `get_food_concentration` over the arena lattice and render a colormapped, alpha-blended field showing the gradient geometry (e.g. Fick vs exponential)

#### Scenario: Selectable field

- **WHEN** a different field (predator / temperature / oxygen / pheromone) is selected for the heatmap
- **THEN** the heatmap SHALL sample that field's getter instead

### Requirement: Gradient quiver overlay

The continuous renderer SHALL provide a gradient-vector (quiver) overlay sampling the gradient
on a coarse lattice and drawing up-gradient arrows whose length reflects gradient strength. It
SHALL be keyboard-toggleable and default to off to keep the live render responsive.

#### Scenario: Quiver toggled on

- **WHEN** the quiver overlay is toggled on
- **THEN** the renderer SHALL draw coarse-lattice arrows from `get_separated_gradients` pointing up-gradient, scaled by gradient strength

#### Scenario: Quiver off by default

- **WHEN** the renderer starts
- **THEN** the quiver overlay SHALL be off until toggled

### Requirement: Sensor-zone overlays

The continuous renderer SHALL draw the klinotaxis head-sweep sample points (left/right,
perpendicular to heading at the configured sweep amplitude) and predator detection/damage
range rings around each predator.

#### Scenario: Klinotaxis sample points

- **WHEN** klinotaxis sensing is active
- **THEN** the renderer SHALL draw the left and right head-sweep sample points relative to the worm's heading and sweep amplitude

#### Scenario: Predator range rings

- **WHEN** predators are present
- **THEN** the renderer SHALL draw detection-radius and damage-radius rings around each predator at the world-scaled radii

### Requirement: Adaptive-sensor readout

When the adaptive chemosensory sensor is enabled, the continuous renderer SHALL display its
state — the tracked background, the current readout value, and the readout mode — in the status
bar.

#### Scenario: Adaptive readout shown

- **WHEN** the adaptive sensor is enabled
- **THEN** the status bar SHALL include the tracked background, the current readout, and the readout mode (e.g. fold_change / contrast / log)

#### Scenario: Adaptive disabled

- **WHEN** the adaptive sensor is disabled
- **THEN** the adaptive readout SHALL be omitted (no error)

### Requirement: Decoupled render-state snapshot

The continuous renderer SHALL be fed a frozen render-state snapshot (`ContinuousRenderState`)
built by the agent/orchestrator each step, and SHALL read only that snapshot and the
environment (for fields, food, and predators) — never the agent's internal sensor objects. A
public accessor SHALL expose the adaptive sensor's tracked background so the snapshot builder
does not reach into private integrator state.

#### Scenario: Renderer reads only the snapshot + env

- **WHEN** the renderer renders a frame
- **THEN** it SHALL obtain worm pose, klinotaxis sample points, and adaptive-sensor state from the `ContinuousRenderState` snapshot, not from agent private attributes

#### Scenario: Public adaptive accessor

- **WHEN** the snapshot is built
- **THEN** the adaptive background SHALL be read via a public accessor on the adaptive sensor (not its private integrator)

### Requirement: Continuous frame export and offline figures

The change SHALL provide a single-frame PNG export of a continuous render (headless-capable via
a dummy SDL video driver) and offline matplotlib figure helpers — trajectory, field heatmap,
and gradient quiver — for logbook/validation figures. Animated (gif/mp4) export is out of scope.

#### Scenario: PNG export

- **WHEN** the continuous screenshot export is run
- **THEN** it SHALL render one continuous frame headlessly and save it as a PNG

#### Scenario: Offline figures

- **WHEN** an offline figure helper is called with a continuous environment / trajectory
- **THEN** it SHALL produce the corresponding matplotlib figure (trajectory, field heatmap, or gradient quiver) without requiring a display

### Requirement: Lawn rendering

Under the lawn food model, the continuous renderer SHALL draw every lawn: each lawn cell filled with
a colour blended from the soil toward lawn green by the cell's remaining density, read each frame,
and each lawn's edge outlined.

#### Scenario: Grazing shows

- **WHEN** a lawn cell's density falls between two frames
- **THEN** the cell SHALL be drawn closer to the soil colour in the later frame

### Requirement: Internal-state status lines

The status bar SHALL show the last step's intake and whether the worm is on a lawn when the lawn
food model is on, and the serotonin and PDF concentrations when the brain carries the modulator
field. Each line SHALL be absent when its quantity does not apply.

#### Scenario: Lines appear only where they apply

- **WHEN** a point-food run with a brain without modulators is rendered
- **THEN** no intake, lawn or modulator line SHALL appear

### Requirement: Posture overlay

A keyboard toggle (`P`) SHALL show, for the kinematic body, the 12 segment curvatures from head to
tail and the body's first two eigenworm amplitudes. It SHALL be off by default.

#### Scenario: The overlay toggles

- **WHEN** `P` is pressed while a kinematic body is rendered
- **THEN** the posture panel SHALL be drawn on the next frame, and pressing `P` again SHALL remove it

### Requirement: Run frame recording

`run_simulation.py` SHALL accept `--record-frames <dir>` and `--record-every <n>` (default 1). With
them, the continuous renderer SHALL save every n-th rendered frame as a PNG named by run and step.
`--record-frames` SHALL be refused with the `headless` theme, and SHALL run without a window under a
dummy SDL video driver. `--record-follow` SHALL start a recording on the agent-following camera, and
`--render-ppm` SHALL set the continuous renderer's scale, live or recorded.

#### Scenario: Frames are saved

- **WHEN** a run renders 4 steps with `--record-every 2`
- **THEN** 2 PNG files SHALL be written to the directory, named by run and step

#### Scenario: Headless refuses recording

- **WHEN** `--record-frames` is given with `--theme headless`
- **THEN** the command SHALL exit with an error naming the conflict
