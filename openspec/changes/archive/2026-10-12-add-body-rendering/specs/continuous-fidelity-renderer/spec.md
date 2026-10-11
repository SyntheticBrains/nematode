## MODIFIED Requirements

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

## ADDED Requirements

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
