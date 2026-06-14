## MODIFIED Requirements

### Requirement: Worm rendering with continuous heading

The continuous renderer SHALL draw the worm as a **path-following, tapered, undulating body** plus a distinct **heading indicator**, both derived from the continuous pose (rather than the grid renderer's four-way `Direction` head sprite). The renderer SHALL maintain a short history of the worm's recent real-valued positions (accumulated across frames from the `pos_continuous` it already receives — no new render-state field) and draw the body backbone through that history so the body **trails the worm's actual path** (curving through where the head has been). A small sinusoidal lateral undulation (phase advancing per frame) SHALL be overlaid for the crawl wave, with the body length scaled by `body_length_mm`. The body history SHALL be **reset at an episode boundary**, detected by a position-jump discontinuity (larger than any single legal step). The **head end SHALL be visually distinct** from the tapering tail, and the **heading indicator SHALL be a colour that clearly contrasts** the body/head marker. The worm remains a point kinematically — the body is a pure visual overlay and SHALL NOT affect physics, sensing, or any brain.

#### Scenario: Heading indicator follows continuous heading

- **WHEN** the worm has heading `heading_rad`
- **THEN** the heading indicator SHALL point along `heading_rad`, rotate smoothly as the heading changes, and be drawn in a colour that visibly contrasts the worm's body/head marker

#### Scenario: Body trails the worm's path

- **WHEN** the worm moves over several frames (including through a turn)
- **THEN** the body SHALL be drawn through the worm's recent real-valued positions (tapered head→tail, with a sinusoidal undulation overlay), so it curves along the path the head travelled rather than rigidly snapping to the current heading

#### Scenario: Body history resets at an episode boundary

- **WHEN** the worm position jumps discontinuously (e.g. reset to the arena centre at a new episode), by more than any single legal step
- **THEN** the renderer SHALL clear the accumulated body history so the body does not streak across the reset
