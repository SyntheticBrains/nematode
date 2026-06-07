# multi-agent-visualization Specification

## Purpose

Defines requirements for rendering multi-agent simulations in the Pygame pixel theme, including per-agent colored sprites, viewport agent switching, pheromone concentration overlays, and multi-agent status bar.

## Requirements

### Requirement: Theme Support for Multi-Agent

The system SHALL select the appropriate renderer for a multi-agent simulation based on the requested theme, creating a PygameRenderer for `--theme pixel`, no renderer for `--theme headless`, and falling back to headless mode for any other theme.

#### Scenario: Pixel theme

- **WHEN** a multi-agent simulation is started with `--theme pixel`

- **THEN** the system SHALL create a PygameRenderer and pass it to the MultiAgentSimulation orchestrator

- **AND** the renderer SHALL display all visible agents within the viewport

#### Scenario: Headless theme

- **WHEN** a multi-agent simulation is started with `--theme headless`

- **THEN** no renderer SHALL be created

- **AND** behavior SHALL be identical to current headless multi-agent mode

#### Scenario: Other theme

- **WHEN** a multi-agent simulation is started with any other theme

- **THEN** the system SHALL log a warning

- **AND** fall back to headless mode

### Requirement: Multi-Agent Frame Rendering

The renderer SHALL draw all agents within the viewport bounds with distinct palette colors and SHALL compute the followed agent's viewport centered on its position while clamping to grid boundaries.

#### Scenario: Render agents in viewport

- **WHEN** `render_multi_agent_frame()` is called with a list of `AgentRenderState` objects

- **THEN** the renderer SHALL draw all agents whose positions fall within the viewport bounds, including dead agents

- **AND** each agent SHALL be rendered with a distinct color from the 8-color palette based on their `color_index`

- **AND** dead agents SHALL be rendered with a semi-transparent gray overlay applied on top of the base colored sprite

#### Scenario: Compute followed agent viewport

- **WHEN** the followed agent's viewport is computed

- **THEN** the viewport SHALL be centered on the followed agent's position

- **AND** the viewport SHALL clamp to grid boundaries (no wraparound)

- **AND** existing rendering layers (background, temperature zones, oxygen zones, toxic zones, food, predators) SHALL render identically to single-agent mode

### Requirement: Agent Switching

The renderer SHALL allow the user to switch the followed agent via arrow keys and number keys, SHALL update the status bar and handle followed-agent removal, and SHALL return the current followed agent ID.

#### Scenario: Next agent

- **WHEN** the user presses the right arrow key during a multi-agent rendering session

- **THEN** the viewport SHALL switch to follow the next agent in the `AgentRenderState` list order (wrapping from last to first)

#### Scenario: Previous agent

- **WHEN** the user presses the left arrow key during a multi-agent rendering session

- **THEN** the viewport SHALL switch to follow the previous agent in the `AgentRenderState` list order (wrapping from first to last)

#### Scenario: Jump to agent by number

- **WHEN** the user presses number key N (1-9) during a multi-agent rendering session

- **THEN** the viewport SHALL switch to follow the agent at index N-1 in the `AgentRenderState` list (if present)

- **AND** if index N-1 does not exist, the keypress SHALL be ignored

#### Scenario: Status bar update on switch

- **WHEN** the followed agent changes via any input method

- **THEN** the status bar SHALL update to show the newly followed agent's metrics

#### Scenario: Followed agent removed

- **WHEN** the followed agent is removed from the `AgentRenderState` list between frames

- **THEN** the renderer SHALL fall back to following the first agent in the list

#### Scenario: Return followed agent ID

- **WHEN** `render_multi_agent_frame()` returns

- **THEN** it SHALL return the current followed agent ID (which may have changed due to keyboard input or fallback)

### Requirement: Colored Agent Sprites

The system SHALL generate per-color tinted directional head sprites and SHALL tint body segments to match each agent's palette color, with backward-compatible defaults.

#### Scenario: Tinted head sprites

- **WHEN** tinted head sprites are generated for a color index

- **THEN** the system SHALL produce 4 directional sprites (up, down, left, right) tinted to the palette color

- **AND** the sprites SHALL be 32x32 pixels with SRCALPHA transparency

- **AND** the `color_index` SHALL be computed as `agent_index % 8` to support arbitrary agent counts

#### Scenario: Tinted body segments

- **WHEN** body segments are rendered for a colored agent

- **THEN** the body color, outline, and highlight SHALL be tinted to match the agent's palette color

- **AND** the existing `draw_body_segment()` function SHALL accept optional color parameters with backward-compatible defaults

### Requirement: Pheromone Overlay

The renderer SHALL compute and display pheromone concentration overlays per viewport cell when the overlay is enabled, and SHALL perform no pheromone computation or rendering when it is disabled.

#### Scenario: Overlay enabled

- **WHEN** the pheromone overlay is enabled (toggled by 'P' key)

- **THEN** the renderer SHALL compute pheromone concentration at each viewport cell for each active pheromone field

- **AND** food pheromone concentration SHALL be displayed as green semi-transparent overlay

- **AND** alarm pheromone concentration SHALL be displayed as red semi-transparent overlay

- **AND** aggregation pheromone concentration SHALL be displayed as blue semi-transparent overlay

- **AND** overlay alpha SHALL be proportional to the concentration value

#### Scenario: Overlay disabled

- **WHEN** the pheromone overlay is disabled (default state)

- **THEN** no pheromone concentration computation or rendering SHALL occur

### Requirement: Viewport Bounds for Arbitrary Agents

The environment SHALL provide `get_viewport_bounds_for(agent_id)` returning viewport bounds centered on the specified agent and clamped to grid boundaries, with `get_viewport_bounds()` delegating to it for the default agent.

#### Scenario: Bounds for specified agent

- **WHEN** `get_viewport_bounds_for(agent_id)` is called on the environment
- **THEN** it SHALL return viewport bounds centered on the specified agent's position
- **AND** the bounds SHALL clamp to grid boundaries
- **AND** the existing `get_viewport_bounds()` SHALL delegate to `get_viewport_bounds_for(DEFAULT_AGENT_ID)`

### Requirement: Orchestrator Rendering Integration

`MultiAgentSimulation` SHALL integrate the renderer as a typed dataclass field, SHALL invoke it once per step with data assembled from the current env and per-agent tracking state, and SHALL handle renderer window closure consistently with single-agent mode.

#### Scenario: Renderer integration and per-step rendering

- **WHEN** `MultiAgentSimulation` is created with a renderer parameter

- **THEN** the renderer field SHALL be a dataclass field typed `PygameRenderer | None = None` with a `TYPE_CHECKING` import guard

- **AND** the renderer SHALL be called once per step via `_render_step()` after all agent effects are applied

- **AND** the rendering data SHALL be assembled from `self.env` (current env reference) and per-agent tracking state

- **AND** `AgentRenderState.direction` SHALL be converted from the `Direction` enum to its `.value` string when building snapshots

#### Scenario: Renderer window closed

- **WHEN** the renderer window is closed by the user

- **THEN** `renderer.closed` SHALL become True

- **AND** `MultiAgentSimulation` SHALL set a `_renderer_closed` flag and return early from `run_episode()`

- **AND** `_run_multi_agent()` SHALL check `sim.renderer_closed` after each episode and break the run loop

- **AND** this SHALL match the single-agent window close behavior

#### Scenario: Show last frame only flag

- **WHEN** a multi-agent simulation is started with `--show-last-frame-only`

- **THEN** the flag SHALL be ignored for multi-agent rendering (render every frame or not at all)

### Requirement: Multi-Agent Status Bar

The multi-agent status bar SHALL display the followed agent's full metrics, a compact per-agent summary, and an agent switcher indicator, word-wrapping lines that exceed the available window width.

#### Scenario: Status bar rendering

- **WHEN** the multi-agent status bar is rendered

- **THEN** it SHALL display the followed agent's full metrics (step, food collected, HP, satiety)

- **AND** it SHALL display a compact summary line for all agents (ID, food count, alive/dead)

- **AND** it SHALL display the agent switcher indicator ("Following: agent_N [N/total]", "\</> to switch, 1-9 to jump")

- **AND** lines exceeding the available window width SHALL be word-wrapped across multiple lines, with character-level breaking as fallback for single tokens wider than the window
