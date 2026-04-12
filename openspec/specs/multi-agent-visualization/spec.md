## Purpose

Defines requirements for rendering multi-agent simulations in the Pygame pixel theme, including per-agent colored sprites, viewport agent switching, pheromone concentration overlays, and multi-agent status bar.

## MODIFIED Requirement: Theme Support for Multi-Agent

- WHEN a multi-agent simulation is started with `--theme pixel`

- THEN the system SHALL create a PygameRenderer and pass it to the MultiAgentSimulation orchestrator

- AND the renderer SHALL display all visible agents within the viewport

- WHEN a multi-agent simulation is started with `--theme headless`

- THEN no renderer SHALL be created

- AND behavior SHALL be identical to current headless multi-agent mode

- WHEN a multi-agent simulation is started with any other theme

- THEN the system SHALL log a warning

- AND fall back to headless mode

## ADDED Requirement: Multi-Agent Frame Rendering

- WHEN `render_multi_agent_frame()` is called with a list of `AgentRenderState` objects

- THEN the renderer SHALL draw all agents whose positions fall within the viewport bounds, including dead agents

- AND each agent SHALL be rendered with a distinct color from the 8-color palette based on their `color_index`

- AND dead agents SHALL be rendered with a semi-transparent gray overlay applied on top of the base colored sprite

- WHEN the followed agent's viewport is computed

- THEN the viewport SHALL be centered on the followed agent's position

- AND the viewport SHALL clamp to grid boundaries (no wraparound)

- AND existing rendering layers (background, temperature zones, oxygen zones, toxic zones, food, predators) SHALL render identically to single-agent mode

## ADDED Requirement: Agent Switching

- WHEN the user presses the right arrow key during a multi-agent rendering session

- THEN the viewport SHALL switch to follow the next agent in the `AgentRenderState` list order (wrapping from last to first)

- WHEN the user presses the left arrow key during a multi-agent rendering session

- THEN the viewport SHALL switch to follow the previous agent in the `AgentRenderState` list order (wrapping from first to last)

- WHEN the user presses number key N (1-9) during a multi-agent rendering session

- THEN the viewport SHALL switch to follow the agent at index N-1 in the `AgentRenderState` list (if present)

- AND if index N-1 does not exist, the keypress SHALL be ignored

- WHEN the followed agent changes via any input method

- THEN the status bar SHALL update to show the newly followed agent's metrics

- WHEN the followed agent is removed from the `AgentRenderState` list between frames

- THEN the renderer SHALL fall back to following the first agent in the list

- WHEN `render_multi_agent_frame()` returns

- THEN it SHALL return the current followed agent ID (which may have changed due to keyboard input or fallback)

## ADDED Requirement: Colored Agent Sprites

- WHEN tinted head sprites are generated for a color index

- THEN the system SHALL produce 4 directional sprites (up, down, left, right) tinted to the palette color

- AND the sprites SHALL be 32x32 pixels with SRCALPHA transparency

- AND the `color_index` SHALL be computed as `agent_index % 8` to support arbitrary agent counts

- WHEN body segments are rendered for a colored agent

- THEN the body color, outline, and highlight SHALL be tinted to match the agent's palette color

- AND the existing `draw_body_segment()` function SHALL accept optional color parameters with backward-compatible defaults

## ADDED Requirement: Pheromone Overlay

- WHEN the pheromone overlay is enabled (toggled by 'P' key)

- THEN the renderer SHALL compute pheromone concentration at each viewport cell for each active pheromone field

- AND food pheromone concentration SHALL be displayed as green semi-transparent overlay

- AND alarm pheromone concentration SHALL be displayed as red semi-transparent overlay

- AND aggregation pheromone concentration SHALL be displayed as blue semi-transparent overlay

- AND overlay alpha SHALL be proportional to the concentration value

- WHEN the pheromone overlay is disabled (default state)

- THEN no pheromone concentration computation or rendering SHALL occur

## ADDED Requirement: Viewport Bounds for Arbitrary Agents

- WHEN `get_viewport_bounds_for(agent_id)` is called on the environment
- THEN it SHALL return viewport bounds centered on the specified agent's position
- AND the bounds SHALL clamp to grid boundaries
- AND the existing `get_viewport_bounds()` SHALL delegate to `get_viewport_bounds_for(DEFAULT_AGENT_ID)`

## ADDED Requirement: Orchestrator Rendering Integration

- WHEN `MultiAgentSimulation` is created with a renderer parameter

- THEN the renderer field SHALL be a dataclass field typed `PygameRenderer | None = None` with a `TYPE_CHECKING` import guard

- AND the renderer SHALL be called once per step via `_render_step()` after all agent effects are applied

- AND the rendering data SHALL be assembled from `self.env` (current env reference) and per-agent tracking state

- AND `AgentRenderState.direction` SHALL be converted from the `Direction` enum to its `.value` string when building snapshots

- WHEN the renderer window is closed by the user

- THEN `renderer.closed` SHALL become True

- AND `MultiAgentSimulation` SHALL set a `_renderer_closed` flag and return early from `run_episode()`

- AND `_run_multi_agent()` SHALL check `sim.renderer_closed` after each episode and break the run loop

- AND this SHALL match the single-agent window close behavior

- WHEN a multi-agent simulation is started with `--show-last-frame-only`

- THEN the flag SHALL be ignored for multi-agent rendering (render every frame or not at all)

## ADDED Requirement: Multi-Agent Status Bar

- WHEN the multi-agent status bar is rendered

- THEN it SHALL display the followed agent's full metrics (step, food collected, HP, satiety)

- AND it SHALL display a compact summary line for all agents (ID, food count, alive/dead)

- AND it SHALL display the agent switcher indicator ("Following: agent_N [N/total]", "\</> to switch, 1-9 to jump")

- AND lines exceeding the available window width SHALL be word-wrapped across multiple lines, with character-level breaking as fallback for single tokens wider than the window
