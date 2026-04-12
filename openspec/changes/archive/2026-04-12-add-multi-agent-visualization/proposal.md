## Why

Multi-agent simulations currently run headless only — there is no visual rendering. Single-agent mode has a mature Pygame pixel renderer (layered surfaces, viewport tracking, procedural sprites at 32x32px), but the `MultiAgentSimulation` orchestrator has no rendering integration. This makes it impossible to visually observe emergent behaviors (aggregation patterns, alarm response cascades, pheromone trail following) in real time, which is critical for debugging coordination issues and communicating scientific findings.

Phase 4 evaluation revealed several emergent phenomena (social feeding clusters, alarm evasion, food sharing) that were only measurable via CSV metrics. Being able to watch agents interact in a pheromone field would have accelerated hypothesis testing and made results more compelling for documentation.

## What Changes

### 1. Multi-Agent Pygame Rendering

Extend the existing `PygameRenderer` with a `render_multi_agent_frame()` method (parallel to `render_frame()` — single-agent path untouched):

- Renders all agents visible within the viewport with distinct per-agent colors
- Reuses existing rendering layers (background, temperature zones, oxygen zones, toxic zones, food, predators)
- Adds per-agent colored head sprites and body segments (8-color palette, cycling for >8 agents)
- Dead/frozen agents rendered with gray overlay
- Followed agent gets a subtle highlight marker
- Closing the Pygame window terminates the simulation (matches single-agent behavior)

### 2. Viewport with Agent Switching

- Viewport follows one agent at a time (default: first agent), identical to single-agent behavior
- Keyboard controls: left/right arrow keys to cycle between agents, number keys 1-9 for direct jump
- `get_viewport_bounds_for(agent_id)` added to environment for flexible viewport targeting

### 3. Pheromone Overlay (Togglable)

- New rendering layer between zones and entities showing pheromone concentration as colored semi-transparent overlays
- Food pheromone = green, alarm = red, aggregation = blue
- Toggled with 'P' key, default off
- Computes concentration per viewport cell per frame from existing `PheromoneField.get_concentration()`

### 4. Multi-Agent Status Bar

- Shows followed agent's full metrics (step, food, HP, satiety, danger)
- Compact all-agent summary line (ID, food count, alive/dead)
- Agent switcher indicator ("Following: agent_0 [1/4] — Press 1-4 to switch")

### 5. Theme Restrictions

- Only PIXEL and HEADLESS themes supported for multi-agent
- Other themes produce a warning and fall back to HEADLESS

## Capabilities

**Modified**: `environment-simulation` (viewport for arbitrary agents), `multi-agent` (renderer integration in step loop, rendering data flow).

**No new capabilities** — this extends the existing rendering system.

## Impact

**Core code:**

- `quantumnematode/env/env.py` — `get_viewport_bounds_for(agent_id)` method
- `quantumnematode/env/pygame_renderer.py` — `AgentRenderState` dataclass, `render_multi_agent_frame()`, pheromone overlay, agent switching events, multi-agent status bar
- `quantumnematode/env/sprites.py` — Agent color palette, tinted sprite generation, colored body segment support, dead agent overlay
- `quantumnematode/agent/multi_agent.py` — Optional renderer parameter, `_render_step()`, followed-agent tracking
- `scripts/run_simulation.py` — Renderer creation in `_run_multi_agent()`, theme validation/fallback

**Tests:**

- New test file for multi-agent rendering (viewport bounds, sprite generation, render smoke tests, agent switching)

## Breaking Changes

None. Single-agent rendering is untouched. Multi-agent headless mode unchanged. New rendering is additive.

## Backward Compatibility

All existing behavior preserved. Multi-agent defaults to headless. PIXEL theme must be explicitly requested via `--theme pixel` CLI flag.

## Dependencies

None beyond Pygame (already an optional dependency used by single-agent PIXEL theme).

## Post-Implementation Notes

- Unicode arrow characters (`←` `→` `—`) rendered as squares in the monospace pygame font. Replaced with ASCII equivalents (`</>`, `--`) in the status bar.
- Long status bar text (agent switcher indicator, all-agent summary) overflowed the window. Added word-wrapping that splits lines to fit within the available pixel width.
- Extended `scripts/export_screenshot.py` with `--multi-agent` and `--both` flags for documentation screenshots.
- Updated `README.md` and `CONTRIBUTING.md` to document multi-agent features, visualization controls, and testing examples.
