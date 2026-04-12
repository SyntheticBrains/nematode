## Context

The `PygameRenderer` class (491 lines) renders single-agent simulations with a 6-layer compositing pipeline: background soil → temperature zones → oxygen zones → toxic zones → entities (food, predators, nematode) → status bar. The viewport (default 11x11 cells at 32px each) follows the agent via `env.get_viewport_bounds()`. Sprites are procedurally generated at init time.

`MultiAgentSimulation` orchestrates 2-8 agents in a shared `DynamicForagingEnvironment`. The environment stores per-agent state in `env.agents` dict (`AgentState` with position, body, direction, hp, alive). Pheromone fields are shared point-source models (`PheromoneField`) with `get_concentration(pos, step)`.

Currently `MultiAgentSimulation.run_episode()` has no rendering — only stdout logging and CSV export.

## Goals

- Render multi-agent simulations in the Pygame pixel theme
- Visually distinguish agents with per-agent colors
- Allow switching which agent the viewport follows
- Optionally visualize pheromone concentration fields
- Zero regression on single-agent rendering

## Non-Goals

- Text-based themes (ASCII, emoji, rich) for multi-agent — out of scope
- Full-grid view showing entire environment without viewport — performance concern on 100x100 grids
- Recording/replay of frames to video — future work
- Agent trails/path visualization — future work

## Decisions

### Decision 1: Extend PygameRenderer, don't subclass

Add `render_multi_agent_frame()` as a parallel method to `render_frame()`.

**Rationale:** The rendering primitives (background, zones, coordinate conversion, status bar framework) are identical for both modes. A subclass would duplicate 80% of the code. The single-agent `render_frame()` is untouched — no risk of regression.

**Alternative rejected:** Separate `MultiAgentPygameRenderer` class — would require extracting shared primitives into a base class, which is more refactoring than needed for the gain.

### Decision 2: Renderer receives lightweight snapshots, not agent objects

A `AgentRenderState` frozen dataclass carries the minimum data needed to render each agent (position, body, direction, alive, metrics, color_index). Built by the orchestrator before each render call.

**Rationale:** Keeps the renderer decoupled from `QuantumNematodeAgent` internals (brain, STAM, reward calculator). The renderer doesn't need to know about RL — just positions and metrics.

### Decision 3: Orchestrator owns the render call

`MultiAgentSimulation._render_step()` is called once per step from the step loop, after all agents have moved and effects applied. Individual agents never call the renderer.

**Rationale:** Multi-agent rendering must show all agents simultaneously. Per-agent rendering would be confusing and waste frames.

### Decision 4: Pheromone overlay is togglable, default off

Computing pheromone concentration per viewport cell (≈121 cells × 3 fields × up to 200 sources each) is O(72k) per frame. Acceptable at 30fps in Python but non-trivial. Default off avoids surprising slowdowns; 'P' key toggles it.

**Rationale:** Users who want to observe pheromone dynamics can enable it. Users debugging agent behavior get a clean view by default.

### Decision 5: Agent color palette with 8 fixed colors, cycled for >8 agents

Agents are assigned colors by index from a fixed 8-color palette. Color 0 is the default cream (matching single-agent appearance). Colors 1-7 are distinct hues (blue, green, red, orange, purple, cyan, yellow). For agents beyond 8, the palette cycles (`color_index = agent_index % 8`).

**Rationale:** 8 colors provides good visual distinction for the typical 2-8 agent range. Cycling handles arbitrary agent counts without added complexity. Two agents with the same color are still distinguishable by position and status bar labeling.

### Decision 6: Agent switching via arrow keys and number keys

Left/right arrow keys cycle through agents sequentially (wrapping at boundaries). Number keys 1-9 jump directly to agent_0 through agent_8. Events are processed in the renderer's `pump_events()` and the new followed agent ID is returned to the orchestrator.

**Rationale:** Arrow keys provide intuitive navigation regardless of agent count (especially >9). Number keys are a shortcut for quick access. The renderer doesn't own agent management — it just reports the user's intent. The orchestrator decides whether the requested agent exists.

### Decision 7: Window close terminates the simulation

Closing the Pygame window terminates the current episode and stops the simulation, matching single-agent behavior (see `runners.py:782`, `run_simulation.py:778`).

**Rationale:** Consistency with single-agent UX. Users expect closing the window to stop the program, not silently continue headless.

## Risks / Trade-offs

1. **Pheromone overlay performance**: Worst case 72k evaluations per frame. Mitigated by: default off, viewport-only computation, potential source pruning (skip sources >2× viewport radius from center).

2. **Pygame event loop in step loop**: The renderer processes events synchronously, adding ~1ms per step. Negligible compared to brain inference time.

3. **Sprite memory**: 8 colors × 4 directions × 32x32px = 32 head sprites. Plus body color variants. Total \<100KB — negligible.
