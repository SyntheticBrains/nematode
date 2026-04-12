## 1. Environment Viewport Support

- [x] 1.1 Add `get_viewport_bounds_for(agent_id: str)` method to `DynamicForagingEnvironment`
- [x] 1.2 Refactor existing `get_viewport_bounds()` to delegate to `get_viewport_bounds_for(DEFAULT_AGENT_ID)`
- [x] 1.3 Add unit tests for `get_viewport_bounds_for()` with multiple agent positions

## 2. Agent Color Palette and Sprites

- [x] 2.1 Define `AGENT_COLOR_PALETTE` constant (8 colors) in `sprites.py`
- [x] 2.2 Add `create_tinted_head_sprites(pg, tint_color)` function returning dict of 4 directional sprites
- [x] 2.3 Add optional `body_color`, `outline_color`, `highlight_color` kwargs to `draw_body_segment()` with backward-compatible defaults
- [x] 2.4 Add `create_dead_agent_overlay(pg)` for gray semi-transparent overlay
- [x] 2.5 Add unit tests for tinted sprite size and surface format

## 3. Rendering Data Contract

- [x] 3.1 Add `AgentRenderState` frozen dataclass to `pygame_renderer.py` (agent_id, position, body, direction as str via Direction.value, alive, hp, max_hp, foods_collected, satiety, max_satiety, color_index)
- [x] 3.2 Add unit test for `AgentRenderState` construction

## 4. Multi-Agent Renderer Extension

- [x] 4.1 Add tinted sprite caching to `PygameRenderer.__init__()` (lazily populated dict\[int, dict[str, Surface]\])
- [x] 4.2 Add `_pump_multi_agent_events()` method handling agent switching (left/right arrows to cycle, 1-9 keys to jump) and pheromone toggle ('P')
- [x] 4.3 Add `_render_pheromone_overlay()` method computing per-cell concentration for active pheromone fields
- [x] 4.4 Add `_render_multi_agent_entities()` method rendering all agents with colored sprites
- [x] 4.5 Add `_render_multi_agent_status_bar()` with followed agent metrics, all-agent summary, and switcher text ("← → to switch, 1-9 to jump")
- [x] 4.6 Add `render_multi_agent_frame()` public method accepting `current_step` (for pheromone queries), compositing all layers, and returning followed_agent_id
- [ ] 4.7 Add smoke test for `render_multi_agent_frame()` with mock pygame (2-8 agents)
- [ ] 4.8 Add test for agent switching (arrow keys cycle, number keys jump)
- [ ] 4.9 Add test for pheromone overlay toggle

## 5. MultiAgentSimulation Integration

- [x] 5.1 Add `renderer: PygameRenderer | None = None` dataclass field with `TYPE_CHECKING` import guard (after existing defaulted fields)
- [x] 5.2 Add `_followed_agent_id` and `_renderer_closed` runtime fields (init=False)
- [x] 5.3 Add `_render_step()` method building `AgentRenderState` list from `self.env` and per-agent trackers, converting `Direction` enum to `.value` string, passing `current_step` to renderer for pheromone queries
- [x] 5.4 Insert render call in step loop after learning phase; check `renderer.closed` after call and set `_renderer_closed` flag + return early from `run_episode()`
- [x] 5.5 Add `renderer_closed` property for `_run_multi_agent()` to check after each episode
- [x] 5.6 Add unit test for `_render_step()` data assembly

## 6. run_simulation.py Integration

- [x] 6.1 Add renderer creation when `theme == Theme.PIXEL` in `_run_multi_agent()`
- [x] 6.2 Add theme validation — warn and fall back to headless for unsupported themes
- [x] 6.3 Add renderer cleanup (close) after run loop completes
- [x] 6.4 Check `sim.renderer_closed` after each episode and break the run loop if true
- [x] 6.5 Ignore `--show-last-frame-only` for multi-agent (render every frame or not at all)
- [ ] 6.6 Manual smoke test: multi-agent foraging with `--theme pixel`

## 7. Verification

- [ ] 7.1 All existing tests pass (`uv run pytest -m "not nightly"`)
- [ ] 7.2 Pre-commit hooks pass (`uv run pre-commit run -a`)
- [ ] 7.3 Manual test: agents render with distinct colors
- [ ] 7.4 Manual test: arrow keys and number keys switch followed agent
- [ ] 7.5 Manual test: 'P' toggles pheromone overlay
- [ ] 7.6 Manual test: dead agents render with gray overlay
- [ ] 7.7 Manual test: closing window terminates simulation
