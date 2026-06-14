# Tasks

## 1. Implementation

- [ ] 1.1 Add body/heading colour constants (distinct heading indicator; head vs tail shade) alongside
  the existing `WORM_*_COLOR` in `pygame_renderer.py`.
- [ ] 1.2 `Continuous2DRenderer.__init__`: add a `deque(maxlen=N)` body-history buffer, a `_last_pos`
  tracker, and an undulation `_phase` accumulator.
- [ ] 1.3 Per frame (in `render_frame` / `_render_entities`): clear the history on a position-jump
  discontinuity (> a small multiple of `max_step_mm`, with a safe floor), else append `pos`; advance
  `_phase`.
- [ ] 1.4 Draw the body before the head marker: a tapered backbone through the history with a sinusoidal
  lateral undulation overlay (amplitude ∝ `body_length_mm`); then the distinct head marker + the
  recoloured heading indicator.

## 2. Tests (headless, `SDL_VIDEODRIVER=dummy`)

- [ ] 2.1 A few frames render without error on a staged `Continuous2DEnvironment`; `_screen` dimensions
  hold (reuse the existing renderer-smoke guard).
- [ ] 2.2 The body history accumulates over successive frames and **resets** when `pos` jumps
  discontinuously (simulated episode boundary).
- [ ] 2.3 The heading-indicator colour differs from the body/head marker colour.

## 3. Validate + gate

- [ ] 3.1 `openspec validate add-continuous-worm-body-render --strict`.
- [ ] 3.2 Targeted `pre-commit` on changed files; full `pre-commit run -a` before push;
  `uv run pytest -m "not nightly"` for the renderer suite.
- [x] 3.3 **Manual visual confirmation** on `--theme pixel_continuous` (load a trained model): the worm
  shows an undulating body trailing its path, a distinct heading colour, and readable head/tail.

**Visual confirmed (2026-06-15):** path-following undulating body trails the path with a travelling crawl wave; distinct head + contrasting heading line; clean reset per episode. Body drawn as a connected tapered tube (links + rounded joints), not disjoint segments. User approved the look.
