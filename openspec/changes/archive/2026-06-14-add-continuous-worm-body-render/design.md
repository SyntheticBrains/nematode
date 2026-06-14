# Design

A renderer-only visual enhancement. Decisions:

## D1 — Body history lives in the renderer, not the render state

The `Continuous2DRenderer` instance persists across an episode and receives `pos_continuous` every
frame, so it can keep its own `collections.deque(maxlen=N)` of recent positions — no new
`ContinuousRenderState` field and no agent wiring. This keeps the change confined to the renderer and
the snapshot-decoupling contract intact. `N` is sized so the trailing body spans ~`body_length_mm`
(a fixed segment count, e.g. ~16, is sufficient at the render cadence).

## D2 — Episode-reset detection by position jump

`render_frame` does not receive a step index or run number, so a new episode is detected by a
**position discontinuity**: if the new `pos` is farther from the last than any single legal step could
move it (a small multiple of `max_step_mm`, with a safe absolute floor), the history is cleared. This is
robust because legal motion is bounded by `max_step_mm`/frame while a reset teleports the worm to centre.

## D3 — Body geometry: path backbone + undulation overlay, tapered

The backbone is the polyline through the position history (head = newest, tail = oldest). The crawl wave
is a lateral sinusoid added perpendicular to the local backbone direction: `offset(i) = A · sin(phase + k·i)`, with amplitude `A ∝ body_length_mm`, `phase` advanced a fixed increment per frame, and `i` the
segment index. Width tapers head→tail (drawn as decreasing-radius circles or a tapered polygon). The
head end gets a distinct shade/size; the heading indicator is recoloured to a contrasting hue. All
colours are module constants alongside the existing `WORM_*_COLOR`.

## D4 — Cost and scope

Negligible: an ~16-point deque per agent and ~16 segment draws per frame, dwarfed by the per-frame
heatmap lattice. Single-agent continuous renderer only (multi-agent-continuous body rendering is a
documented follow-up, consistent with the renderer's existing single-agent scope). No physics, sensing,
brain, or saved-weights impact — purely how the worm is drawn.
