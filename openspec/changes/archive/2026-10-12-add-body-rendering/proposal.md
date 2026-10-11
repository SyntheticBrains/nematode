## Why

C.5 (tracker, SHOULD) asks the continuous renderer to draw the body the simulation actually runs.
Today it does not.

- **The body drawn is not the simulated one.** The renderer draws a decorative trail of the head's
  last positions with a cosmetic wave, even when `body_model: kinematic` runs a 12-segment body.
- **Lawns cannot be watched at all.** A lawn environment refuses every pixel renderer, so the B.3 and
  D.1 cells can only run headless.

Human reviewers checking behaviour, such as a crawl, a reversal, an omega turn, dwelling on a lawn
or leaving one, have no way to see it. Decided with the user on 2026-10-12, alongside the
per-connection signs rung.

## What Changes

- **The kinematic body is drawn from its state.** The 12 segments come from the body's curvature,
  head position and frame angle. Head and tail are distinct, and a reversal is marked. Omega turns
  show as the posture they are.
  - The point worm keeps today's trail.
  - A public helper on the body module returns the world-frame midline, so the renderer needs no
    private body code.
- **A posture overlay,** toggled with `P`: the body's curvature profile, and its first two
  eigenworm amplitudes from the vendored basis.
- **Lawns are drawn.** Each lawn's cells are shaded by their remaining density, inside the lawn's
  edge. `pixel_continuous` no longer refuses lawns. The grid `pixel` renderer still does, since
  lawns exist only on the continuous substrate.
- **Status lines for internal state:** the last step's intake and whether the worm is on a lawn
  (lawn cells only), and the serotonin and PDF concentrations (brains with the modulator field only).
- **Frame recording.** `--record-frames <dir>` with `--record-every <n>` saves a PNG of every n-th
  rendered step. It needs no new dependency. Under a dummy SDL driver it runs without a window.
- **Headless is unchanged.** Every addition lives in the renderer and in the snapshot the agent
  builds for it, and neither exists under `--theme headless`.

## Capabilities

**Modified**:

- `continuous-fidelity-renderer`: the body drawn from the kinematic body's state, lawns, internal
  state, the posture overlay and frame recording.
- `patchy-lawns`: `pixel_continuous` draws lawns.

## Impact

- `env/body.py`: the public midline helper.
- `env/pygame_renderer.py`: the body, lawns, status lines, the overlay and the recorder.
- `env/env.py`: the lawn refusal narrowed to the grid renderer.
- `agent/agent.py`: the snapshot's new fields.
- `scripts/run_simulation.py`: the two recording flags.
- Tests (headless, under the dummy driver), a screenshot in the docs, the CHANGELOG and the tracker.
