# Tasks: C.5 — the body, lawns and frames in the continuous renderer

- [x] 1. **Body geometry**: `body.midline(state, params)`, the world-frame joints head to tail. Tests:
  13 points, the head first, segment spacing one body length over 12, a straight body along its
  frame angle.
- [x] 2. **Snapshot**: `ContinuousRenderState` gains `body_midline`, `body_reversed`, `intake`,
  `on_lawn` and `modulators`, all optional; the agent fills them. Tests: point worm and headless
  unaffected; the kinematic body's midline equals the helper's.
- [x] 3. **Renderer**: the kinematic body (taper, head, reversal tint and tail marker); lawns with
  density shading; the status lines; the `P` overlay. Tests under the dummy driver: a frame renders
  with a body and lawns; grazing darkens a cell; lines appear only where they apply; the toggle.
- [x] 4. **Lawns accepted**: the refusal narrowed to the grid theme. Tests: `pixel_continuous` places
  lawns; `pixel` is refused.
- [x] 5. **Frame recording**: the two flags, the refusal with headless, the PNGs. Tests: two frames
  of four at every-2; the headless refusal.
- [x] 6. **Docs**: a screenshot of a lawn run with the kinematic body; the CHANGELOG; tracker C.5;
  `docs` on watching and recording a run.
- [ ] 7. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
